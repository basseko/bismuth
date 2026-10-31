// mce: the Metal 4 side of the compatibility layer (see mce.h for the contract). Nothing here runs unless -Dmcopt.metal4=true.
// Built without ARC like the rest. Every MTL4 API used here is macOS 26+; mce_init checks the OS and the device first.
#pragma clang diagnostic ignored "-Wunguarded-availability-new"
#pragma clang diagnostic ignored "-Wunguarded-availability"
#import "mce.h"
#import <objc/runtime.h>
#include <os/lock.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

int mce_metal4;
static int wanted;

int mce_enable(int on) {
	wanted = on;
	return on;
}

// ---- state ----
#define MAX_ALLOCS 48
#define MAX_CHUNKS 64
#define CHUNK_SIZE (8u << 20)
#define RING_ALIGN 256
#define TABLE_BUFFERS 31
#define TABLE_TEXTURES 48
#define TABLE_SAMPLERS 16

static struct {
	Ctx *ctx;
	id<MTLDevice> device;
	id<MTL4CommandQueue> queue;
	id<MTLSharedEvent> timeline;  // signaled with each commit's value after all work before it completed
	uint64_t submitted;  // the last value a commit was given
	os_unfair_lock lock;  // everything below (encoding is mostly on the render thread, but resources are made and released anywhere)
	struct { id<MTL4CommandAllocator> a; uint64_t last; int busy; } allocs[MAX_ALLOCS];
	int nallocs;
	NSMutableArray *cbPool;  // committed MTL4CommandBuffers, reused once complete
	NSMutableArray *cbPoolValues;
	NSMutableArray *tablePool;  // argument tables not attached to an open encoder
	id<MTLResidencySet> set;  // everything made resident through the layer
	int setDirty;
	struct { id<MTLBuffer> buf; uint64_t size, used, last; int inFrame; } chunks[MAX_CHUNKS];
	int nchunks, cur;
	struct { id obj; uint64_t value; } *grave;  // value 0: released during the frame being encoded
	int ngrave, capgrave;
	NSMutableSet *logged;
	CAMetalLayer *layer;
} M = {.lock = OS_UNFAIR_LOCK_INIT};

static uint64_t completedValue(void) { return M.timeline.signaledValue; }

static os_unfair_lock logLock = OS_UNFAIR_LOCK_INIT;
static const char *loggedMsgs[64];
static int nLoggedMsgs;
void mce_log_once(const char *msg) {
	os_unfair_lock_lock(&logLock);
	for (int i = 0; i < nLoggedMsgs; i++) {
		if (loggedMsgs[i] == msg) { os_unfair_lock_unlock(&logLock); return; }
	}
	if (nLoggedMsgs < 64) loggedMsgs[nLoggedMsgs++] = msg;
	os_unfair_lock_unlock(&logLock);
	fprintf(stderr, "mcopt-metal4: %s\n", msg);
}

static void logUnported(const char *what, SEL sel) {
	@autoreleasepool {
		NSString *k = [NSString stringWithFormat:@"%s %s", what, sel_getName(sel)];
		os_unfair_lock_lock(&logLock);
		if (!M.logged) M.logged = [NSMutableSet new];
		int fresh = ![M.logged containsObject:k];
		if (fresh) [M.logged addObject:k];
		os_unfair_lock_unlock(&logLock);
		if (fresh) fprintf(stderr, "mcopt-metal4: unported Metal 3 call -[%s %s] dropped (port the call site to mce_*)\n", what, sel_getName(sel));
	}
}

// ---- setup ----
int mce_init(Ctx *ctx) {
	if (!wanted) return 0;
	if (@available(macOS 26.0, *)) {
		id<MTLDevice> d = ctx->device;
		if (![d supportsFamily:MTLGPUFamilyMetal4]) {
			fprintf(stderr, "mcopt-metal4: -Dmcopt.metal4=true but %s has no Metal 4 support: using Metal 3\n", d.name.UTF8String);
			return 0;
		}
		@autoreleasepool {
			M.ctx = ctx;
			M.device = d;
			MTL4CommandQueueDescriptor *qd = [[MTL4CommandQueueDescriptor new] autorelease];
			qd.label = @"mcopt metal4";
			// Commit feedback (GPU times, completion blocks) on a queue of its own: by default it would go to the main queue, which is
			// Minecraft's render thread on macOS and never idles, so the handlers would never run.
			qd.feedbackQueue = dispatch_queue_create("mcopt.metal4.feedback", dispatch_queue_attr_make_with_qos_class(DISPATCH_QUEUE_SERIAL, QOS_CLASS_USER_INTERACTIVE, 0));
			NSError *e = nil;
			M.queue = [d newMTL4CommandQueueWithDescriptor:qd error:&e];
			if (!M.queue) {
				fprintf(stderr, "mcopt-metal4: no MTL4CommandQueue (%s): using Metal 3\n", e ? e.description.UTF8String : "?");
				return 0;
			}
			M.timeline = [d newSharedEvent];
			M.timeline.signaledValue = 0;
			MTLResidencySetDescriptor *rd = [[MTLResidencySetDescriptor new] autorelease];
			rd.label = @"mcopt metal4 resources";
			rd.initialCapacity = 4096;
			M.set = [d newResidencySetWithDescriptor:rd error:&e];
			if (!M.set) {
				fprintf(stderr, "mcopt-metal4: no residency set (%s): using Metal 3\n", e ? e.description.UTF8String : "?");
				[M.queue release];
				M.queue = nil;
				return 0;
			}
			[M.queue addResidencySet:M.set];
			[M.set requestResidency];
			M.cbPool = [NSMutableArray new];
			M.cbPoolValues = [NSMutableArray new];
			M.tablePool = [NSMutableArray new];
		}
		mce_metal4 = 1;
		fprintf(stderr, "mcopt-metal4: Metal 4 backend on %s (MTL4CommandQueue, argument tables, upload ring, residency sets, "
			"conservative barriers)\n", d.name.UTF8String);
		return 1;
	}
	fprintf(stderr, "mcopt-metal4: -Dmcopt.metal4=true needs macOS 26 or later: using Metal 3\n");
	return 0;
}

void mce4_add_queue_set(id<MTLResidencySet> set) {
	if (mce_metal4 && set) [M.queue addResidencySet:set];
}

void mce4_layer(CAMetalLayer *layer) {
	if (!mce_metal4 || layer == M.layer) return;
	id<MTLResidencySet> s = layer.residencySet;
	if (s) [M.queue addResidencySet:s];
	else mce_log_once("the CAMetalLayer has no residency set: drawables may not be resident");
	M.layer = layer;
}

// ---- residency and deferred release ----
void mce4_make_resident(id allocation) {
	os_unfair_lock_lock(&M.lock);
	[M.set addAllocation:allocation];
	M.setDirty = 1;
	os_unfair_lock_unlock(&M.lock);
}

void mc_release_now(id obj);  // mcmetal.m: today's release (provisioned buffers leave their residency set first)

// Frees graveyard entries whose frame completed. Called with M.lock held; the releases themselves run after it's dropped.
static void collect(id *out, int *nout, int cap) {
	uint64_t done = completedValue();
	int j = 0;
	for (int i = 0; i < M.ngrave; i++) {
		if (M.grave[i].value != 0 && M.grave[i].value <= done && *nout < cap) {
			id o = M.grave[i].obj;
			if ([o conformsToProtocol:@protocol(MTLAllocation)] && [M.set containsAllocation:o]) {
				[M.set removeAllocation:o];
				M.setDirty = 1;
			}
			out[(*nout)++] = o;
		} else {
			M.grave[j++] = M.grave[i];
		}
	}
	M.ngrave = j;
}

static void releaseAll(id *objs, int n) {
	for (int i = 0; i < n; i++) mc_release_now(objs[i]);
}

void mce4_release(id obj) {
	if ([obj isKindOfClass:[MceCmd class]]) {  // a committed command buffer token: nothing the GPU holds
		[obj release];
		return;
	}
	os_unfair_lock_lock(&M.lock);
	if (M.ngrave == M.capgrave) {
		M.capgrave = M.capgrave ? M.capgrave * 2 : 256;
		M.grave = realloc(M.grave, sizeof *M.grave * M.capgrave);
	}
	M.grave[M.ngrave].obj = obj;
	M.grave[M.ngrave].value = 0;
	M.ngrave++;
	os_unfair_lock_unlock(&M.lock);
}

// ---- upload ring ----
static int newChunk(uint64_t size) {
	if (M.nchunks == MAX_CHUNKS) return -1;
	id<MTLBuffer> b = [M.device newBufferWithLength:size options:MTLResourceStorageModeShared | MTLResourceHazardTrackingModeUntracked];
	if (!b) return -1;
	b.label = @"mcopt metal4 upload ring";
	[M.set addAllocation:b];
	M.setDirty = 1;
	int k = M.nchunks++;
	M.chunks[k].buf = b;
	M.chunks[k].size = size;
	M.chunks[k].used = 0;
	M.chunks[k].last = 0;
	M.chunks[k].inFrame = 0;
	return k;
}

// lock held
static uint64_t ringAlloc(NSUInteger length, id<MTLBuffer> *buffer, NSUInteger *offset, uint8_t **ptr) {
	uint64_t need = (length + RING_ALIGN - 1) & ~(uint64_t) (RING_ALIGN - 1);
	if (M.nchunks == 0) M.cur = newChunk(CHUNK_SIZE);
	for (;;) {
		if (M.cur >= 0 && M.chunks[M.cur].used + need <= M.chunks[M.cur].size) break;
		// the current chunk is full: a chunk the GPU is done with, else a new one, else wait for the oldest
		uint64_t done = completedValue();
		int pick = -1;
		for (int i = 0; i < M.nchunks; i++) {
			if (i != M.cur && !M.chunks[i].inFrame && M.chunks[i].last <= done && M.chunks[i].size >= need) { pick = i; break; }
		}
		if (pick < 0) pick = newChunk(need > CHUNK_SIZE ? need : CHUNK_SIZE);
		if (pick < 0) {  // all chunks busy and no room for more: wait for the GPU (only with a runaway frame)
			uint64_t oldest = UINT64_MAX;
			for (int i = 0; i < M.nchunks; i++) if (!M.chunks[i].inFrame && M.chunks[i].last < oldest) oldest = M.chunks[i].last;
			if (oldest == UINT64_MAX) { fprintf(stderr, "mcopt-metal4: upload ring exhausted within one frame\n"); abort(); }
			os_unfair_lock_unlock(&M.lock);
			[M.timeline waitUntilSignaledValue:oldest timeoutMS:5000];
			os_unfair_lock_lock(&M.lock);
			continue;
		}
		M.cur = pick;
		M.chunks[pick].used = 0;
	}
	typeof(M.chunks[0]) *c = &M.chunks[M.cur];
	uint64_t off = c->used;
	c->used += need;
	c->inFrame = 1;
	if (buffer) *buffer = c->buf;
	if (offset) *offset = (NSUInteger) off;
	*ptr = (uint8_t *) c->buf.contents + off;
	return c->buf.gpuAddress + off;
}

static uint64_t upload(const void *bytes, NSUInteger length, id<MTLBuffer> *buffer, NSUInteger *offset) {
	NSUInteger padded = (length + 15) & ~(NSUInteger) 15;
	uint8_t *p;
	os_unfair_lock_lock(&M.lock);
	uint64_t a = ringAlloc(padded, buffer, offset, &p);
	os_unfair_lock_unlock(&M.lock);
	memcpy(p, bytes, length);
	if (padded != length) memset(p + length, 0, padded - length);
	return a;
}

uint64_t mce_upload(const void *bytes, NSUInteger length, id<MTLBuffer> *buffer, NSUInteger *offset) {
	return mce_metal4 ? upload(bytes, length, buffer, offset) : 0;
}

// ---- frame end ----
void mce4_frame_end(void) {
	id rel[512];
	int n = 0;
	os_unfair_lock_lock(&M.lock);
	uint64_t v = M.submitted;  // the frame's last commit: its completion implies every earlier one's
	for (int i = 0; i < M.nchunks; i++) {
		if (M.chunks[i].inFrame) {
			M.chunks[i].inFrame = 0;
			M.chunks[i].last = v;
		}
	}
	for (int i = 0; i < M.ngrave; i++) if (M.grave[i].value == 0) M.grave[i].value = v;
	collect(rel, &n, 512);
	os_unfair_lock_unlock(&M.lock);
	releaseAll(rel, n);
}

// ---- argument tables ----
static id<MTL4ArgumentTable> takeTable(void) {
	id<MTL4ArgumentTable> t = [M.tablePool lastObject];
	if (t) {
		[t retain];
		[M.tablePool removeLastObject];
		return t;
	}
	@autoreleasepool {
		MTL4ArgumentTableDescriptor *d = [[MTL4ArgumentTableDescriptor new] autorelease];
		d.maxBufferBindCount = TABLE_BUFFERS;
		d.maxTextureBindCount = TABLE_TEXTURES;
		d.maxSamplerStateBindCount = TABLE_SAMPLERS;
		d.initializeBindings = YES;
		NSError *e = nil;
		t = [M.device newArgumentTableWithDescriptor:d error:&e];
		if (!t) { fprintf(stderr, "mcopt-metal4: argument table: %s\n", e ? e.description.UTF8String : "?"); abort(); }
		return t;
	}
}

// ---- command buffers ----
@implementation MceCmd
- (void)dealloc {
	[completed release];
	[waits release];
	[signals release];
	[drawable release];
	[super dealloc];
}
// today's MTLCommandBuffer selectors, for call sites not yet ported (prefer the mce_cmd_* / mce_new_*_encoder functions)
- (id)computeCommandEncoder { return [mce4_new_compute_encoder(self, MTLDispatchTypeSerial) autorelease]; }
- (id)computeCommandEncoderWithDispatchType:(MTLDispatchType)t { return [mce4_new_compute_encoder(self, t) autorelease]; }
- (id)computeCommandEncoderWithDescriptor:(MTLComputePassDescriptor *)d { return [mce4_new_compute_encoder(self, d.dispatchType) autorelease]; }
- (id)blitCommandEncoder { return [mce4_new_blit_encoder(self) autorelease]; }
- (id)blitCommandEncoderWithDescriptor:(MTLBlitPassDescriptor *)d { return [mce4_new_blit_encoder(self) autorelease]; }
- (id)renderCommandEncoderWithDescriptor:(MTLRenderPassDescriptor *)rp { return [mce4_new_render_encoder(self, rp) autorelease]; }
- (void)commit { mce4_cmd_commit(self); }
- (void)waitUntilCompleted { mce4_cmd_wait(self); }
- (void)addCompletedHandler:(MTLCommandBufferHandler)block {
	MceCmd *me = self;
	mce4_cmd_on_completed(self, ^(double s, double e) { block((id<MTLCommandBuffer>) me); });
}
- (void)presentDrawable:(id<MTLDrawable>)d { mce4_cmd_present(self, (id<CAMetalDrawable>) d); }
- (void)encodeSignalEvent:(id<MTLEvent>)e value:(uint64_t)v { mce4_cmd_signal_event(self, e, v); }
- (void)encodeWaitForEvent:(id<MTLEvent>)e value:(uint64_t)v { mce4_cmd_wait_event(self, e, v); }
- (MTLCommandBufferStatus)status {
	return !value ? MTLCommandBufferStatusNotEnqueued : mce4_cmd_done(self) ? MTLCommandBufferStatusCompleted : MTLCommandBufferStatusCommitted;
}
- (CFTimeInterval)GPUStartTime { return mce4_cmd_gpu_start(self); }
- (CFTimeInterval)GPUEndTime { return mce4_cmd_gpu_end(self); }
- (CFTimeInterval)kernelStartTime { return 0; }
- (CFTimeInterval)kernelEndTime { return 0; }
- (NSError *)error { return nil; }
- (id<MTLDevice>)device { return M.device; }
- (void)pushDebugGroup:(NSString *)s { [cb pushDebugGroup:s]; }
- (void)popDebugGroup { [cb popDebugGroup]; }
- (void)setLabel:(NSString *)l { [cb setLabel:l]; }
- (NSString *)label { return [cb label]; }
- (NSMethodSignature *)methodSignatureForSelector:(SEL)sel {
	NSMethodSignature *s = [super methodSignatureForSelector:sel];
	if (s) return s;
	struct objc_method_description d = protocol_getMethodDescription(@protocol(MTLCommandBuffer), sel, YES, YES);
	if (!d.types) d = protocol_getMethodDescription(@protocol(MTLCommandBuffer), sel, NO, YES);
	return d.types ? [NSMethodSignature signatureWithObjCTypes:d.types] : nil;
}
- (void)forwardInvocation:(NSInvocation *)inv {
	logUnported("MTLCommandBuffer", inv.selector);
	NSUInteger n = inv.methodSignature.methodReturnLength;
	if (n) { uint8_t zero[64] = {0}; if (n <= sizeof zero) [inv setReturnValue:zero]; }
}
@end

id mce4_new_cmd(Ctx *ctx) {
	MceCmd *c = [MceCmd new];
	os_unfair_lock_lock(&M.lock);
	int slot = -1;
	for (;;) {
		uint64_t done = completedValue();
		for (int i = 0; i < M.nallocs; i++) {
			if (!M.allocs[i].busy && M.allocs[i].last <= done) { slot = i; break; }
		}
		if (slot >= 0) {
			if (M.allocs[slot].last) [M.allocs[slot].a reset];  // every command buffer that used it has completed
			M.allocs[slot].last = 0;
			break;
		}
		if (M.nallocs < MAX_ALLOCS) {
			NSError *e = nil;
			MTL4CommandAllocatorDescriptor *d = [MTL4CommandAllocatorDescriptor new];
			d.label = [NSString stringWithFormat:@"mcopt metal4 allocator %d", M.nallocs];
			id<MTL4CommandAllocator> a = [M.device newCommandAllocatorWithDescriptor:d error:&e];
			[d release];
			if (!a) { fprintf(stderr, "mcopt-metal4: command allocator: %s\n", e ? e.description.UTF8String : "?"); abort(); }
			slot = M.nallocs++;
			M.allocs[slot].a = a;
			M.allocs[slot].last = 0;
			break;
		}
		// every allocator is encoding or in flight: wait for the oldest in-flight one
		uint64_t oldest = UINT64_MAX;
		for (int i = 0; i < M.nallocs; i++) if (!M.allocs[i].busy && M.allocs[i].last < oldest) oldest = M.allocs[i].last;
		if (oldest == UINT64_MAX) { fprintf(stderr, "mcopt-metal4: %d command buffers open at once\n", M.nallocs); abort(); }
		os_unfair_lock_unlock(&M.lock);
		[M.timeline waitUntilSignaledValue:oldest timeoutMS:5000];
		os_unfair_lock_lock(&M.lock);
	}
	M.allocs[slot].busy = 1;
	id<MTL4CommandBuffer> cb = nil;
	if (M.cbPool.count && [M.cbPoolValues[0] unsignedLongLongValue] <= completedValue()) {  // oldest first
		cb = [M.cbPool[0] retain];
		[M.cbPool removeObjectAtIndex:0];
		[M.cbPoolValues removeObjectAtIndex:0];
	}
	os_unfair_lock_unlock(&M.lock);
	if (!cb) cb = [M.device newCommandBuffer];
	[cb beginCommandBufferWithAllocator:M.allocs[slot].a];
	c->cb = cb;
	c->alloc = slot;
	return c;
}

void mce4_cmd_on_completed(id h, void (^block)(double, double)) {
	MceCmd *c = h;
	if (!c->completed) c->completed = [NSMutableArray new];
	id b = [block copy];
	[c->completed addObject:b];
	[b release];
}

void mce4_cmd_present(id h, id<CAMetalDrawable> d) {
	MceCmd *c = h;
	if (c->drawable) mce_log_once("two presents in one command buffer: only the last is shown");
	[c->drawable release];
	c->drawable = [d retain];
}

void mce4_cmd_signal_event(id h, id<MTLEvent> e, uint64_t v) {
	MceCmd *c = h;
	if (!c->signals) c->signals = [NSMutableArray new];
	[c->signals addObject:e];
	[c->signals addObject:@(v)];
}

void mce4_cmd_wait_event(id h, id<MTLEvent> e, uint64_t v) {
	MceCmd *c = h;
	if (!c->waits) c->waits = [NSMutableArray new];
	[c->waits addObject:e];
	[c->waits addObject:@(v)];
}

void mce4_cmd_commit(id h) {
	MceCmd *c = h;
	if (!c->cb) return;
	@autoreleasepool {
		id<MTL4CommandBuffer> cb = c->cb;
		[cb endCommandBuffer];
		MTL4CommitOptions *o = [[MTL4CommitOptions new] autorelease];
		NSArray *blocks = [[c->completed copy] autorelease];
		MceCmd *me = [c retain];  // released by the feedback handler
		[o addFeedbackHandler:^(id<MTL4CommitFeedback> f) {
			if (f.error) fprintf(stderr, "mcopt-metal4: command buffer failed: %s\n", f.error.description.UTF8String);
			me->gpuStart = f.GPUStartTime;
			me->gpuEnd = f.GPUEndTime;
			__atomic_store_n(&me->timesReady, 1, __ATOMIC_RELEASE);
			for (void (^b)(double, double) in blocks) b(f.GPUStartTime, f.GPUEndTime);
			[me release];
		}];
		os_unfair_lock_lock(&M.lock);
		if (M.setDirty) {
			[M.set commit];
			M.setDirty = 0;
		}
		for (NSUInteger i = 0; i + 1 < c->waits.count; i += 2) [M.queue waitForEvent:c->waits[i] value:[c->waits[i + 1] unsignedLongLongValue]];
		if (c->drawable) [M.queue waitForDrawable:c->drawable];
		[M.queue commit:&cb count:1 options:o];
		uint64_t v = ++M.submitted;
		[M.queue signalEvent:M.timeline value:v];
		c->value = v;
		for (NSUInteger i = 0; i + 1 < c->signals.count; i += 2) [M.queue signalEvent:c->signals[i] value:[c->signals[i + 1] unsignedLongLongValue]];
		if (c->drawable) {
			[M.queue signalDrawable:c->drawable];
			[c->drawable present];
			[c->drawable release];
			c->drawable = nil;
		}
		M.allocs[c->alloc].busy = 0;
		M.allocs[c->alloc].last = v;
		[M.cbPool addObject:cb];
		[M.cbPoolValues addObject:@(v)];
		os_unfair_lock_unlock(&M.lock);
		[cb release];
		c->cb = nil;
	}
}

void mce4_cmd_wait(id h) {
	MceCmd *c = h;
	if (!c->value) return;
	while (![M.timeline waitUntilSignaledValue:c->value timeoutMS:5000])
		fprintf(stderr, "mcopt-metal4: still waiting for command buffer %llu (completed %llu, submitted %llu)\n", c->value, completedValue(), M.submitted);
}

int mce4_cmd_done(id h) {
	MceCmd *c = h;
	return c->value && completedValue() >= c->value;
}

// The feedback handler lands shortly after the timeline signal: wait for it briefly (only when asked for the times).
static void waitTimes(MceCmd *c) {
	if (!c->value || __atomic_load_n(&c->timesReady, __ATOMIC_ACQUIRE)) return;
	mce4_cmd_wait(c);
	for (int i = 0; i < 2000 && !__atomic_load_n(&c->timesReady, __ATOMIC_ACQUIRE); i++) usleep(50);
}
double mce4_cmd_gpu_start(id h) { waitTimes(h); return ((MceCmd *) h)->gpuStart; }
double mce4_cmd_gpu_end(id h) { waitTimes(h); return ((MceCmd *) h)->gpuEnd; }

// ---- encoders ----
#define ENCODER_STAGES (MTLStageDispatch | MTLStageBlit)

@implementation MceEncoder
- (void)dealloc {
	if (!ended && real) {
		mce_log_once("an encoder was released without mce_end_encoder: ended now");
		mce4_end_encoder(self);
	}
	[super dealloc];
}
static void serialPoint(MceEncoder *e) {
	if (e->kind == MCE_KIND_SERIAL || e->kind == MCE_KIND_BLIT) {
		if (e->pending) [(id<MTL4ComputeCommandEncoder>) e->real barrierAfterEncoderStages:ENCODER_STAGES beforeEncoderStages:ENCODER_STAGES
			visibilityOptions:MTL4VisibilityOptionDevice];
		e->pending = 1;
	}
}
// identical selectors that need the serial barrier first (sent by ported and unported code alike)
- (void)dispatchThreads:(MTLSize)g threadsPerThreadgroup:(MTLSize)t { serialPoint(self); [real dispatchThreads:g threadsPerThreadgroup:t]; }
- (void)dispatchThreadgroups:(MTLSize)g threadsPerThreadgroup:(MTLSize)t { serialPoint(self); [real dispatchThreadgroups:g threadsPerThreadgroup:t]; }
- (void)copyFromBuffer:(id<MTLBuffer>)s sourceOffset:(NSUInteger)so toBuffer:(id<MTLBuffer>)d destinationOffset:(NSUInteger)dO size:(NSUInteger)n {
	serialPoint(self);
	[real copyFromBuffer:s sourceOffset:so toBuffer:d destinationOffset:dO size:n];
}
- (void)copyFromBuffer:(id<MTLBuffer>)s sourceOffset:(NSUInteger)o sourceBytesPerRow:(NSUInteger)r sourceBytesPerImage:(NSUInteger)i sourceSize:(MTLSize)z
	toTexture:(id<MTLTexture>)t destinationSlice:(NSUInteger)sl destinationLevel:(NSUInteger)l destinationOrigin:(MTLOrigin)org {
	serialPoint(self);
	[real copyFromBuffer:s sourceOffset:o sourceBytesPerRow:r sourceBytesPerImage:i sourceSize:z toTexture:t destinationSlice:sl destinationLevel:l destinationOrigin:org];
}
- (void)copyFromBuffer:(id<MTLBuffer>)s sourceOffset:(NSUInteger)o sourceBytesPerRow:(NSUInteger)r sourceBytesPerImage:(NSUInteger)i sourceSize:(MTLSize)z
	toTexture:(id<MTLTexture>)t destinationSlice:(NSUInteger)sl destinationLevel:(NSUInteger)l destinationOrigin:(MTLOrigin)org options:(MTLBlitOption)op {
	serialPoint(self);
	[real copyFromBuffer:s sourceOffset:o sourceBytesPerRow:r sourceBytesPerImage:i sourceSize:z toTexture:t destinationSlice:sl destinationLevel:l
		destinationOrigin:org options:op];
}
- (void)copyFromTexture:(id<MTLTexture>)s sourceSlice:(NSUInteger)ss sourceLevel:(NSUInteger)sl sourceOrigin:(MTLOrigin)so sourceSize:(MTLSize)z
	toBuffer:(id<MTLBuffer>)d destinationOffset:(NSUInteger)o destinationBytesPerRow:(NSUInteger)r destinationBytesPerImage:(NSUInteger)i {
	serialPoint(self);
	[real copyFromTexture:s sourceSlice:ss sourceLevel:sl sourceOrigin:so sourceSize:z toBuffer:d destinationOffset:o destinationBytesPerRow:r destinationBytesPerImage:i];
}
- (void)copyFromTexture:(id<MTLTexture>)s sourceSlice:(NSUInteger)ss sourceLevel:(NSUInteger)sl sourceOrigin:(MTLOrigin)so sourceSize:(MTLSize)z
	toTexture:(id<MTLTexture>)d destinationSlice:(NSUInteger)ds destinationLevel:(NSUInteger)dl destinationOrigin:(MTLOrigin)dorg {
	serialPoint(self);
	[real copyFromTexture:s sourceSlice:ss sourceLevel:sl sourceOrigin:so sourceSize:z toTexture:d destinationSlice:ds destinationLevel:dl destinationOrigin:dorg];
}
- (void)copyFromTexture:(id<MTLTexture>)s sourceSlice:(NSUInteger)ss sourceLevel:(NSUInteger)sl toTexture:(id<MTLTexture>)d destinationSlice:(NSUInteger)ds
	destinationLevel:(NSUInteger)dl sliceCount:(NSUInteger)sc levelCount:(NSUInteger)lc {
	serialPoint(self);
	[real copyFromTexture:s sourceSlice:ss sourceLevel:sl toTexture:d destinationSlice:ds destinationLevel:dl sliceCount:sc levelCount:lc];
}
- (void)copyFromTexture:(id<MTLTexture>)s toTexture:(id<MTLTexture>)d { serialPoint(self); [real copyFromTexture:s toTexture:d]; }
- (void)fillBuffer:(id<MTLBuffer>)b range:(NSRange)r value:(uint8_t)v { serialPoint(self); [real fillBuffer:b range:r value:v]; }
- (void)generateMipmapsForTexture:(id<MTLTexture>)t { serialPoint(self); [real generateMipmapsForTexture:t]; }
- (void)endEncoding { mce4_end_encoder(self); }
// everything else MTL4 has under the same name goes straight to the real encoder; Metal 3-only selectors are logged and dropped
- (id)forwardingTargetForSelector:(SEL)sel { return [real respondsToSelector:sel] ? real : nil; }
- (NSMethodSignature *)methodSignatureForSelector:(SEL)sel {
	NSMethodSignature *s = [super methodSignatureForSelector:sel];
	if (s) return s;
	Protocol *ps[] = {@protocol(MTLRenderCommandEncoder), @protocol(MTLComputeCommandEncoder), @protocol(MTLBlitCommandEncoder), @protocol(MTLCommandEncoder)};
	for (int i = 0; i < 4; i++) {
		struct objc_method_description d = protocol_getMethodDescription(ps[i], sel, YES, YES);
		if (!d.types) d = protocol_getMethodDescription(ps[i], sel, NO, YES);
		if (d.types) return [NSMethodSignature signatureWithObjCTypes:d.types];
	}
	return nil;
}
- (void)forwardInvocation:(NSInvocation *)inv {
	logUnported(kind == MCE_KIND_RENDER ? "MTLRenderCommandEncoder" : kind == MCE_KIND_BLIT ? "MTLBlitCommandEncoder" : "MTLComputeCommandEncoder", inv.selector);
	NSUInteger n = inv.methodSignature.methodReturnLength;
	if (n) { uint8_t zero[64] = {0}; if (n <= sizeof zero) [inv setReturnValue:zero]; }
}
@end

// The conservative inter-encoder barrier: nothing in this encoder starts before all earlier work on the queue is done.
static void openBarrier(MceEncoder *e) {
	[(id<MTL4CommandEncoder>) e->real barrierAfterQueueStages:MTLStageAll beforeStages:MTLStageAll visibilityOptions:MTL4VisibilityOptionDevice];
}

static MceEncoder *wrap(id real, int kind) {
	MceEncoder *e = [MceEncoder new];
	e->real = [real retain];
	e->kind = kind;
	os_unfair_lock_lock(&M.lock);
	if (kind == MCE_KIND_RENDER) {
		static const MTLRenderStages stages[MCE_STAGES] = {MTLRenderStageVertex, MTLRenderStageFragment, MTLRenderStageObject, MTLRenderStageMesh, MTLRenderStageTile};
		for (int i = 0; i < MCE_STAGES; i++) {
			e->tables[i] = takeTable();
			[(id<MTL4RenderCommandEncoder>) real setArgumentTable:e->tables[i] atStages:stages[i]];
		}
	} else if (kind != MCE_KIND_BLIT) {
		e->tables[0] = takeTable();
		[(id<MTL4ComputeCommandEncoder>) real setArgumentTable:e->tables[0]];
	}
	os_unfair_lock_unlock(&M.lock);
	openBarrier(e);
	return e;
}

id mce4_new_render_encoder(id cmd, MTLRenderPassDescriptor *rp) {
	MceCmd *c = cmd;
	@autoreleasepool {
		MTL4RenderPassDescriptor *d = [[MTL4RenderPassDescriptor new] autorelease];
		for (int i = 0; i < MAX_COLORS; i++) {
			MTLRenderPassColorAttachmentDescriptor *a = rp.colorAttachments[i];
			if (a.texture) d.colorAttachments[i] = a;  // copies
		}
		if (rp.depthAttachment.texture) d.depthAttachment = rp.depthAttachment;
		if (rp.stencilAttachment.texture) d.stencilAttachment = rp.stencilAttachment;
		d.renderTargetWidth = rp.renderTargetWidth;
		d.renderTargetHeight = rp.renderTargetHeight;
		d.defaultRasterSampleCount = rp.defaultRasterSampleCount;
		d.renderTargetArrayLength = rp.renderTargetArrayLength;
		d.imageblockSampleLength = rp.imageblockSampleLength;
		d.threadgroupMemoryLength = rp.threadgroupMemoryLength;
		d.tileWidth = rp.tileWidth;
		d.tileHeight = rp.tileHeight;
		d.rasterizationRateMap = rp.rasterizationRateMap;
		d.visibilityResultBuffer = rp.visibilityResultBuffer;
		id<MTL4RenderCommandEncoder> r = [c->cb renderCommandEncoderWithDescriptor:d];
		if (!r) { fprintf(stderr, "mcopt-metal4: render encoder creation failed\n"); abort(); }
		return wrap(r, MCE_KIND_RENDER);
	}
}

id mce4_new_compute_encoder(id cmd, MTLDispatchType type) {
	MceCmd *c = cmd;
	return wrap([c->cb computeCommandEncoder], type == MTLDispatchTypeConcurrent ? MCE_KIND_CONCURRENT : MCE_KIND_SERIAL);
}

id mce4_new_blit_encoder(id cmd) {
	MceCmd *c = cmd;
	return wrap([c->cb computeCommandEncoder], MCE_KIND_BLIT);
}

void mce4_end_encoder(id h) {
	MceEncoder *e = h;
	if (e->ended) return;
	e->ended = 1;
	// and the other half of the conservative barrier: later encoders wait for all of this one
	[(id<MTL4CommandEncoder>) e->real barrierAfterStages:MTLStageAll beforeQueueStages:MTLStageAll visibilityOptions:MTL4VisibilityOptionDevice];
	[(id<MTL4CommandEncoder>) e->real endEncoding];
	[e->real release];
	e->real = nil;
	os_unfair_lock_lock(&M.lock);
	for (int i = 0; i < MCE_STAGES; i++) {
		if (e->tables[i]) {  // Metal snapshotted it at each draw/dispatch: free for the next encoder
			[M.tablePool addObject:e->tables[i]];
			[e->tables[i] release];
			e->tables[i] = nil;
		}
	}
	os_unfair_lock_unlock(&M.lock);
}

// ---- bindings ----
static inline id<MTL4ArgumentTable> table(id h, int stage) {
	MceEncoder *e = h;
	id<MTL4ArgumentTable> t = e->tables[stage];
	if (!t) logUnported("mce (binding on an encoder kind without that stage)", @selector(description));
	return t;
}

static int bufferIndexOk(NSUInteger i) {
	if (i < TABLE_BUFFERS) return 1;
	mce_log_once("buffer binding index >= 31: dropped");
	return 0;
}

void mce4_set_bytes(id h, int stage, const void *bytes, NSUInteger length, NSUInteger index) {
	if (!bufferIndexOk(index)) return;
	[table(h, stage) setAddress:upload(bytes, length, NULL, NULL) atIndex:index];
}

void mce4_set_bytes2(id h, int stageA, int stageB, const void *bytes, NSUInteger length, NSUInteger index) {
	if (!bufferIndexOk(index)) return;
	uint64_t a = upload(bytes, length, NULL, NULL);
	[table(h, stageA) setAddress:a atIndex:index];
	[table(h, stageB) setAddress:a atIndex:index];
}

void mce4_set_buffer(id h, int stage, id<MTLBuffer> buffer, NSUInteger offset, NSUInteger index) {
	if (!bufferIndexOk(index)) return;
	[table(h, stage) setAddress:buffer ? buffer.gpuAddress + offset : 0 atIndex:index];
}

void mce4_set_texture(id h, int stage, id<MTLTexture> texture, NSUInteger index) {
	if (index >= TABLE_TEXTURES) { mce_log_once("texture binding index >= 48: dropped"); return; }
	MTLResourceID none = {0};
	[table(h, stage) setTexture:texture ? texture.gpuResourceID : none atIndex:index];
}

void mce4_set_sampler(id h, int stage, id<MTLSamplerState> sampler, NSUInteger index) {
	if (index >= TABLE_SAMPLERS) { mce_log_once("sampler binding index >= 16: dropped"); return; }
	MTLResourceID none = {0};
	[table(h, stage) setSamplerState:sampler ? sampler.gpuResourceID : none atIndex:index];
}

// ---- draws and dispatches ----
void mce4_draw_indexed(id h, MTLPrimitiveType p, NSUInteger count, MTLIndexType t, id<MTLBuffer> ib, NSUInteger off, NSUInteger inst, NSInteger bv, NSUInteger bi) {
	[(id<MTL4RenderCommandEncoder>) ((MceEncoder *) h)->real drawIndexedPrimitives:p indexCount:count indexType:t indexBuffer:ib.gpuAddress + off
		indexBufferLength:ib.length - off instanceCount:inst baseVertex:bv baseInstance:bi];
}

void mce4_draw_indirect(id h, MTLPrimitiveType p, id<MTLBuffer> ind, NSUInteger indOff) {
	[(id<MTL4RenderCommandEncoder>) ((MceEncoder *) h)->real drawPrimitives:p indirectBuffer:ind.gpuAddress + indOff];
}

void mce4_draw_indexed_indirect(id h, MTLPrimitiveType p, MTLIndexType t, id<MTLBuffer> ib, NSUInteger off, id<MTLBuffer> ind, NSUInteger indOff) {
	[(id<MTL4RenderCommandEncoder>) ((MceEncoder *) h)->real drawIndexedPrimitives:p indexType:t indexBuffer:ib.gpuAddress + off
		indexBufferLength:ib.length - off indirectBuffer:ind.gpuAddress + indOff];
}

void mce4_draw_mesh_indirect(id h, id<MTLBuffer> ind, NSUInteger indOff, MTLSize obj, MTLSize mesh) {
	[(id<MTL4RenderCommandEncoder>) ((MceEncoder *) h)->real drawMeshThreadgroupsWithIndirectBuffer:ind.gpuAddress + indOff
		threadsPerObjectThreadgroup:obj threadsPerMeshThreadgroup:mesh];
}

void mce4_dispatch_threads(id h, MTLSize grid, MTLSize tg) { [(MceEncoder *) h dispatchThreads:grid threadsPerThreadgroup:tg]; }
void mce4_dispatch_threadgroups(id h, MTLSize groups, MTLSize tg) { [(MceEncoder *) h dispatchThreadgroups:groups threadsPerThreadgroup:tg]; }
void mce4_dispatch_threadgroups_indirect(id h, id<MTLBuffer> ind, NSUInteger indOff, MTLSize tg) {
	MceEncoder *e = h;
	serialPoint(e);
	[(id<MTL4ComputeCommandEncoder>) e->real dispatchThreadgroupsWithIndirectBuffer:ind.gpuAddress + indOff threadsPerThreadgroup:tg];
}

// ---- barriers and fences ----
void mce4_memory_barrier(id h) {
	MceEncoder *e = h;
	[(id<MTL4CommandEncoder>) e->real barrierAfterEncoderStages:ENCODER_STAGES beforeEncoderStages:ENCODER_STAGES visibilityOptions:MTL4VisibilityOptionDevice];
	e->pending = 0;
}

void mce4_memory_barrier_render(id h, MTLRenderStages after, MTLRenderStages before) {
	// MTLRenderStages and MTLStages share the bits of vertex, fragment, tile, object and mesh.
	[(id<MTL4CommandEncoder>) ((MceEncoder *) h)->real barrierAfterEncoderStages:(MTLStages) after beforeEncoderStages:(MTLStages) before
		visibilityOptions:MTL4VisibilityOptionDevice];
}

void mce4_barrier_full(id h) {
	MceEncoder *e = h;
	[(id<MTL4CommandEncoder>) e->real barrierAfterQueueStages:MTLStageAll beforeStages:MTLStageAll visibilityOptions:MTL4VisibilityOptionDevice];
	if (e->kind != MCE_KIND_RENDER) mce4_memory_barrier(h);
}

static MTLStages fenceStages(MceEncoder *e, int render, MTLRenderStages s) {
	return render ? (MTLStages) s : ENCODER_STAGES;
}

void mce4_update_fence(id h, id<MTLFence> f, int render, MTLRenderStages stages) {
	MceEncoder *e = h;
	[(id<MTL4CommandEncoder>) e->real updateFence:f afterEncoderStages:fenceStages(e, render, stages)];
}

void mce4_wait_fence(id h, id<MTLFence> f, int render, MTLRenderStages stages) {
	MceEncoder *e = h;
	[(id<MTL4CommandEncoder>) e->real waitForFence:f beforeEncoderStages:fenceStages(e, render, stages)];
}
