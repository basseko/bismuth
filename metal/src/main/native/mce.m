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
	NSMutableArray *tablePool, *tablePoolBig;  // argument tables (small, large) not attached to an open encoder
	id<MTLResidencySet> set;  // everything made resident through the layer
	int setDirty;
	int setRemoved;  // removals not committed yet
	double setRemovedAt;
	struct { id<MTLBuffer> buf; uint64_t size, used, last; int inFrame; } chunks[MAX_CHUNKS];
	int nchunks, cur;
	struct { id obj; uint64_t value; } *grave;  // value 0: released during the frame being encoded
	int ngrave, capgrave;
	NSMutableSet *logged;
	CAMetalLayer *layer;
} M = {.lock = OS_UNFAIR_LOCK_INIT};

// MEASUREMENT: MCE_STATS=1 prints the layer's per-frame counts every 2 s
static struct { long commits, setCommits, encoders, fences, uploads, uploadBytes, resident, released, presents, tableGrows; double last; } ST;
static int stats(void) { static int m = -1; if (m < 0) m = getenv("MCE_STATS") != NULL; return m; }


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
			M.tablePoolBig = [NSMutableArray new];
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
static void churnNote(const char *what, id o) {
	if (!stats()) return;
	static NSMutableDictionary *seen;
	static double last;
	static os_unfair_lock lk = OS_UNFAIR_LOCK_INIT;
	os_unfair_lock_lock(&lk);
	if (!seen) seen = [NSMutableDictionary new];
	NSString *kind = [o conformsToProtocol:@protocol(MTLTexture)] ? [NSString stringWithFormat:@"tex %lux%lu fmt %lu", (unsigned long) ((id<MTLTexture>) o).width,
		(unsigned long) ((id<MTLTexture>) o).height, (unsigned long) ((id<MTLTexture>) o).pixelFormat]
		: [o conformsToProtocol:@protocol(MTLBuffer)] ? [NSString stringWithFormat:@"buf %lu", (unsigned long) ((id<MTLBuffer>) o).length] : NSStringFromClass([o class]);
	NSString *k = [NSString stringWithFormat:@"%s %@ [%@]", what, kind, [o respondsToSelector:@selector(label)] ? [o label] : @""];
	seen[k] = @([seen[k] longValue] + 1);
	double now = CFAbsoluteTimeGetCurrent();
	if (!last) last = now;
	if (now - last > 5.0) {
		NSArray *ks = [seen keysSortedByValueUsingComparator:^NSComparisonResult(NSNumber *a, NSNumber *b) { return [b compare:a]; }];
		for (NSUInteger i = 0; i < ks.count && i < 12; i++) fprintf(stderr, "mcopt-metal4 churn 5s: %6ld  %s\n", [seen[ks[i]] longValue], [ks[i] UTF8String]);
		[seen removeAllObjects];
		last = now;
	}
	os_unfair_lock_unlock(&lk);
}

void mce4_make_resident(id allocation) {
	ST.resident++;
	churnNote("new", allocation);
	// a texture view, or a texture over a buffer, shares its parent's allocation: the parent's residency covers it (and per-frame views would churn the set)
	if ([allocation conformsToProtocol:@protocol(MTLTexture)] && (((id<MTLTexture>) allocation).parentTexture || ((id<MTLTexture>) allocation).buffer)) return;
	os_unfair_lock_lock(&M.lock);
	// (a pooled buffer handed out again is already resident: no commit of the set for it; a residency commit is a driver operation that stalls)
	if (![M.set containsAllocation:allocation]) {
		[M.set addAllocation:allocation];
		M.setDirty = 1;
	}
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
				M.setRemoved++;  // (committed lazily: a freed allocation may stay resident a little longer, a new one must not wait)
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
	ST.released++;
	churnNote("free", obj);
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
	ST.uploads++;
	ST.uploadBytes += (long) length;
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
// Metal snapshots every attached table at each draw / dispatch, at its full size: vanilla's ~1,450 draws a frame pay for 31 / 48 / 16 slots
// per stage although its shaders bind only low indices (+11 % fps with small tables). So tables come in two sizes: an encoder starts with small
// ones, and the first bind beyond a stage's small range swaps in a large table for that stage, carrying over the bindings made so far (mirrored in
// the encoder while small). MCE_TABLE_SMALL=buffers,textures,samplers (each <= MCE_SMALL_MAX) for measurement.
static int smallN[3] = {12, 12, 8};
static const int bigN[3] = {TABLE_BUFFERS, TABLE_TEXTURES, TABLE_SAMPLERS};

static id<MTL4ArgumentTable> takeTable(int big) {
	NSMutableArray *pool = big ? M.tablePoolBig : M.tablePool;
	id<MTL4ArgumentTable> t = [pool lastObject];
	if (t) {
		[t retain];
		[pool removeLastObject];
		return t;
	}
	@autoreleasepool {
		static int once;
		const char *v = getenv("MCE_TABLE_SMALL");
		if (!once++ && v) {
			sscanf(v, "%d,%d,%d", &smallN[0], &smallN[1], &smallN[2]);
			for (int i = 0; i < 3; i++) smallN[i] = smallN[i] < 1 ? 1 : smallN[i] > MCE_SMALL_MAX ? MCE_SMALL_MAX : smallN[i];
			fprintf(stderr, "mcopt-metal4: MEASUREMENT small tables %d / %d / %d\n", smallN[0], smallN[1], smallN[2]);
		}
		const int *n = big ? bigN : smallN;
		MTL4ArgumentTableDescriptor *d = [[MTL4ArgumentTableDescriptor new] autorelease];
		d.maxBufferBindCount = (NSUInteger) n[0];
		d.maxTextureBindCount = (NSUInteger) n[1];
		d.maxSamplerStateBindCount = (NSUInteger) n[2];
		d.initializeBindings = YES;
		NSError *e = nil;
		t = [M.device newArgumentTableWithDescriptor:d error:&e];
		if (!t) { fprintf(stderr, "mcopt-metal4: argument table: %s\n", e ? e.description.UTF8String : "?"); abort(); }
		return t;
	}
}

static void giveTables(MceEncoder *e) {
	for (int i = 0; i < MCE_STAGES; i++) {
		if (e->tables[i]) {  // Metal snapshotted it at each draw/dispatch: free for the next encoder
			[(e->bigMask >> i & 1 ? M.tablePoolBig : M.tablePool) addObject:e->tables[i]];
			[e->tables[i] release];
			e->tables[i] = nil;
		}
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

static void ptResolve(id heapObj, NSArray *labels, int presented);
@class MceCmd;
static void closeOpen(MceCmd *c);

static void statsTick(int presented) {
	if (!stats()) return;
	ST.presents += presented;
	double now = CFAbsoluteTimeGetCurrent();
	if (!ST.last) ST.last = now;
	if (now - ST.last > 2.0 && ST.presents) {
		double f = (double) ST.presents;
		fprintf(stderr, "mcopt-metal4 stats per presented frame: commits %.2f, residency commits %.2f, encoders %.1f, fences %.1f, uploads %.1f (%.1f KB), made resident %.2f, released %.2f, tables grown %.1f\n",
			ST.commits / f, ST.setCommits / f, ST.encoders / f, ST.fences / f, ST.uploads / f, ST.uploadBytes / f / 1024.0, ST.resident / f, ST.released / f, ST.tableGrows / f);
		memset(&ST, 0, sizeof ST);
		ST.last = now;
	}
}

void mce4_cmd_commit(id h) {
	MceCmd *c = h;
	if (!c->cb) return;
	@autoreleasepool {
		closeOpen(c);
		id<MTL4CommandBuffer> cb = c->cb;
		[cb endCommandBuffer];
		MTL4CommitOptions *o = [[MTL4CommitOptions new] autorelease];
		NSArray *blocks = [[c->completed copy] autorelease];
		MceCmd *me = [c retain];  // released by the feedback handler
		id ptHeap = c->heap;
		NSArray *ptLabels = [c->heapLabels copy];
		int ptPresented = c->drawable != nil;
		statsTick(ptPresented);
		c->heap = nil;
		[c->heapLabels release];
		c->heapLabels = nil;
		[o addFeedbackHandler:^(id<MTL4CommitFeedback> f) {
			if (f.error) fprintf(stderr, "mcopt-metal4: command buffer failed: %s\n", f.error.description.UTF8String);
			me->gpuStart = f.GPUStartTime;
			me->gpuEnd = f.GPUEndTime;
			__atomic_store_n(&me->timesReady, 1, __ATOMIC_RELEASE);
			for (void (^b)(double, double) in blocks) b(f.GPUStartTime, f.GPUEndTime);
			if (ptHeap) {
				ptResolve(ptHeap, ptLabels, ptPresented);
				[ptHeap release];
			}
			[ptLabels release];
			[me release];
		}];
		os_unfair_lock_lock(&M.lock);
		ST.commits++;
		double nowc = CFAbsoluteTimeGetCurrent();
		if (M.setRemoved && (M.setRemoved > 256 || nowc - M.setRemovedAt > 1.0)) M.setDirty = 1;
		if (M.setDirty) {
			ST.setCommits++;
			M.setRemoved = 0;
			M.setRemovedAt = nowc;
			[M.set commit];
			M.setDirty = 0;
		}
		for (NSUInteger i = 0; i + 1 < c->waits.count; i += 2) [M.queue waitForEvent:c->waits[i] value:[c->waits[i + 1] unsignedLongLongValue]];
		if (c->drawable && !getenv("MCE_NODRAWWAIT")) [M.queue waitForDrawable:c->drawable];  // (MCE_NODRAWWAIT: MEASUREMENT only, may tear)
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
// MEASUREMENT ONLY (unsafe): MCE_BARRIERS in the environment, a mask of the conservative barriers kept: 1 at encoder open, 2 at encoder end, 4 between a
// serial encoder's commands (default 7, all). Turning any off can race; it exists to see what each class costs.
static int barMask(void) {
	static int m = -1;
	if (m < 0) {
		const char *v = getenv("MCE_BARRIERS");
		m = v ? atoi(v) : 7;
		if (m != 7) fprintf(stderr, "mcopt-metal4: MEASUREMENT barrier mask %d (unsafe)\n", m);
	}
	return m;
}

static void flushWaits(MceEncoder *e);
static void track(MceEncoder *e, id obj, int isTex, int stage);
static void use(MceEncoder *e, id obj, int write, int isTex, MTLStages stages);
static void serialPoint(MceEncoder *e) {
	flushWaits(e);
	if ((e->kind == MCE_KIND_SERIAL || e->kind == MCE_KIND_BLIT) && (barMask() & 4)) {
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
	use(self, s, 0, 0, 0); use(self, d, 1, 0, 0); flushWaits(self);
	[real copyFromBuffer:s sourceOffset:so toBuffer:d destinationOffset:dO size:n];
}
- (void)copyFromBuffer:(id<MTLBuffer>)s sourceOffset:(NSUInteger)o sourceBytesPerRow:(NSUInteger)r sourceBytesPerImage:(NSUInteger)i sourceSize:(MTLSize)z
	toTexture:(id<MTLTexture>)t destinationSlice:(NSUInteger)sl destinationLevel:(NSUInteger)l destinationOrigin:(MTLOrigin)org {
	serialPoint(self);
	use(self, s, 0, 0, 0); use(self, t, 1, 1, 0); flushWaits(self);
	[real copyFromBuffer:s sourceOffset:o sourceBytesPerRow:r sourceBytesPerImage:i sourceSize:z toTexture:t destinationSlice:sl destinationLevel:l destinationOrigin:org];
}
- (void)copyFromBuffer:(id<MTLBuffer>)s sourceOffset:(NSUInteger)o sourceBytesPerRow:(NSUInteger)r sourceBytesPerImage:(NSUInteger)i sourceSize:(MTLSize)z
	toTexture:(id<MTLTexture>)t destinationSlice:(NSUInteger)sl destinationLevel:(NSUInteger)l destinationOrigin:(MTLOrigin)org options:(MTLBlitOption)op {
	serialPoint(self);
	use(self, s, 0, 0, 0); use(self, t, 1, 1, 0); flushWaits(self);
	[real copyFromBuffer:s sourceOffset:o sourceBytesPerRow:r sourceBytesPerImage:i sourceSize:z toTexture:t destinationSlice:sl destinationLevel:l
		destinationOrigin:org options:op];
}
- (void)copyFromTexture:(id<MTLTexture>)s sourceSlice:(NSUInteger)ss sourceLevel:(NSUInteger)sl sourceOrigin:(MTLOrigin)so sourceSize:(MTLSize)z
	toBuffer:(id<MTLBuffer>)d destinationOffset:(NSUInteger)o destinationBytesPerRow:(NSUInteger)r destinationBytesPerImage:(NSUInteger)i {
	serialPoint(self);
	use(self, s, 0, 1, 0); use(self, d, 1, 0, 0); flushWaits(self);
	[real copyFromTexture:s sourceSlice:ss sourceLevel:sl sourceOrigin:so sourceSize:z toBuffer:d destinationOffset:o destinationBytesPerRow:r destinationBytesPerImage:i];
}
- (void)copyFromTexture:(id<MTLTexture>)s sourceSlice:(NSUInteger)ss sourceLevel:(NSUInteger)sl sourceOrigin:(MTLOrigin)so sourceSize:(MTLSize)z
	toTexture:(id<MTLTexture>)d destinationSlice:(NSUInteger)ds destinationLevel:(NSUInteger)dl destinationOrigin:(MTLOrigin)dorg {
	serialPoint(self);
	use(self, s, 0, 1, 0); use(self, d, 1, 1, 0); flushWaits(self);
	[real copyFromTexture:s sourceSlice:ss sourceLevel:sl sourceOrigin:so sourceSize:z toTexture:d destinationSlice:ds destinationLevel:dl destinationOrigin:dorg];
}
- (void)copyFromTexture:(id<MTLTexture>)s sourceSlice:(NSUInteger)ss sourceLevel:(NSUInteger)sl toTexture:(id<MTLTexture>)d destinationSlice:(NSUInteger)ds
	destinationLevel:(NSUInteger)dl sliceCount:(NSUInteger)sc levelCount:(NSUInteger)lc {
	serialPoint(self);
	use(self, s, 0, 1, 0); use(self, d, 1, 1, 0); flushWaits(self);
	[real copyFromTexture:s sourceSlice:ss sourceLevel:sl toTexture:d destinationSlice:ds destinationLevel:dl sliceCount:sc levelCount:lc];
}
- (void)copyFromTexture:(id<MTLTexture>)s toTexture:(id<MTLTexture>)d { use(self, s, 0, 1, 0); use(self, d, 1, 1, 0); serialPoint(self); [real copyFromTexture:s toTexture:d]; }
- (void)fillBuffer:(id<MTLBuffer>)b range:(NSRange)r value:(uint8_t)v { use(self, b, 1, 0, 0); serialPoint(self); [real fillBuffer:b range:r value:v]; }
- (void)generateMipmapsForTexture:(id<MTLTexture>)t { use(self, t, 1, 1, 0); serialPoint(self); [real generateMipmapsForTexture:t]; }
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
static int barMode(void) {
	static int m = -1;
	if (m < 0) { const char *v = getenv("MCE_BARMODE"); m = v ? atoi(v) : 0; if (m) fprintf(stderr, "mcopt-metal4: MEASUREMENT barrier mode %d\n", m); }
	return m;
}

static void openBarrier(MceEncoder *e) {
	if (!(barMask() & 1)) return;
	int mode = barMode();
	if (mode == 3 || mode == 4) {
		[(id<MTL4CommandEncoder>) e->real barrierAfterQueueStages:MTLStageAll beforeStages:MTLStageAll visibilityOptions:MTL4VisibilityOptionNone];
		return;
	}
	if (mode == 1 || mode == 2) {
		// 1: compute waits for compute only, render for compute + fragment; 2: compute waits for everything, render for compute + fragment
		if (e->kind == MCE_KIND_RENDER) {
			[(id<MTL4CommandEncoder>) e->real barrierAfterQueueStages:MTLStageDispatch | MTLStageBlit | MTLStageFragment
				beforeStages:MTLStageVertex | MTLStageObject | MTLStageMesh | MTLStageFragment | MTLStageTile visibilityOptions:MTL4VisibilityOptionDevice];
		} else {
			[(id<MTL4CommandEncoder>) e->real barrierAfterQueueStages:(mode == 1 ? MTLStageDispatch | MTLStageBlit : MTLStageAll)
				beforeStages:MTLStageDispatch | MTLStageBlit visibilityOptions:MTL4VisibilityOptionDevice];
		}
		return;
	}
	[(id<MTL4CommandEncoder>) e->real barrierAfterQueueStages:MTLStageAll beforeStages:MTLStageAll visibilityOptions:MTL4VisibilityOptionDevice];
}

// ---- tracking ----
// Metal 4 has no hazard tracking. The layer tracks per resource the last encoder that used it (bound it, drew from it, copied it, rendered to it):
// an encoder that uses a resource last used by an earlier, ended encoder of the SAME command buffer waits on that encoder's fence right before its
// next command (Apple: a wait as close to the consuming command as possible; independent encoders keep overlapping). Across command buffers the first
// encoder of each command buffer carries one consumer queue barrier (all stages after all stages): a command buffer committed earlier (the frame's pre
// buffer, the previous frames) is ordered before it, and no fence is ever waited on whose update comes later in queue order (that would hang the GPU).
// Every use is treated as a write (conservative: two readers are ordered too). The upload ring (setBytes) is not tracked: each range is used once.
// Resources that shaders reach only through addresses stored in other buffers are not seen: none in mcopt (everything is bound directly).
// MCE_SYNC=barriers (environment) restores the old conservative barriers around every encoder.
@interface MceSync : NSObject {
@public
	id fence;
	void *cmd;
	void *enc;
	int ended;
}
@end
@implementation MceSync
- (void)dealloc { [fence release]; [super dealloc]; }
@end

static int syncBarriers(void) {
	static int m = -1;
	if (m < 0) {
		const char *v = getenv("MCE_SYNC");
		m = v && strcmp(v, "barriers") == 0;
		if (m) fprintf(stderr, "mcopt-metal4: conservative barriers around every encoder (MCE_SYNC=barriers)\n");
	}
	return m;
}

static MTLStages encStages(MceEncoder *e) {
	return e->kind == MCE_KIND_RENDER ? MTLStageVertex | MTLStageFragment | MTLStageObject | MTLStageMesh | MTLStageTile : MTLStageDispatch | MTLStageBlit;
}

// per resource: its last writer and the readers since (MceUse); a read waits for the writer, a write for the writer and those readers
@interface MceUse : NSObject {
@public
	MceSync *writer;
	MceSync *readers[8];
	int nreaders;  // > 8: too many to wait on one by one (a writer then takes a queue barrier)
}
@end
@implementation MceUse
- (void)dealloc {
	[writer release];
	for (int i = 0; i < nreaders && i < 8; i++) [readers[i] release];
	[super dealloc];
}
@end

// ---- pass timing (MEASUREMENT: MCE_PASSTIME=1 in the environment) ----
// Every encoder writes a timestamp into its command buffer's counter heap when it opens and when it ends; the commit's feedback handler (after the GPU
// finished) resolves the heap and adds each encoder's duration to its label; every 2 s the mean GPU time per presented frame of each label is printed.
#define PT_CAP 1024
static int passTime(void) {
	static int m = -1;
	if (m < 0) { const char *v = getenv("MCE_PASSTIME"); m = v && atoi(v) > 0; if (m) fprintf(stderr, "mcopt-metal4: MEASUREMENT pass timing on\n"); }
	return m;
}
static struct { os_unfair_lock lock; NSMutableDictionary *us, *n; uint64_t frames; double last; double freq; } PT = {.lock = OS_UNFAIR_LOCK_INIT};

static void ptStart(MceEncoder *e) {
	MceCmd *c = e->cmd;
	e->tsIndex = -1;
	if (!passTime() || !c) return;
	if (!c->heap) {
		MTL4CounterHeapDescriptor *d = [[MTL4CounterHeapDescriptor new] autorelease];
		d.type = MTL4CounterHeapTypeTimestamp;
		d.count = PT_CAP * 2;
		NSError *err = nil;
		c->heap = [M.device newCounterHeapWithDescriptor:d error:&err];
		if (!c->heap) { mce_log_once("pass timing: no counter heap"); return; }
		c->heapLabels = [NSMutableArray new];
	}
	NSUInteger i = c->heapLabels.count;
	if (i >= PT_CAP) return;
	[c->heapLabels addObject:@"?"];
	e->tsIndex = (int) i;
	if (e->kind == MCE_KIND_RENDER)
		[(id<MTL4RenderCommandEncoder>) e->real writeTimestampWithGranularity:MTL4TimestampGranularityRelaxed afterStage:MTLRenderStageVertex intoHeap:c->heap atIndex:i * 2];
	else
		[(id<MTL4ComputeCommandEncoder>) e->real writeTimestampWithGranularity:MTL4TimestampGranularityRelaxed intoHeap:c->heap atIndex:i * 2];
}

static void ptEnd(MceEncoder *e) {
	MceCmd *c = e->cmd;
	if (e->tsIndex < 0 || !c || !c->heap) return;
	NSString *l = [(id<MTL4CommandEncoder>) e->real label];
	c->heapLabels[e->tsIndex] = l.length ? l : (e->kind == MCE_KIND_RENDER ? @"render (no label)" : @"compute (no label)");
	NSUInteger i = (NSUInteger) e->tsIndex;
	if (e->kind == MCE_KIND_RENDER)
		[(id<MTL4RenderCommandEncoder>) e->real writeTimestampWithGranularity:MTL4TimestampGranularityRelaxed afterStage:MTLRenderStageFragment intoHeap:c->heap atIndex:i * 2 + 1];
	else
		[(id<MTL4ComputeCommandEncoder>) e->real writeTimestampWithGranularity:MTL4TimestampGranularityRelaxed intoHeap:c->heap atIndex:i * 2 + 1];
}

// on the feedback queue, after the GPU finished the command buffer
static void ptResolve(id heapObj, NSArray *labels, int presented) {
	id<MTL4CounterHeap> heap = heapObj;
	if (!PT.freq) PT.freq = (double) [M.device queryTimestampFrequency];
	NSData *d = labels.count ? [heap resolveCounterRange:NSMakeRange(0, labels.count * 2)] : nil;
	const MTL4TimestampHeapEntry *t = d.bytes;
	os_unfair_lock_lock(&PT.lock);
	if (!PT.us) { PT.us = [NSMutableDictionary new]; PT.n = [NSMutableDictionary new]; }
	for (NSUInteger i = 0; t && i < labels.count; i++) {
		uint64_t a = t[i * 2].timestamp, b = t[i * 2 + 1].timestamp;
		if (!a || !b || b < a) continue;
		double us = (double) (b - a) * 1e6 / PT.freq;
		NSString *k = labels[i];
		PT.us[k] = @([PT.us[k] doubleValue] + us);
		PT.n[k] = @([PT.n[k] longLongValue] + 1);
	}
	PT.frames += presented;
	double now = CFAbsoluteTimeGetCurrent();
	if (PT.last == 0) PT.last = now;
	if (now - PT.last > 2.0 && PT.frames > 0) {
		NSArray *keys = [PT.us keysSortedByValueUsingComparator:^NSComparisonResult(NSNumber *x, NSNumber *y) { return [y compare:x]; }];
		double total = 0;
		for (NSString *k in keys) total += [PT.us[k] doubleValue];
		fprintf(stderr, "mcopt-metal4 passtime: %llu frames, sum of passes %.3f ms/frame\n", (unsigned long long) PT.frames, total / 1000.0 / PT.frames);
		int shown = 0;
		for (NSString *k in keys) {
			if (shown++ >= 30) break;
			fprintf(stderr, "mcopt-metal4 passtime:   %8.1f us/frame  %6.2f /frame  %s\n", [PT.us[k] doubleValue] / PT.frames, [PT.n[k] doubleValue] / PT.frames, k.UTF8String);
		}
		[PT.us removeAllObjects];
		[PT.n removeAllObjects];
		PT.frames = 0;
		PT.last = now;
	}
	os_unfair_lock_unlock(&PT.lock);
}

static CFMutableDictionaryRef trackMap;  // canonical resource (unretained key) -> MceUse (retained)

// (isTex: the call site knows what it binds; asking the runtime, conformsToProtocol:, on every bind was 70 % of the render thread on vanilla's many draws)
static void *canon(id obj, int isTex) {
	if (isTex) {
		id<MTLTexture> t = obj;
		while (t.parentTexture) t = t.parentTexture;  // (views alias their texture)
		if (t.buffer) return (__bridge void *) t.buffer;  // (texture over a buffer)
		return (__bridge void *) t;
	}
	return (__bridge void *) obj;
}

// e waits for s's encoder, if that is an earlier, ended encoder of e's command buffer (anything older is ordered by the command buffer barrier)
// stages: the stages of this encoder that consume the resource (a render pass waits only where it reads: its attachments at the fragment stage, a
// vertex buffer at the vertex stage; waiting at every stage kept a pass's vertex work behind the previous pass's last fragment)
static void waitFor(MceEncoder *e, MceSync *s, MTLStages stages) {
	if (!s || s == e->sync || !s->ended || s->cmd != (__bridge void *) e->cmd || !s->fence) return;
	for (int i = 0; i < e->nwaits; i++) if (e->waits[i] == s->fence && (e->waitStages[i] & stages) == stages) return;
	// encoded right here, before the command this use is for (a draw can be a plain drawPrimitives that goes straight to the real encoder)
	[(id<MTL4CommandEncoder>) e->real waitForFence:s->fence beforeEncoderStages:stages];
	for (int i = 0; i < e->nwaits; i++) if (e->waits[i] == s->fence) { e->waitStages[i] |= stages; return; }
	if (e->nwaits < 16) { e->waits[e->nwaits] = s->fence; e->waitStages[e->nwaits++] = stages; }  // (only to skip repeats)
}

static void use(MceEncoder *e, id obj, int write, int isTex, MTLStages stages) {
	if (!obj || syncBarriers()) return;
	if (!trackMap) trackMap = CFDictionaryCreateMutable(NULL, 0, NULL, &kCFTypeDictionaryValueCallBacks);
	void *k = canon(obj, isTex);
	MceUse *u = (MceUse *) CFDictionaryGetValue(trackMap, k);
	if (!u) {
		u = [MceUse new];
		CFDictionarySetValue(trackMap, k, (__bridge const void *) u);
		[u release];
	}
	MceSync *me = e->sync;
	if (!stages) stages = encStages(e);
	waitFor(e, u->writer, stages);
	if (!write) {
		if (u->writer == me) return;
		for (int i = 0; i < u->nreaders && i < 8; i++) if (u->readers[i] == me) return;
		if (u->nreaders < 8) u->readers[u->nreaders] = [me retain];
		u->nreaders++;
		return;
	}
	if (u->nreaders > 8) {
		// too many readers to wait on each: everything earlier on the queue first
		[(id<MTL4CommandEncoder>) e->real barrierAfterQueueStages:MTLStageAll beforeStages:stages visibilityOptions:MTL4VisibilityOptionDevice];
	} else {
		for (int i = 0; i < u->nreaders; i++) waitFor(e, u->readers[i], stages);
	}
	for (int i = 0; i < u->nreaders && i < 8; i++) [u->readers[i] release];
	u->nreaders = 0;
	if (u->writer != me) {
		[u->writer release];
		u->writer = [me retain];
	}
}

// a bind: compute binds may be written (treated as writes); in a render pass a buffer is read, a texture written only if it was made shader-writable
static const MTLStages stageBits[MCE_STAGES] = {MTLStageVertex, MTLStageFragment, MTLStageObject, MTLStageMesh, MTLStageTile};

// stage: the bind's stage (MCE_VERTEX...), or -1 for every stage of the encoder
static void track(MceEncoder *e, id obj, int isTex, int stage) {
	if (!obj) return;
	int write = e->kind != MCE_KIND_RENDER;
	if (!write && isTex) write = (((id<MTLTexture>) obj).usage & MTLTextureUsageShaderWrite) != 0;
	use(e, obj, write, isTex, e->kind == MCE_KIND_RENDER && stage >= 0 && stage < MCE_STAGES ? stageBits[stage] : 0);
}

// (the waits are encoded by track itself; nothing is left pending)
static void flushWaits(MceEncoder *e) {
	(void) e;
}

static MceEncoder *wrap(id real, int kind) {
	ST.encoders++;
	MceEncoder *e = [MceEncoder new];
	e->real = [real retain];
	e->kind = kind;
	os_unfair_lock_lock(&M.lock);
	if (kind == MCE_KIND_RENDER) {
		// vertex and fragment tables at once; object / mesh / tile only when something is bound to them (table()): Metal snapshots the attached tables
		// at each draw, so unused ones are paid for at every draw
		static const MTLRenderStages stages[MCE_STAGES] = {MTLRenderStageVertex, MTLRenderStageFragment, MTLRenderStageObject, MTLRenderStageMesh, MTLRenderStageTile};
		for (int i = 0; i < 2; i++) {
			e->tables[i] = takeTable(0);
			[(id<MTL4RenderCommandEncoder>) real setArgumentTable:e->tables[i] atStages:stages[i]];
		}
	} else if (kind != MCE_KIND_BLIT) {
		e->tables[0] = takeTable(0);
		[(id<MTL4ComputeCommandEncoder>) real setArgumentTable:e->tables[0]];
	}
	os_unfair_lock_unlock(&M.lock);
	if (syncBarriers()) openBarrier(e);
	return e;
}

// fences are reused once the frame that last used them completed on the GPU (one per encoder: creating and freeing them was ~1600 objects a second)
static struct { id f; MceCmd *cmd; } fencePool[512];
static int nFencePool;

static id takeFence(void) {
	uint64_t done = completedValue();
	for (int i = 0; i < nFencePool; i++) {
		MceCmd *c = fencePool[i].cmd;
		if (c && c->value && c->value <= done) {
			id f = fencePool[i].f;
			[c release];
			fencePool[i] = fencePool[--nFencePool];
			return f;
		}
	}
	ST.fences++;
	return [M.device newFence];
}

static void giveFence(id f, MceCmd *cmd) {
	if (nFencePool < 512) {
		fencePool[nFencePool].f = f;
		fencePool[nFencePool].cmd = [cmd retain];
		nFencePool++;
	} else {
		mce4_release(f);
	}
}

static int mergeCompute(void) {
	static int m = -1;
	if (m < 0) { const char *v = getenv("MCE_MERGE"); m = !(v && strcmp(v, "0") == 0); }
	return m && !syncBarriers() && !passTime();
}

// ends the compute encoder kept open on c (its fence updated: encoders that depend on what ran in it wait on it)
static void closeOpen(MceCmd *c) {
	if (!c || !c->openReal) return;
	id<MTL4ComputeCommandEncoder> r = c->openReal;
	[r updateFence:c->openFence afterEncoderStages:MTLStageDispatch | MTLStageBlit];
	[r endEncoding];
	[r release];
	c->openReal = nil;
	giveFence(c->openFence, c);
	c->openFence = nil;
	[(MceSync *) c->openSync release];
	c->openSync = nil;
}

void mce4_cmd_flush(id cmd) {
	closeOpen(cmd);
}

// the tracking state of a new encoder on command buffer cmd (wrap's caller)
static MceEncoder *tracked(MceEncoder *e, id cmd) {
	if (syncBarriers()) return e;
	MceCmd *c = cmd;
	e->cmd = cmd;
	e->fence = takeFence();
	MceSync *s = [MceSync new];
	s->fence = [e->fence retain];
	s->cmd = (__bridge void *) cmd;
	s->enc = (__bridge void *) e;
	e->sync = s;
	if (!c->opened) {
		c->opened = 1;
		[(id<MTL4CommandEncoder>) e->real barrierAfterQueueStages:MTLStageAll beforeStages:MTLStageAll visibilityOptions:MTL4VisibilityOptionDevice];
	}
	ptStart(e);
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
		closeOpen(c);
		id<MTL4RenderCommandEncoder> r = [c->cb renderCommandEncoderWithDescriptor:d];
		if (!r) { fprintf(stderr, "mcopt-metal4: render encoder creation failed\n"); abort(); }
		MceEncoder *e = tracked(wrap(r, MCE_KIND_RENDER), cmd);
		// the attachments: the pass loads / stores them, so their earlier users are waited for before anything of this pass
		for (int i = 0; i < MAX_COLORS; i++) if (rp.colorAttachments[i].texture) use(e, rp.colorAttachments[i].texture, 1, 1, MTLStageFragment);
		if (rp.depthAttachment.texture) use(e, rp.depthAttachment.texture, 1, 1, MTLStageFragment);
		if (rp.stencilAttachment.texture) use(e, rp.stencilAttachment.texture, 1, 1, MTLStageFragment);
		flushWaits(e);
		return e;
	}
}

id mce4_new_compute_encoder(id cmd, MTLDispatchType type) {
	MceCmd *c = cmd;
	int kind = type == MTLDispatchTypeConcurrent ? MCE_KIND_CONCURRENT : MCE_KIND_SERIAL;
	if (mergeCompute() && c->openReal) {
		// continue in the open encoder: one intra-pass barrier orders this encoder's commands after the previous one's (what separate encoders did)
		id<MTL4ComputeCommandEncoder> r = c->openReal;
		[r barrierAfterEncoderStages:MTLStageDispatch | MTLStageBlit beforeEncoderStages:MTLStageDispatch | MTLStageBlit visibilityOptions:MTL4VisibilityOptionDevice];
		MceEncoder *e = wrap(r, kind);
		e->merged = 1;
		e->cmd = cmd;
		e->fence = [c->openFence retain];
		e->sync = [c->openSync retain];
		return e;
	}
	closeOpen(c);
	return tracked(wrap([c->cb computeCommandEncoder], kind), cmd);
}

id mce4_new_blit_encoder(id cmd) {
	MceCmd *c = cmd;
	if (mergeCompute() && c->openReal) {
		id<MTL4ComputeCommandEncoder> r = c->openReal;
		[r barrierAfterEncoderStages:MTLStageDispatch | MTLStageBlit beforeEncoderStages:MTLStageDispatch | MTLStageBlit visibilityOptions:MTL4VisibilityOptionDevice];
		MceEncoder *e = wrap(r, MCE_KIND_BLIT);
		e->merged = 1;
		e->cmd = cmd;
		e->fence = [c->openFence retain];
		e->sync = [c->openSync retain];
		return e;
	}
	closeOpen(c);
	return tracked(wrap([c->cb computeCommandEncoder], MCE_KIND_BLIT), cmd);
}

void mce4_end_encoder(id h) {
	MceEncoder *e = h;
	if (e->ended) return;
	e->ended = 1;
	ptEnd(e);
	if (mergeCompute() && e->kind != MCE_KIND_RENDER && e->cmd && ((MceCmd *) e->cmd)->cb) {
		MceCmd *c = e->cmd;
		if (!e->merged) {
			// the first of a run: its real encoder, fence and record become the command buffer's open encoder
			c->openReal = [e->real retain];
			c->openFence = [e->fence retain];
			c->openSync = [e->sync retain];
		}
		((MceSync *) e->sync)->ended = 1;  // (later encoders of another kind close it first, so its fence is updated before they wait on it)
		[(MceSync *) e->sync release];
		e->sync = nil;
		[e->fence release];
		e->fence = nil;
		[e->real release];
		e->real = nil;
		os_unfair_lock_lock(&M.lock);
		giveTables(e);
		os_unfair_lock_unlock(&M.lock);
		return;
	}
	if (!syncBarriers() && e->fence) {
		flushWaits(e);  // (waits for resources bound but never used by a command: harmless)
		[(id<MTL4CommandEncoder>) e->real updateFence:e->fence afterEncoderStages:encStages(e)];
		((MceSync *) e->sync)->ended = 1;
		[(MceSync *) e->sync release];  // (the resource map keeps it while it is some resource's last user)
		e->sync = nil;
		giveFence(e->fence, e->cmd);  // (reused once that command buffer completed)
		e->fence = nil;
	}
	// and the other half of the conservative barrier: later encoders wait for all of this one
	if (syncBarriers() && (barMask() & 2)) [(id<MTL4CommandEncoder>) e->real barrierAfterStages:MTLStageAll beforeQueueStages:MTLStageAll
		visibilityOptions:(barMode() == 4 ? MTL4VisibilityOptionNone : MTL4VisibilityOptionDevice)];
	[(id<MTL4CommandEncoder>) e->real endEncoding];
	[e->real release];
	e->real = nil;
	os_unfair_lock_lock(&M.lock);
	giveTables(e);
	os_unfair_lock_unlock(&M.lock);
}

// ---- bindings ----
static const MTLRenderStages renderStages[MCE_STAGES] = {MTLRenderStageVertex, MTLRenderStageFragment, MTLRenderStageObject, MTLRenderStageMesh, MTLRenderStageTile};

static void attach(MceEncoder *e, int stage, id<MTL4ArgumentTable> t) {
	if (e->kind == MCE_KIND_RENDER) [(id<MTL4RenderCommandEncoder>) e->real setArgumentTable:t atStages:renderStages[stage]];
	else [(id<MTL4ComputeCommandEncoder>) e->real setArgumentTable:t];
}

// what = 0 buffer, 1 texture, 2 sampler: the stage's table, made large first if index is beyond the small range; nil if beyond the large one
static id<MTL4ArgumentTable> table(MceEncoder *e, int stage, int what, NSUInteger index) {
	static const char *const over[3] = {"buffer binding index >= 31: dropped", "texture binding index >= 48: dropped", "sampler binding index >= 16: dropped"};
	if (index >= (NSUInteger) bigN[what]) { mce_log_once(over[what]); return nil; }
	id<MTL4ArgumentTable> t = e->tables[stage];
	if (!t) {
		if (e->kind != MCE_KIND_RENDER || stage < 2 || stage >= MCE_STAGES) {
			logUnported("mce (binding on an encoder kind without that stage)", @selector(description));
			return nil;
		}
		os_unfair_lock_lock(&M.lock);
		t = e->tables[stage] = takeTable(0);
		os_unfair_lock_unlock(&M.lock);
		attach(e, stage, t);
	}
	if (e->bigMask >> stage & 1 || index < (NSUInteger) smallN[what]) return t;
	ST.tableGrows++;
	os_unfair_lock_lock(&M.lock);
	id<MTL4ArgumentTable> big = takeTable(1);
	[M.tablePool addObject:t];  // (Metal snapshotted it at the draws so far)
	os_unfair_lock_unlock(&M.lock);
	[t release];
	for (int i = 0; i < smallN[0]; i++) [big setAddress:e->shB[stage][i] atIndex:(NSUInteger) i];
	for (int i = 0; i < smallN[1]; i++) [big setTexture:e->shT[stage][i] atIndex:(NSUInteger) i];
	for (int i = 0; i < smallN[2]; i++) [big setSamplerState:e->shS[stage][i] atIndex:(NSUInteger) i];
	e->tables[stage] = big;
	e->bigMask |= 1 << stage;
	attach(e, stage, big);
	return big;
}

static inline void setAddr(MceEncoder *e, int stage, uint64_t a, NSUInteger index) {
	id<MTL4ArgumentTable> t = table(e, stage, 0, index);
	if (!t) return;
	if (!(e->bigMask >> stage & 1)) e->shB[stage][index] = a;
	[t setAddress:a atIndex:index];
}

void mce4_set_bytes(id h, int stage, const void *bytes, NSUInteger length, NSUInteger index) {
	if (index >= TABLE_BUFFERS) { mce_log_once("buffer binding index >= 31: dropped"); return; }
	setAddr(h, stage, upload(bytes, length, NULL, NULL), index);
}

void mce4_set_bytes2(id h, int stageA, int stageB, const void *bytes, NSUInteger length, NSUInteger index) {
	if (index >= TABLE_BUFFERS) { mce_log_once("buffer binding index >= 31: dropped"); return; }
	uint64_t a = upload(bytes, length, NULL, NULL);
	setAddr(h, stageA, a, index);
	setAddr(h, stageB, a, index);
}

void mce4_set_buffer(id h, int stage, id<MTLBuffer> buffer, NSUInteger offset, NSUInteger index) {
	if (index >= TABLE_BUFFERS) { mce_log_once("buffer binding index >= 31: dropped"); return; }
	track(h, buffer, 0, stage);
	setAddr(h, stage, buffer ? buffer.gpuAddress + offset : 0, index);
}

void mce4_set_texture(id h, int stage, id<MTLTexture> texture, NSUInteger index) {
	MceEncoder *e = h;
	id<MTL4ArgumentTable> t = table(e, stage, 1, index);
	if (!t) return;
	MTLResourceID none = {0}, r = texture ? texture.gpuResourceID : none;
	track(h, texture, 1, stage);
	if (!(e->bigMask >> stage & 1)) e->shT[stage][index] = r;
	[t setTexture:r atIndex:index];
}

void mce4_set_sampler(id h, int stage, id<MTLSamplerState> sampler, NSUInteger index) {
	MceEncoder *e = h;
	id<MTL4ArgumentTable> t = table(e, stage, 2, index);
	if (!t) return;
	MTLResourceID none = {0}, r = sampler ? sampler.gpuResourceID : none;
	if (!(e->bigMask >> stage & 1)) e->shS[stage][index] = r;
	[t setSamplerState:r atIndex:index];
}

// ---- draws and dispatches ----
void mce4_draw_indexed(id h, MTLPrimitiveType p, NSUInteger count, MTLIndexType t, id<MTLBuffer> ib, NSUInteger off, NSUInteger inst, NSInteger bv, NSUInteger bi) {
	use(h, ib, 0, 0, MTLStageVertex | MTLStageObject | MTLStageMesh);
	flushWaits(h);
	[(id<MTL4RenderCommandEncoder>) ((MceEncoder *) h)->real drawIndexedPrimitives:p indexCount:count indexType:t indexBuffer:ib.gpuAddress + off
		indexBufferLength:ib.length - off instanceCount:inst baseVertex:bv baseInstance:bi];
}

void mce4_draw_indirect(id h, MTLPrimitiveType p, id<MTLBuffer> ind, NSUInteger indOff) {
	use(h, ind, 0, 0, MTLStageVertex | MTLStageObject | MTLStageMesh);
	flushWaits(h);
	[(id<MTL4RenderCommandEncoder>) ((MceEncoder *) h)->real drawPrimitives:p indirectBuffer:ind.gpuAddress + indOff];
}

void mce4_draw_indexed_indirect(id h, MTLPrimitiveType p, MTLIndexType t, id<MTLBuffer> ib, NSUInteger off, id<MTLBuffer> ind, NSUInteger indOff) {
	use(h, ib, 0, 0, MTLStageVertex | MTLStageObject | MTLStageMesh);
	use(h, ind, 0, 0, MTLStageVertex | MTLStageObject | MTLStageMesh);
	flushWaits(h);
	[(id<MTL4RenderCommandEncoder>) ((MceEncoder *) h)->real drawIndexedPrimitives:p indexType:t indexBuffer:ib.gpuAddress + off
		indexBufferLength:ib.length - off indirectBuffer:ind.gpuAddress + indOff];
}

void mce4_draw_mesh_indirect(id h, id<MTLBuffer> ind, NSUInteger indOff, MTLSize obj, MTLSize mesh) {
	use(h, ind, 0, 0, MTLStageVertex | MTLStageObject | MTLStageMesh);
	flushWaits(h);
	[(id<MTL4RenderCommandEncoder>) ((MceEncoder *) h)->real drawMeshThreadgroupsWithIndirectBuffer:ind.gpuAddress + indOff
		threadsPerObjectThreadgroup:obj threadsPerMeshThreadgroup:mesh];
}

void mce4_dispatch_threads(id h, MTLSize grid, MTLSize tg) { [(MceEncoder *) h dispatchThreads:grid threadsPerThreadgroup:tg]; }
void mce4_dispatch_threadgroups(id h, MTLSize groups, MTLSize tg) { [(MceEncoder *) h dispatchThreadgroups:groups threadsPerThreadgroup:tg]; }
void mce4_dispatch_threadgroups_indirect(id h, id<MTLBuffer> ind, NSUInteger indOff, MTLSize tg) {
	MceEncoder *e = h;
	use(e, ind, 0, 0, 0);
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
