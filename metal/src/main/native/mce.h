// mce: the Metal 3 / Metal 4 compatibility layer (-Dmcopt.metal4=true selects Metal 4; default Metal 3).
//
// Every function here has TODAY's (Metal 3) semantics. With Metal 3 selected (mce_metal4 == 0) each one is a direct call of the
// Metal 3 method it is named after (static inline: one predictable branch, no allocation, no extra message). With Metal 4 it is
// translated onto the MTL4 API (mce.m). Both backends live in the same binary; the choice is made once, before mc_create.
//
// HANDLES. Under Metal 3 a handle is the plain Metal object. Under Metal 4:
//   - a command buffer handle (mc_frame_cmd, mc_pre, mce_new_cmd, enc->cmd/enc->pre) is an MceCmd: an MTL4CommandBuffer with its
//     own MTL4CommandAllocator (from a pool, reset only once the GPU finished every command buffer that used it).
//   - an encoder handle (mce_new_*_encoder, enc->render/enc->blit/enc->preCompute/enc->preBlit, mc_profiled_compute/blit,
//     mc_render_suspend) is an MceEncoder wrapping an MTL4RenderCommandEncoder or MTL4ComputeCommandEncoder (blits run on a compute
//     encoder in Metal 4) plus its argument tables.
//   Keep typing them as today (id<MTLRenderCommandEncoder> etc.): they are objects (retain/release/autorelease work as usual).
//
// WHAT TO CALL. Port every Metal call on a handle to the mce_* function below with the same name/meaning. Selectors that exist with
// identical signatures in Metal 4 (setRenderPipelineState:, setDepthStencilState:, setCullMode:, setTriangleFillMode:, setDepthBias:
// slopeScale:clamp:, setDepthClipMode:, setScissorRect:, setViewport:, setFrontFacingWinding:, setColorStoreAction:atIndex:,
// setDepthStoreAction:, setVisibilityResultMode:offset:, setComputePipelineState:, setThreadgroupMemoryLength:atIndex: (compute),
// setObjectThreadgroupMemoryLength:atIndex:, drawPrimitives:vertexStart:vertexCount:[instanceCount:[baseInstance:]],
// drawMeshThreadgroups:threadsPerObjectThreadgroup:threadsPerMeshThreadgroup:, push/popDebugGroup, insertDebugSignpost:, label)
// go to [mce_raw(handle) ...]: the real encoder, no wrapper hop. Under Metal 4 any Metal 3-only selector sent straight to an
// encoder handle is logged once ("mcopt-metal4: unported ...") and DROPPED (so an unported file runs but renders wrong, it does
// not crash). Command buffer handles do accept today's Metal 3 command buffer selectors under Metal 4 (computeCommandEncoder[...],
// blitCommandEncoder[...], renderCommandEncoderWithDescriptor: (an MTLRenderPassDescriptor), commit, waitUntilCompleted,
// addCompletedHandler:, presentDrawable:, encodeSignalEvent:value:, encodeWaitForEvent:value:, status, GPUStartTime/GPUEndTime),
// but prefer the mce_cmd_* functions.
//
// RULES (Metal 4 path; all hold trivially on Metal 3):
//  1. Bytes: mce_set_*_bytes copies `bytes` immediately (into a per-frame upload ring, 256-byte aligned, length rounded up to 16 with
//     zero padding) and binds its GPU address. The caller's memory can be reused at once. No 4 KiB limit.
//  2. Bindings persist per encoder, as in Metal 3: each encoder has its own argument tables (one per stage: vertex, fragment, object,
//     mesh, tile; compute: one), set once on the encoder. Metal snapshots the table at each draw/dispatch, so rebinding between
//     draws works exactly like Metal 3. A new encoder starts with unbound (nil) slots. Limits: buffers index < 31, textures < 48,
//     samplers < 16 (beyond: logged and dropped).
//  3. Residency: Metal 4 has no implicit residency. Every buffer/texture/heap the GPU touches must be in a residency set the queue
//     carries. Call mce_make_resident(obj) right after creating any MTLBuffer/MTLTexture (newBuffer*, newTexture*, texture views,
//     buffer-backed textures); mcmetal.m's allocation functions (mc_buffer_new, mc_buffer_private, mc_texture_new, mc_texture_view,
//     mc_texture_buffer) already do. useResource:/useResources: are no-ops under Metal 4 (mce_use_resource_*): residency is what
//     makes GPU-address-only (bindless) access legal. Release resources through mc_release (or mce_release): under Metal 4 the
//     object is only evicted and released once the GPU finished every frame that could use it (MTL4 command buffers retain nothing).
//     A native-side [obj release] of a GPU resource that may still be in flight must become mce_release(obj).
//  4. Barriers (correctness first): Metal 4 tracks no hazards. The layer puts a full queue barrier (all stages after all stages,
//     device visibility) at the start and end of EVERY encoder, so encoders and command buffers execute strictly in order, like
//     Metal 3 tracked resources (more conservative: frames no longer overlap on the GPU). Inside an encoder:
//       - a SERIAL compute encoder (mce_new_compute_encoder(.., MTLDispatchTypeSerial), Metal 3's default) and every blit encoder get
//         a dispatch/blit barrier before each command after the first, as Metal 3 serial semantics require. Call dispatches and
//         blits through mce_dispatch_* / mce_copy_* / mce_fill_buffer / mce_generate_mipmaps (sending the identical selector to the
//         handle itself also works: the wrapper intercepts them; do NOT send them to mce_raw(handle)).
//       - a CONCURRENT compute encoder has no implicit ordering (as in Metal 3): keep today's memoryBarrierWithScope: calls, as
//         mce_memory_barrier(handle, scope).
//       - render encoders: today's memoryBarrierWithScope:afterStages:beforeStages: becomes mce_memory_barrier_render(...).
//       - fences: mce_update_fence/mce_wait_fence (+ _render variants with stages).
//     mce_barrier_full(handle) adds an explicit all-stages barrier (no-op on Metal 3) for spots that need one beyond the above.
//  5. Command buffers commit in call order on one MTL4CommandQueue; `pre` before `cmd` in mc_enc_commit as today. Standalone ones
//     (today's [ctx->queue commandBuffer]) are mce_new_cmd(ctx) + mce_cmd_commit; never use ctx->queue directly under Metal 4 (it is
//     a Metal 3 queue: nothing orders it against the Metal 4 queue).
//  6. Encoders: one open encoder per command buffer at a time (as in Metal 3). mce_new_*_encoder returns a RETAINED (+1) handle; end
//     it with mce_end_encoder(h) and then [h release].
//  7. Not available under Metal 4 (logged once, then off): GPU timestamp profiling (mc_profile_begin returns 0), GPU counting
//     (-Dmcopt.metal.gpuCount), -Dmcopt.metal.passTest, -Dmcopt.cpu.cmdAhead, -Dmcopt.metal.cbdiag command buffer reports,
//     MetalFX (mcfx_supported returns 0; MetalFX has MTL4FXSpatialScaler for later), the second present queue
//     (-Dmcopt.metal.presentQueue: presents go on the frame's queue instead).
#ifndef MCE_H
#define MCE_H
#import "mcmetal.h"
#import <QuartzCore/CAMetalLayer.h>

extern int mce_metal4;  // 1: Metal 4 backend (set by mce_enable before mc_create; never changes after)

// Called from Java (Native.create) before mc_create when -Dmcopt.metal4=true. mc_create then checks the device and the OS and falls
// back to Metal 3 (logged) if Metal 4 isn't there. Returns the flag as set.
int mce_enable(int on);
// mc_create's Metal 4 setup (queue, timeline event, pools, residency set); 0 on failure (then mce_metal4 is 0 again).
int mce_init(Ctx *ctx);

enum { MCE_VERTEX, MCE_FRAGMENT, MCE_OBJECT, MCE_MESH, MCE_TILE, MCE_STAGES };
#define MCE_COMPUTE MCE_VERTEX  // a compute encoder's single table
enum { MCE_KIND_RENDER = 1, MCE_KIND_SERIAL, MCE_KIND_CONCURRENT, MCE_KIND_BLIT };

// The Metal 4 wrappers (only ever instantiated under Metal 4). Fields public for the inline helpers; don't touch them elsewhere.
@interface MceEncoder : NSObject {
@public
	id real;  // id<MTL4RenderCommandEncoder> (MCE_KIND_RENDER) or id<MTL4ComputeCommandEncoder> (the others)
	int kind;
	int pending;  // serial/blit: a command was encoded since the last intra-encoder barrier
	int ended;
	id tables[MCE_STAGES];  // id<MTL4ArgumentTable>, nil where the kind has none
	// automatic tracking (mce.m "tracking"): this encoder's fence (updated at its end), the command buffer it is on, the fences of earlier encoders
	// of that command buffer it must wait for before its next command
	id fence;
	id cmd;
	id sync;  // MceSync, the record the resource map points at
	id waits[16];
	int nwaits;
	int tsIndex;  // MCE_PASSTIME: its start timestamp's index in the command buffer's heap (-1 none)
	int merged;  // its real encoder is shared with the previous compute encoder(s) of the command buffer
}
@end

@interface MceCmd : NSObject {
@public
	id cb;  // id<MTL4CommandBuffer>; nil once committed
	int alloc;  // allocator pool slot
	uint64_t value;  // timeline value signaled when this command buffer (and all before it) finished; 0 until committed
	int timesReady;  // set (atomically) by the commit feedback handler
	double gpuStart, gpuEnd;
	NSMutableArray *completed, *waits, *signals;  // completion blocks; {event, value} pairs to wait on before / signal after the commit
	id drawable;  // id<CAMetalDrawable> to present with this commit
	int opened;  // an encoder was opened on it (the first gets the queue barrier that orders it after earlier command buffers)
	id heap;  // MCE_PASSTIME: id<MTL4CounterHeap> of its encoders' timestamps
	// compute encoders merged (Apple: group compute-class work, every encoder boundary costs): a compute encoder that ended is kept open here
	// until the next encoder of another kind, a commit, or mce4_cmd_flush; the next compute encoder continues in it after an intra-pass barrier
	id openReal, openSync, openFence;
	NSMutableArray *heapLabels;  // one label per encoder (timestamps 2i, 2i + 1)
}
@end

// ---- handles ----
// The encoder to send identical-signature selectors to (see WHAT TO CALL).
static inline id mce_raw(id h) { return mce_metal4 && h ? ((MceEncoder *) h)->real : h; }

// ---- command buffers ----
id mce4_new_cmd(Ctx *ctx);
void mce4_cmd_commit(id c);
void mce4_cmd_flush(id cmd);  // ends a compute encoder the layer kept open for merging (before encoding into the command buffer directly)
void mce4_cmd_wait(id c);
int mce4_cmd_done(id c);
double mce4_cmd_gpu_start(id c);
double mce4_cmd_gpu_end(id c);
void mce4_cmd_on_completed(id c, void (^block)(double gpuStart, double gpuEnd));
void mce4_cmd_present(id c, id<CAMetalDrawable> drawable);
void mce4_cmd_signal_event(id c, id<MTLEvent> event, uint64_t value);
void mce4_cmd_wait_event(id c, id<MTLEvent> event, uint64_t value);

// A standalone command buffer (today: [[ctx->queue commandBuffer] retain]). Retained (+1).
static inline id mce_new_cmd(Ctx *ctx) {
	if (mce_metal4) return mce4_new_cmd(ctx);
	@autoreleasepool { return [[ctx->queue commandBuffer] retain]; }
}
static inline void mce_cmd_commit(id c) { if (mce_metal4) mce4_cmd_commit(c); else [(id<MTLCommandBuffer>) c commit]; }
static inline void mce_cmd_wait(id c) { if (mce_metal4) mce4_cmd_wait(c); else [(id<MTLCommandBuffer>) c waitUntilCompleted]; }
static inline int mce_cmd_done(id c) {
	return mce_metal4 ? mce4_cmd_done(c) : ((id<MTLCommandBuffer>) c).status >= MTLCommandBufferStatusCompleted;
}
static inline double mce_cmd_gpu_start(id c) { return mce_metal4 ? mce4_cmd_gpu_start(c) : ((id<MTLCommandBuffer>) c).GPUStartTime; }
static inline double mce_cmd_gpu_end(id c) { return mce_metal4 ? mce4_cmd_gpu_end(c) : ((id<MTLCommandBuffer>) c).GPUEndTime; }
// Runs block when the command buffer completed (Metal 3: addCompletedHandler; Metal 4: the commit's feedback handler), with its
// GPU start and end times (host clock, seconds). Any thread.
static inline void mce_cmd_on_completed(id c, void (^block)(double gpuStart, double gpuEnd)) {
	if (mce_metal4) mce4_cmd_on_completed(c, block);
	else [(id<MTLCommandBuffer>) c addCompletedHandler:^(id<MTLCommandBuffer> b) { block(b.GPUStartTime, b.GPUEndTime); }];
}
// presentDrawable: (Metal 4: waitForDrawable before the commit, signalDrawable + present after it).
static inline void mce_cmd_present(id c, id<CAMetalDrawable> d) {
	if (mce_metal4) mce4_cmd_present(c, d);
	else [(id<MTLCommandBuffer>) c presentDrawable:d];
}
// encodeSignalEvent / encodeWaitForEvent: Metal 4 has them only at queue level, so they apply at this command buffer's commit
// (signal: after all of it; wait: before any of it). Same meaning as Metal 3's at the end / start of the command buffer only.
static inline void mce_cmd_signal_event(id c, id<MTLEvent> e, uint64_t v) {
	if (mce_metal4) mce4_cmd_signal_event(c, e, v);
	else [(id<MTLCommandBuffer>) c encodeSignalEvent:e value:v];
}
static inline void mce_cmd_wait_event(id c, id<MTLEvent> e, uint64_t v) {
	if (mce_metal4) mce4_cmd_wait_event(c, e, v);
	else [(id<MTLCommandBuffer>) c encodeWaitForEvent:e value:v];
}

// ---- encoders (all return RETAINED handles: mce_end_encoder, then release) ----
id mce4_new_render_encoder(id cmd, MTLRenderPassDescriptor *rp);
id mce4_new_compute_encoder(id cmd, MTLDispatchType type);
id mce4_new_blit_encoder(id cmd);
void mce4_end_encoder(id h);

// rp is today's MTLRenderPassDescriptor (converted under Metal 4: attachments, render target size, sample count, tile and
// threadgroup sizes, rasterization rate map, visibility buffer; sample buffer attachments are dropped).
static inline id mce_new_render_encoder(id cmd, MTLRenderPassDescriptor *rp) {
	if (mce_metal4) return mce4_new_render_encoder(cmd, rp);
	return [[(id<MTLCommandBuffer>) cmd renderCommandEncoderWithDescriptor:rp] retain];
}
// type: MTLDispatchTypeSerial (= today's computeCommandEncoder) or MTLDispatchTypeConcurrent.
static inline id mce_new_compute_encoder(id cmd, MTLDispatchType type) {
	if (mce_metal4) return mce4_new_compute_encoder(cmd, type);
	return [(type == MTLDispatchTypeSerial ? [(id<MTLCommandBuffer>) cmd computeCommandEncoder]
		: [(id<MTLCommandBuffer>) cmd computeCommandEncoderWithDispatchType:type]) retain];
}
static inline id mce_new_blit_encoder(id cmd) {
	if (mce_metal4) return mce4_new_blit_encoder(cmd);
	return [[(id<MTLCommandBuffer>) cmd blitCommandEncoder] retain];
}
static inline void mce_end_encoder(id h) { if (mce_metal4) mce4_end_encoder(h); else [(id<MTLCommandEncoder>) h endEncoding]; }

// ---- bindings (h: an encoder handle) ----
void mce4_set_bytes(id h, int stage, const void *bytes, NSUInteger length, NSUInteger index);
void mce4_set_bytes2(id h, int stageA, int stageB, const void *bytes, NSUInteger length, NSUInteger index);
void mce4_set_buffer(id h, int stage, id<MTLBuffer> buffer, NSUInteger offset, NSUInteger index);
void mce4_set_texture(id h, int stage, id<MTLTexture> texture, NSUInteger index);
void mce4_set_sampler(id h, int stage, id<MTLSamplerState> sampler, NSUInteger index);

#define MCE_R(h) ((id<MTLRenderCommandEncoder>) (h))
#define MCE_C(h) ((id<MTLComputeCommandEncoder>) (h))
#define MCE_B(h) ((id<MTLBlitCommandEncoder>) (h))

// vertex
static inline void mce_set_vertex_bytes(id h, const void *b, NSUInteger n, NSUInteger i) { if (mce_metal4) mce4_set_bytes(h, MCE_VERTEX, b, n, i); else [MCE_R(h) setVertexBytes:b length:n atIndex:i]; }
static inline void mce_set_vertex_buffer(id h, id<MTLBuffer> b, NSUInteger o, NSUInteger i) { if (mce_metal4) mce4_set_buffer(h, MCE_VERTEX, b, o, i); else [MCE_R(h) setVertexBuffer:b offset:o atIndex:i]; }
static inline void mce_set_vertex_texture(id h, id<MTLTexture> t, NSUInteger i) { if (mce_metal4) mce4_set_texture(h, MCE_VERTEX, t, i); else [MCE_R(h) setVertexTexture:t atIndex:i]; }
static inline void mce_set_vertex_sampler(id h, id<MTLSamplerState> s, NSUInteger i) { if (mce_metal4) mce4_set_sampler(h, MCE_VERTEX, s, i); else [MCE_R(h) setVertexSamplerState:s atIndex:i]; }
// fragment
static inline void mce_set_fragment_bytes(id h, const void *b, NSUInteger n, NSUInteger i) { if (mce_metal4) mce4_set_bytes(h, MCE_FRAGMENT, b, n, i); else [MCE_R(h) setFragmentBytes:b length:n atIndex:i]; }
static inline void mce_set_fragment_buffer(id h, id<MTLBuffer> b, NSUInteger o, NSUInteger i) { if (mce_metal4) mce4_set_buffer(h, MCE_FRAGMENT, b, o, i); else [MCE_R(h) setFragmentBuffer:b offset:o atIndex:i]; }
static inline void mce_set_fragment_texture(id h, id<MTLTexture> t, NSUInteger i) { if (mce_metal4) mce4_set_texture(h, MCE_FRAGMENT, t, i); else [MCE_R(h) setFragmentTexture:t atIndex:i]; }
static inline void mce_set_fragment_sampler(id h, id<MTLSamplerState> s, NSUInteger i) { if (mce_metal4) mce4_set_sampler(h, MCE_FRAGMENT, s, i); else [MCE_R(h) setFragmentSamplerState:s atIndex:i]; }
// the same bytes to vertex and fragment (today: setVertexBytes + setFragmentBytes; Metal 4: one copy, bound twice)
static inline void mce_set_vertex_fragment_bytes(id h, const void *b, NSUInteger n, NSUInteger i) {
	if (mce_metal4) mce4_set_bytes2(h, MCE_VERTEX, MCE_FRAGMENT, b, n, i);
	else { [MCE_R(h) setVertexBytes:b length:n atIndex:i]; [MCE_R(h) setFragmentBytes:b length:n atIndex:i]; }
}
// object (mesh pipelines' object stage)
static inline void mce_set_object_bytes(id h, const void *b, NSUInteger n, NSUInteger i) { if (mce_metal4) mce4_set_bytes(h, MCE_OBJECT, b, n, i); else [MCE_R(h) setObjectBytes:b length:n atIndex:i]; }
static inline void mce_set_object_buffer(id h, id<MTLBuffer> b, NSUInteger o, NSUInteger i) { if (mce_metal4) mce4_set_buffer(h, MCE_OBJECT, b, o, i); else [MCE_R(h) setObjectBuffer:b offset:o atIndex:i]; }
static inline void mce_set_object_texture(id h, id<MTLTexture> t, NSUInteger i) { if (mce_metal4) mce4_set_texture(h, MCE_OBJECT, t, i); else [MCE_R(h) setObjectTexture:t atIndex:i]; }
static inline void mce_set_object_sampler(id h, id<MTLSamplerState> s, NSUInteger i) { if (mce_metal4) mce4_set_sampler(h, MCE_OBJECT, s, i); else [MCE_R(h) setObjectSamplerState:s atIndex:i]; }
// mesh
static inline void mce_set_mesh_bytes(id h, const void *b, NSUInteger n, NSUInteger i) { if (mce_metal4) mce4_set_bytes(h, MCE_MESH, b, n, i); else [MCE_R(h) setMeshBytes:b length:n atIndex:i]; }
static inline void mce_set_mesh_buffer(id h, id<MTLBuffer> b, NSUInteger o, NSUInteger i) { if (mce_metal4) mce4_set_buffer(h, MCE_MESH, b, o, i); else [MCE_R(h) setMeshBuffer:b offset:o atIndex:i]; }
static inline void mce_set_mesh_texture(id h, id<MTLTexture> t, NSUInteger i) { if (mce_metal4) mce4_set_texture(h, MCE_MESH, t, i); else [MCE_R(h) setMeshTexture:t atIndex:i]; }
static inline void mce_set_mesh_sampler(id h, id<MTLSamplerState> s, NSUInteger i) { if (mce_metal4) mce4_set_sampler(h, MCE_MESH, s, i); else [MCE_R(h) setMeshSamplerState:s atIndex:i]; }
// tile (tile shaders: lodseam.m)
static inline void mce_set_tile_bytes(id h, const void *b, NSUInteger n, NSUInteger i) { if (mce_metal4) mce4_set_bytes(h, MCE_TILE, b, n, i); else [MCE_R(h) setTileBytes:b length:n atIndex:i]; }
static inline void mce_set_tile_buffer(id h, id<MTLBuffer> b, NSUInteger o, NSUInteger i) { if (mce_metal4) mce4_set_buffer(h, MCE_TILE, b, o, i); else [MCE_R(h) setTileBuffer:b offset:o atIndex:i]; }
static inline void mce_set_tile_texture(id h, id<MTLTexture> t, NSUInteger i) { if (mce_metal4) mce4_set_texture(h, MCE_TILE, t, i); else [MCE_R(h) setTileTexture:t atIndex:i]; }
static inline void mce_set_tile_sampler(id h, id<MTLSamplerState> s, NSUInteger i) { if (mce_metal4) mce4_set_sampler(h, MCE_TILE, s, i); else [MCE_R(h) setTileSamplerState:s atIndex:i]; }
// compute
static inline void mce_set_compute_bytes(id h, const void *b, NSUInteger n, NSUInteger i) { if (mce_metal4) mce4_set_bytes(h, MCE_COMPUTE, b, n, i); else [MCE_C(h) setBytes:b length:n atIndex:i]; }
static inline void mce_set_compute_buffer(id h, id<MTLBuffer> b, NSUInteger o, NSUInteger i) { if (mce_metal4) mce4_set_buffer(h, MCE_COMPUTE, b, o, i); else [MCE_C(h) setBuffer:b offset:o atIndex:i]; }
static inline void mce_set_compute_texture(id h, id<MTLTexture> t, NSUInteger i) { if (mce_metal4) mce4_set_texture(h, MCE_COMPUTE, t, i); else [MCE_C(h) setTexture:t atIndex:i]; }
static inline void mce_set_compute_sampler(id h, id<MTLSamplerState> s, NSUInteger i) { if (mce_metal4) mce4_set_sampler(h, MCE_COMPUTE, s, i); else [MCE_C(h) setSamplerState:s atIndex:i]; }

// ---- residency (no-ops on Metal 3) ----
void mce4_make_resident(id allocation);
void mce4_release(id obj);
// Adds a buffer/texture/heap to the layer's residency set (committed before the next command buffer is). Call right after creating it.
static inline void mce_make_resident(id allocation) { if (mce_metal4 && allocation) mce4_make_resident(allocation); }
// Releases a GPU object the GPU may still be using: Metal 3 [obj release]; Metal 4 evicts and releases once every frame encoded so far
// has completed. mc_release does this for Java-owned handles.
static inline void mce_release(id obj) { if (!obj) return; if (mce_metal4) mce4_release(obj); else [obj release]; }
// useResource: (bindless access). Metal 4: nothing to do (the resource must be resident: mce_make_resident).
static inline void mce_use_resource_render(id h, id<MTLResource> r, MTLResourceUsage u, MTLRenderStages s) { if (!mce_metal4) [MCE_R(h) useResource:r usage:u stages:s]; }
static inline void mce_use_resource_compute(id h, id<MTLResource> r, MTLResourceUsage u) { if (!mce_metal4) [MCE_C(h) useResource:r usage:u]; }

// ---- draws (render handles) ----
// The identical drawPrimitives forms go to mce_raw(h). Indexed and indirect forms take a buffer + offset as today (Metal 4: GPU addresses).
void mce4_draw_indexed(id h, MTLPrimitiveType p, NSUInteger count, MTLIndexType t, id<MTLBuffer> ib, NSUInteger off, NSUInteger inst, NSInteger bv, NSUInteger bi);
void mce4_draw_indirect(id h, MTLPrimitiveType p, id<MTLBuffer> ind, NSUInteger indOff);
void mce4_draw_indexed_indirect(id h, MTLPrimitiveType p, MTLIndexType t, id<MTLBuffer> ib, NSUInteger off, id<MTLBuffer> ind, NSUInteger indOff);
void mce4_draw_mesh_indirect(id h, id<MTLBuffer> ind, NSUInteger indOff, MTLSize obj, MTLSize mesh);
static inline void mce_draw_indexed(id h, MTLPrimitiveType p, NSUInteger count, MTLIndexType t, id<MTLBuffer> ib, NSUInteger off) {
	if (mce_metal4) mce4_draw_indexed(h, p, count, t, ib, off, 1, 0, 0);
	else [MCE_R(h) drawIndexedPrimitives:p indexCount:count indexType:t indexBuffer:ib indexBufferOffset:off];
}
static inline void mce_draw_indexed_instanced(id h, MTLPrimitiveType p, NSUInteger count, MTLIndexType t, id<MTLBuffer> ib, NSUInteger off, NSUInteger inst) {
	if (mce_metal4) mce4_draw_indexed(h, p, count, t, ib, off, inst, 0, 0);
	else [MCE_R(h) drawIndexedPrimitives:p indexCount:count indexType:t indexBuffer:ib indexBufferOffset:off instanceCount:inst];
}
static inline void mce_draw_indexed_base(id h, MTLPrimitiveType p, NSUInteger count, MTLIndexType t, id<MTLBuffer> ib, NSUInteger off,
	NSUInteger inst, NSInteger baseVertex, NSUInteger baseInstance) {
	if (mce_metal4) mce4_draw_indexed(h, p, count, t, ib, off, inst, baseVertex, baseInstance);
	else [MCE_R(h) drawIndexedPrimitives:p indexCount:count indexType:t indexBuffer:ib indexBufferOffset:off instanceCount:inst baseVertex:baseVertex baseInstance:baseInstance];
}
static inline void mce_draw_indirect(id h, MTLPrimitiveType p, id<MTLBuffer> ind, NSUInteger indOff) {
	if (mce_metal4) mce4_draw_indirect(h, p, ind, indOff);
	else [MCE_R(h) drawPrimitives:p indirectBuffer:ind indirectBufferOffset:indOff];
}
static inline void mce_draw_indexed_indirect(id h, MTLPrimitiveType p, MTLIndexType t, id<MTLBuffer> ib, NSUInteger off, id<MTLBuffer> ind, NSUInteger indOff) {
	if (mce_metal4) mce4_draw_indexed_indirect(h, p, t, ib, off, ind, indOff);
	else [MCE_R(h) drawIndexedPrimitives:p indexType:t indexBuffer:ib indexBufferOffset:off indirectBuffer:ind indirectBufferOffset:indOff];
}
static inline void mce_draw_mesh_threadgroups_indirect(id h, id<MTLBuffer> ind, NSUInteger indOff, MTLSize obj, MTLSize mesh) {
	if (mce_metal4) mce4_draw_mesh_indirect(h, ind, indOff, obj, mesh);
	else [MCE_R(h) drawMeshThreadgroupsWithIndirectBuffer:ind indirectBufferOffset:indOff threadsPerObjectThreadgroup:obj threadsPerMeshThreadgroup:mesh];
}
// tile dispatch inside a render encoder (same selector; listed for completeness)
static inline void mce_dispatch_threads_per_tile(id h, MTLSize s) { [MCE_R(mce_raw(h)) dispatchThreadsPerTile:s]; }

// ---- dispatches (compute handles; serial encoders get their barriers here) ----
void mce4_dispatch_threads(id h, MTLSize grid, MTLSize tg);
void mce4_dispatch_threadgroups(id h, MTLSize groups, MTLSize tg);
void mce4_dispatch_threadgroups_indirect(id h, id<MTLBuffer> ind, NSUInteger indOff, MTLSize tg);
static inline void mce_dispatch_threads(id h, MTLSize grid, MTLSize tg) { if (mce_metal4) mce4_dispatch_threads(h, grid, tg); else [MCE_C(h) dispatchThreads:grid threadsPerThreadgroup:tg]; }
static inline void mce_dispatch_threadgroups(id h, MTLSize groups, MTLSize tg) { if (mce_metal4) mce4_dispatch_threadgroups(h, groups, tg); else [MCE_C(h) dispatchThreadgroups:groups threadsPerThreadgroup:tg]; }
static inline void mce_dispatch_threadgroups_indirect(id h, id<MTLBuffer> ind, NSUInteger indOff, MTLSize tg) {
	if (mce_metal4) mce4_dispatch_threadgroups_indirect(h, ind, indOff, tg);
	else [MCE_C(h) dispatchThreadgroupsWithIndirectBuffer:ind indirectBufferOffset:indOff threadsPerThreadgroup:tg];
}

// ---- blits (blit handles; under Metal 4 a compute encoder, serial: each command after the first waits for the previous) ----
// These selectors are identical in MTL4ComputeCommandEncoder, so the handle itself takes them in both backends; the wrapper adds
// the serial barrier. The functions only spell them out.
static inline void mce_copy_buffer(id h, id<MTLBuffer> src, NSUInteger so, id<MTLBuffer> dst, NSUInteger dO, NSUInteger size) {
	[MCE_B(h) copyFromBuffer:src sourceOffset:so toBuffer:dst destinationOffset:dO size:size];
}
static inline void mce_copy_buffer_to_texture(id h, id<MTLBuffer> src, NSUInteger off, NSUInteger bpr, NSUInteger bpi, MTLSize size,
	id<MTLTexture> dst, NSUInteger slice, NSUInteger level, MTLOrigin origin) {
	[MCE_B(h) copyFromBuffer:src sourceOffset:off sourceBytesPerRow:bpr sourceBytesPerImage:bpi sourceSize:size toTexture:dst
		destinationSlice:slice destinationLevel:level destinationOrigin:origin];
}
static inline void mce_copy_texture_to_buffer(id h, id<MTLTexture> src, NSUInteger slice, NSUInteger level, MTLOrigin origin, MTLSize size,
	id<MTLBuffer> dst, NSUInteger off, NSUInteger bpr, NSUInteger bpi) {
	[MCE_B(h) copyFromTexture:src sourceSlice:slice sourceLevel:level sourceOrigin:origin sourceSize:size toBuffer:dst destinationOffset:off
		destinationBytesPerRow:bpr destinationBytesPerImage:bpi];
}
static inline void mce_copy_texture(id h, id<MTLTexture> src, NSUInteger sSlice, NSUInteger sLevel, MTLOrigin sOrigin, MTLSize size,
	id<MTLTexture> dst, NSUInteger dSlice, NSUInteger dLevel, MTLOrigin dOrigin) {
	[MCE_B(h) copyFromTexture:src sourceSlice:sSlice sourceLevel:sLevel sourceOrigin:sOrigin sourceSize:size toTexture:dst
		destinationSlice:dSlice destinationLevel:dLevel destinationOrigin:dOrigin];
}
static inline void mce_copy_texture_all(id h, id<MTLTexture> src, id<MTLTexture> dst) { [MCE_B(h) copyFromTexture:src toTexture:dst]; }
static inline void mce_fill_buffer(id h, id<MTLBuffer> b, NSRange r, uint8_t v) { [MCE_B(h) fillBuffer:b range:r value:v]; }
static inline void mce_generate_mipmaps(id h, id<MTLTexture> t) { [MCE_B(h) generateMipmapsForTexture:t]; }

// ---- barriers and fences ----
void mce4_memory_barrier(id h);
void mce4_memory_barrier_render(id h, MTLRenderStages after, MTLRenderStages before);
void mce4_barrier_full(id h);
void mce4_update_fence(id h, id<MTLFence> f, int render, MTLRenderStages stages);
void mce4_wait_fence(id h, id<MTLFence> f, int render, MTLRenderStages stages);
// compute/blit: today's [c memoryBarrierWithScope:scope]. Metal 4: dispatch+blit -> dispatch+blit encoder barrier (scope ignored: all memory).
static inline void mce_memory_barrier(id h, MTLBarrierScope scope) { if (mce_metal4) mce4_memory_barrier(h); else [MCE_C(h) memoryBarrierWithScope:scope]; }
// render: today's [r memoryBarrierWithScope:scope afterStages:a beforeStages:b]. Metal 4: encoder barrier a -> b (same stage bits).
static inline void mce_memory_barrier_render(id h, MTLBarrierScope scope, MTLRenderStages a, MTLRenderStages b) {
	if (mce_metal4) mce4_memory_barrier_render(h, a, b); else [MCE_R(h) memoryBarrierWithScope:scope afterStages:a beforeStages:b];
}
// An explicit full barrier before the next command of h (all earlier work on the queue). Metal 3: nothing (hazard tracking).
static inline void mce_barrier_full(id h) { if (mce_metal4) mce4_barrier_full(h); }
static inline void mce_update_fence(id h, id<MTLFence> f) { if (mce_metal4) mce4_update_fence(h, f, 0, 0); else [MCE_C(h) updateFence:f]; }
static inline void mce_wait_fence(id h, id<MTLFence> f) { if (mce_metal4) mce4_wait_fence(h, f, 0, 0); else [MCE_C(h) waitForFence:f]; }
static inline void mce_update_fence_render(id h, id<MTLFence> f, MTLRenderStages after) { if (mce_metal4) mce4_update_fence(h, f, 1, after); else [MCE_R(h) updateFence:f afterStages:after]; }
static inline void mce_wait_fence_render(id h, id<MTLFence> f, MTLRenderStages before) { if (mce_metal4) mce4_wait_fence(h, f, 1, before); else [MCE_R(h) waitForFence:f beforeStages:before]; }

// ---- upload ring (Metal 4 only) ----
// Copies length bytes into this frame's upload ring; returns the GPU address (256-aligned) and, if asked, the ring buffer and
// offset. Valid until the GPU finished the frame being encoded. Metal 3: returns 0 (use setBytes).
uint64_t mce_upload(const void *bytes, NSUInteger length, id<MTLBuffer> *buffer, NSUInteger *offset);

// ---- core frame loop hooks (mcmetal.m) ----
// The frame's command buffers are committed: everything released or uploaded during it is retired against the last commit's value.
void mce4_frame_end(void);
// Called when a CAMetalLayer is configured: the layer's residency set goes on the Metal 4 queue.
void mce4_layer(CAMetalLayer *layer);
// A residency set made elsewhere (provisioned buffers, animation frames): also on the Metal 4 queue.
void mce4_add_queue_set(id<MTLResidencySet> set);
// Logs msg once per distinct msg pointer ("mcopt-metal4: ..."): for features switched off under Metal 4.
void mce_log_once(const char *msg);
#endif
