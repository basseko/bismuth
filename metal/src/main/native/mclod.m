// Far terrain (mcopt.metal.lod, -Dmcopt.lod=true): its library and pipeline states, the column walk (a compute dispatch in
// the pre command buffer) and its composite (one draw into the level's open render encoder). Only called with -Dmcopt.lod;
// the backend's own paths never reach it. Composite pipeline states are made per attachment layout of the encoder they
// draw in (shaders off: the main color and depth; the native shading pipeline: its G-buffer pass).
#import "mcmetal.h"
#import "mce.h"
#include <string.h>
#include "lodmesh.h"

#define LOD_MAX_LAYOUTS 8

// lodseam.m: -Dmcopt.lod.taa's history texture and constants for the mesh's fragment stage (0 / nil without the flag)
extern int gSeamTaaOn;
extern id<MTLTexture> gSeamTaaHist;
extern uint8_t gSeamTaaBytes[96];

static void lodErr(NSError *e, const char *what, char *err, int cap) {
	if (err && cap > 0) snprintf(err, cap, "%s: %s", what, e ? e.description.UTF8String : "unknown error");
}

// Attachment count of the open render encoder (0: none open), for the caller to pick the flavor.
int mcl_attachments(Enc *enc) {
	return enc->render ? enc->colorCount : 0;
}

// A shared buffer the CPU writes and the GPU reads (Apple silicon: the same memory).
id<MTLBuffer> mcl_buffer(Ctx *ctx, uint64_t size) {
	id<MTLBuffer> b = [ctx->device newBufferWithLength:size options:MTLResourceStorageModeShared | MTLResourceHazardTrackingModeUntracked];
	mce_make_resident(b);
	return b;
}

void *mcl_contents(id<MTLBuffer> b) {
	return b.contents;
}

void mcl_release(id obj) {
	mce_release(obj);
}

// ---- far terrain per screen column (columns.metal): a compute walk in the pre command buffer, a composite in the level pass ----

// what a layout's pipeline draws: the walk's composite, the mesh's opaque faces, its plants
#define LOD_PSO_COMP 0
#define LOD_PSO_MESH 1
#define LOD_PSO_PLANT 2

typedef struct {
	MTLPixelFormat colors[MAX_COLORS];
	MTLPixelFormat depth;
	int colorCount;
	int gbuffer;
	int kind;
	id<MTLRenderPipelineState> pso;
} ColLayout;

typedef struct {
	Ctx *ctx;
	id<MTLLibrary> library;
	id<MTLComputePipelineState> march, segMax, segFill, rows;
	id<MTLComputePipelineState> meshReset, meshCull, meshFinish, meshQuads, meshFinish2, hzClear, hzPrefix, meshEmit, meshFinishList;
	id<MTLFunction> meshVs, meshFsVanilla, meshFsGbuffer, plantVs, plantFsVanilla, plantFsGbuffer;
	id<MTLBuffer> meshIdx, plantIdx;   // {0, 1, 2, 2, 3, 0} + 4q for the survivors' 64-quad and the plants' 32-quad instances
	id<MTLDepthStencilState> depthTest;
	// -Dmcopt.lod.meshFences: the cull's outputs untracked, ordered by two fences instead: the draw's vertex stage waits for
	// this frame's cull (cullDone), the next cull waits for this frame's vertex stage (drawDone). The level pass's earlier
	// draws (the real terrain) then overlap the cull instead of waiting for it.
	id<MTLFence> cullDone, drawDone;
	id<MTLFunction> vs, fsVanilla, fsGbuffer;
	id<MTLDepthStencilState> depth;
	id<MTLSamplerState> atlasSampler;  // the block atlas as the game samples blocks: nearest texels, blended mips
	ColLayout layouts[LOD_MAX_LAYOUTS];
	int layoutCount;
	// the culls' GPU time (the position-keyed lists' load-based switch): the next cull's encoder samples its start and end
	// into slot cullSlot (-1: none) of cullTimes
	id<MTLCounterSampleBuffer> cullTimes;
	int cullSlot, cullTimesTried;
	MTLTimestamp cullCpu0, cullGpu0;
	uint64_t cullLast[16];
} LodCols;

LodCols *mcl_cols_new(Ctx *ctx, const char *source, char *err, int errCap) {
	@autoreleasepool {
		MTLCompileOptions *o = [[MTLCompileOptions new] autorelease];
		o.languageVersion = MTLLanguageVersion3_1;
		// safe math: under fast math the compiler may assume no NaN or infinity, and a walk whose exit test meets one may never
		// end (a GPU that never finishes a frame takes WindowServer down with it)
		o.mathMode = MTLMathModeSafe;
		NSError *e = nil;
		id<MTLLibrary> lib = [ctx->device newLibraryWithSource:[NSString stringWithUTF8String:source] options:o error:&e];
		if (!lib) {
			lodErr(e, "columns.metal", err, errCap);
			return NULL;
		}
		id<MTLFunction> march = [[lib newFunctionWithName:@"lod_columns"] autorelease];
		if (!march) {
			snprintf(err, errCap, "columns.metal: no lod_columns");
			return NULL;
		}
		id<MTLComputePipelineState> pso = [ctx->device newComputePipelineStateWithFunction:march error:&e];
		if (!pso) {
			lodErr(e, "lod_columns pipeline", err, errCap);
			return NULL;
		}
		LodCols *l = calloc(1, sizeof(LodCols));
		l->ctx = ctx;
		l->cullSlot = -1;
		l->library = lib;
		l->march = pso;
		id<MTLFunction> f1 = [[lib newFunctionWithName:@"lod_seg_max"] autorelease], f2 = [[lib newFunctionWithName:@"lod_seg_fill"] autorelease];
		l->segMax = f1 ? [ctx->device newComputePipelineStateWithFunction:f1 error:&e] : nil;
		l->segFill = f2 ? [ctx->device newComputePipelineStateWithFunction:f2 error:&e] : nil;
		if (!l->segMax || !l->segFill) {
			lodErr(e, "lod_seg pipelines", err, errCap);
			return NULL;
		}
		id<MTLFunction> f3 = [[lib newFunctionWithName:@"lod_rows"] autorelease];
		l->rows = f3 ? [ctx->device newComputePipelineStateWithFunction:f3 error:&e] : nil;
		if (!l->rows) {
			lodErr(e, "lod_rows pipeline", err, errCap);
			return NULL;
		}
		l->vs = [lib newFunctionWithName:@"lod_comp_vs"];
		l->fsVanilla = [lib newFunctionWithName:@"lod_comp_vanilla"];
		l->fsGbuffer = [lib newFunctionWithName:@"lod_comp_gbuffer"];
		MTLDepthStencilDescriptor *dd = [[MTLDepthStencilDescriptor new] autorelease];
		dd.depthCompareFunction = MTLCompareFunctionGreaterEqual;  // the game's projection is reversed Z
		dd.depthWriteEnabled = YES;
		l->depth = [ctx->device newDepthStencilStateWithDescriptor:dd];
		MTLSamplerDescriptor *sd = [[MTLSamplerDescriptor new] autorelease];
		sd.minFilter = MTLSamplerMinMagFilterNearest;
		sd.magFilter = MTLSamplerMinMagFilterNearest;
		sd.mipFilter = MTLSamplerMipFilterLinear;
		sd.sAddressMode = MTLSamplerAddressModeClampToEdge;
		sd.tAddressMode = MTLSamplerAddressModeClampToEdge;
		l->atlasSampler = [ctx->device newSamplerStateWithDescriptor:sd];
		if (!l->vs || !l->fsVanilla || !l->fsGbuffer) {
			snprintf(err, errCap, "columns.metal: missing entry points");
			return NULL;
		}
		// the mesh (-Dmcopt.lod.render=mesh)
		const char *kernels[] = {"lod_mesh_reset", "lod_mesh_cull", "lod_mesh_finish", "lod_mesh_quads", "lod_mesh_finish2", "lod_mesh_hz_clear", "lod_mesh_hz_prefix",
			"lod_mesh_emit", "lod_mesh_finish_list"};
		id<MTLComputePipelineState> *slots[] = {&l->meshReset, &l->meshCull, &l->meshFinish, &l->meshQuads, &l->meshFinish2, &l->hzClear, &l->hzPrefix,
			&l->meshEmit, &l->meshFinishList};
		for (int i = 0; i < 9; i++) {
			id<MTLFunction> fn = [[lib newFunctionWithName:[NSString stringWithUTF8String:kernels[i]]] autorelease];
			*slots[i] = fn ? [ctx->device newComputePipelineStateWithFunction:fn error:&e] : nil;
			if (!*slots[i]) {
				lodErr(e, kernels[i], err, errCap);
				return NULL;
			}
		}
		l->meshVs = [lib newFunctionWithName:@"lod_mesh_vs2"];
		l->meshFsVanilla = [lib newFunctionWithName:@"lod_mesh_vanilla"];
		l->meshFsGbuffer = [lib newFunctionWithName:@"lod_mesh_gbuffer"];
		l->plantVs = [lib newFunctionWithName:@"lod_mesh_plant_vs"];
		l->plantFsVanilla = [lib newFunctionWithName:@"lod_mesh_plant_vanilla"];
		l->plantFsGbuffer = [lib newFunctionWithName:@"lod_mesh_plant_gbuffer"];
		if (!l->meshVs || !l->meshFsVanilla || !l->meshFsGbuffer || !l->plantVs || !l->plantFsVanilla || !l->plantFsGbuffer) {
			snprintf(err, errCap, "columns.metal: missing mesh entry points");
			return NULL;
		}
		uint16_t idx[64 * 6];
		for (int q = 0; q < 64; q++) {
			uint16_t b = (uint16_t) (q * 4);
			uint16_t k[6] = {b, (uint16_t) (b + 1), (uint16_t) (b + 2), (uint16_t) (b + 2), (uint16_t) (b + 3), b};
			memcpy(idx + q * 6, k, sizeof k);
		}
		l->meshIdx = [ctx->device newBufferWithBytes:idx length:sizeof idx options:MTLResourceStorageModeShared];
		mce_make_resident(l->meshIdx);
		l->cullDone = [ctx->device newFence];
		l->drawDone = [ctx->device newFence];
		l->plantIdx = [ctx->device newBufferWithBytes:idx length:32 * 6 * sizeof(uint16_t) options:MTLResourceStorageModeShared];
		mce_make_resident(l->plantIdx);
		return l;
	}
}

static id<MTLRenderPipelineState> colPipeline(LodCols *l, Enc *enc, int gbuffer, int kind, char *err, int errCap) {
	for (int i = 0; i < l->layoutCount; i++) {
		ColLayout *k = &l->layouts[i];
		if (k->colorCount != enc->colorCount || k->gbuffer != gbuffer || k->kind != kind) continue;
		if ((enc->depth ? enc->depth.pixelFormat : MTLPixelFormatInvalid) != k->depth) continue;
		int same = 1;
		for (int c = 0; c < enc->colorCount; c++)
			if ((enc->colors[c] ? enc->colors[c].pixelFormat : MTLPixelFormatInvalid) != k->colors[c]) same = 0;
		if (same) return k->pso;
	}
	@autoreleasepool {
		MTLRenderPipelineDescriptor *pd = [[MTLRenderPipelineDescriptor new] autorelease];
		pd.label = gbuffer ? @"lod columns gbuffer" : @"lod columns vanilla";
		pd.vertexFunction = kind == LOD_PSO_MESH ? l->meshVs : kind == LOD_PSO_PLANT ? l->plantVs : l->vs;
		if (kind == LOD_PSO_MESH) pd.fragmentFunction = gbuffer ? l->meshFsGbuffer : l->meshFsVanilla;
		else if (kind == LOD_PSO_PLANT) pd.fragmentFunction = gbuffer ? l->plantFsGbuffer : l->plantFsVanilla;
		else pd.fragmentFunction = gbuffer ? l->fsGbuffer : l->fsVanilla;
		for (int c = 0; c < enc->colorCount; c++) {
			if (!enc->colors[c]) continue;
			pd.colorAttachments[c].pixelFormat = enc->colors[c].pixelFormat;
			int written = gbuffer ? (c >= 1 && c <= 4) : c == 0;
			pd.colorAttachments[c].writeMask = written ? MTLColorWriteMaskAll : MTLColorWriteMaskNone;
		}
		pd.depthAttachmentPixelFormat = enc->depth ? enc->depth.pixelFormat : MTLPixelFormatInvalid;
		pd.inputPrimitiveTopology = MTLPrimitiveTopologyClassTriangle;
		NSError *e = nil;
		id<MTLRenderPipelineState> pso = [l->ctx->device newRenderPipelineStateWithDescriptor:pd error:&e];
		if (!pso) {
			lodErr(e, "lod columns pipeline", err, errCap);
			return nil;
		}
		if (l->layoutCount == LOD_MAX_LAYOUTS) {
			mce_release(l->layouts[0].pso);
			memmove(&l->layouts[0], &l->layouts[1], sizeof(ColLayout) * (LOD_MAX_LAYOUTS - 1));
			l->layoutCount--;
		}
		ColLayout *k = &l->layouts[l->layoutCount++];
		memset(k, 0, sizeof *k);
		k->colorCount = enc->colorCount;
		for (int c = 0; c < enc->colorCount; c++) k->colors[c] = enc->colors[c] ? enc->colors[c].pixelFormat : MTLPixelFormatInvalid;
		k->depth = pd.depthAttachmentPixelFormat;
		k->gbuffer = gbuffer;
		k->kind = kind;
		k->pso = pso;
		return pso;
	}
}

// The column walk, in an encoder of its own in the pre command buffer (it runs before anything in the frame's command
// buffer; `out` is hazard-tracked, so the composite that reads it waits for it, and the next frame's walk for the composite).
// segments < 0: the row-split walk, -segments threads per column (lod_rows);
// segments 0: the serial walk (a thread per column); else the segmented walk: `segments` (the level count) threads per
// column, a pass for each stretch's highest elevation into segMax, then the fill.
void mcl_cols_march(LodCols *l, Enc *enc, const void *frame, int frameLength, id<MTLBuffer> geom, id<MTLBuffer> color, id<MTLBuffer> maxmip,
	id<MTLBuffer> mask, id<MTLBuffer> out, id<MTLBuffer> dbg, id<MTLBuffer> crowns, int columns, int segments, id<MTLBuffer> segMax,
	id<MTLBuffer> texWords, id<MTLBuffer> palette) {
	if (columns <= 0) return;
	mc_pre_end_encoders(enc);
	@autoreleasepool {
		id c = mce_new_compute_encoder(mc_pre(enc), MTLDispatchTypeSerial);
		[mce_raw(c) setLabel:@"lod columns"];
		mce_set_compute_bytes(c, frame, frameLength, 0);
		mce_set_compute_buffer(c, geom, 0, 1);
		mce_set_compute_buffer(c, color, 0, 2);
		mce_set_compute_buffer(c, maxmip, 0, 3);
		mce_set_compute_buffer(c, mask, 0, 4);
		mce_set_compute_buffer(c, dbg, 0, 6);
		mce_set_compute_buffer(c, crowns, 0, 7);
		// level 0's texture words and plant words, and the palette (plant profiles)
		mce_set_compute_buffer(c, texWords, 0, 9);
		mce_set_compute_buffer(c, palette, 0, 10);
		if (segments < 0) {
			[mce_raw(c) setComputePipelineState:l->rows];
			mce_set_compute_buffer(c, out, 0, 5);
			NSUInteger n = (NSUInteger) columns * (NSUInteger) (-segments);
			mce_dispatch_threads(c, MTLSizeMake(n, 1, 1), MTLSizeMake(MIN((NSUInteger) 64, l->rows.maxTotalThreadsPerThreadgroup), 1, 1));
		} else if (segments == 0) {
			[mce_raw(c) setComputePipelineState:l->march];
			mce_set_compute_buffer(c, out, 0, 5);
			NSUInteger tg = MIN((NSUInteger) 64, l->march.maxTotalThreadsPerThreadgroup);
			mce_dispatch_threads(c, MTLSizeMake(columns, 1, 1), MTLSizeMake(tg, 1, 1));
		} else {
			NSUInteger n = (NSUInteger) columns * segments;
			[mce_raw(c) setComputePipelineState:l->segMax];
			mce_set_compute_buffer(c, segMax, 0, 5);
			mce_dispatch_threads(c, MTLSizeMake(n, 1, 1), MTLSizeMake(MIN((NSUInteger) 64, l->segMax.maxTotalThreadsPerThreadgroup), 1, 1));
			mce_memory_barrier(c, MTLBarrierScopeBuffers);
			[mce_raw(c) setComputePipelineState:l->segFill];
			mce_set_compute_buffer(c, out, 0, 5);
			mce_set_compute_buffer(c, segMax, 0, 8);
			mce_dispatch_threads(c, MTLSizeMake(n, 1, 1), MTLSizeMake(MIN((NSUInteger) 64, l->segFill.maxTotalThreadsPerThreadgroup), 1, 1));
		}
		mce_end_encoder(c); [c release];
	}
}

// The composite: a quad over the band's rows into the open render encoder (shaders off: color 0 and depth; native shading:
// its G-buffer). Leaves pipeline, depth and cull state changed: the caller puts the game's back. 0, or -1 with err set.
int mcl_cols_composite(LodCols *l, Enc *enc, int gbuffer, const void *frame, int frameLength, id<MTLBuffer> out, id<MTLBuffer> geom,
	id<MTLBuffer> color, id<MTLBuffer> crowns, id<MTLBuffer> texWords, id<MTLBuffer> palette, id<MTLTexture> atlas, char *err, int errCap) {
	id r = enc->render;
	if (!r) return 0;
	id<MTLRenderPipelineState> pso = colPipeline(l, enc, gbuffer, LOD_PSO_COMP, err, errCap);
	if (!pso) return -1;
	[mce_raw(r) pushDebugGroup:@"lod columns"];
	[mce_raw(r) setRenderPipelineState:pso];
	[mce_raw(r) setDepthStencilState:l->depth];
	[mce_raw(r) setCullMode:MTLCullModeNone];
	[mce_raw(r) setTriangleFillMode:MTLTriangleFillModeFill];
	[mce_raw(r) setDepthBias:0 slopeScale:0 clamp:0];
	mce_set_vertex_bytes(r, frame, frameLength, 22);
	mce_set_fragment_bytes(r, frame, frameLength, 22);
	mce_set_fragment_buffer(r, out, 0, 23);
	mce_set_fragment_buffer(r, geom, 0, 24);
	mce_set_fragment_buffer(r, color, 0, 25);
	mce_set_fragment_buffer(r, crowns, 0, 26);
	mce_set_fragment_buffer(r, texWords, 0, 27);
	mce_set_fragment_buffer(r, palette, 0, 28);
	mce_set_fragment_texture(r, atlas, 20);
	mce_set_fragment_sampler(r, l->atlasSampler, 14);
	[mce_raw(r) drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:6];
	[mce_raw(r) popDebugGroup];
	return 0;
}

// A GPU-only, hazard-tracked buffer (the walk's output: written by one command buffer, read by the next).
id<MTLBuffer> mcl_private_buffer(Ctx *ctx, uint64_t size) {
	id<MTLBuffer> b = [ctx->device newBufferWithLength:size options:MTLResourceStorageModePrivate];
	mce_make_resident(b);
	return b;
}

// A GPU-only buffer without hazard tracking: the mesh's cull outputs when fences order them (mcl_mesh_cull / mcl_mesh_draw).
id<MTLBuffer> mcl_private_untracked_buffer(Ctx *ctx, uint64_t size) {
	id<MTLBuffer> b = [ctx->device newBufferWithLength:size options:MTLResourceStorageModePrivate | MTLResourceHazardTrackingModeUntracked];
	mce_make_resident(b);
	return b;
}

// The open render encoder's viewport, width << 16 | height (0: none open).
int mcl_enc_size(Enc *enc) {
	return enc->render ? (enc->width << 16) | (enc->height & 0xFFFF) : 0;
}

// ---- far terrain as meshes (-Dmcopt.lod.render=mesh) ----

// Meshes one tile of the clipmap (lodmesh.c), from any thread: the quads' count, or -1 when cap is too small.
int mcl_mesh_tile(const uint32_t *geom, const uint32_t *crown, const uint32_t *runs, const uint32_t *plantA, const uint32_t *plantB, int logN, int level,
	int tx, int tz, int neighbors, int block, uint32_t *header, uint32_t *quads, int cap) {
	LmIn in = {geom, crown, runs, plantA, plantB, logN, level, tx, tz, neighbors, block};
	return lm_mesh(&in, header, quads, cap);
}

static id<MTLComputePipelineState> pkPso(LodCols *l, const char *name);

// ---- the culls' GPU time ----

#define CULL_SLOTS 16

// The cull's encoder in the pre command buffer, sampling its GPU start and end when a slot was asked for.
static id cullEncoder(LodCols *l, Enc *enc) {
	id cb = mc_pre(enc);
	int slot = l->cullSlot;
	l->cullSlot = -1;
	if (mce_metal4 || slot < 0 || !l->cullTimes) return mce_new_compute_encoder(cb, MTLDispatchTypeSerial);
	MTLComputePassDescriptor *d = [MTLComputePassDescriptor computePassDescriptor];
	d.sampleBufferAttachments[0].sampleBuffer = l->cullTimes;
	d.sampleBufferAttachments[0].startOfEncoderSampleIndex = (NSUInteger) (2 * slot);
	d.sampleBufferAttachments[0].endOfEncoderSampleIndex = (NSUInteger) (2 * slot + 1);
	return [[cb computeCommandEncoderWithDescriptor:d] retain];
}

// The next cull (mcl_mesh_cull or mcl_pk_cull) samples its GPU start and end into slot (0-15): 0, or -1 when the GPU can't.
int mcl_cull_time_slot(LodCols *l, int slot) {
	if (mce_metal4) {  // no GPU counter sampling under Metal 4 (mce.h rule 7)
		mce_log_once("lod cull GPU timing off");
		return -1;
	}
	if (!l->cullTimesTried) {
		l->cullTimesTried = 1;
		id<MTLDevice> device = l->ctx->device;
		if ([device supportsCounterSampling:MTLCounterSamplingPointAtStageBoundary]) {
			@autoreleasepool {
				for (id<MTLCounterSet> set in device.counterSets) {
					if (![set.name isEqualToString:MTLCommonCounterSetTimestamp]) continue;
					MTLCounterSampleBufferDescriptor *d = [[MTLCounterSampleBufferDescriptor new] autorelease];
					d.counterSet = set;
					d.storageMode = MTLStorageModeShared;
					d.sampleCount = CULL_SLOTS * 2;
					l->cullTimes = [device newCounterSampleBufferWithDescriptor:d error:nil];
					[device sampleTimestamps:&l->cullCpu0 gpuTimestamp:&l->cullGpu0];
					break;
				}
			}
		}
	}
	if (!l->cullTimes || slot < 0 || slot >= CULL_SLOTS) return -1;
	l->cullSlot = slot;
	return 0;
}

// The GPU time (ms) of the cull that sampled into slot, once its frame completed; -1 when there's no new sample there.
double mcl_cull_time_read(LodCols *l, int slot) {
	if (!l->cullTimes || slot < 0 || slot >= CULL_SLOTS) return -1;
	MTLTimestamp cpu1, gpu1;
	[l->ctx->device sampleTimestamps:&cpu1 gpuTimestamp:&gpu1];
	// (timestamps are in GPU ticks; the CPU side of the calibration pair is in nanoseconds)
	if (gpu1 <= l->cullGpu0 || cpu1 <= l->cullCpu0) return -1;
	double nsPerTick = (double) (cpu1 - l->cullCpu0) / (double) (gpu1 - l->cullGpu0);
	double ms = -1;
	@autoreleasepool {
		NSData *data = [l->cullTimes resolveCounterRange:NSMakeRange((NSUInteger) (2 * slot), 2)];
		if (data.length >= 2 * sizeof(MTLCounterResultTimestamp)) {
			const MTLCounterResultTimestamp *ts = data.bytes;
			uint64_t a = ts[0].timestamp, b = ts[1].timestamp;
			if (a != MTLCounterErrorValue && b != MTLCounterErrorValue && a != 0 && b > a && a != l->cullLast[slot]) {
				l->cullLast[slot] = a;
				ms = (double) (b - a) * nsPerTick / 1e6;
			}
		}
	}
	return ms;
}

// The mesh's per-frame selection, in an encoder of its own in the pre command buffer: the block cull (blocks: threads, a
// block of every level's window each), then every candidate quad (an indirect dispatch the block cull sizes), leaving the
// draws' indirect arguments in args. args, inst, plantInst and survivors are GPU-only and hazard-tracked: the draws in
// this frame's level pass wait for them.
void mcl_mesh_cull(LodCols *l, Enc *enc, const void *frame, int frameLength, const void *comp, int compLength, id<MTLBuffer> table, id<MTLBuffer> arena,
	id<MTLBuffer> mask, id<MTLBuffer> args, id<MTLBuffer> inst, id<MTLBuffer> plantInst, id<MTLBuffer> survivors, id<MTLBuffer> horizon, id<MTLBuffer> list,
	id<MTLBuffer> geom, id<MTLBuffer> crowns, int blocks, int fences) {
	mc_pre_end_encoders(enc);
	@autoreleasepool {
		id c = cullEncoder(l, enc);
		[mce_raw(c) setLabel:@"lod mesh cull"];
		if (fences) mce_wait_fence(c, l->drawDone);
		mce_set_compute_bytes(c, frame, frameLength, 0);
		mce_set_compute_buffer(c, table, 0, 1);
		mce_set_compute_buffer(c, arena, 0, 2);
		mce_set_compute_buffer(c, mask, 0, 3);
		mce_set_compute_buffer(c, args, 0, 4);
		mce_set_compute_buffer(c, inst, 0, 5);
		mce_set_compute_buffer(c, plantInst, 0, 6);
		mce_set_compute_buffer(c, survivors, 0, 7);
		mce_set_compute_bytes(c, comp, compLength, 22);
		mce_set_compute_buffer(c, inst, 0, 23);
		mce_set_compute_buffer(c, arena, 0, 24);
		mce_set_compute_buffer(c, horizon, 0, 8);
		mce_set_compute_buffer(c, list, 0, 9);
		mce_set_compute_buffer(c, geom, 0, 12);
		mce_set_compute_buffer(c, crowns, 0, 13);
		[mce_raw(c) setComputePipelineState:l->meshReset];
		mce_dispatch_threads(c, MTLSizeMake(1, 1, 1), MTLSizeMake(1, 1, 1));
		[mce_raw(c) setComputePipelineState:l->hzClear];
		mce_dispatch_threads(c, MTLSizeMake(4096 * 128, 1, 1), MTLSizeMake(256, 1, 1));
		mce_memory_barrier(c, MTLBarrierScopeBuffers);
		// (the real terrain as an occluder: the far terrain's lowest tangent per horizon bin first)
		id<MTLComputePipelineState> farMin = (((const int32_t *) frame)[98] & 16) ? pkPso(l, "lod_mesh_far_min") : nil;
		if (farMin) {
			[mce_raw(c) setComputePipelineState:farMin];
			mce_dispatch_threads(c, MTLSizeMake((NSUInteger) blocks, 1, 1), MTLSizeMake(64, 1, 1));
			mce_memory_barrier(c, MTLBarrierScopeBuffers);
		}
		[mce_raw(c) setComputePipelineState:l->meshCull];
		mce_dispatch_threads(c, MTLSizeMake((NSUInteger) blocks, 1, 1), MTLSizeMake(64, 1, 1));
		mce_memory_barrier(c, MTLBarrierScopeBuffers);
		[mce_raw(c) setComputePipelineState:l->meshFinishList];
		mce_dispatch_threads(c, MTLSizeMake(1, 1, 1), MTLSizeMake(1, 1, 1));
		[mce_raw(c) setComputePipelineState:l->hzPrefix];
		mce_dispatch_threads(c, MTLSizeMake(4096, 1, 1), MTLSizeMake(64, 1, 1));
		mce_memory_barrier(c, MTLBarrierScopeBuffers);
		[mce_raw(c) setComputePipelineState:l->meshEmit];
		mce_dispatch_threadgroups_indirect(c, args, 19 * 4, MTLSizeMake(64, 1, 1));
		mce_memory_barrier(c, MTLBarrierScopeBuffers);
		[mce_raw(c) setComputePipelineState:l->meshFinish];
		mce_dispatch_threads(c, MTLSizeMake(1, 1, 1), MTLSizeMake(1, 1, 1));
		mce_memory_barrier(c, MTLBarrierScopeBuffers);
		[mce_raw(c) setComputePipelineState:l->meshQuads];
		mce_dispatch_threadgroups_indirect(c, args, 16 * 4, MTLSizeMake(64, 1, 1));
		mce_memory_barrier(c, MTLBarrierScopeBuffers);
		[mce_raw(c) setComputePipelineState:l->meshFinish2];
		mce_dispatch_threads(c, MTLSizeMake(1, 1, 1), MTLSizeMake(1, 1, 1));
		if (fences) mce_update_fence(c, l->cullDone);
		mce_end_encoder(c); [c release];
	}
}

// The mesh's two indirect draws into the open render encoder (shaders off: color 0 and depth; native shading: its
// G-buffer): the opaque faces that survived the culls, then the plants (two-sided, cut to their sprites). Leaves pipeline,
// depth and cull state changed: the caller puts the game's back. 0, or -1 with err set.
int mcl_mesh_draw(LodCols *l, Enc *enc, int gbuffer, const void *comp, int compLength, id<MTLBuffer> arena, id<MTLBuffer> args, id<MTLBuffer> inst,
	id<MTLBuffer> plantInst, id<MTLBuffer> survivors, id<MTLBuffer> geom, id<MTLBuffer> color, id<MTLBuffer> crowns, id<MTLBuffer> texWords,
	id<MTLBuffer> palette, id<MTLTexture> atlas, id<MTLBuffer> table, const void *meshFrame, int meshFrameLength, int fences, char *err, int errCap) {
	id r = enc->render;
	if (!r) return 0;
	id<MTLRenderPipelineState> pso = colPipeline(l, enc, gbuffer, LOD_PSO_MESH, err, errCap);
	id<MTLRenderPipelineState> ppso = pso ? colPipeline(l, enc, gbuffer, LOD_PSO_PLANT, err, errCap) : nil;
	if (!pso || !ppso) return -1;
	[mce_raw(r) pushDebugGroup:@"lod mesh"];
	if (fences) mce_wait_fence_render(r, l->cullDone, MTLRenderStageVertex);
	[mce_raw(r) setDepthStencilState:l->depth];
	[mce_raw(r) setTriangleFillMode:MTLTriangleFillModeFill];
	[mce_raw(r) setDepthBias:0 slopeScale:0 clamp:0];
	mce_set_vertex_bytes(r, comp, compLength, 22);
	mce_set_fragment_bytes(r, comp, compLength, 22);
	mce_set_vertex_buffer(r, arena, 0, 24);
	mce_set_vertex_buffer(r, args, 0, 25);
	mce_set_vertex_buffer(r, survivors, 0, 26);
	mce_set_fragment_buffer(r, geom, 0, 24);
	mce_set_fragment_buffer(r, color, 0, 25);
	mce_set_fragment_buffer(r, crowns, 0, 26);
	mce_set_fragment_buffer(r, texWords, 0, 27);
	mce_set_fragment_buffer(r, palette, 0, 28);
	mce_set_fragment_texture(r, atlas, 20);
	mce_set_fragment_sampler(r, l->atlasSampler, 14);
	// the vertex stage's skirt feet: the tile table, the clipmap's words, MeshFrame
	mce_set_vertex_buffer(r, table, 0, 27);
	mce_set_fragment_buffer(r, table, 0, 29);   // (-Dmcopt.lod.dissolve: the fragment stage reads install times; unused otherwise)
	// (-Dmcopt.lod.taa, lodseam.m: the far terrain's temporal filter's history and constants; set only with the flag)
	if (gSeamTaaOn && gSeamTaaHist) {
		mce_set_fragment_texture(r, gSeamTaaHist, 21);
		mce_set_fragment_bytes(r, gSeamTaaBytes, sizeof(gSeamTaaBytes), 30);
	}
	mce_set_vertex_buffer(r, geom, 0, 28);
	mce_set_vertex_buffer(r, crowns, 0, 29);
	mce_set_vertex_bytes(r, meshFrame, meshFrameLength, 30);
	[mce_raw(r) setRenderPipelineState:pso];
	[mce_raw(r) setCullMode:MTLCullModeBack];
	mce_set_vertex_buffer(r, inst, 0, 23);
	mce_draw_indexed_indirect(r, MTLPrimitiveTypeTriangle, MTLIndexTypeUInt16, l->meshIdx, 0, args, 0);
	[mce_raw(r) setRenderPipelineState:ppso];
	[mce_raw(r) setCullMode:MTLCullModeNone];
	mce_set_vertex_buffer(r, plantInst, 0, 23);
	mce_draw_indexed_indirect(r, MTLPrimitiveTypeTriangle, MTLIndexTypeUInt16, l->plantIdx, 0, args, 5 * 4);
	if (fences) mce_update_fence_render(r, l->drawDone, MTLRenderStageVertex);
	[mce_raw(r) popDebugGroup];
	return 0;
}

// ---- position-keyed candidate lists (-Dmcopt.lod.pk; columns.metal lod_pk_*) ----

static id<MTLComputePipelineState> pkPso(LodCols *l, const char *name) {
	static id<MTLComputePipelineState> cache[24];
	static const char *names[24];
	static int count;
	for (int i = 0; i < count; i++)
		if (!strcmp(names[i], name)) return cache[i];
	@autoreleasepool {
		NSError *e = nil;
		id<MTLFunction> fn = [[l->library newFunctionWithName:[NSString stringWithUTF8String:name]] autorelease];
		id<MTLComputePipelineState> p = fn ? [l->ctx->device newComputePipelineStateWithFunction:fn error:&e] : nil;
		if (!p) {
			fprintf(stderr, "mcopt-lod: %s: %s\n", name, e ? e.description.UTF8String : "missing");
			return nil;
		}
		if (count < 24) {
			cache[count] = p;
			names[count++] = name;
		}
		return p;
	}
}

// A shared (CPU-readable), hazard-tracked buffer: the position-keyed lists' sector table.
id<MTLBuffer> mcl_shared_tracked_buffer(Ctx *ctx, uint64_t size) {
	id<MTLBuffer> b = [ctx->device newBufferWithLength:size options:MTLResourceStorageModeShared];
	mce_make_resident(b);
	return b;
}

// The position-keyed lists' passes, in an encoder of their own in the pre command buffer: the position pass over the sectors
// params name (rebuild != 0), then (draw != 0) the live cull (the full cull's kernels over the live ring's blocks and the
// stale sectors' in view) and the per-frame pass over the drawable sectors in view, which append to the same draws'
// arguments in frameArgs (mcl_mesh_draw draws them: inst = pkInst, which also holds the sectors' stand-ins from
// PkParams.standBase on, plantInst = framePlants, survivors = frameSurv). bufs:
// table, arena, mask, pkArgs, pkInst, plantSec, recs, horizon, blockList, geom, crowns, sec, stand, list, frameArgs,
// frameSurv, framePlants. Returns -1 when a pipeline is missing (nothing encoded).
int mcl_pk_cull(LodCols *l, Enc *enc, const void *frame, int frameLength, const void *comp, int compLength, const void *params, int paramsLength,
	const int64_t *bufs, int rebuild, int draw, int blocks) {
	id<MTLComputePipelineState> reset = pkPso(l, "lod_pk_reset"), hzClear = pkPso(l, "lod_pk_hz_clear"), cull = pkPso(l, "lod_pk_cull"),
		emit = pkPso(l, "lod_pk_emit"), quads = pkPso(l, "lod_pk_quads"), select = pkPso(l, "lod_pk_select"), framePass = pkPso(l, "lod_pk_frame"),
		plants = pkPso(l, "lod_pk_plants"), finish = pkPso(l, "lod_pk_finish"), live = pkPso(l, "lod_pk_live"), prefix = pkPso(l, "lod_pk_hz_prefix");
	if (!reset || !hzClear || !cull || !emit || !quads || !select || !framePass || !plants || !finish || !live || !prefix) return -1;
	// (the real terrain as an occluder: the far terrain's lowest tangent per horizon bin before the occluders)
	int realOcc = (((const int32_t *) frame)[98] & 16) != 0;
	id<MTLComputePipelineState> pkFarMin = realOcc ? pkPso(l, "lod_pk_far_min") : nil, liveFarMin = realOcc ? pkPso(l, "lod_pk_live_far_min") : nil;
	if (realOcc && (!pkFarMin || !liveFarMin)) return -1;
#define B(i) ((id<MTLBuffer>) (void *) bufs[i])
	mc_pre_end_encoders(enc);
	@autoreleasepool {
		id c = cullEncoder(l, enc);
		[mce_raw(c) setLabel:@"lod pk"];
		mce_set_compute_bytes(c, frame, frameLength, 0);
		mce_set_compute_bytes(c, params, paramsLength, 14);
		if (rebuild) {
			mce_set_compute_buffer(c, B(0), 0, 1);
			mce_set_compute_buffer(c, B(1), 0, 2);
			mce_set_compute_buffer(c, B(2), 0, 3);
			mce_set_compute_buffer(c, B(3), 0, 4);
			mce_set_compute_buffer(c, B(4), 0, 5);
			mce_set_compute_buffer(c, B(5), 0, 6);
			mce_set_compute_buffer(c, B(6), 0, 7);
			mce_set_compute_buffer(c, B(7), 0, 8);
			mce_set_compute_buffer(c, B(8), 0, 9);
			mce_set_compute_buffer(c, B(9), 0, 12);
			mce_set_compute_buffer(c, B(10), 0, 13);
			mce_set_compute_buffer(c, B(11), 0, 15);
			mce_set_compute_buffer(c, B(12), 0, 16);
			[mce_raw(c) setComputePipelineState:reset];
			mce_dispatch_threads(c, MTLSizeMake(128, 1, 1), MTLSizeMake(64, 1, 1));
			[mce_raw(c) setComputePipelineState:hzClear];
			mce_dispatch_threads(c, MTLSizeMake(4096 * 128, 1, 1), MTLSizeMake(256, 1, 1));
			mce_memory_barrier(c, MTLBarrierScopeBuffers);
			if (realOcc) {
				[mce_raw(c) setComputePipelineState:pkFarMin];
				mce_dispatch_threads(c, MTLSizeMake((NSUInteger) blocks, 1, 1), MTLSizeMake(64, 1, 1));
				mce_memory_barrier(c, MTLBarrierScopeBuffers);
			}
			[mce_raw(c) setComputePipelineState:cull];
			mce_dispatch_threads(c, MTLSizeMake((NSUInteger) blocks, 1, 1), MTLSizeMake(64, 1, 1));
			mce_memory_barrier(c, MTLBarrierScopeBuffers);
			[mce_raw(c) setComputePipelineState:l->meshFinishList];
			mce_dispatch_threads(c, MTLSizeMake(1, 1, 1), MTLSizeMake(1, 1, 1));
			[mce_raw(c) setComputePipelineState:prefix];
			mce_dispatch_threads(c, MTLSizeMake(4096, 1, 1), MTLSizeMake(64, 1, 1));
			mce_memory_barrier(c, MTLBarrierScopeBuffers);
			[mce_raw(c) setComputePipelineState:emit];
			mce_dispatch_threadgroups_indirect(c, B(3), 19 * 4, MTLSizeMake(64, 1, 1));
			mce_memory_barrier(c, MTLBarrierScopeBuffers);
			[mce_raw(c) setComputePipelineState:l->meshFinish];
			mce_dispatch_threads(c, MTLSizeMake(1, 1, 1), MTLSizeMake(1, 1, 1));
			mce_memory_barrier(c, MTLBarrierScopeBuffers);
			[mce_raw(c) setComputePipelineState:quads];
			mce_dispatch_threadgroups_indirect(c, B(3), 16 * 4, MTLSizeMake(64, 1, 1));
			mce_memory_barrier(c, MTLBarrierScopeBuffers);
		}
		if (draw == 2) {
			// the live cull: the full cull's kernels over the stale sectors' blocks (and a live ring's, when maskDist.w has one)
			// (args, survivors and plants from the start; the lists append theirs after)
			mce_set_compute_buffer(c, B(0), 0, 1);
			mce_set_compute_buffer(c, B(1), 0, 2);
			mce_set_compute_buffer(c, B(2), 0, 3);
			mce_set_compute_buffer(c, B(14), 0, 4);
			mce_set_compute_buffer(c, B(4), 0, 5);
			mce_set_compute_buffer(c, B(16), 0, 6);
			mce_set_compute_buffer(c, B(15), 0, 7);
			mce_set_compute_buffer(c, B(7), 0, 8);
			mce_set_compute_buffer(c, B(8), 0, 9);
			mce_set_compute_buffer(c, B(9), 0, 12);
			mce_set_compute_buffer(c, B(10), 0, 13);
			mce_set_compute_bytes(c, comp, compLength, 22);
			mce_set_compute_buffer(c, B(4), 0, 23);
			mce_set_compute_buffer(c, B(1), 0, 24);
			[mce_raw(c) setComputePipelineState:l->meshReset];
			mce_dispatch_threads(c, MTLSizeMake(1, 1, 1), MTLSizeMake(1, 1, 1));
			[mce_raw(c) setComputePipelineState:l->hzClear];
			mce_dispatch_threads(c, MTLSizeMake(4096 * 128, 1, 1), MTLSizeMake(256, 1, 1));
			mce_memory_barrier(c, MTLBarrierScopeBuffers);
			if (realOcc) {
				[mce_raw(c) setComputePipelineState:liveFarMin];
				mce_dispatch_threads(c, MTLSizeMake((NSUInteger) blocks, 1, 1), MTLSizeMake(64, 1, 1));
				mce_memory_barrier(c, MTLBarrierScopeBuffers);
			}
			[mce_raw(c) setComputePipelineState:live];
			mce_dispatch_threads(c, MTLSizeMake((NSUInteger) blocks, 1, 1), MTLSizeMake(64, 1, 1));
			mce_memory_barrier(c, MTLBarrierScopeBuffers);
			[mce_raw(c) setComputePipelineState:l->meshFinishList];
			mce_dispatch_threads(c, MTLSizeMake(1, 1, 1), MTLSizeMake(1, 1, 1));
			[mce_raw(c) setComputePipelineState:l->hzPrefix];
			mce_dispatch_threads(c, MTLSizeMake(4096, 1, 1), MTLSizeMake(64, 1, 1));
			mce_memory_barrier(c, MTLBarrierScopeBuffers);
			[mce_raw(c) setComputePipelineState:l->meshEmit];
			mce_dispatch_threadgroups_indirect(c, B(14), 19 * 4, MTLSizeMake(64, 1, 1));
			mce_memory_barrier(c, MTLBarrierScopeBuffers);
			[mce_raw(c) setComputePipelineState:l->meshFinish];
			mce_dispatch_threads(c, MTLSizeMake(1, 1, 1), MTLSizeMake(1, 1, 1));
			mce_memory_barrier(c, MTLBarrierScopeBuffers);
			[mce_raw(c) setComputePipelineState:l->meshQuads];
			mce_dispatch_threadgroups_indirect(c, B(14), 16 * 4, MTLSizeMake(64, 1, 1));
			mce_memory_barrier(c, MTLBarrierScopeBuffers);
		}
		if (draw) {
			// the lists (lod_pk_select starts the draws' arguments when the live cull didn't run)
			mce_set_compute_bytes(c, comp, compLength, 22);
			mce_set_compute_buffer(c, B(0), 0, 1);
			mce_set_compute_buffer(c, B(1), 0, 2);
			mce_set_compute_buffer(c, B(14), 0, 4);
			mce_set_compute_buffer(c, B(5), 0, 6);
			mce_set_compute_buffer(c, B(6), 0, 7);
			mce_set_compute_buffer(c, B(13), 0, 10);
			mce_set_compute_buffer(c, B(15), 0, 11);
			mce_set_compute_buffer(c, B(16), 0, 12);
			mce_set_compute_buffer(c, B(11), 0, 15);
			mce_set_compute_buffer(c, B(12), 0, 16);
			[mce_raw(c) setComputePipelineState:select];
			mce_dispatch_threadgroups(c, MTLSizeMake(1, 1, 1), MTLSizeMake(128, 1, 1));
			mce_memory_barrier(c, MTLBarrierScopeBuffers);
			[mce_raw(c) setComputePipelineState:framePass];
			mce_dispatch_threadgroups_indirect(c, B(14), 16 * 4, MTLSizeMake(64, 1, 1));
			[mce_raw(c) setComputePipelineState:plants];
			mce_dispatch_threadgroups_indirect(c, B(14), 19 * 4, MTLSizeMake(64, 1, 1));
			mce_memory_barrier(c, MTLBarrierScopeBuffers);
			[mce_raw(c) setComputePipelineState:finish];
			mce_dispatch_threads(c, MTLSizeMake(1, 1, 1), MTLSizeMake(1, 1, 1));
		}
		mce_end_encoder(c); [c release];
	}
#undef B
	return 0;
}

// ---- the lists' visible set (-Dmcopt.lod.pkVis; columns.metal lod_pk_facet_*) ----

// The facets' resources, grown on demand (one LodCols in practice).
static struct {
	id<MTLRenderPipelineState> pso;
	id<MTLDepthStencilState> depth;
	id<MTLTexture> dist, depthT;
	int w, h;
	id<MTLBuffer> surv, out, args, cnt, near, tiles;
	int stride, tilesLen;
} pkVis;

static id<MTLTexture> pkVisTexture(Ctx *ctx, MTLPixelFormat f, int w, int h, MTLTextureUsage usage, MTLStorageMode mode) {
	MTLTextureDescriptor *d = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:f width:(NSUInteger) w height:(NSUInteger) h mipmapped:NO];
	d.usage = usage;
	d.storageMode = mode;
	id<MTLTexture> t = [ctx->device newTextureWithDescriptor:d];
	if (mode != MTLStorageModeMemoryless) mce_make_resident(t);
	return t;
}

// Prunes the records of azimuth az's sectors in bandMask (1 the hand-off band, 2 the far band) against a depth facet: the
// records of the azimuth and its two neighbors, both bands, of the sectors in use (use: 4 x 32-bit masks), drawn from facet
// (a CompFrame: the facet's camera-relative viewProj, screen.xy its size fw x fh, origin and camFrac the build camera's).
// eps0, eps1: the bands' margins (blocks). stride: max(recCap, recCap0). bufs: as mcl_pk_cull's. Encoded in the pre command
// buffer, before the frame's cull. Returns -1 when a pipeline or a resource is missing (nothing encoded).
int mcl_pk_vis(LodCols *l, Enc *enc, const void *frame, int frameLength, const void *facet, int facetLength, const void *params, int paramsLength,
	const int64_t *bufs, int az, int bandMask, float eps0, float eps1, const void *use, int fw, int fh, int stride) {
	id<MTLComputePipelineState> gather = pkPso(l, "lod_pk_facet_gather"), args = pkPso(l, "lod_pk_facet_args"), reduce = pkPso(l, "lod_pk_facet_reduce"),
		prune = pkPso(l, "lod_pk_facet_prune"), commit = pkPso(l, "lod_pk_facet_commit");
	if (!gather || !args || !reduce || !prune || !commit || fw <= 0 || fh <= 0 || stride <= 0) return -1;
	Ctx *ctx = l->ctx;
	@autoreleasepool {
		if (!pkVis.pso) {
			MTLRenderPipelineDescriptor *pd = [[MTLRenderPipelineDescriptor new] autorelease];
			pd.label = @"lod pk facet";
			pd.vertexFunction = l->meshVs;
			pd.fragmentFunction = [[l->library newFunctionWithName:@"lod_pk_facet_fs"] autorelease];
			if (!pd.fragmentFunction) return -1;
			pd.colorAttachments[0].pixelFormat = MTLPixelFormatR32Float;
			pd.depthAttachmentPixelFormat = MTLPixelFormatDepth32Float;
			NSError *e = nil;
			pkVis.pso = [ctx->device newRenderPipelineStateWithDescriptor:pd error:&e];
			if (!pkVis.pso) {
				fprintf(stderr, "mcopt-lod: lod pk facet pipeline: %s\n", e ? e.description.UTF8String : "?");
				return -1;
			}
			MTLDepthStencilDescriptor *dd = [[MTLDepthStencilDescriptor new] autorelease];
			dd.depthCompareFunction = MTLCompareFunctionGreaterEqual;
			dd.depthWriteEnabled = YES;
			pkVis.depth = [ctx->device newDepthStencilStateWithDescriptor:dd];
			pkVis.args = [ctx->device newBufferWithLength:128 options:MTLResourceStorageModePrivate];
			mce_make_resident(pkVis.args);
			pkVis.cnt = [ctx->device newBufferWithLength:16 options:MTLResourceStorageModePrivate];
			mce_make_resident(pkVis.cnt);
			pkVis.near = [ctx->device newBufferWithLength:16 options:MTLResourceStorageModePrivate];
			mce_make_resident(pkVis.near);
		}
		if (fw > pkVis.w || fh > pkVis.h) {
			mce_release(pkVis.dist);
			mce_release(pkVis.depthT);
			pkVis.w = fw > pkVis.w ? fw : pkVis.w;
			pkVis.h = fh > pkVis.h ? fh : pkVis.h;
			pkVis.dist = pkVisTexture(ctx, MTLPixelFormatR32Float, pkVis.w, pkVis.h, MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead,
				MTLStorageModePrivate);
			pkVis.depthT = pkVisTexture(ctx, MTLPixelFormatDepth32Float, pkVis.w, pkVis.h, MTLTextureUsageRenderTarget, MTLStorageModeMemoryless);
			int tl = ((pkVis.w + 7) / 8) * ((pkVis.h + 7) / 8) * 4;
			if (tl > pkVis.tilesLen) {
				mce_release(pkVis.tiles);
				pkVis.tiles = [ctx->device newBufferWithLength:(NSUInteger) tl options:MTLResourceStorageModePrivate];
				mce_make_resident(pkVis.tiles);
				pkVis.tilesLen = tl;
			}
		}
		if (stride > pkVis.stride) {
			mce_release(pkVis.surv);
			mce_release(pkVis.out);
			pkVis.surv = [ctx->device newBufferWithLength:(NSUInteger) stride * 6 * 16 options:MTLResourceStorageModePrivate];
			mce_make_resident(pkVis.surv);
			pkVis.out = [ctx->device newBufferWithLength:(NSUInteger) stride * 2 * 16 options:MTLResourceStorageModePrivate];
			mce_make_resident(pkVis.out);
			pkVis.stride = stride;
		}
		if (!pkVis.dist || !pkVis.depthT || !pkVis.tiles || !pkVis.surv || !pkVis.out || !pkVis.args || !pkVis.cnt || !pkVis.near) return -1;
#define B(i) ((id<MTLBuffer>) (void *) bufs[i])
		mc_pre_end_encoders(enc);
		id cb = mc_pre(enc);
		id b = mce_new_blit_encoder(cb);
		[mce_raw(b) setLabel:@"lod pk facet reset"];
		mce_fill_buffer(b, pkVis.args, NSMakeRange(0, 128), 0);
		mce_fill_buffer(b, pkVis.cnt, NSMakeRange(0, 16), 0);
		// (0x7f7f7f7f: a float past any distance)
		mce_fill_buffer(b, pkVis.near, NSMakeRange(0, 4), 0x7f);
		mce_end_encoder(b);
		[b release];
		uint32_t a = (uint32_t) az, cap = (uint32_t) stride * 6, sel[2] = {(uint32_t) az, (uint32_t) bandMask}, size[2] = {(uint32_t) fw, (uint32_t) fh};
		float eps[2] = {eps0, eps1};
		id c = mce_new_compute_encoder(cb, MTLDispatchTypeSerial);
		[mce_raw(c) setLabel:@"lod pk facet gather"];
		mce_set_compute_bytes(c, params, paramsLength, 14);
		mce_set_compute_buffer(c, B(11), 0, 15);
		mce_set_compute_buffer(c, B(6), 0, 7);
		mce_set_compute_buffer(c, pkVis.surv, 0, 11);
		mce_set_compute_buffer(c, pkVis.args, 0, 4);
		mce_set_compute_bytes(c, &a, 4, 20);
		mce_set_compute_bytes(c, &cap, 4, 21);
		mce_set_compute_bytes(c, use, 16, 9);
		[mce_raw(c) setComputePipelineState:gather];
		mce_dispatch_threads(c, MTLSizeMake((NSUInteger) cap, 1, 1), MTLSizeMake(256, 1, 1));
		mce_memory_barrier(c, MTLBarrierScopeBuffers);
		[mce_raw(c) setComputePipelineState:args];
		mce_dispatch_threads(c, MTLSizeMake(1, 1, 1), MTLSizeMake(1, 1, 1));
		mce_end_encoder(c); [c release];
		MTLRenderPassDescriptor *rp = [MTLRenderPassDescriptor renderPassDescriptor];
		rp.colorAttachments[0].texture = pkVis.dist;
		rp.colorAttachments[0].loadAction = MTLLoadActionClear;
		rp.colorAttachments[0].clearColor = MTLClearColorMake(1e30, 0, 0, 0);
		rp.colorAttachments[0].storeAction = MTLStoreActionStore;
		rp.depthAttachment.texture = pkVis.depthT;
		rp.depthAttachment.loadAction = MTLLoadActionClear;
		rp.depthAttachment.clearDepth = 0.0;
		rp.depthAttachment.storeAction = MTLStoreActionDontCare;
		id r = mce_new_render_encoder(cb, rp);
		[mce_raw(r) setLabel:@"lod pk facet"];
		[mce_raw(r) setViewport:(MTLViewport) {0, 0, (double) fw, (double) fh, 0, 1}];
		[mce_raw(r) setScissorRect:(MTLScissorRect) {0, 0, (NSUInteger) fw, (NSUInteger) fh}];
		[mce_raw(r) setRenderPipelineState:pkVis.pso];
		[mce_raw(r) setDepthStencilState:pkVis.depth];
		[mce_raw(r) setCullMode:MTLCullModeNone];
		mce_set_vertex_bytes(r, facet, facetLength, 22);
		mce_set_vertex_buffer(r, B(4), 0, 23);
		mce_set_vertex_buffer(r, B(1), 0, 24);
		mce_set_vertex_buffer(r, pkVis.args, 0, 25);
		mce_set_vertex_buffer(r, pkVis.surv, 0, 26);
		mce_set_vertex_buffer(r, B(0), 0, 27);
		mce_set_vertex_buffer(r, B(9), 0, 28);
		mce_set_vertex_buffer(r, B(10), 0, 29);
		mce_set_vertex_bytes(r, frame, frameLength, 30);
		mce_draw_indexed_indirect(r, MTLPrimitiveTypeTriangle, MTLIndexTypeUInt16, l->meshIdx, 0, pkVis.args, 0);
		mce_end_encoder(r);
		[r release];
		c = mce_new_compute_encoder(cb, MTLDispatchTypeSerial);
		[mce_raw(c) setLabel:@"lod pk facet prune"];
		mce_set_compute_texture(c, pkVis.dist, 0);
		mce_set_compute_buffer(c, pkVis.tiles, 0, 0);
		mce_set_compute_buffer(c, pkVis.near, 0, 1);
		mce_set_compute_bytes(c, size, 8, 2);
		[mce_raw(c) setComputePipelineState:reduce];
		mce_dispatch_threads(c, MTLSizeMake((NSUInteger) (fw + 7) / 8, (NSUInteger) (fh + 7) / 8, 1), MTLSizeMake(8, 8, 1));
		mce_memory_barrier(c, MTLBarrierScopeBuffers);
		mce_set_compute_bytes(c, facet, facetLength, 22);
		mce_set_compute_bytes(c, frame, frameLength, 0);
		mce_set_compute_bytes(c, params, paramsLength, 14);
		mce_set_compute_buffer(c, B(11), 0, 15);
		mce_set_compute_buffer(c, B(6), 0, 7);
		mce_set_compute_buffer(c, B(12), 0, 16);
		mce_set_compute_buffer(c, B(1), 0, 2);
		mce_set_compute_buffer(c, B(0), 0, 1);
		mce_set_compute_buffer(c, pkVis.tiles, 0, 8);
		mce_set_compute_buffer(c, pkVis.out, 0, 11);
		mce_set_compute_buffer(c, pkVis.cnt, 0, 12);
		mce_set_compute_bytes(c, sel, 8, 20);
		mce_set_compute_bytes(c, eps, 8, 21);
		mce_set_compute_buffer(c, pkVis.near, 0, 9);
		[mce_raw(c) setComputePipelineState:prune];
		mce_dispatch_threads(c, MTLSizeMake((NSUInteger) stride * 2, 1, 1), MTLSizeMake(256, 1, 1));
		mce_memory_barrier(c, MTLBarrierScopeBuffers);
		[mce_raw(c) setComputePipelineState:commit];
		mce_dispatch_threads(c, MTLSizeMake((NSUInteger) stride * 2, 1, 1), MTLSizeMake(256, 1, 1));
		mce_end_encoder(c); [c release];
#undef B
	}
	return 0;
}
