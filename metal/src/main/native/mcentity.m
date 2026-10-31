// Shaders milestone 5 (-Dmcopt.shaders=true; mcopt.metal.EntityShade): vanilla-pipeline geometry (entities, block entities, items) takes the terrain's
// sun shadows and casts into its shadow map.
//   receive: mce_publish / mce_off set the frame block (the terrain's ShadeFrame + the entity extras) every patched draw binds at fragment buffer 29
//            together with this frame's shadow map at fragment texture 24 (mce_bind); mce_map is called by mcown.m's mcs_frame with that map.
//   cast:    mce_begin opens a depth-only render pass on slice `slice` of that map in the pre command buffer, loading what the terrain's shadow pass
//            wrote there, and returns an Enc the usual mc_r_* calls encode into; mce_end closes it.
#import "mcmetal.h"
#import "mce.h"
#include <stdlib.h>
#include <string.h>

#define MCE_BLOCK 400

static struct {
	id<MTLTexture> map;         // this frame's shadow map (mcown.m sh.cur)
	id<MTLTexture> dummy;       // a 1 layer depth array bound when there is no map (the shader does not sample it then)
	id<MTLDepthStencilState> depthLess;
	uint8_t block[MCE_BLOCK];
	int on;                     // a block is published (the sun applies in this frame's world pass)
} ent;

// mcown.m, mcs_frame: this frame's shadow map
void mce_map(id<MTLTexture> map) {
	ent.map = map;
}

void mce_publish(const void *bytes, int length) {
	if (length > MCE_BLOCK) length = MCE_BLOCK;
	memset(ent.block, 0, sizeof ent.block);
	memcpy(ent.block, bytes, (size_t) length);
	ent.on = 1;
}

// nothing applies (no sun, the hand, the GUI): patched draws get a zero block, whose ent.x is 0: the shader returns the colour untouched
void mce_off(void) {
	ent.on = 0;
}

// Before a draw of a patched pipeline, in the open render encoder.
void mce_bind(Enc *enc) {
	id<MTLRenderCommandEncoder> r = enc->render;
	if (!r) return;
	static const uint8_t zero[MCE_BLOCK];
	id<MTLTexture> t = ent.on ? ent.map : nil;
	if (!t) {
		if (!ent.dummy) {
			MTLTextureDescriptor *d = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatDepth32Float width:1 height:1 mipmapped:NO];
			d.textureType = MTLTextureType2DArray;
			d.arrayLength = 1;
			d.usage = MTLTextureUsageShaderRead;
			d.storageMode = MTLStorageModePrivate;
			ent.dummy = [enc->ctx->device newTextureWithDescriptor:d];
			mce_make_resident(ent.dummy);
			ent.dummy.label = @"shade entity dummy shadow map";
		}
		t = ent.dummy;
	}
	mce_set_fragment_bytes(r, ent.on ? ent.block : zero, MCE_BLOCK, 29);
	mce_set_fragment_texture(r, t, 24);
}

// A depth-only render pass on `slice` of the shadow map (contents loaded) in the pre command buffer; the Enc to encode into (NULL: no shadow map).
// Standard Z (Less, depth write), no culling, no bias, the whole map as viewport: as the terrain's own shadow draws (mcown.m mcs_shadow_pass).
Enc *mce_begin(Enc *enc, int slice) {
	id<MTLTexture> map = ent.map;
	if (!map) return NULL;  // (casters are only recorded in frames whose shadow map was rendered; the receive block may be off again by now: the hand)
	@autoreleasepool {
		if (!ent.depthLess) {
			MTLDepthStencilDescriptor *dd = [[MTLDepthStencilDescriptor new] autorelease];
			dd.depthCompareFunction = MTLCompareFunctionLess;
			dd.depthWriteEnabled = YES;
			ent.depthLess = [enc->ctx->device newDepthStencilStateWithDescriptor:dd];
		}
		mc_pre_end_encoders(enc);
		MTLRenderPassDescriptor *rp = [MTLRenderPassDescriptor renderPassDescriptor];
		rp.depthAttachment.texture = map;
		rp.depthAttachment.slice = (NSUInteger) slice;
		rp.depthAttachment.loadAction = MTLLoadActionLoad;
		rp.depthAttachment.storeAction = MTLStoreActionStore;
		id<MTLRenderCommandEncoder> r = mce_new_render_encoder(mc_pre(enc), rp);
		if (!r) return NULL;
		[r retain];
		[mce_raw(r) setLabel:[NSString stringWithFormat:@"shade entity shadow c%d", slice]];
		[mce_raw(r) setViewport:(MTLViewport) {0, 0, (double) map.width, (double) map.height, 0, 1}];
		[mce_raw(r) setDepthClipMode:MTLDepthClipModeClamp];
		[mce_raw(r) setDepthStencilState:ent.depthLess];
		[mce_raw(r) setCullMode:MTLCullModeNone];
		Enc *f = calloc(1, sizeof(Enc));
		f->ctx = enc->ctx;
		f->render = r;
		f->depth = map;
		f->colorCount = 0;
		f->width = (int) map.width;
		f->height = (int) map.height;
		return f;
	}
}

void mce_end(Enc *f) {
	if (!f) return;
	mce_end_encoder(f->render);
	[f->render release];
	free(f);
}
