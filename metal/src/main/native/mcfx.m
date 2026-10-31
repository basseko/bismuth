// MetalFX spatial upscaling (opt-in: -Dmcopt.metalfx=spatial; mcopt.metal.MetalFx). The world renders into a reduced-size target; this encodes
// MTLFXSpatialScaler from that target's colour into the full-size main target's colour on the frame's command buffer (no encoder open), after
// which the GUI draws over the full-size image as usual. Nothing here runs unless the Java side asks.
#import <MetalFX/MetalFX.h>
#import "mcmetal.h"

static id<MTLFXSpatialScaler> fxScaler;
static int fxIw, fxIh, fxOw, fxOh;
static MTLPixelFormat fxFormat;

// 1 if the device can run the spatial scaler.
int mcfx_supported(Ctx *ctx) {
	return [MTLFXSpatialScalerDescriptor supportsDevice:ctx->device] ? 1 : 0;
}

// The usage bits the scaler needs of its colour input and its output texture (for the Java side's log).
int mcfx_usage(Ctx *ctx, int out, int w, int h, int ow, int oh) {
	@autoreleasepool {
		MTLFXSpatialScalerDescriptor *d = [[MTLFXSpatialScalerDescriptor new] autorelease];
		d.colorTextureFormat = d.outputTextureFormat = MTLPixelFormatRGBA8Unorm;
		d.inputWidth = w; d.inputHeight = h; d.outputWidth = ow; d.outputHeight = oh;
		id<MTLFXSpatialScaler> s = [d newSpatialScalerWithDevice:ctx->device];
		if (!s) return -1;
		int u = (int) (out ? s.outputTextureUsage : s.colorTextureUsage);
		[s release];
		return u;
	}
}

// Upscales in (iw x ih, the top-left content) into out (ow x oh). 0 on success, <0 on failure (-1: no scaler, -2: texture usage).
int mcfx_spatial(Enc *enc, id<MTLTexture> in, id<MTLTexture> out) {
	@autoreleasepool {
		int iw = (int) in.width, ih = (int) in.height, ow = (int) out.width, oh = (int) out.height;
		if (!fxScaler || iw != fxIw || ih != fxIh || ow != fxOw || oh != fxOh || in.pixelFormat != fxFormat) {
			[fxScaler release];
			fxScaler = nil;
			MTLFXSpatialScalerDescriptor *d = [[MTLFXSpatialScalerDescriptor new] autorelease];
			d.colorTextureFormat = in.pixelFormat;
			d.outputTextureFormat = out.pixelFormat;
			d.inputWidth = iw; d.inputHeight = ih; d.outputWidth = ow; d.outputHeight = oh;
			d.colorProcessingMode = MTLFXSpatialScalerColorProcessingModePerceptual;
			fxScaler = [d newSpatialScalerWithDevice:enc->ctx->device];
			if (!fxScaler) return -1;
			fxIw = iw; fxIh = ih; fxOw = ow; fxOh = oh; fxFormat = in.pixelFormat;
		}
		if ((in.usage & fxScaler.colorTextureUsage) != fxScaler.colorTextureUsage || (out.usage & fxScaler.outputTextureUsage) != fxScaler.outputTextureUsage) return -2;
		fxScaler.colorTexture = in;
		fxScaler.outputTexture = out;
		fxScaler.inputContentWidth = iw;
		fxScaler.inputContentHeight = ih;
		[fxScaler encodeToCommandBuffer:mc_frame_cmd(enc)];
		fxScaler.colorTexture = nil;
		fxScaler.outputTexture = nil;
		return 0;
	}
}
