// MetalFX spatial upscaling (opt-in: -Dmcopt.metalfx=spatial; mcopt.metal.MetalFx). The world renders into a reduced-size target; this encodes
// MTLFXSpatialScaler from that target's colour into the full-size main target's colour on the frame's command buffer (no encoder open), after
// which the GUI draws over the full-size image as usual. Nothing here runs unless the Java side asks.
#pragma clang diagnostic ignored "-Wunguarded-availability-new"
#import <MetalFX/MetalFX.h>
#import "mcmetal.h"
#import "mce.h"

static id<MTLFXSpatialScalerBase> fxScaler;  // id<MTLFXSpatialScaler> (Metal 3) or id<MTL4FXSpatialScaler> (Metal 4): the shared properties are on MTLFXSpatialScalerBase
static id<MTL4Compiler> fxCompiler;
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
			if (mce_metal4) {
				if (!fxCompiler) {
					NSError *e = nil;
					MTL4CompilerDescriptor *cd = [[MTL4CompilerDescriptor new] autorelease];
					fxCompiler = [enc->ctx->device newCompilerWithDescriptor:cd error:&e];
					if (!fxCompiler) { fprintf(stderr, "mcopt-metalfx: MTL4Compiler: %s\n", e ? e.description.UTF8String : "?"); return -1; }
				}
				fxScaler = [d newSpatialScalerWithDevice:enc->ctx->device compiler:fxCompiler];
			} else {
				fxScaler = [d newSpatialScalerWithDevice:enc->ctx->device];
			}
			if (!fxScaler) return -1;
			fxIw = iw; fxIh = ih; fxOw = ow; fxOh = oh; fxFormat = in.pixelFormat;
		}
		if ((in.usage & fxScaler.colorTextureUsage) != fxScaler.colorTextureUsage || (out.usage & fxScaler.outputTextureUsage) != fxScaler.outputTextureUsage) return -2;
		fxScaler.colorTexture = in;
		fxScaler.outputTexture = out;
		fxScaler.inputContentWidth = iw;
		fxScaler.inputContentHeight = ih;
		if (mce_metal4) {
			// the layer's textures are resident; an empty encoder on each side of the scaler gives the full barriers the layer puts around encoders
			MceCmd *c = (MceCmd *) mc_frame_cmd(enc);
			id e1 = mce_new_compute_encoder(c, MTLDispatchTypeSerial);
			mce_barrier_full(e1);
			mce_end_encoder(e1);
			[e1 release];
			mce4_cmd_flush(c);  // (the layer may keep a compute encoder open for merging: none may be open while MetalFX encodes)
			[(id<MTL4FXSpatialScaler>) fxScaler encodeToCommandBuffer:c->cb];
			id e2 = mce_new_compute_encoder(c, MTLDispatchTypeSerial);
			mce_barrier_full(e2);
			mce_end_encoder(e2);
			[e2 release];
		} else {
			[(id<MTLFXSpatialScaler>) fxScaler encodeToCommandBuffer:mc_frame_cmd(enc)];
		}
		fxScaler.colorTexture = nil;
		fxScaler.outputTexture = nil;
		return 0;
	}
}
