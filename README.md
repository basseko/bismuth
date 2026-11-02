# Bismuth

A Metal 4 renderer and shaders for Minecraft: Java Edition on Apple Silicon Macs. Bismuth draws the world with its own
native Metal renderer (Metal 4 where the GPU supports it, Metal 3 otherwise), at native resolution with real frames: no
upscaling or frame generation unless you turn them on. **Very much alpha:** expect bugs, and back up your worlds.

Bismuth is a fork of [mcopt](https://github.com/noahdunnagan/mcopt) by Noah Dunnagan. It adds the Metal 4 backend,
optional shaders (sun shadows, water, sky and fog, clouds, foliage, volumetric light, colour grading) and an opt-in
MetalFX path on top of mcopt 0.3.0-alpha.2. mcopt's settings names (`mcopt.properties`, `-Dmcopt.*`) are kept for now.

## Requirements

- An Apple Silicon Mac on macOS 26 or later
- Java 25 (the official launcher and Prism download it for 26.3; nothing to install)
- Minecraft 26.3 with Fabric Loader 0.19.5 or newer

## Install

1. Install Fabric for Minecraft 26.3: https://fabricmc.net/use/installer/ (or in Prism Launcher: new instance, 26.3,
   loader Fabric).
2. Download the Bismuth jar from [Releases](../../releases).
3. Put it in `~/Library/Application Support/minecraft/mods`. Create the `mods` folder if it's missing
   (Finder: Go > Go to Folder). In Prism: Edit the instance > Mods > Add file.
4. Start Minecraft with the Fabric profile.

Sodium isn't needed: Bismuth won't start alongside it, so take Sodium out of that folder if it's there. Remove mcopt
too: the two can't run together. Fabric API
isn't required; keep it if your other mods need it.

To check it's working: press F3 (fn + F3 on a Mac keyboard); the bottom right says `Metal 4 (Bismuth)` (or `Metal 3 (Bismuth)` on
Macs without Metal 4).

Shows an fps counter in the top-left corner (hidden with F1 or F3; fn + F1 or fn + F3 on a Mac keyboard).

## Settings

- The perf profile is on by default.
- `profile=none` in `~/Library/Application Support/minecraft/config/mcopt.properties` (in Prism, the instance's
  `minecraft/config` folder) turns it off. The first launch writes that file.
- Shaders are an opt-in: `mcopt.shaders=true` in the same file.
- `mcopt.metal4=false` uses the Metal 3 backend instead of Metal 4.
- Far terrain is an experimental opt-in: `mcopt.lod=true` in the same file, for Macs with 10 or more GPU cores.

## Numbers

mcopt's numbers (before the fork), fps spinning / flying, same test world, 1920x1080, render distance 16, VSync off, at least two runs each:

| Mac | vanilla | Sodium 0.9.3 | 0.2.0-alpha.3 | 0.3.0-alpha.1 |
|---|---|---|---|---|
| Mac mini M4 (10-core GPU, 16 GB) | 156 / 146 | 275-281 / 253-267 | 1202-1223 / 1068-1074 | 1209-1211 / 1068-1069 |
| MacBook Neo (A18 Pro, 5-core GPU, 8 GB) | 79-81 / 69-72 | 133-146 / 85-91 | 514-579 / 422-449 | 591-638 / 477-504 |
| Mac mini M6 (12-core GPU, 16 GB) | 297-328 / 283-291 | 454-460 / 406-413 | 1988-1991 / 1738-1742 | 2020-2022 / 1690-1704 |

The mini and Neo cells ran on 6c133ecc, whose only difference from 9865a0a3 is vGroup, which is gated off on those GPUs. The Neo's vanilla and Sodium cells come from the da566d00 run. This build differs from 9865a0a3 only in code that runs when Distant Horizons is installed.

For the most fps: Options > Video Settings, VSync off and Max Framerate Unlimited. The numbers above are with VSync off.

## Known issues

- Mods that require Sodium (for example Better Block Entities or Iris) can't be used with this build.
- Mods that call OpenGL directly can't draw on the Metal backend. To switch Bismuth's renderer off, add
  `mcopt.metal=false` to `config/mcopt.properties` (or launch with `-Dmcopt.metal=false`).
- Distant Horizons runs on the Metal renderer next to Bismuth's own terrain. With DH set to its OpenGL renderer,
  Bismuth switches its renderer off for it.
- Far terrain is experimental. On Macs with fewer than 10 GPU cores, it costs most of the fps.

## Plans

- Far terrain on by default, with tiers for smaller Macs
- Faster world generation (native noise, bit-exact)
- More exact server-tick and chunk optimizations, held for later alphas
- Faster startup
- Wider hardware testing

## Licence and credits

[Apache License 2.0](LICENSE). Keep the [NOTICE](NOTICE) file with any copy. Bismuth is built on mcopt by Noah Dunnagan;
the NOTICE credits both.
