Bismuth @VERSION@

A Metal 4 renderer and shaders for Minecraft on Apple Silicon Macs. No Sodium needed. VERY MUCH ALPHA: expect bugs, and back up your
worlds.

You need
- An Apple Silicon Mac (M1 or newer) on macOS 26 or later
- Java 25 (the official launcher and Prism download it for 26.3; nothing to install)
- Minecraft: Java Edition 26.3
- Fabric Loader 0.19.5 or newer for 26.3

Install
1. Install Fabric for Minecraft 26.3: https://fabricmc.net/use/installer/
   (or in Prism Launcher: new instance, version 26.3, loader Fabric)
2. Put bismuth-@VERSION@.jar in your mods folder.
   Official launcher: ~/Library/Application Support/minecraft/mods
   (in Finder: Go > Go to Folder, paste that path; create the mods folder if it's missing)
   In Prism: Edit the instance > Mods > Add file.
3. Start Minecraft with the Fabric profile.

No Sodium needed: Bismuth won't start alongside it (Fabric says so), so take Sodium out of the mods folder.
Remove mcopt too: the two can't run together.
Fabric API isn't required; keep it if your other mods need it.
Mods that require Sodium (for example Better Block Entities or Iris) can't be used with this build.
Distant Horizons runs on the Metal renderer next to Bismuth's own terrain. With DH set to its OpenGL renderer, Bismuth
switches its renderer off for it.

The jar is a mod, not an app: double-clicking it only shows these install steps.

For the most fps: Options > Video Settings, VSync off and Max Framerate Unlimited.
Shows an fps counter in the top-left corner (hidden with F1 or F3; fn + F1 or fn + F3 on a Mac keyboard).

On a computer the mod can't run on (not an Apple Silicon Mac, or macOS older than 26), the game stops at launch
and says what's missing; remove the Bismuth jar from the mods folder to play there.
To check it's working: press F3 (fn + F3 on a Mac keyboard); the bottom right says Metal 4 (Bismuth),
or Metal 3 (Bismuth) on Macs without Metal 4.

The first launch writes config/mcopt.properties in your game folder (in Prism, the instance's minecraft/config
folder); profile=none there turns the perf profile off.
Shaders are opt-in: add mcopt.shaders=true to that file.
To switch Bismuth's renderer off, add mcopt.metal=false to that file (or launch with -Dmcopt.metal=false).

Licence: Apache-2.0 (LICENSE, NOTICE). A fork of mcopt by Noah Dunnagan: https://github.com/noahdunnagan/mcopt
Source: https://github.com/basseko/bismuth
