# Starts the title on an NVIDIA card with the Series S settings measured on an RTX 5070 (about 18-22 fps, no device loss).
#   scripts\play-nvidia.ps1 -Exe <path of your own copy's executable>
# -NoSmoothMotion  run target\<build>\victory.exe itself. By default a copy named victory_sm.exe is started instead,
#                  so a Smooth Motion profile set for that exe in the NVIDIA App applies every time.
# -SeriesX         the Series X profile instead (slower: about 7 fps at half scale)
# -Quality         fewer stretched/jumping bodies and less rainbow skin, at about 13 fps instead of about 20
# -RayTracing      keep the title's ray-tracing passes (off by default: see below)
# -Resolution      the height the game renders at before the title's own upscaler: 2160, 1440, 1080, 720, 540 or 480.
#                  Series S outputs 1080p (so up to 1080 there), Series X 2160p. Default: half the output (540 on S,
#                  1080 on X). The play-S-*.ps1 and play-X-*.ps1 scripts next to this one call it with each.
# -Adapter         on a PC with more than one card, the one to use: part of its name (-Adapter 5060) or its index
#                  (-Adapter '#1'). Default: Windows' default card (the one the main display is on).
# -Build           target\diag (the default, the build all measurements are of) or target\release
#                  (the release build ignores the ray-tracing skip list: VICTORY_SKIP_PIPELINES is a diagnostics switch)
# -More            other settings: -More @{VICTORY_PANEL='1'}
# Run it from a normal window, not as administrator: the driver then keeps its shader cache elsewhere.
param(
    [Parameter(Mandatory)][string]$Exe,
    [switch]$NoSmoothMotion,
    [switch]$SeriesX,
    [switch]$Quality,
    [switch]$RayTracing,
    [ValidateSet('2160', '1440', '1080', '720', '540', '480')][string]$Resolution,
    [string]$Adapter,
    # Local: no render-scale or upscaler patch at all (the title's own resolution, as our VIctory-dexnom builds run).
    # Render scale 1.0 through the patch stops at a breakpoint at start.
    [switch]$NativeScale,
    # Local: VICTORY_CARD_STEPS after -Quality (0 off, 1 constant-buffer fills only, 2 also buffer copies).
    # A text value, since -More's hashtable cannot be passed through powershell -File from a .bat.
    [ValidateSet('0', '1', '2')][string]$CardSteps,
    # Local: open PIX with these settings in its environment instead of starting the game (PIX launches it).
    [switch]$Pix,
    # Local: disarm the title's watchdog (VICTORY_NO_WATCHDOG; on with -Pix). It ends the process, by a write to address
    # 0 at image+0x23cf683, when the main thread has not advanced for about 30 s: a capture or a long stall trips it.
    [switch]$NoWatchdog,
    # Local: other settings as NAME=VALUE, comma-separated (a .bat cannot pass -More's hashtable):
    # -Set VICTORY_PLACED_ORDER=1,VICTORY_X=2
    [string[]]$Set = @(),
    [ValidateSet('diag', 'release')][string]$Build = 'diag',
    [hashtable]$More = @{}
)
$settings = @{
    # This fork's play set (scripts\play.ps1, not -Safe): the speed set.
    VICTORY_TEXTURE_NO_WAIT = '1'; VICTORY_RELAY = '2'; VICTORY_CARD_STEPS = '2'
    VICTORY_WIDE_BACK = '1'; VICTORY_CARD_REHASH_RUNS = '1'
    VICTORY_JOIN_COPIES = '1'; VICTORY_INDIRECT_WINDOW = '1'; VICTORY_RELAY_COPY = '1'
    VICTORY_INDIRECT_BATCH = '2'; VICTORY_MAP_THREADS = '1'; VICTORY_TITLE_PRIORITIES = '0'; VICTORY_CARD_AWAKE = '1'
    VICTORY_SHOW_AFTER = '1'; VICTORY_AMBIENT = '0.8'; VICTORY_D3D12_DRED = 'light'; VICTORY_SKIP_PIPELINES = 'none'
    VICTORY_THIN = '1'; VICTORY_JOIN_PARTS = '8'; VICTORY_UPLOAD_AHEAD = '1'
    # Cheap picture fixes, on in both sets:
    # - Overbright views: without rays the light probes keep the sky's light indoors. The fog is bounded by 1.5 times
    #   the light around, and what the title left by 2 times the estimate (was 4; ambient.rs).
    VICTORY_AMBIENT_AIR = '1.5'; VICTORY_AMBIENT_MOST = '2'
    # - Black blobs: a NaN a shader writes is made 0 before blur and the temporal upscaler spread it (emit/lower.rs).
    VICTORY_NAN_SCRUB = '1'
    # Wait for new shaders rather than skip their draws (a hitch the first time a place is seen). Each pipeline is
    # waited for at most once (VICTORY_PIPELINE_WAIT_ONCE, on by default): one that takes longer no longer stops the
    # picture in every frame it is in view (pipeline/make.rs).
    VICTORY_PIPELINE_WAIT_MS = '10000'
    # NVIDIA caps the shader-visible view heap at 1,000,000: without more dynamic slots thousands of draws a frame
    # are dropped (descriptor/host.rs dynamic_slots).
    VICTORY_DYNAMIC_SLOTS = '524288'
    # Half render scale with the title's own upscaler (Series S needs it with a scale below 1; victory-loader hooks.rs).
    VICTORY_RENDER_SCALE = '0.5'; VICTORY_OWN_UPSCALER = '1'
    # Ray walks end at the root while the structure builder is off (emit/intersect.rs).
    VICTORY_BVH_MISS = '1'
    # Memory compares on the card instead of the CPU: 11 -> about 22 fps on Series S.
    VICTORY_CARD_BUFFERS_MIN_MB = '1'; VICTORY_CARD_BUFFERS_MB = '1536'
    # Series S presents the interface as a second plane: laid over the game (compose.rs), else black before gameplay.
    VICTORY_COMPOSE = '1'
}
# Ray tracing mostly off. The structure builder cannot run on NVIDIA (its 64-lane programs are refused; upstream
# Marvisak/VIctory pull request 2311), so every ray misses, and the passes that turn misses into light made rooms go
# white depending on the view angle. rt_off_hashes.txt lists the 1,000 CS_RaytracingResolve and 34 traced-shadow
# shaders, which are skipped. The screen-probe and visibility traces keep running: skipping those too left their
# outputs as garbage (rainbow glitter everywhere).
$rtList = Join-Path $PSScriptRoot 'rt_off_hashes.txt'
if (-not $RayTracing -and (Test-Path $rtList)) {
    $settings.VICTORY_SKIP_PIPELINES = (Get-Content $rtList -Raw).Trim()
}
if ($Quality) {
    # Stretched/jumping bodies: relay stage 1 and no card steps (avatarmanz/VIctory-7900XTX issue 13,
    # fxceless/victory-fork issue 89). Rainbow skin: texture refreshes wait for the runtime's remapping.
    # Costs about 7 fps (the submitting thread waits about 30 ms a frame for the card).
    $settings.VICTORY_RELAY = '1'; $settings.VICTORY_CARD_STEPS = '0'; $settings.VICTORY_TEXTURE_NO_WAIT = '0'
}
# The render scale is the rendered height over the output height (Series S 1080, Series X 2160).
$outHeight = if ($SeriesX) { 2160 } else { 1080 }
$console = if ($SeriesX) { 'X' } else { 'S' }
if ($Resolution) {
    if ([int]$Resolution -gt $outHeight) {
        Write-Host "Series $console outputs ${outHeight}p: ${Resolution}p is above it."
        return
    }
    $settings.VICTORY_RENDER_SCALE = ([int]$Resolution / $outHeight).ToString('0.####', [Globalization.CultureInfo]::InvariantCulture)
}
if ($Adapter) { $settings.VICTORY_ADAPTER = $Adapter }
if ($CardSteps) { $settings.VICTORY_CARD_STEPS = $CardSteps }
if ($NoWatchdog -or $Pix) { $settings.VICTORY_NO_WATCHDOG = '1' }
foreach ($pair in ($Set | ForEach-Object { $_ -split ',' } | Where-Object { $_ })) {
    $name, $value = $pair -split '=', 2
    if (-not $name -or $null -eq $value) { Write-Host "-Set wants NAME=VALUE, not '$pair'."; return }
    $settings[$name.Trim()] = $value.Trim()
    Write-Host "Set $($name.Trim())=$($value.Trim())"
}
foreach ($k in $More.Keys) { $settings[$k] = $More[$k] }
if ($NativeScale) {
    $settings.Remove('VICTORY_RENDER_SCALE'); $settings.Remove('VICTORY_OWN_UPSCALER')
    Write-Host "Series $console, the title's own resolution (no render-scale or upscaler patch)"
} else {
    $scale = [double]::Parse($settings.VICTORY_RENDER_SCALE, [Globalization.CultureInfo]::InvariantCulture)
    Write-Host "Series $console, rendering at $([int]($outHeight * $scale))p (render scale $($settings.VICTORY_RENDER_SCALE))"
}

$victory = Join-Path $PSScriptRoot "..\target\$Build\victory.exe"
if (-not (Test-Path $victory)) { Write-Host "No $victory. Build it first (README.md)."; return }
if (-not (Test-Path $Exe)) { Write-Host "No $Exe."; return }
if (-not $NoSmoothMotion) {
    # A copy under the name the NVIDIA App's Smooth Motion profile is set for, refreshed when the build changes.
    $sm = Join-Path (Split-Path $victory) 'victory_sm.exe'
    if (-not (Test-Path $sm) -or (Get-Item $sm).LastWriteTime -lt (Get-Item $victory).LastWriteTime) {
        Copy-Item $victory $sm -Force
    }
    $victory = $sm
}
# (Other settings left over in this window go; where your files are stays.)
$paths = 'VICTORY_CONFIG', 'VICTORY_GAME_DIR', 'VICTORY_CACHE_DIR', 'VICTORY_DATA_DIR'
Get-ChildItem Env:VICTORY_* | Where-Object { $_.Name -notin $paths } | ForEach-Object { Remove-Item "Env:$($_.Name)" }
foreach ($k in $settings.Keys) { Set-Item "Env:$k" $settings[$k] }
$Exe = (Resolve-Path $Exe).Path
$runArgs = @('run', $Exe, '--execute')
if (-not $SeriesX) { $runArgs += @('--console', 'series-s') }
if ($Pix) {
    $root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
    $pixExe = Get-ChildItem 'C:\Program Files\Microsoft PIX\*\WinPix.exe' | Sort-Object FullName | Select-Object -Last 1
    $shown = ($runArgs | ForEach-Object { if ($_ -match ' ') { '"' + $_ + '"' } else { $_ } }) -join ' '
    Write-Host "Fill these into PIX's 'Launch Win32' tab:"
    Write-Host "  Path:              $((Resolve-Path $victory).Path)"
    Write-Host "  Working directory: $root"
    Write-Host "  Arguments:         $shown"
    Write-Host "Tick 'For GPU capture', then Launch. In the game: load the save, stand still, press Print Screen once."
    Write-Host "Save the capture in C:\Games\xbox\captures."
    Start-Process $pixExe.FullName -WorkingDirectory $root
    return
}
Push-Location (Join-Path $PSScriptRoot '..')   # victory.toml is looked for here
try { & $victory @runArgs } finally { Pop-Location }
