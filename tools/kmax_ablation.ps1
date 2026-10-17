# Re-render an existing checkpoint with a different per-tile candidate cap (K_MAX)
# and measure it, to test whether K_MAX is what actually limits render quality.
#
# Why this is needed: notes/results.md reported that K_MAX=256 "basically removes"
# the 16x16 tile seams -- but that was measured at iteration 700, where the scene
# has few, large gaussians and the cap never binds.  At the delivery checkpoints
# the seams are plainly visible at K=256, so the claim had to be re-tested where
# it matters.  It was false: see tools/seam_metric.py, which measures the seam
# directly (boundary gradient / interior gradient).
#
# The render goes into a *sibling* model directory, so the K=256 renders are never
# overwritten and stay available as the baseline.
#
# Note on the budget: the rasterizer computes
#     kmax = min(K_MAX, counts_max, BUDGET // (n_tiles * 256))
# so raising K alone is not enough -- BUDGET must be raised too, or the cap is
# silently clamped back down.  For 245x136 there are 144 tiles; the upstream
# default BUDGET of 6e6 would allow only 162.
#
# Usage:
#   powershell -NoProfile -ExecutionPolicy Bypass -File tools\kmax_ablation.ps1 -K 1024
#   powershell -NoProfile -ExecutionPolicy Bypass -File tools\kmax_ablation.ps1 -K 1024 `
#       -Source output\truck_eval -BothSplits

param(
    [int]$Iteration = 15000,
    [int]$K = 1024,
    [string]$Source = "output\truck_full",
    [switch]$BothSplits
)

$ErrorActionPreference = "Continue"

$w      = Split-Path -Parent $PSScriptRoot        # <root>\tools -> <root>
$repo   = "$w\gaussian-splatting"
$py     = "$w\venv\Scripts\python.exe"
$logs   = "$w\logs"
$srcName = Split-Path -Leaf $Source
$src    = if ([IO.Path]::IsPathRooted($Source)) { $Source } else { "$w\$Source" }
$dst    = "$w\output\${srcName}_k$K"
$status = "$logs\kmax_ablation_${srcName}_k$K.log"

# 245x136 render -> ceil(245/16)*ceil(136/16) = 16*9 = 144 tiles, 256 px each.
$tiles  = 144
$budget = [Math]::Max(16000000, [int]($K * $tiles * 256))

$env:PYTHONPATH       = "$w\cpu_port"
$env:PYTHONIOENCODING = "utf-8"
$env:TORCH_HOME       = "$w\torch_home"
$env:MPLCONFIGDIR     = "$w\mplcache"
$env:DSH_3DGS_KMAX    = "$K"
$env:DSH_3DGS_BUDGET  = "$budget"
$env:DSH_3DGS_THREADS = "8"

function Note($msg) {
    $line = "[{0}] {1}" -f (Get-Date -Format "HH:mm:ss"), $msg
    Add-Content -Path $status -Value $line
    Write-Host $line
}

Set-Location $repo
$kmaxBudget = [int]($budget / ($tiles * 256))
Note "K_MAX ablation: source=$srcName checkpoint=$Iteration K=$K BUDGET=$budget (kmax_budget=$kmaxBudget)"
if ($kmaxBudget -lt $K) { Note "  WARNING: budget clamps K down to $kmaxBudget -- raise BUDGET for a valid control" }

# A minimal model dir is enough: Scene skips writing input.ply/cameras.json when a
# load_iteration is given, and reads only cfg_args + the point cloud.
$plySrc = "$src\point_cloud\iteration_$Iteration\point_cloud.ply"
if (-not (Test-Path $plySrc)) { Note "ABORT: no checkpoint at $plySrc"; exit 1 }

New-Item -ItemType Directory -Force -Path "$dst\point_cloud\iteration_$Iteration" | Out-Null
Copy-Item "$src\cfg_args" "$dst\cfg_args" -Force
Copy-Item $plySrc "$dst\point_cloud\iteration_$Iteration\point_cloud.ply" -Force
Note "prepared $dst"

$renderArgs = @("-m", $dst, "--iteration", $Iteration)
if (-not $BothSplits) { $renderArgs += "--skip_test" }
$splitLabel = if ($BothSplits) { "train+test" } else { "train" }

Note "render $splitLabel @ $Iteration with K=$K"
$sw = [Diagnostics.Stopwatch]::StartNew()
& $py render.py @renderArgs *> "$logs\render_${srcName}_kmax_k$K.log"
Note ("  render exit={0}  elapsed={1:N1} min" -f $LASTEXITCODE, $sw.Elapsed.TotalMinutes)

New-Item -ItemType Directory -Force -Path "$w\notes\eval" | Out-Null
New-Item -ItemType Directory -Force -Path "$w\notes\probe" | Out-Null

$splits = if ($BothSplits) { @("train", "test") } else { @("train") }
foreach ($sp in $splits) {
    Note "metrics: $sp split"
    & $py "$w\tools\eval_split_metrics.py" --model "$dst" --iteration $Iteration --split $sp `
        --json "$w\notes\eval\metrics_${sp}_${srcName}_K${K}_$Iteration.json" `
        *> "$logs\metrics_${srcName}_${sp}_k$K.log"
    Note "  metrics exit=$LASTEXITCODE"
    Get-Content "$logs\metrics_${srcName}_${sp}_k$K.log" -EA SilentlyContinue |
        Select-String -Pattern 'PSNR|SSIM|LPIPS|views' | ForEach-Object { Note "  $($_.Line.Trim())" }
}

Note "montage for visual comparison"
& $py "$w\tools\make_montage.py" --model "$dst" --iteration $Iteration --split train `
    --scale 4 --views 0 60 120 180 `
    --out "$w\notes\probe\kmax_${srcName}_${Iteration}_K$K.png" `
    --single-out "$w\notes\probe\render_${srcName}_${Iteration}_K$K.png" *> "$logs\montage_${srcName}_k$K.log"
Note "  montage exit=$LASTEXITCODE"

# Direct seam measurement: boundary gradient / interior gradient, tile 16x16.
Note "seam metric (boundary/interior gradient ratio)"
foreach ($sp in $splits) {
    $rd = "$dst\$sp\ours_$Iteration\renders"
    if (Test-Path $rd) {
        & $py "$w\tools\seam_metric.py" "$rd" *> "$logs\seam_${srcName}_${sp}_k$K.log"
        Get-Content "$logs\seam_${srcName}_${sp}_k$K.log" -EA SilentlyContinue |
            Select-Object -Skip 3 | ForEach-Object { Note "  $_" }
    }
}

Note "ABLATION COMPLETE ($srcName, K=$K)"
