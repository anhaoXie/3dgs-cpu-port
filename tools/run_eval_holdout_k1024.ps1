# Retrain the held-out (--eval) protocol with K_MAX=1024 *in training*, so the
# cap used to optimise and the cap used to score are the same cap.
#
# Why: the delivered truck_eval run was trained under K_MAX=256 truncation and
# then rendered at K_MAX=1024 (notes/kmax-ablation.md sections 7-8). The weights
# therefore never saw the candidate list that scored them. This repeats that run
# as a single-variable A/B against tools/run_eval_holdout.ps1 -- same scene,
# same resolution, same iteration count, same densification defaults, same
# threads. Only K changes.
#
# Non-destructive: output goes to output/truck_eval_k1024 and its own log files,
# so output/truck_eval (the delivered K=256 model) is untouched.
#
# MEASURED ON THE REFERENCE MACHINE (15.7 GB, no GPU): this does NOT fit. During
# training the commit charge reached 29.36 of a 29.72 GB limit, free physical
# memory fell to 0.44 GB, and throughput collapsed to 4.38 s/it -- 8.5 h for 7000
# iterations versus 52 min at K=256 (a 9.8x slowdown caused by paging, not by the
# extra arithmetic). The run was aborted at iteration 30 with no checkpoint.
# Keep this script for a machine with more memory. On the reference box there is
# no K that is both self-consistent (train cap == scoring cap) and seam-converged,
# because convergence needs K >= 1024 and training at K >= 1024 does not fit.
# See notes/kmax-ablation.md section 11.
#
# ASCII only -- PowerShell 5.1 decodes BOM-less .ps1 as GBK, which would mangle
# any non-ASCII comment or path.

$ErrorActionPreference = "Continue"

$w      = Split-Path -Parent $PSScriptRoot
$repo   = "$w\gaussian-splatting"
$py     = "$w\venv\Scripts\python.exe"
$logs   = "$w\logs"
$status = "$logs\eval_k1024_status.log"
$model  = "$w\output\truck_eval_k1024"
$K      = 1024
# The rasterizer additionally clamps K by DSH_3DGS_BUDGET // (n_tiles * 256).
# At 245x136 there are ceil(245/16) * ceil(136/16) = 16 * 9 = 144 tiles, so the
# budget must reach K * 144 * 256 = 37748736 or K=1024 is silently reduced.
$budget = $K * 144 * 256

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
Note "K=1024 held-out retrain started (pid $PID, budget=$budget)"
Note "  model dir: $model"

if (Test-Path $model) {
    Note "removing stale $model"
    Remove-Item $model -Recurse -Force -ErrorAction SilentlyContinue
}

Note "training: -r 4 --eval, 7000 iterations, upstream densification defaults, K=1024"
$sw = [Diagnostics.Stopwatch]::StartNew()
& $py train.py -s "$w\data\truck" -m $model `
    --data_device cpu --disable_viewer -r 4 --eval `
    --iterations 7000 --test_iterations 7000 --save_iterations 7000 `
    *> "$logs\eval_k1024_train.log"
$trainExit = $LASTEXITCODE
Note ("  train exit={0}  elapsed={1:N1} min" -f $trainExit, $sw.Elapsed.TotalMinutes)
if ($trainExit -ne 0) {
    Note "TRAIN FAILED -- stopping (see logs\eval_k1024_train.log)"
    exit 1
}

Note "split sanity (official reader + LLFF rule)"
& $py "$w\tools\check_split.py" --model_path $model *> "$logs\eval_k1024_split_check.log"
Note "  check_split exit=$LASTEXITCODE"

Note "render.py at iteration 7000 (same cap as training)"
& $py render.py -m $model --iteration 7000 *> "$logs\render_eval_k1024.log"
Note "  render exit=$LASTEXITCODE"

Note "official metrics.py (test split only)"
& $py metrics.py -m $model *> "$logs\metrics_eval_k1024.log"
Note "  metrics.py exit=$LASTEXITCODE"

Note "paired train/test metrics at the same iteration"
New-Item -ItemType Directory -Force -Path "$w\notes\eval_k1024" | Out-Null
foreach ($split in @("test", "train")) {
    $json = "$w\notes\eval_k1024\metrics_${split}_7000.json"
    & $py "$w\tools\eval_split_metrics.py" --model $model --iteration 7000 `
        --split $split --json $json *> "$logs\eval_k1024_metrics_$split.log"
    Note ("  {0} metrics exit={1}" -f $split, $LASTEXITCODE)
}

Note "seam ratio on the retrained model (should be near converged)"
& $py "$w\tools\seam_metric.py" "$model\train\ours_7000\renders" `
    *> "$logs\eval_k1024_seam.log"
Get-Content "$logs\eval_k1024_seam.log" -ErrorAction SilentlyContinue |
    Select-Object -Last 3 | ForEach-Object { Note "  $_" }

Note "K=1024 RETRAIN COMPLETE"
