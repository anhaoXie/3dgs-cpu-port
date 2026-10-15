# Deliver the remaining artefacts, with explicit status writes (no reliance on
# buffered stdout, which is what made the previous tail script opaque).
#
#   1. wait for iteration_15000 -> render train split + metrics(montage)
#   2. wait for iteration_30000 -> render train split + metrics(montage) + curves
#   3. held-out `--eval` run at the same resolution/iterations -> official metrics.py
#
# Status is appended to logs/remaining_status.log after every step.

$ErrorActionPreference = "Continue"

# IMPORTANT: derive the workspace root from the script location rather than
# hardcoding it.  Windows PowerShell 5.1 reads BOM-less .ps1 files as ANSI/GBK,
# which turns a non-ASCII user name in a hardcoded path into mojibake and makes
# every Test-Path/Set-Location fail.  $PSScriptRoot is already decoded correctly.
$w = Split-Path -Parent $PSScriptRoot          # <root>\tools -> <root>
$repo = "$w\gaussian-splatting"
$py = "$w\venv\Scripts\python.exe"
$logs = "$w\logs"
$status = "$logs\remaining_status.log"

$env:PYTHONPATH = "$w\cpu_port"
$env:PYTHONIOENCODING = "utf-8"
$env:TORCH_HOME = "$w\torch_home"
$env:MPLCONFIGDIR = "$w\mplcache"
$env:DSH_3DGS_KMAX = "256"
$env:DSH_3DGS_BUDGET = "16000000"
$env:DSH_3DGS_THREADS = "8"

function Note($msg) {
    $line = "[{0}] {1}" -f (Get-Date -Format "HH:mm:ss"), $msg
    Add-Content -Path $status -Value $line
    Write-Host $line
}

function Wait-Iter($iter, $hours) {
    $f = "$w\output\truck_full\point_cloud\iteration_$iter\point_cloud.ply"
    $sw = [Diagnostics.Stopwatch]::StartNew()
    while (-not (Test-Path $f) -and $sw.Elapsed.TotalHours -lt $hours) { Start-Sleep -Seconds 60 }
    if (Test-Path $f) {
        Note ("checkpoint $iter present after {0:N1} min ({1:N1} MB)" -f $sw.Elapsed.TotalMinutes, ((Get-Item $f).Length / 1MB))
        Start-Sleep -Seconds 90
        return $true
    }
    Note "gave up waiting for checkpoint $iter"
    return $false
}

Set-Location $repo
Note "remaining-delivery script started (pid $PID)"

foreach ($iter in @(15000, 30000)) {
    if (-not (Wait-Iter $iter 14)) { continue }

    Note "render train split @ $iter"
    & $py render.py -m "$w\output\truck_full" --iteration $iter --skip_test *> "$logs\render_full_$iter.log"
    Note "  render exit=$LASTEXITCODE"

    Note "metrics(montage) train split @ $iter"
    & $py "$w\tools\make_montage.py" --model "$w\output\truck_full" --iteration $iter `
        --split train --scale 4 --views 0 100 200 250 *> "$logs\montage_$iter.log"
    Note "  montage exit=$LASTEXITCODE"
    Get-Content "$logs\montage_$iter.log" -ErrorAction SilentlyContinue |
        Select-Object -Last 6 | ForEach-Object { Note "  $_" }

    Note "loss curves @ $iter"
    & $py "$w\tools\extract_curves.py" --model "$w\output\truck_full" `
        --log "$logs\full_train.log" --outdir "$w\notes" *> "$logs\curves_full.log"
    Note "  curves exit=$LASTEXITCODE"
}

# ---------------------------------------------------------------- held-out run
Note "starting held-out (--eval) training at -r 4 / 7000 iters"
& $py train.py -s "$w\data\truck" -m "$w\output\truck_eval" `
    --data_device cpu --disable_viewer -r 4 --eval `
    --iterations 7000 --test_iterations 7000 --save_iterations 7000 `
    *> "$logs\eval_train.log"
Note "  eval train exit=$LASTEXITCODE"

& $py render.py -m "$w\output\truck_eval" --iteration 7000 *> "$logs\render_eval.log"
Note "  render eval exit=$LASTEXITCODE"

& $py metrics.py -m "$w\output\truck_eval" *> "$logs\metrics_eval.log"
Note "  metrics.py exit=$LASTEXITCODE"
Get-Content "$logs\metrics_eval.log" -ErrorAction SilentlyContinue |
    Select-Object -Last 12 | ForEach-Object { Note "  $_" }

& $py "$w\tools\make_montage.py" --model "$w\output\truck_eval" --iteration 7000 `
    --split test --scale 4 --views 0 8 16 24 *> "$logs\montage_eval_test.log"
Note "  montage eval-test exit=$LASTEXITCODE"
Get-Content "$logs\montage_eval_test.log" -ErrorAction SilentlyContinue |
    Select-Object -Last 6 | ForEach-Object { Note "  $_" }

& $py "$w\tools\extract_curves.py" --model "$w\output\truck_eval" `
    --log "$logs\eval_train.log" --outdir "$w\notes\eval" *> "$logs\curves_eval.log"
Note "  curves eval exit=$LASTEXITCODE"

Note "REMAINING DELIVERY COMPLETE"
