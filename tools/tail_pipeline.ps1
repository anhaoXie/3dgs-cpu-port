# Staged tail pipeline for the 3DGS CPU reproduction.
#
# The CPU rasterizer scales ~linearly with the Gaussian count, and adaptive
# densification grows the model ~4-6x by iteration 30000, so a full 30k run
# takes on the order of 10+ hours on this 12-thread CPU.  Rather than block on
# it, this script delivers progressively:
#
#   stage A : as soon as iteration 7000 exists
#               render + metrics(montage) on the train split, loss curves
#   stage B : a separate held-out run WITH upstream's --eval protocol
#               (every 8th camera is a test view) so upstream metrics.py can run
#   stage C : iteration 15000, if the main run gets there
#   stage D : iteration 30000, if the main run gets there
#
# Every step is logged; failures never abort later stages.

$ErrorActionPreference = "Continue"
# Derive the root from the script location: Windows PowerShell 5.1 decodes
# BOM-less .ps1 files as ANSI/GBK, so a hardcoded non-ASCII path would be mangled.
$w = Split-Path -Parent $PSScriptRoot          # <root>\tools -> <root>
$repo = "$w\gaussian-splatting"
$py = "$w\venv\Scripts\python.exe"
$logs = "$w\logs"

$env:PYTHONPATH = "$w\cpu_port"
$env:PYTHONIOENCODING = "utf-8"
$env:TORCH_HOME = "$w\torch_home"
# Must match training, else render.py truncates tile candidate lists differently.
$env:DSH_3DGS_KMAX = "256"
$env:DSH_3DGS_BUDGET = "16000000"
# 8 was the measured optimum on this hybrid P/E-core CPU (4 threads tied;
# 1 and 2 threads were ~2x and ~1.25x slower respectively).
$env:DSH_3DGS_THREADS = "8"

function Wait-ForIter($iter, $hours) {
    $p = "$w\output\truck_full\point_cloud\iteration_$iter\point_cloud.ply"
    $sw = [Diagnostics.Stopwatch]::StartNew()
    while (-not (Test-Path $p) -and $sw.Elapsed.TotalHours -lt $hours) { Start-Sleep -Seconds 60 }
    if (Test-Path $p) {
        Write-Output ("[{0}] checkpoint present after {1:N1} min ({2:N1} MB)" -f `
            (Get-Date -Format HH:mm:ss), $sw.Elapsed.TotalMinutes, ((Get-Item $p).Length / 1MB))
        Start-Sleep -Seconds 90
        return $true
    }
    Write-Output "[$([DateTime]::Now.ToString('HH:mm:ss'))] giving up waiting for iteration_$iter"
    return $false
}

function Deliver($iter) {
    Set-Location $repo
    Write-Output "--- render train split @ $iter ---"
    & $py render.py -m "$w\output\truck_full" --iteration $iter --skip_test *> "$logs\render_full_$iter.log"
    Write-Output "render exit=$LASTEXITCODE"
    Write-Output "--- metrics(montage) train split @ $iter ---"
    & $py "$w\tools\make_montage.py" --model "$w\output\truck_full" --iteration $iter `
        --split train --scale 4 --views 0 100 200 250 *> "$logs\montage_$iter.log"
    Write-Output "montage exit=$LASTEXITCODE"
    Get-Content "$logs\montage_$iter.log" -ErrorAction SilentlyContinue | Select-Object -Last 8
}

# ------------------------------------------------------------------ stage A
if (Wait-ForIter 7000 5) {
    Deliver 7000

    Write-Output "--- loss curves (main run) ---"
    & $py "$w\tools\extract_curves.py" --model "$w\output\truck_full" `
        --log "$logs\full_train.log" --outdir "$w\notes" *> "$logs\curves_full.log"
    Write-Output "curves exit=$LASTEXITCODE"
    Get-Content "$logs\curves_full.log" -ErrorAction SilentlyContinue | Select-Object -Last 12
}

# ------------------------------------------------------------------ stage B
# Held-out protocol at the SAME resolution/iterations as the stage-A deliverable,
# so train-split and test-split numbers are directly comparable.
Write-Output "--- held-out (--eval) training run at -r 4 / 7000 iters ---"
& $py train.py -s "$w\data\truck" -m "$w\output\truck_eval" `
    --data_device cpu --disable_viewer -r 4 --eval `
    --iterations 7000 --test_iterations 7000 --save_iterations 7000 `
    *> "$logs\eval_train.log"
Write-Output "eval train exit=$LASTEXITCODE"

Set-Location $repo
& $py render.py -m "$w\output\truck_eval" --iteration 7000 *> "$logs\render_eval.log"
Write-Output "render eval exit=$LASTEXITCODE"

& $py metrics.py -m "$w\output\truck_eval" *> "$logs\metrics_eval.log"
Write-Output "metrics eval exit=$LASTEXITCODE"
Get-Content "$logs\metrics_eval.log" -ErrorAction SilentlyContinue | Select-Object -Last 14

& $py "$w\tools\make_montage.py" --model "$w\output\truck_eval" --iteration 7000 `
    --split test --scale 4 --views 0 8 16 24 *> "$logs\montage_eval_test.log"
Write-Output "montage eval-test exit=$LASTEXITCODE"
Get-Content "$logs\montage_eval_test.log" -ErrorAction SilentlyContinue | Select-Object -Last 8

& $py "$w\tools\extract_curves.py" --model "$w\output\truck_eval" `
    --log "$logs\eval_train.log" --outdir "$w\notes\eval" *> "$logs\curves_eval.log"
Write-Output "curves eval exit=$LASTEXITCODE"

# ------------------------------------------------------------------ stage C / D
if (Wait-ForIter 15000 12) { Deliver 15000 }
if (Wait-ForIter 30000 20) {
    Deliver 30000
    & $py "$w\tools\extract_curves.py" --model "$w\output\truck_full" `
        --log "$logs\full_train.log" --outdir "$w\notes" *> "$logs\curves_full.log"
    Write-Output "final curves exit=$LASTEXITCODE"
}

Write-Output "STAGED TAIL PIPELINE COMPLETE"
