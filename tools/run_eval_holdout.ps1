# Re-run the held-out (--eval) protocol for the truck scene.
#
# Why this is needed at all: the official metrics.py only walks the `test/`
# split, and train.py only creates one when --eval is passed.  Without it,
# metrics.py fails silently inside a bare try/except.
#
# The first attempt at this was killed mid-run by a harness fault, so this
# version writes a status line to logs/eval_status.log after every step instead
# of relying on buffered stdout, and derives the root from $PSScriptRoot (never
# hardcode a non-ASCII path in a .ps1 -- PowerShell 5.1 decodes BOM-less scripts
# as GBK and mangles it).

$ErrorActionPreference = "Continue"

$w    = Split-Path -Parent $PSScriptRoot        # <root>\tools -> <root>
$repo = "$w\gaussian-splatting"
$py   = "$w\venv\Scripts\python.exe"
$logs = "$w\logs"
$status = "$logs\eval_status.log"

$env:PYTHONPATH       = "$w\cpu_port"
$env:PYTHONIOENCODING = "utf-8"
$env:TORCH_HOME       = "$w\torch_home"
$env:MPLCONFIGDIR     = "$w\mplcache"
$env:DSH_3DGS_KMAX    = "256"
$env:DSH_3DGS_BUDGET  = "16000000"
$env:DSH_3DGS_THREADS = "8"

function Note($msg) {
    $line = "[{0}] {1}" -f (Get-Date -Format "HH:mm:ss"), $msg
    Add-Content -Path $status -Value $line
    Write-Host $line
}

Set-Location $repo
Note "held-out re-run started (pid $PID)"

# Start from a clean slate: a partial run from the killed attempt may be present.
if (Test-Path "$w\output\truck_eval") {
    Note "removing partial output\truck_eval from the previous attempt"
    Remove-Item "$w\output\truck_eval" -Recurse -Force -ErrorAction SilentlyContinue
}

Note "training with --eval (every 8th camera held out), -r 4, 7000 iterations"
$sw = [Diagnostics.Stopwatch]::StartNew()
& $py train.py -s "$w\data\truck" -m "$w\output\truck_eval" `
    --data_device cpu --disable_viewer -r 4 --eval `
    --iterations 7000 --test_iterations 7000 --save_iterations 7000 `
    *> "$logs\eval_train.log"
Note ("  train exit={0}  elapsed={1:N1} min" -f $LASTEXITCODE, $sw.Elapsed.TotalMinutes)

Note "sanity: how many cameras landed in each split?"
# check_split.py calls the same reader train.py/render.py use and cross-checks the
# counts against the LLFF rule; it exits non-zero on a mismatch.  (An earlier
# version inlined this with python -c and passed a Namespace where an
# ArgumentParser was expected, so it died with AttributeError and the split was
# never actually verified.)
& $py "$w\tools\check_split.py" --model_path "$w\output\truck_eval" *> "$logs\eval_split_check.log"
Note "  check_split exit=$LASTEXITCODE"
Get-Content "$logs\eval_split_check.log" -ErrorAction SilentlyContinue |
    Select-String -Pattern 'cameras|total|expected|rule' |
    ForEach-Object { Note "  $($_.Line)" }

Note "render.py (renders both splits)"
& $py render.py -m "$w\output\truck_eval" --iteration 7000 *> "$logs\render_eval.log"
Note "  render exit=$LASTEXITCODE"

Note "official metrics.py  <-- the actual deliverable of this run"
& $py metrics.py -m "$w\output\truck_eval" *> "$logs\metrics_eval.log"
Note "  metrics.py exit=$LASTEXITCODE"
Get-Content "$logs\metrics_eval.log" -ErrorAction SilentlyContinue |
    Select-Object -Last 10 | ForEach-Object { Note "  $_" }

Note "montage + full-metric recompute on the test split"
& $py "$w\tools\make_montage.py" --model "$w\output\truck_eval" --iteration 7000 `
    --split test --scale 4 --views 0 8 16 24 *> "$logs\montage_eval_test.log"
Note "  montage exit=$LASTEXITCODE"
Get-Content "$logs\montage_eval_test.log" -ErrorAction SilentlyContinue |
    Select-Object -Last 6 | ForEach-Object { Note "  $_" }

Note "loss curves for the eval run"
New-Item -ItemType Directory -Force -Path "$w\notes\eval" | Out-Null
& $py "$w\tools\extract_curves.py" --model "$w\output\truck_eval" `
    --log "$logs\eval_train.log" --outdir "$w\notes\eval" *> "$logs\curves_eval.log"
Note "  curves exit=$LASTEXITCODE"

Note "HELD-OUT RUN COMPLETE"
