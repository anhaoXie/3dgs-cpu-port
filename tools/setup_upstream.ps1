# Clone the upstream 3D Gaussian Splatting repository at the exact commit this
# CPU port was developed and verified against.
#
# The upstream code is deliberately NOT vendored into this repository:
#   * it is licensed for non-commercial research use by INRIA, so it is cleaner
#     to reference it than to redistribute it;
#   * it carries a 170 MB .git and a 79 MB assets/ tree;
#   * pinning the commit is a stronger guarantee than copying files -- the whole
#     claim of this project is that the official train.py / render.py /
#     metrics.py were used BYTE-FOR-BYTE UNMODIFIED.
#
# Usage:  powershell -NoProfile -ExecutionPolicy Bypass -File tools\setup_upstream.ps1

$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot          # <root>\tools -> <root>
$dest = Join-Path $root "gaussian-splatting"

$url    = "https://github.com/graphdeco-inria/gaussian-splatting.git"
$commit = "54c035f"        # HEAD at the time this port was written

# Expected SHA-256 of the three files that must stay unmodified.  If these match
# after checkout, the "official code untouched" claim is verified, not asserted.
$expected = @{
    "train.py"   = "078CB36BDE69090F"
    "render.py"  = "E756466AB0A19AFC"
    "metrics.py" = "A1334B4F421DC8E5"
}

if (Test-Path $dest) {
    Write-Host "Already present: $dest"
} else {
    Write-Host "Cloning $url ..."
    git clone $url $dest
    if ($LASTEXITCODE -ne 0) { throw "git clone failed" }
}

Write-Host "Checking out $commit ..."
git -C $dest checkout $commit
if ($LASTEXITCODE -ne 0) { throw "git checkout $commit failed" }

Write-Host ""
Write-Host "Verifying the three files that this project requires to be unmodified:"
$ok = $true
foreach ($f in $expected.Keys) {
    $p = Join-Path $dest $f
    if (-not (Test-Path $p)) { Write-Host ("  MISSING  {0}" -f $f); $ok = $false; continue }
    $hash = (Get-FileHash $p -Algorithm SHA256).Hash
    $got = $hash.Substring(0, 16)
    if ($got -eq $expected[$f]) {
        Write-Host ("  OK       {0,-12} {1}" -f $f, $got)
    } else {
        Write-Host ("  MISMATCH {0,-12} got {1}, expected {2}" -f $f, $got, $expected[$f])
        $ok = $false
    }
}

Write-Host ""
Write-Host "Working tree status (should be empty apart from untracked build output):"
git -C $dest status --porcelain

Write-Host ""
if ($ok) {
    Write-Host "Upstream ready and verified at $commit."
} else {
    Write-Host "WARNING: hash mismatch -- upstream may have changed since this port was written."
    exit 1
}
