$ErrorActionPreference = "Stop"

Write-Host "[Jarvis Gate] Running Phase 8 regression gate..." -ForegroundColor Cyan

if (Test-Path ".venv\Scripts\Activate.ps1") {
    . .\.venv\Scripts\Activate.ps1
}

python tests\phase8_regression.py

Write-Host "[Jarvis Gate] Completed." -ForegroundColor Green
