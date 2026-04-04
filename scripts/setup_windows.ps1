param(
    [switch]$InstallSpeechExtras
)

$ErrorActionPreference = "Stop"

Write-Host "[Jarvis Setup] Starting Windows setup..." -ForegroundColor Cyan

if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    throw "Python is not available in PATH. Install Python 3.12+ and retry."
}

Write-Host "[Jarvis Setup] Using global Python from PATH (no virtual environment)." -ForegroundColor Yellow

Write-Host "[Jarvis Setup] Upgrading pip..." -ForegroundColor Yellow
python -m pip install --upgrade pip

Write-Host "[Jarvis Setup] Installing requirements..." -ForegroundColor Yellow
python -m pip install -r requirements.txt

if ($InstallSpeechExtras) {
    Write-Host "[Jarvis Setup] Installing optional speech extras..." -ForegroundColor Yellow
    python -m pip install transformers huggingface-hub sentencepiece
}

if (-not (Test-Path ".env") -and (Test-Path ".env.example")) {
    Write-Host "[Jarvis Setup] Creating .env from .env.example..." -ForegroundColor Yellow
    Copy-Item ".env.example" ".env"
}

Write-Host "[Jarvis Setup] Running doctor checks..." -ForegroundColor Yellow
python core\doctor.py

Write-Host "[Jarvis Setup] Completed." -ForegroundColor Green
Write-Host "Next: start Ollama (ollama serve), then run python core\\orchestrator.py" -ForegroundColor Green
