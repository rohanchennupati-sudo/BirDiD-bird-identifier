$ErrorActionPreference = "Stop"

Write-Host "Starting BirDiD..."
Write-Host ""

if (-not (Test-Path ".venv")) {
    Write-Host "Virtual environment not found."
    Write-Host "Create it with: python -m venv .venv"
    exit 1
}

& ".\.venv\Scripts\Activate.ps1"

python -c "import torch; print('PyTorch:', torch.__version__)"
python -c "import fastapi; print('FastAPI:', fastapi.__version__)"

python -m uvicorn app.main:app --reload
