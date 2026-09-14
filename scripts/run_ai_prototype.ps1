$ErrorActionPreference = "Stop"

Set-Location (Join-Path $PSScriptRoot "..")

if (Test-Path ".\.venv\Scripts\Activate.ps1") {
    & ".\.venv\Scripts\Activate.ps1"
}

Write-Host "Installation des dépendances IA..." -ForegroundColor Cyan
python -m pip install -r requirements_ai.txt

Write-Host "Construction des embeddings..." -ForegroundColor Cyan
python -m src.ai.build_context_embeddings

Write-Host "Evaluation IA..." -ForegroundColor Cyan
python -m src.evaluation.evaluate_ai

Write-Host ""
Write-Host "Prototype prêt." -ForegroundColor Green
Write-Host "Lance : streamlit run app_ai.py"
