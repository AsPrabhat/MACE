# Bootstrap script for the MACE pedagogical repository

Write-Host "Setting up the MACE environment using uv..." -ForegroundColor Cyan

# Check if uv is installed
if (-not (Get-Command "uv" -ErrorAction SilentlyContinue)) {
    Write-Host "uv is not installed. Please install uv first: https://docs.astral.sh/uv/" -ForegroundColor Red
    exit 1
}

# Create virtual environment and install dependencies
Write-Host "Creating virtual environment and installing dependencies from pyproject.toml..."
uv venv
uv pip install -e .

# Register the IPython kernel for Jupyter notebooks
Write-Host "Registering Jupyter kernel 'mace-env'..."
uv run python -m ipykernel install --user --name=mace-env --display-name="Python (MACE)"

Write-Host "Environment setup complete!" -ForegroundColor Green
Write-Host "Activate the environment using: .\.venv\Scripts\Activate.ps1" -ForegroundColor Yellow
