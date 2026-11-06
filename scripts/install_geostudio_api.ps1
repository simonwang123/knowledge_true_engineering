param(
    [string]$GeoStudioDirectory = "C:\Program Files\Seequent\GeoStudio 2025.2",
    [string]$PythonEnvironment = ".venv"
)

$ErrorActionPreference = "Stop"
$apiDirectory = Join-Path $GeoStudioDirectory "API"
$requirements = Join-Path $apiDirectory "requirements.txt"
if (-not (Test-Path -LiteralPath $requirements -PathType Leaf)) {
    throw "GeoStudio API requirements file was not found: $requirements"
}

if (-not (Test-Path -LiteralPath $PythonEnvironment -PathType Container)) {
    uv venv --python 3.12 $PythonEnvironment
}

$python = Join-Path $PythonEnvironment "Scripts\python.exe"
if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
    throw "Python environment is invalid: $PythonEnvironment"
}

$version = & $python -c "import sys; print('.'.join(map(str, sys.version_info[:3])))"
if ([version]$version -lt [version]"3.12.1" -or [version]$version -ge [version]"3.13") {
    throw "Python $version is unsupported; GeoStudio requires >=3.12.1,<3.13"
}

uv pip install --python $python --requirement $requirements
Write-Host "Installed the GeoStudio API listed by $requirements into $PythonEnvironment"
