Set-Location -LiteralPath $PSScriptRoot
python -m uvicorn workbench.server:app --host 127.0.0.1 --port 8765
