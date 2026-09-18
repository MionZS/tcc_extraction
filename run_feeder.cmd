@echo off
setlocal
cd /d "%~dp0"
set PYTHONPATH=%~dp0;%PYTHONPATH%

echo ====================================================================
echo  Executando Pipeline Completo para o Alimentador Fonte Nova
echo ====================================================================
echo.

python scripts\run_feeder_pipeline.py %*

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ERRO] O pipeline falhou com codigo %ERRORLEVEL%.
) else (
    echo.
    echo [SUCESSO] Pipeline e treinamento do modelo concluidos!
)

echo.
pause
