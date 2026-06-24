@echo off
setlocal enabledelayedexpansion

REM ============================================================================
REM  check_last_date.cmd
REM
REM  Checks the last (most recent) date available in:
REM    1. Local pipeline output  (output\refined\reports\)
REM    2. Published target       (OneDrive machine_learning_pipeline\input)
REM
REM  The main pipeline (run_daily_araucaria_pipeline) publishes
REM  araucaria_daily_report_YYYYMMDD.csv via src/main.py --publish-target.
REM
REM  Writes the result to last_date_check.txt in the project root.
REM ============================================================================

set PROJECT_ROOT=%~dp0..
set OUTPUT_DIR=%PROJECT_ROOT%\output\refined\reports
set PUBLISH_DIR=E:\mion\OneDrive - copel.com\transfer_area\machine_learning_pipeline\input
set RESULT_FILE=%PROJECT_ROOT%\last_date_check.txt

set MAX_LOCAL_DATE=00000000
set MAX_PUBLISH_DATE=00000000

echo.
echo ========================================
echo  Checking last available dates
echo ========================================

REM ── Local output ──────────────────────────────────────────────────────────
echo.
echo [Local] Scanning: %OUTPUT_DIR%

if not exist "%OUTPUT_DIR%" (
    echo [Local] ERROR: Directory not found
    set MAX_LOCAL_DATE=N/A (dir not found)
) else (
    for /f "usebackq delims=" %%f in (`dir /b "%OUTPUT_DIR%\araucaria_daily_report_*.csv" 2^>nul`) do (
        set "FILE=%%~nf"
        REM "araucaria_daily_report_" = 23 chars, so date starts at offset 23
        set "DATE=!FILE:~23,8!"
        if !DATE! gtr !MAX_LOCAL_DATE! (
            set MAX_LOCAL_DATE=!DATE!
        )
    )
    if "!MAX_LOCAL_DATE!"=="00000000" (
        echo [Local] No files found
        set MAX_LOCAL_DATE=N/A (no files)
    ) else (
        echo [Local] Latest: !MAX_LOCAL_DATE!
    )
)

REM ── Published target ──────────────────────────────────────────────────────
echo.
echo [Published] Scanning: %PUBLISH_DIR%

if not exist "%PUBLISH_DIR%" (
    echo [Published] ERROR: Directory not found
    set MAX_PUBLISH_DATE=N/A (dir not found)
) else (
    for /f "usebackq delims=" %%f in (`dir /b "%PUBLISH_DIR%\araucaria_daily_report_*.csv" 2^>nul`) do (
        set "FILE=%%~nf"
        REM "araucaria_daily_report_" = 23 chars, so date starts at offset 23
        set "DATE=!FILE:~23,8!"
        if !DATE! gtr !MAX_PUBLISH_DATE! (
            set MAX_PUBLISH_DATE=!DATE!
        )
    )
    if "!MAX_PUBLISH_DATE!"=="00000000" (
        echo [Published] No files found
        set MAX_PUBLISH_DATE=N/A (no files)
    ) else (
        echo [Published] Latest: !MAX_PUBLISH_DATE!
    )
)

REM ── Write result ─────────────────────────────────────────────────────────
echo.
echo ========================================
echo  Writing result to %RESULT_FILE%
echo ========================================

(
    echo Last Date Check - %date% %time%
    echo ----------------------------------------
    echo Local output   : !MAX_LOCAL_DATE!
    echo Published      : !MAX_PUBLISH_DATE!
    echo ----------------------------------------
    echo.
    echo Local path     : %OUTPUT_DIR%
    echo Publish path   : %PUBLISH_DIR%
) > "%RESULT_FILE%"

echo.
echo Done! Result saved.
echo.
echo   Local output   : !MAX_LOCAL_DATE!
echo   Published      : !MAX_PUBLISH_DATE!
echo.
echo   See %RESULT_FILE% for details.
echo.

endlocal
