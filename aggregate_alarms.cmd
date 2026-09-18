@echo off
rem Wrapper to run the alarm aggregation script with project PYTHONPATH
setlocal
rem Ensure we are in the project root (the script location)
pushd %~dp0..\
rem Add project root to PYTHONPATH
set PYTHONPATH=%cd%;%PYTHONPATH%
echo Running alarm aggregation script...
python scripts\aggregate_alarms.py %*
popd
endlocal

