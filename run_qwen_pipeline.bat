@echo off
rem Stages 0-3 on the dev model (Qwen3.5-4B), end to end, logged to
rem results\pipeline_log.txt. Double-click, or run from any shell.
cd /d "%~dp0"
set "PY="
for %%p in ("%USERPROFILE%\.conda\envs\jlens-filler\python.exe" ^
            "%USERPROFILE%\miniconda3\envs\jlens-filler\python.exe" ^
            "%USERPROFILE%\anaconda3\envs\jlens-filler\python.exe" ^
            "%~dp0.venv\Scripts\python.exe") do (
  if not defined PY if exist "%%~p" set "PY=%%~p"
)
if not defined PY set "PY=python"
echo using %PY%
"%PY%" -u run_qwen_pipeline.py
echo.
echo (log: results\pipeline_log.txt)
pause
