@echo off
rem DeepSeek V4 Flash prompt preflight on the tokenizer alone (no weights, no
rem GPU): reasoning off, post-filler tail tokens, numeric decode mode, BOS.
rem Logged to results\deepseek_preflight_log.txt. Double-click, or run from any shell.
cd /d "%~dp0"
set "PY="
for %%p in ("%USERPROFILE%\.conda\envs\jlens-filler\python.exe" ^
            "%USERPROFILE%\miniconda3\envs\jlens-filler\python.exe" ^
            "%USERPROFILE%\anaconda3\envs\jlens-filler\python.exe" ^
            "%~dp0.venv\Scripts\python.exe") do (
  if not defined PY if exist "%%~p" set "PY=%%~p"
)
if not defined PY set "PY=python"
if not exist results mkdir results
set PYTHONIOENCODING=utf-8
"%PY%" -u 00_smoke_test.py --model deepseek --tokenizer-only > results\deepseek_preflight_log.txt 2>&1
type results\deepseek_preflight_log.txt
echo.
echo (log: results\deepseek_preflight_log.txt)
pause
