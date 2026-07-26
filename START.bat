@echo off
setlocal
title Equity Analysis Report Generator

REM pushd (not cd) so this also works if the folder is opened over a UNC path
pushd "%~dp0"

echo ==============================================================
echo    Equity Analysis Report Generator
echo ==============================================================
echo.

REM ---------- 1. locate Python ----------------------------------
set "PY="
where python >nul 2>&1 && set "PY=python" && goto :havepy
where py     >nul 2>&1 && set "PY=py"     && goto :havepy
goto :nopython
:havepy

REM The Microsoft Store stub answers `where` but fails to run, so verify.
%PY% --version >nul 2>&1
if errorlevel 1 goto :nopython

echo [1/4] Python:
for /f "delims=" %%v in ('%PY% --version 2^>^&1') do echo       %%v

REM ---------- 2. dependencies -----------------------------------
echo [2/4] Checking dependencies...
%PY% -c "import flask, requests, bs4, lxml, anthropic, openai" >nul 2>&1
if errorlevel 1 (
    echo       Missing packages - installing now ^(first run only, may take a minute^)...
    %PY% -m pip install --quiet --disable-pip-version-check -r requirements.txt
    if errorlevel 1 goto :pipfailed
    %PY% -c "import flask, requests, bs4, lxml, anthropic, openai" >nul 2>&1
    if errorlevel 1 goto :pipfailed
    echo       Installed.
) else (
    echo       All packages present.
)

REM ---------- 3. API key ----------------------------------------
REM The provider is chosen in .env (LLM_PROVIDER); any provider key satisfies this.
echo [3/4] Checking API key...
if defined ANTHROPIC_API_KEY   ( echo       Found ANTHROPIC_API_KEY in the environment.   & goto :keyok )
if defined OPENROUTER_API_KEY  ( echo       Found OPENROUTER_API_KEY in the environment.  & goto :keyok )
if defined DEEPSEEK_API_KEY    ( echo       Found DEEPSEEK_API_KEY in the environment.    & goto :keyok )
if defined LLM_API_KEY         ( echo       Found LLM_API_KEY in the environment.         & goto :keyok )
if exist ".env" (
    findstr /r /b /c:"ANTHROPIC_API_KEY=" /c:"OPENROUTER_API_KEY=" /c:"DEEPSEEK_API_KEY=" /c:"LLM_API_KEY=" ".env" >nul 2>&1
    if errorlevel 1 (
        echo       [!] .env exists but has no provider API key line.
    ) else (
        echo       Found .env file.
    )
    goto :keyok
)
echo.
echo       [!] No API key found.
echo           Create a file named .env in this folder. For the default
echo           (Anthropic) provider it should contain:
echo.
echo               ANTHROPIC_API_KEY=sk-ant-your-key-here
echo.
echo           To use OpenRouter or DeepSeek instead, set LLM_PROVIDER and
echo           the matching key — see .env.example and README.md.
echo.
echo           Screener scraping will still work, but the analysis
echo           step will fail until a key is set.
echo.
:keyok

REM ---------- 4. launch -----------------------------------------
echo [4/4] Starting server...
echo.
echo    Opening http://127.0.0.1:5057 in your browser.
echo    Reports are saved to the outputs\ folder.
echo.
echo    Keep this window open while you use the app.
echo    Press Ctrl+C or close this window to stop the server.
echo.
echo --------------------------------------------------------------

REM Open the browser a few seconds later, once the server is listening.
start "" /min powershell -NoProfile -WindowStyle Hidden -Command "Start-Sleep -Seconds 4; Start-Process 'http://127.0.0.1:5057/'"

%PY% "app\server.py"

echo.
echo Server stopped.
popd
pause
exit /b 0

REM ---------- error paths ---------------------------------------
:nopython
echo.
echo   [X] Python was not found.
echo.
echo       Install Python 3.10 or newer from:
echo           https://www.python.org/downloads/
echo       During setup, tick "Add python.exe to PATH".
echo.
popd
pause
exit /b 1

:pipfailed
echo.
echo   [X] Could not install the required packages.
echo.
echo       Try running this manually to see the full error:
echo           %PY% -m pip install -r requirements.txt
echo.
popd
pause
exit /b 1
