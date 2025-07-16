@echo off
setlocal

rem navigate to SignalExtraction directory
set "workDir=%~1%"

if "%workDir%"=="" (
    echo No working directory passed. Running from current directory %cd%
    echo Usage : %~nx0 "C:\Path\to\extraction\tool" "C:\Path\to\root"
    exit /b 1 
)

cd /d "%workDir%" || (
    echo Failed to change directory to "%workDir%"
    exit /b 1
)
echo Currently running from: %cd%

rem root directory passed as argument:
set "rootDir=%~2"

if "%rootDir%"=="" (
    echo [ERROR] Please provide a valid path to the directory containing the files to be processed
    echo Usage: %~nx0 "C:\Path\to\extraction\tool" "C:\Path\to\root"
    rem n stands for filename, x for extension, 0 to show this file 
    exit /b 1
)

if not exist "%rootDir%" (
    echo [ERROR] Directory "%rootDir%" does not exist or could not be found
    exit /b 1
)

if not exist "%workDir%\SignalExtraction.exe" (
    echo [ERROR] SignalExtraction.exe not found in "%workDir%"
    exit /b 1
)

echo Running SignalExtraction on subdirectories of "%rootDir%"
echo.

rem loop through each method folder
for /D %%i in ("%rootDir%\*") do (
    echo Processing: %%~ni
    SignalExtraction "%%i"
)

echo.
echo Complete.
endlocal