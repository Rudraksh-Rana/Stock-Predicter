@echo off
cd /d "d:\Stock market"

REM Check if git is initialized
if not exist .git (
    echo Initializing git repository...
    git init
    echo Git initialized
) else (
    echo Git repository already exists
)

REM Add all files
echo Adding files to git...
git add .

REM Check status
echo.
echo Current git status:
git status

REM Configure remote if not already done
echo.
echo Checking remote configuration...
git remote -v
if %ERRORLEVEL% EQU 0 (
    git remote -v | findstr origin >nul
    if %ERRORLEVEL% NEQ 0 (
        echo Adding remote origin...
        git remote add origin https://github.com/Rudraksh-Rana/Stock-Predicter.git
    ) else (
        echo Remote origin already configured
    )
) else (
    echo Adding remote origin...
    git remote add origin https://github.com/Rudraksh-Rana/Stock-Predicter.git
)

REM Commit changes
echo.
echo Committing changes...
git commit -m "Initial commit: Stock price predictor project" --author="Copilot <223556219+Copilot@users.noreply.github.com>"

REM Push to GitHub
echo.
echo Pushing to GitHub...
git push -u origin main

if %ERRORLEVEL% NEQ 0 (
    echo Push failed, trying master branch...
    git push -u origin master
)

echo.
echo Done!
