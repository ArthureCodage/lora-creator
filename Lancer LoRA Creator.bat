@echo off
title LoRA Creator
rem Python de kohya_ss : par defaut %USERPROFILE%\kohya_ss, sinon definir KOHYA_DIR.
if not defined KOHYA_DIR set "KOHYA_DIR=%USERPROFILE%\kohya_ss"
if not exist "%KOHYA_DIR%\venv\Scripts\python.exe" (
  echo kohya_ss introuvable dans "%KOHYA_DIR%".
  echo Installe kohya_ss ou definis la variable KOHYA_DIR.
  pause
  exit /b 1
)
cd /d "%~dp0app"
"%KOHYA_DIR%\venv\Scripts\python.exe" server.py
if errorlevel 1 pause
