@echo off
setlocal EnableExtensions

cd /d "%~dp0"

if exist ".venv\Scripts\activate.bat" (
    call ".venv\Scripts\activate.bat"
)

call :run_step "01 Data preprocessing" "scripts\01_data_preprocessing.py"
if errorlevel 1 goto :pipeline_error

call :run_step "02 Pseudo label generation" "scripts\02_pseudo_label_generation.py"
if errorlevel 1 goto :pipeline_error

call :run_step "03 Sequence dataset builder" "scripts\03_sequence_dataset_builder.py"
if errorlevel 1 goto :pipeline_error

call :run_step "04 Model training" "scripts\04_model_training.py"
if errorlevel 1 goto :pipeline_error

call :run_step "05 Model evaluation and XAI" "scripts\05_model_evaluation_and_xai.py"
if errorlevel 1 goto :pipeline_error

call :run_step "06 Forensic reporting" "scripts\06_forensic_reporting.py"
if errorlevel 1 goto :pipeline_error

echo.
echo Pipeline completed successfully.
exit /b 0

:pipeline_error
echo.
echo Pipeline stopped because a step failed.
exit /b 1

:run_step
set "STEP_NAME=%~1"
set "SCRIPT_PATH=%~2"

echo.
echo Running: %STEP_NAME%
if not exist "%SCRIPT_PATH%" (
    echo [ERROR] Missing script: %SCRIPT_PATH%
    exit /b 1
)

python "%SCRIPT_PATH%"
if errorlevel 1 (
    echo [ERROR] Step failed: %STEP_NAME%
    exit /b 1
)

exit /b 0