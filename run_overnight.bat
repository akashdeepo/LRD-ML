@echo off
setlocal enabledelayedexpansion
REM Overnight run after the 2026-10-22 data refresh (ledger runs 33 and 30):
REM   module 5   ML learners on the refreshed panel      (~6 h on 12 cores)
REM   module 23  within-learner feature ablation         (~6 h; resumable)
REM   modules 6, 12, 17, 10  evaluation tables and figures that use ML rows
REM   paper tables and the refresh diff
REM Stops at the first failure. Optional argument: the step to resume from,
REM e.g.  run_overnight.bat module23_ml_ablation
REM Module 23 skips finished (learner, feature set, horizon) combinations on
REM its own; module 5 does not, so a module-5 crash means rerunning it.
REM Log: results\intermediate\overnight_log.txt
REM Before starting: plug in, and set Windows sleep to Never while plugged in
REM (Settings > System > Power), or the job stops when the machine sleeps.
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
set LOG=results\intermediate\overnight_log.txt
set START=%1
if "%START%"=="" (set SKIP=0) else (set SKIP=1)
echo ===== overnight started !date! !time! from "%START%" >> %LOG%
for %%M in (module5_ml_models module23_ml_ablation module6_forecast_eval module12_incremental_tests module17_eval_checks module10_plots) do (
    if "%%M"=="%START%" set SKIP=0
    if !SKIP!==0 (
        echo ----- %%M start !time! >> %LOG%
        python -u -m modules.%%M >> %LOG% 2>&1
        if errorlevel 1 (
            echo ##### %%M FAILED !time! >> %LOG%
            exit /b 1
        )
        echo ----- %%M done !time! >> %LOG%
    )
)
echo ----- paper tables start !time! >> %LOG%
python paper_jrfm_v3\make_tables.py >> %LOG% 2>&1
echo ----- refresh diff start !time! >> %LOG%
python -m modules._diff_refresh >> %LOG% 2>&1
echo ===== overnight finished !date! !time! >> %LOG%
