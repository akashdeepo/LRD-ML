@echo off
setlocal enabledelayedexpansion
REM Data-refresh rerun (ledger runs 31 and 32, 2026-10-22): every module that
REM does not fit the ML learners, in dependency order. Stops at the first
REM failure. Optional argument: the module to resume from, e.g.
REM     run_refresh.bat module19_matched_robustness
REM The ML learners (module 5, run-33) and the ablation (module 23, run-30)
REM run separately on more cores. Log: results\intermediate\refresh_log.txt
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
set LOG=results\intermediate\refresh_log.txt
set START=%1
if "%START%"=="" (set SKIP=0) else (set SKIP=1)
echo ===== refresh started !date! !time! from "%START%" >> %LOG%
for %%M in (module1_data_description module1b_preliminary_diagnostics module2_lrd_estimation module3_feature_engineering module4_benchmarks module4b_garch module13_candidates module6_forecast_eval module9_robustness module12_incremental_tests module17_eval_checks module18_rolling_gw module19_matched_robustness module20_window_checks module14_window_inclusion module15_timing module16_roughness module22_timing_roughness_checks module10_plots module11_economic module21_portfolio_checks) do (
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
echo ===== refresh finished !date! !time! >> %LOG%
