@echo off
REM Overnight ML feature ablations (ledger run-30). Safe to re-run: finished
REM combinations are skipped. Log: results\intermediate\ml_ablation_log.txt
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
python -u -m modules.module23_ml_ablation >> results\intermediate\ml_ablation_log.txt 2>&1
echo FINISHED %date% %time% >> results\intermediate\ml_ablation_log.txt
