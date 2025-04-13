@echo off
echo Running Hyperparameter Tuning Experiments...

REM Activate conda environment if needed
REM call conda activate thesis

REM Define the different values for the parameters
set neg_balanced=0 1
set we_values=64 128 256
set win_values=2
set min_cnt_values=1
set sg_values=0 1
set depth_values=32 128
set epochs_values=1000
set batch_size_values=32 128
set lr_values=0.005 0.001 0.0005
set lr_scheduler_values=reduce_on_plateau
set cnn_values=0 1
set kernel_values=2 4 6

REM Create results directory
if not exist results mkdir results
if not exist results\models mkdir results\models

echo.
echo ================================================================================
echo Starting hyperparameter tuning experiment
echo ================================================================================
echo.

REM Loop through parameters
REM Note: CMD loops are more limited than bash, so this is a simplified version
REM You may want to selectively test specific scheduler types rather than all combinations
REM Example below runs with default scheduler (constant) plus one additional scheduler for testing

REM This is a simplified example that only varies learning rate and scheduler
REM for %%l in (%lr_values%) do (
REM     for %%s in (%lr_scheduler_values%) do (
REM         echo ================================================================================
REM         echo Running with parameters:
REM         echo we=512, win=2, min_cnt=1, sg=1, depth=64, epochs=1000, batch_size=32, lr=%%l, lr_scheduler=%%s, cnn=0
REM         echo ================================================================================
REM         python main.py --we 512 --win 2 --min_cnt 1 --sg 1 --depth 64 --epochs 1000 --batch_size 32 --lr %%l --lr_scheduler %%s --cnn 0
REM     )
REM )

REM Uncomment below for a more comprehensive parameter search 
REM Note: This will take a very long time to run with all combinations!
for %%n in (%neg_balanced%) do (
    for %%w in (%we_values%) do (
        for %%i in (%win_values%) do (
            for %%m in (%min_cnt_values%) do (
                for %%s in (%sg_values%) do (
                    for %%d in (%depth_values%) do (
                        for %%e in (%epochs_values%) do (
                            for %%b in (%batch_size_values%) do (
                                for %%l in (%lr_values%) do (
                                    for %%r in (reduce_on_plateau) do (
                                        for %%c in (%cnn_values%) do (
                                            if %%c==1 (
                                                for %%k in (%kernel_values%) do (
                                                    echo ================================================================================
                                                    echo Running with parameters:
                                                    echo neg_balanced=%%n we=%%w, win=%%i, min_cnt=%%m, sg=%%s, depth=%%d, epochs=%%e, batch_size=%%b, lr=%%l, lr_scheduler=%%r, cnn=%%c, kernel=%%k
                                                    echo ================================================================================
                                                    python main.py --neg_balanced %%n --we %%w --win %%i --min_cnt %%m --sg %%s --depth %%d --epochs %%e --batch_size %%b --lr %%l --lr_scheduler %%r --cnn %%c --kernel %%k
                                                )
                                            ) else (
                                                echo ================================================================================
                                                echo Running with parameters:
                                                echo neg_balanced=%%n we=%%w, win=%%i, min_cnt=%%m, sg=%%s, depth=%%d, epochs=%%e, batch_size=%%b, lr=%%l, lr_scheduler=%%r, cnn=%%c
                                                echo ================================================================================
                                                python main.py --neg_balanced %%n --we %%w --win %%i --min_cnt %%m --sg %%s --depth %%d --epochs %%e --batch_size %%b --lr %%l --lr_scheduler %%r --cnn %%c
                                            )
                                        )
                                    )
                                )
                            )
                        )
                    )
                )
            )
        )
    )
)

REM Aggregate results
echo.
echo ================================================================================
echo Aggregating results
echo ================================================================================
echo.

python -c "from drug_repo.utils.results import aggregate_results; from drug_repo.visualization.plots import plot_hyperparameter_results; df = aggregate_results(); plot_hyperparameter_results('results/results_table.csv')"

echo.
echo ================================================================================
echo Hyperparameter tuning complete
echo ================================================================================
echo.

pause