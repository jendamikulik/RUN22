Jak to spustit

planted + heur pro n=60 a 120, m≈4.33n, 20 pokusů, max 312 kroků:

python run_runner.py --modes planted heur --n 60 120 --ratio 4.33 --trials 20 --max_steps 312


jen heuristika na větším n:

python run_runner.py --modes heur --n 240 360 --ratio 4.33 --trials 30 --max_steps 900

Co dostaneš

runner_results.csv – řádek = jedna jízda (včetně času, kroků, FinalUnsat, EndPhi/EndR/kappa).

runner_summary.csv – agregace po (mode, n, m): success-rate, mediány kroků/času.

runner_success.png – jednoduchý graf úspěšnosti proti m/n pro planted vs. heuristic.

Když budeš chtít ještě variantu, která kromě CSV ukládá i snapshot vstupních instancí a finálních přiřazení (pro pozdější „replay“), přidám ti to jako volitelné --dump_dir.