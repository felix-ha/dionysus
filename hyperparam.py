import optuna

def f(x, y):
    return (1 - x) ** 2 + 100 * (y - x ** 2) ** 2


def objective(trial):
    x = trial.suggest_float("x", -2, 2)
    y = trial.suggest_float("y", -2, 2)
    return f(x, y)
    

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=1000)

print(study.best_params)