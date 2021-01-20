import uuid
from argparse import ArgumentParser

import optuna
from double_dqn import get_args as double_dqn_get_args
from double_dqn import train_double_dqn
from dqn import get_args as dqn_get_args
from dqn import train_dqn


def dqn_parser_hook(parser: ArgumentParser, trail: optuna.Trial):
    parser.set_defaults(
        name=str(uuid.uuid1()),
        lr=trail.suggest_categorical("lr", [1e-7, 1e-6, 1e-5, 1e-4, 1e-3]),
        activation=trail.suggest_categorical(
            "activation", ["elu", "relu", "selu", "tanh", "ident"]
        ),
        layer_num=trail.suggest_int("layer_num", 1, 10, step=1),
        hidden_size=trail.suggest_int("hidden_size", 16, 256, step=16),
    )


def double_dqn_parser_hook(parser: ArgumentParser, trail: optuna.Trial):
    parser.set_defaults(
        name=str(uuid.uuid1()),
        lr=trail.suggest_categorical("lr", [1e-7, 1e-6, 1e-5, 1e-4, 1e-3]),
        activation=trail.suggest_categorical(
            "activation", ["elu", "relu", "selu", "tanh", "ident"]
        ),
        layer_num=trail.suggest_int("layer_num", 1, 10, step=1),
        hidden_size=trail.suggest_int("hidden_size", 16, 256, step=16),
        target_update_freq=trail.suggest_categorical(
            "target_update_freq", [1, 16, 32, 64, 128]
        ),
        tau=trail.suggest_float("tau", 0.0, 1.0),
    )


def dqn_objective(trail: optuna.Trial) -> float:
    args = dqn_get_args(lambda parser: dqn_parser_hook(parser, trail))
    return train_dqn(args)


def double_dqn_objective(trail: optuna.Trial) -> float:
    args = double_dqn_get_args(lambda parser: dqn_parser_hook(parser, trail))
    return train_double_dqn(args)


def optimize_dqn(trials: int):
    study = optuna.create_study(
        storage="sqlite:///dqn.db",
        study_name="dqn",
        direction="maximize",
        load_if_exists=True,
    )
    study.optimize(dqn_objective, n_trials=trials)

    print(study.best_params)
    print(study.best_value)


def optimize_double_dqn(trials: int):
    study = optuna.create_study(
        storage="sqlite:///dqn.db",
        study_name="double_dqn",
        direction="maximize",
        load_if_exists=True,
    )
    study.optimize(double_dqn_objective, n_trials=trials)

    trial = study.best_trial
    print("  Value: ", trial.value)
    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))


def main():
    trials = 1000
    optimize_dqn(trials)
    optimize_double_dqn(trials)


if __name__ == "__main__":
    main()
