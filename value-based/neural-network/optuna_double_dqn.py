from argparse import ArgumentParser

import optuna
from double_dqn import get_args, train_double_dqn
from optuna_dqn import dqn_parser_hook, dump_best_study


def double_dqn_parser_hook(parser: ArgumentParser, trail: optuna.Trial):
    dqn_parser_hook("double_dqn", parser, trail)
    parser.set_defaults(
        target_update_freq=trail.suggest_categorical(
            "target_update_freq", [1, 16, 32, 64, 128]
        ),
        tau=trail.suggest_float("tau", 0.1, 1.0),
    )


def double_dqn_objective(trail: optuna.Trial) -> float:
    args = get_args(lambda parser: double_dqn_parser_hook(parser, trail))
    return train_double_dqn(args)


def optimize_double_dqn(trials: int) -> optuna.Study:
    study = optuna.create_study(
        storage="sqlite:///double_dqn.db",
        study_name="double_dqn",
        direction="maximize",
        load_if_exists=True,
    )
    study.optimize(double_dqn_objective, n_trials=trials)
    return study


def main():
    trials = 500

    double_dqn = optimize_double_dqn(trials)
    dump_best_study(double_dqn)


if __name__ == "__main__":
    main()
