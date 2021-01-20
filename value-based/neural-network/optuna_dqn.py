import uuid
from argparse import ArgumentParser

import optuna
from dqn import get_args, train_dqn


def dqn_parser_hook(name_prefix: str, parser: ArgumentParser, trail: optuna.Trial):
    parser.set_defaults(
        name=name_prefix + "_" + str(uuid.uuid1()),
        lr=trail.suggest_categorical("lr", [1e-6, 1e-5]),
        activation=trail.suggest_categorical("activation", ["relu", "selu", "ident"]),
        layer_num=trail.suggest_int("layer_num", 1, 5, step=1),
        hidden_size=trail.suggest_categorical("hidden_size", [32, 64, 128, 256, 512]),
        dueling=trail.suggest_categorical("dueling", [False, True]),
    )


def dqn_objective(trail: optuna.Trial) -> float:
    args = get_args(lambda parser: dqn_parser_hook("dqn", parser, trail))
    reward = train_dqn(args)
    print(trail.params, reward)
    return reward


def optimize_dqn(trials: int) -> optuna.Study:
    study = optuna.create_study(
        storage="sqlite:///dqn.db",
        study_name="dqn",
        direction="maximize",
        load_if_exists=True,
    )
    study.optimize(dqn_objective, n_trials=trials)
    return study


def dump_best_study(study):
    trial = study.best_trial
    print("  Best Value: ", trial.value)
    print("  Best Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))


def main():
    trials = 200

    dqn_study = optimize_dqn(trials)
    dump_best_study(dqn_study)


if __name__ == "__main__":
    main()
