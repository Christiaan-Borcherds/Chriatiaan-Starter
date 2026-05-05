import itertools
import random


def manual_search(config, model_name):
    return list(config.HP_MANUAL[model_name])


def grid_search(config, model_name):
    space = config.HP_SPACE[model_name]
    keys = list(space.keys())
    values = list(space.values())

    candidates = []
    for combination in itertools.product(*values):
        candidates.append(dict(zip(keys, combination)))

    return candidates


def random_search(config, model_name):
    space = config.HP_SPACE[model_name]
    rng = random.Random(config.HP_SEARCH["random_seed"])

    candidates = []
    for _ in range(config.HP_SEARCH["n_trials"]):
        candidates.append({
            key: rng.choice(values)
            for key, values in space.items()
        })

    return candidates


def generate_hp_candidates(config, model_name):
    strategy = config.HP_SEARCH["strategy"]

    if strategy == "manual":
        return manual_search(config, model_name)

    if strategy == "grid":
        return grid_search(config, model_name)

    if strategy == "random":
        return random_search(config, model_name)

    raise ValueError(f"Unsupported HP search strategy: {strategy}")
