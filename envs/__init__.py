from .monty import MontyEnv
from .bandit import BanditEnv
from .stag_hunt import StagHuntEnv
from .ipd import IPDEnv
from .trust_game import TrustGameEnv
from .machine import MachineEnv
from .chicken import ChickenEnv
from .bos import BoSEnv
from .pennies import PenniesEnv

ENVS = {
    "monty": MontyEnv,
    "bandit": BanditEnv,
    "stag_hunt": StagHuntEnv,
    "ipd": IPDEnv,
    "trust_game": TrustGameEnv,
    "machine": MachineEnv,      # 2026-07-19: de-socialized twin of trust_game
    "chicken": ChickenEnv,      # 2026-07-19: anti-coordination strategic env
    "bos": BoSEnv,              # 2026-07-24: Bach-or-Stravinsky, stag_hunt twin at tau=1/3
    "pennies": PenniesEnv,      # 2026-07-25: Matching Pennies, strictly-opposed payoffs, tau=1/2
}


def make_env(name, rng, T, seed=None, **kwargs):
    """seed: the session's integer seed, used ONLY by envs that stratify their
    hidden-parameter draw across seeds (class attr stratify_types). All call
    sites (run_all, run_steer, smoke_test) must pass it, or the stratifying
    env falls back to an iid draw and replays will mismatch the sweep."""
    cls = ENVS[name]
    if seed is not None and getattr(cls, "stratify_types", False):
        kwargs["seed"] = seed
    return cls(rng, T, **kwargs)
