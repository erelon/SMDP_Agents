# Agents package initialization
from importlib import import_module

from .base import Agent, MAX_REWARDS
from .oracle import Oracle
from .random_agent import RandomAgent
from .q_learning import QLearning, ContinuousQLearning
from .r_learning import ContinuousRLearning, RLearning
from .smart_r import SMART
from .relaxed_smart import RelaxedSMART
from .harmonic_r import Harmonic, WeightedHarmonic
from .deep_q_wrapper import DeepQWrapper
from .ppo import PPO, RsmartPPO, SmartPPO, HarmonicPPO, RolloutBuffer

_mab_epsilon = import_module(".mab-epsilon", __name__)
_mab_ucb = import_module(".mab-ucb", __name__)

MAB = _mab_epsilon.MAB
ContinuesMAB = _mab_epsilon.ContinuesMAB
UCB = _mab_ucb.UCB
ContinuosUCB = _mab_ucb.ContinuosUCB

__all__ = [
    'Agent', 'MAX_REWARDS',
    'Oracle',
    'RandomAgent',
    'QLearning', 'ContinuousQLearning',
    'RLearning', 'ContinuousRLearning',
    'SMART', 'RelaxedSMART',
    'Harmonic', 'WeightedHarmonic',
    'MAB', 'ContinuesMAB', 'UCB', 'ContinuosUCB',
    'DeepQWrapper',
    'PPO', 'RsmartPPO', 'SmartPPO', 'HarmonicPPO', 'RolloutBuffer',
]
