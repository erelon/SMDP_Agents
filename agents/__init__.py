# Agents package initialization
from .base import Agent, MAX_REWARDS
from .oracle import Oracle
from .random_agent import RandomAgent
from .q_learning import QLearning, ContinuousQLearning
from .r_learning import ContinuousRLearning, RLearning
from .smart_r import SMART, RelaxedSMART
from .harmonic_r import Harmonic, WeightedHarmonic
from .bandits import MAB, ContinuesMAB, UCB, ContinuosUCB
from .deep_q_wrapper import DeepQWrapper

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
]
