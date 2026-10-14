from . import checkpoint
from .base import Agent, MAX_REWARDS
from .oracle import Oracle
from .random_agent import RandomAgent
from .q_learning import QLearning, ContinuousQLearning
from .r_learning import ContinuousRLearning, RLearning
from .smart_r import SMART, SmoothedSMART
from .relaxed_smart import RelaxedSMART
from .harmonic_r import (CumulativeHarmonic, CumulativeWeightedHarmonic,
                         Harmonic, WeightedHarmonic)
from .experemental_harmonic_r import (ExperimentalCumulativeWeightedHarmonic,
                                      ExperimentalWeightedHarmonic)
from .deep_q_wrapper import DeepQWrapper
from .gaussian_mlp import CategoricalMLP, GaussianMLP
from .ppo import PPO, RolloutBuffer
from .deep_smart import (APO, DeepSMART, CumulativeHarmonicDeepSMART,
                         CumulativeWeightedHarmonicDeepSMART, DiscreteAPO,
                         DiscreteHarmonicDeepSMART, DiscreteRsmartDeepSMART,
                         DiscreteSmartDeepSMART,
                         DiscreteSmoothedSmartDeepSMART,
                         ExperimentalCumulativeWeightedHarmonicDeepSMART,
                         ExperimentalWeightedHarmonicDeepSMART,
                         HarmonicDeepSMART, RsmartDeepSMART, SmartDeepSMART,
                         SmoothedSmartDeepSMART, WeightedHarmonicDeepSMART)
from .mab_epsilon import EpsilonGreedyMAB, ContinuousEpsilonGreedyMAB
from .mab_ucb import UCB, ContinuosUCB

__all__ = [
    'Agent', 'MAX_REWARDS', 'checkpoint',
    'Oracle',
    'RandomAgent',
    'QLearning', 'ContinuousQLearning',
    'RLearning', 'ContinuousRLearning',
    'SMART', 'RelaxedSMART', 'SmoothedSMART',
    'Harmonic', 'WeightedHarmonic',
    'CumulativeHarmonic', 'CumulativeWeightedHarmonic',
    'ExperimentalWeightedHarmonic', 'ExperimentalCumulativeWeightedHarmonic',
    'EpsilonGreedyMAB', 'ContinuousEpsilonGreedyMAB', 'UCB', 'ContinuosUCB',
    'DeepQWrapper',
    'DeepSMART', 'APO', 'RsmartDeepSMART', 'SmartDeepSMART', 'HarmonicDeepSMART',
    'SmoothedSmartDeepSMART', 'WeightedHarmonicDeepSMART', 'CumulativeHarmonicDeepSMART',
    'CumulativeWeightedHarmonicDeepSMART', 'ExperimentalWeightedHarmonicDeepSMART',
    'ExperimentalCumulativeWeightedHarmonicDeepSMART',
    'DiscreteAPO', 'DiscreteRsmartDeepSMART', 'DiscreteSmartDeepSMART',
    'DiscreteHarmonicDeepSMART', 'DiscreteSmoothedSmartDeepSMART',
    'GaussianMLP', 'CategoricalMLP',
    'PPO',
    'RolloutBuffer',
]
