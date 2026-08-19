from .r_learning import ContinuousRLearning
from .average_rates import CumulativeWeightedHarmonicRate, WeightedHarmonicRate


class WeightedHarmonic(ContinuousRLearning):
    def __init__(self, name: str, action_space=None, learning_rate=0.1, exploration_rate=0.1, with_rho_trick=True,
                 rho_learning_rate=0.3, **kwargs):
        super().__init__(name, action_space, learning_rate, exploration_rate, with_rho_trick, rho_learning_rate,
                         **kwargs)
        self.hma = WeightedHarmonicRate(rho_learning_rate)
        self.reset()

    def reset(self):
        super().reset()
        self.hma.reset()

    def calc_new_rho(self, reward, time, td_target, td_error):
        self.rho = self.hma.update(reward, time, reward)  # Weighted HMA with weight = reward


class Harmonic(WeightedHarmonic):
    def calc_new_rho(self, reward, time, td_target, td_error):
        self.rho = self.hma.update(reward, time)  # Unweighted HMA: the default weight of 1.0


class CumulativeWeightedHarmonic(WeightedHarmonic):
    """``WeightedHarmonic`` over the whole history instead of an EMA.

    Stands to :class:`WeightedHarmonic` as SMART stands to RelaxedSMART: the same
    rho, averaged without forgetting.  ``rho_learning_rate`` is still accepted --
    every agent in this family takes it -- but nothing reads it, since a
    cumulative average has no gain to set.

    Note the weight makes this one's *rho* degenerate on a domain whose rewards
    are all strictly positive: the ``r`` in ``(tau / r) * r`` cancels and it
    becomes ``sum(r) / sum(tau)``, which is exactly SMART's.  The unweighted
    :class:`CumulativeHarmonic` stays a harmonic mean there.
    """

    def __init__(self, name: str, action_space=None, learning_rate=0.1, exploration_rate=0.1,
                 with_rho_trick=True, rho_learning_rate=0.3, **kwargs):
        super().__init__(name, action_space, learning_rate, exploration_rate, with_rho_trick,
                         rho_learning_rate, **kwargs)
        self.hma = CumulativeWeightedHarmonicRate()
        self.reset()


class CumulativeHarmonic(CumulativeWeightedHarmonic):
    """``Harmonic`` over the whole history: the harmonic mean of the rates.

    With unit weight and no forgetting, and while no reward is negative or zero,
    rho is ``n / sum(tau_i / r_i)`` -- the third averaging of the same samples
    beside SMART's ``sum(r) / sum(tau)`` and the mean of ``r_i / tau_i``.
    """

    def calc_new_rho(self, reward, time, td_target, td_error):
        self.rho = self.hma.update(reward, time)  # Unweighted: the default weight of 1.0
