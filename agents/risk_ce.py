import math
from .value_checks import require_positive, require_finite

# CRRA: Constant Relative Risk Aversion (CRRA)
"""
           x^(1-\theta) - 1
 crra(x)= -------------------
             1 - \theta

has *certainty equivalent*
 
    crra_inverse(E[crra(x)]) = (E[x^(1-\theta)])^(1/(1-\theta))
"""

def crra(rate: float, theta: float) -> float:
    """Return CRRA utility of a finite, strictly positive rate."""

    rate = require_positive("rate", rate)
    theta = require_finite("theta", theta)

    # handle simple cases
    # -------------------
    # risk neutral
    if theta == 0:
        return (rate - 1.0)

    # strong risk aversion (harmonic)
    if theta == 2:
        return (1.0 - (1.0/rate))

    # mild risk aversion (logarithmic), a special case of DARA (Decreasing Absolute Risk Aversion)
    if theta == 1:
        return math.log(rate)

    p = 1.0 - theta
    try:
        value = math.expm1(p * math.log(rate)) / p
    except OverflowError as error:
        raise ValueError("CRRA utility is not finite; overflow") from error
    if not math.isfinite(value):
        raise ValueError("CRRA utility is not finite")
    return value


def crra_invert(utility: float, theta: float) -> float:
    """Map a valid CRRA utility value to its certainty-equivalent rate."""

    utility = require_finite("utility", utility)
    theta = require_finite("theta", theta)

    # handle simple cases
    # -------------------
    # risk neutral    
    if theta == 0:
        return require_positive("certainty-equivalent rate", utility + 1.0)
    
    # strong risk aversion (harmonic)
    if theta == 2:
        if utility >= 1.0:
            raise ValueError("utility is outside the inverse CRRA domain")
        return require_positive("certainty-equivalent rate", 1.0/(1.0-utility))
        
    # mild risk aversion (logarithmic), a special case of DARA (Decreasing Absolute Risk Aversion)
    if theta == 1:
        try:
            rate = math.exp(utility)
        except OverflowError as error:
            raise ValueError("certainty-equivalent rate is not finite") from error
    else:
        p = 1.0 - theta
        base = 1.0 + p * utility
        if base <= 0:
            raise ValueError("utility is outside the inverse CRRA domain")
        try:
            rate = math.exp(math.log1p(p * utility) / p)
        except (OverflowError, ValueError) as error:
            raise ValueError("certainty-equivalent rate is not finite") from error
    return require_positive("certainty-equivalent rate", rate)
