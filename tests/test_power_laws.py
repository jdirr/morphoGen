"""Power-law tails by maximum likelihood (Clauset, Shalizi & Newman 2009): the exponents of
exact Pareto and zeta laws within their standard errors, the error of the discrete
estimate against the variance of ln x under the law, the lower bound of a tail above a
body found by the Kolmogorov-Smirnov distance, and the errors."""

import math

import numpy as np
import pytest

from morphogen.analysis import PowerLawTail, power_law_tail


def test_continuous_tails():
    rng = np.random.default_rng(1)
    n = 20000
    for a in (2.0, 2.5, 3.5):
        x = 2.0 * (1.0 - rng.random(n)) ** (-1.0 / (a - 1.0))  # Pareto of exponent a above 2
        fit = power_law_tail(x, x_min=2.0)
        assert isinstance(fit, PowerLawTail) and fit.n_tail == n and not fit.discrete
        assert abs(fit.exponent - a) < 3.0 * fit.error
        assert fit.error == pytest.approx((fit.exponent - 1.0) / math.sqrt(n))
        assert fit.ks < 1.63 / math.sqrt(n)  # the 1 % quantile of Kolmogorov's law
    # a tail above a body: uniform values below 1, a Pareto tail of exponent 3 above. Every
    # bound in the tail fits a power law; the least distance picks one there, not in the body
    body = rng.random(5000)
    tail = (1.0 - rng.random(5000)) ** -0.5
    fit = power_law_tail(np.concatenate([body, tail]))
    assert 1.0 <= fit.x_min < 2.0 and abs(fit.exponent - 3.0) < 3.5 * fit.error
    assert fit.ks < power_law_tail(np.concatenate([body, tail]), x_min=0.5).ks
    assert fit.to_dict()["n_tail"] == fit.n_tail


def test_discrete_tails():
    rng = np.random.default_rng(2)
    n = 20000
    k = rng.zipf(3.0, n)  # x^-3 / zeta(3) on 1, 2, 3...
    fit = power_law_tail(k, discrete=True, x_min=1)
    assert fit.discrete and fit.x_min == 1.0 and abs(fit.exponent - 3.0) < 3.0 * fit.error
    # the error, 1 / sqrt(n Var[ln x]) at the estimate: the variance by the sums of the law
    x = np.arange(1, 2_000_001, dtype=float)
    p = x**-fit.exponent
    p /= p.sum()
    var = np.sum(p * np.log(x) ** 2) - np.sum(p * np.log(x)) ** 2
    assert fit.error == pytest.approx(1.0 / math.sqrt(n * var), rel=1e-3)
    auto = power_law_tail(k, discrete=True)
    assert abs(auto.exponent - 3.0) < 3.5 * auto.error and auto.n_tail >= 10
    # the zeta law above a bound: the continuous estimate of the integers is biased there
    # (Clauset et al.: about x_min > 6), the discrete one is not
    tail = k[k >= 3]
    exact = power_law_tail(tail, discrete=True, x_min=3)
    rough = power_law_tail(tail, x_min=3)
    assert abs(exact.exponent - 3.0) < 3.5 * exact.error
    assert abs(rough.exponent - 3.0) > 5.0 * rough.error


def test_errors():
    with pytest.raises(ValueError, match="integers"):
        power_law_tail([1.5, 2.0, 3.0], discrete=True)
    with pytest.raises(ValueError, match="at least"):
        power_law_tail([1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="min_tail"):
        power_law_tail(np.arange(1.0, 50.0), min_tail=1)
    with pytest.raises(ValueError, match="above x_min"):
        power_law_tail(np.arange(1.0, 50.0), x_min=49.0)
    with pytest.raises(ValueError, match="distinct"):
        power_law_tail(np.ones(30))
