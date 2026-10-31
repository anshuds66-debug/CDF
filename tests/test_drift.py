import pytest
import math
from cdf.metrics.drift import DriftMetrics

def test_reliability():
    assert DriftMetrics.calculate_reliability(100, 50) == 0.5
    assert DriftMetrics.calculate_reliability(0, 0) == 0.0

def test_exponential_fit_memoryless():
    # For a memoryless agent with p0 = 0.95
    p0 = 0.95
    horizons = [1, 2, 4, 8, 16]
    n_h = 100000
    n_h_list = [n_h] * len(horizons)
    s_h_list = [int(n_h * (p0 ** h)) for h in horizons]
    
    fit = DriftMetrics.fit_exponential_drift(horizons, n_h_list, s_h_list)
    
    expected_lambda = -math.log(p0)
    
    # Should be close
    assert math.isclose(fit["lambda"], expected_lambda, rel_tol=1e-2)
    assert math.isclose(fit["A"], 1.0, rel_tol=1e-2)

def test_empirical_half_life():
    horizons = [1, 2, 4, 8]
    s_hat_list = [0.9, 0.7, 0.4, 0.1]
    # crosses 0.5 between h=2 and h=4
    # ln(S) between ln(0.7) and ln(0.4)
    # m = (ln(0.4) - ln(0.7)) / 2 = -0.2798
    # c = ln(0.7) - m*2
    # h_50 = (ln(0.5) - c) / m
    h_50 = DriftMetrics.calculate_empirical_half_life(horizons, s_hat_list)
    assert 2.0 < h_50 < 4.0

def test_empirical_half_life_no_cross():
    horizons = [1, 2, 4]
    s_hat_list = [0.9, 0.8, 0.7]
    assert DriftMetrics.calculate_empirical_half_life(horizons, s_hat_list) == ">4"

def test_excess_drift():
    # If S(h) exactly follows exponential, delta(h) should be ~0
    p0 = 0.95
    horizons = [1, 2, 4, 8, 16]
    n_h = 100000
    n_h_list = [n_h] * len(horizons)
    s_h_list = [int(n_h * (p0 ** h)) for h in horizons]
    
    delta = DriftMetrics.calculate_excess_drift(horizons, n_h_list, s_h_list)
    
    for h in horizons:
        assert math.isclose(delta[h], 0.0, abs_tol=1e-2)
