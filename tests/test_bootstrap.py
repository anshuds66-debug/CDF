import pytest
from cdf.metrics.bootstrap import BootstrapMetrics
import math

def test_bootstrap():
    p0 = 0.95
    horizons = [1, 2, 4, 8]
    n_h_list = [1000] * 4
    s_h_list = [int(1000 * (p0 ** h)) for h in horizons]
    
    ci = BootstrapMetrics.parametric_bootstrap_ci(horizons, n_h_list, s_h_list, B=100)
    expected_lambda = -math.log(p0)
    
    lambda_ci = ci["lambda"]
    # lambda_ci should be a tuple (low, high)
    assert lambda_ci[0] <= expected_lambda <= lambda_ci[1] or math.isclose(lambda_ci[0], expected_lambda, abs_tol=0.01)
