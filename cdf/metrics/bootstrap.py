import numpy as np
from typing import List, Dict, Any, Tuple
from .drift import DriftMetrics

class BootstrapMetrics:
    @staticmethod
    def parametric_bootstrap_ci(horizons: List[int], n_h_list: List[int], s_h_list: List[int], B: int = 500, seed: int = 42) -> Dict[str, Tuple[float, float]]:
        """
        Calculates 95% bootstrap CI for lambda.
        Resamples s_h ~ Binomial(n_h, S_hat(h))
        """
        rng = np.random.RandomState(seed)
        
        lambda_estimates = []
        A_estimates = []
        
        # Calculate base S_hat
        s_hat_base = []
        for n_h, s_h in zip(n_h_list, s_h_list):
            if n_h == 0:
                s_hat_base.append(0.0)
            else:
                s_hat_base.append(s_h / n_h)
                
        for _ in range(B):
            b_s_h_list = []
            for n_h, s_hat in zip(n_h_list, s_hat_base):
                if n_h == 0:
                    b_s_h_list.append(0)
                else:
                    b_s_h_list.append(rng.binomial(n_h, s_hat))
                    
            fit = DriftMetrics.fit_exponential_drift(horizons, n_h_list, b_s_h_list)
            if not np.isnan(fit["lambda"]):
                lambda_estimates.append(fit["lambda"])
            if not np.isnan(fit["A"]):
                A_estimates.append(fit["A"])
                
        if len(lambda_estimates) == 0:
            return {"lambda": (float('nan'), float('nan')), "A": (float('nan'), float('nan'))}
            
        lambda_ci = (float(np.percentile(lambda_estimates, 2.5)), float(np.percentile(lambda_estimates, 97.5)))
        A_ci = (float(np.percentile(A_estimates, 2.5)), float(np.percentile(A_estimates, 97.5)))
        
        return {
            "lambda": lambda_ci,
            "A": A_ci
        }
