import math
import numpy as np
from typing import Dict, Any, Tuple, List

class DriftMetrics:
    @staticmethod
    def calculate_reliability(n_h: int, s_h: int) -> float:
        if n_h == 0:
            return 0.0
        return s_h / n_h

    @staticmethod
    def fit_exponential_drift(horizons: List[int], n_h_list: List[int], s_h_list: List[int]) -> Dict[str, float]:
        """
        Fits S(h) = A * exp(-lambda * h) using weighted least squares on log-linear form.
        w_h = n_h * S_hat(h) / (1 - S_hat(h))
        Excludes horizons with s_h == 0.
        Handles S_hat(h) == 1 by adding a small epsilon or capping.
        """
        valid_h = []
        log_S = []
        weights = []
        
        for h, n_h, s_h in zip(horizons, n_h_list, s_h_list):
            if n_h == 0 or s_h == 0:
                continue
                
            S_hat = s_h / n_h
            
            # Handle S_hat = 1 to prevent division by zero in weights
            if S_hat == 1.0:
                S_hat = 1.0 - 1/(2*n_h) # Continuity correction equivalent
            
            log_s_val = math.log(S_hat)
            weight = n_h * S_hat / (1 - S_hat)
            
            valid_h.append(h)
            log_S.append(log_s_val)
            weights.append(weight)
            
        if len(valid_h) < 2:
            return {"lambda": float('nan'), "A": float('nan'), "r": float('nan'), "H_1_2": float('nan')}
            
        # Weighted least squares for y = c + m*x (where y = log_S, x = h, m = -lambda, c = ln_A)
        H = np.array(valid_h)
        Y = np.array(log_S)
        W = np.array(weights)
        
        # Calculate WLS manually or via numpy
        # sum(w), sum(w*x), sum(w*y), sum(w*x*x), sum(w*x*y)
        sum_w = np.sum(W)
        sum_wx = np.sum(W * H)
        sum_wy = np.sum(W * Y)
        sum_wxx = np.sum(W * H * H)
        sum_wxy = np.sum(W * H * Y)
        
        denominator = (sum_w * sum_wxx - sum_wx * sum_wx)
        if denominator == 0:
            return {"lambda": float('nan'), "A": float('nan'), "r": float('nan'), "H_1_2": float('nan')}
            
        m = (sum_w * sum_wxy - sum_wx * sum_wy) / denominator
        c = (sum_wy - m * sum_wx) / sum_w
        
        lambda_val = -m
        A = math.exp(c)
        
        r = math.exp(-lambda_val)
        h_half = math.log(2) / lambda_val if lambda_val > 0 else float('inf')
        
        return {
            "lambda": lambda_val,
            "A": A,
            "r": r,
            "H_1_2": h_half
        }

    @staticmethod
    def calculate_empirical_half_life(horizons: List[int], s_hat_list: List[float]) -> Any:
        """
        Model-free empirical half-life H_hat_50 using log-linear interpolation.
        Returns a float if crosses 0.5, else a string like ">32".
        """
        for i in range(len(horizons) - 1):
            h1, h2 = horizons[i], horizons[i+1]
            s1, s2 = s_hat_list[i], s_hat_list[i+1]
            
            if s1 >= 0.5 and s2 < 0.5:
                # Log-linear interpolation
                # ln(S) = m*h + c
                if s1 == 0 or s2 == 0:
                    continue # Should not happen around 0.5, but safety
                
                m = (math.log(s2) - math.log(s1)) / (h2 - h1)
                if m == 0:
                    continue
                
                c = math.log(s1) - m * h1
                h_50 = (math.log(0.5) - c) / m
                return h_50
                
        # If it never crosses 0.5
        max_h = horizons[-1] if horizons else 0
        if all(s >= 0.5 for s in s_hat_list):
            return f">{max_h}"
        elif all(s < 0.5 for s in s_hat_list):
            return f"<{horizons[0]}"
        return "Unknown"

    @staticmethod
    def calculate_excess_drift(horizons: List[int], n_h_list: List[int], s_h_list: List[int]) -> Dict[int, float]:
        """
        Fit exponential model only on short horizons (h <= 4).
        Calculate delta(h) = (ln A_s - lambda_s * h) - ln S(h)
        """
        short_h = [h for h in horizons if h <= 4]
        short_idx = [i for i, h in enumerate(horizons) if h <= 4]
        short_n = [n_h_list[i] for i in short_idx]
        short_s = [s_h_list[i] for i in short_idx]
        
        fit_s = DriftMetrics.fit_exponential_drift(short_h, short_n, short_s)
        lambda_s = fit_s["lambda"]
        A_s = fit_s["A"]
        
        if math.isnan(lambda_s) or math.isnan(A_s):
            return {}
            
        ln_A_s = math.log(A_s)
        
        delta = {}
        for h, n_h, s_h in zip(horizons, n_h_list, s_h_list):
            if n_h == 0 or s_h == 0:
                continue
            
            S_hat = s_h / n_h
            if S_hat == 1.0:
                S_hat = 1.0 - 1/(2*n_h)
                
            ln_S_h = math.log(S_hat)
            baseline = ln_A_s - lambda_s * h
            delta[h] = baseline - ln_S_h
            
        return delta
