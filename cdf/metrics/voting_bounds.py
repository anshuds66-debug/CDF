import math
from typing import List

class VotingBounds:
    @staticmethod
    def _comb(n: int, k: int) -> int:
        return math.comb(n, k)

    @staticmethod
    def majority_bound(k: int, p: float) -> float:
        """
        S_maj(k,p) = sum_{j=ceil(k/2)}^k C(k,j) p^j (1-p)^{k-j}
        """
        if k == 0:
            return 0.0
        
        limit = math.ceil(k / 2)
        prob = 0.0
        for j in range(limit, k + 1):
            prob += VotingBounds._comb(k, j) * (p ** j) * ((1 - p) ** (k - j))
        return prob

    @staticmethod
    def plurality_bound(k: int, p: float) -> float:
        """
        S_plur(k,p) = 1 - (1-p)^k
        """
        return 1.0 - ((1.0 - p) ** k)
