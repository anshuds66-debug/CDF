import pytest
import math
from cdf.metrics.voting_bounds import VotingBounds
from cdf.evaluation.voting import Voting

def test_majority_bound():
    p = 0.8
    # k=1 -> p
    assert math.isclose(VotingBounds.majority_bound(1, p), 0.8)
    # k=3, need 2 or 3 correct
    # 3*(0.8^2)*0.2 + 0.8^3 = 3*0.64*0.2 + 0.512 = 0.384 + 0.512 = 0.896
    assert math.isclose(VotingBounds.majority_bound(3, p), 0.896)

def test_plurality_bound():
    p = 0.8
    # k=1 -> p
    assert math.isclose(VotingBounds.plurality_bound(1, p), 0.8)
    # k=3 -> 1 - 0.2^3 = 1 - 0.008 = 0.992
    assert math.isclose(VotingBounds.plurality_bound(3, p), 0.992)

def test_plurality_vote():
    # 5 rollouts
    t1 = [10, 20, 30]
    t2 = [10, 20, 30]
    t3 = [10, 99, 30]
    t4 = [10, 20, 30]
    t5 = [10, 20, 40]
    
    voted = Voting.aggregate_plurality_vote([t1, t2, t3, t4, t5], horizon=3)
    assert voted == [10, 20, 30]

def test_plurality_vote_with_failures():
    t1 = [10, 20] # failed at 3
    t2 = [10, 20]
    t3 = [10, 20, 30]
    
    voted = Voting.aggregate_plurality_vote([t1, t2, t3], horizon=3)
    assert voted == [10, 20]
