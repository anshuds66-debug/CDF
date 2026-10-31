from typing import List, Dict, Any, Tuple
from collections import Counter

class Voting:
    @staticmethod
    def aggregate_plurality_vote(parsed_trajectories: List[List[int]], horizon: int) -> List[int]:
        """
        Votes over the FULL VECTOR of intermediate answers.
        Takes a list of parsed trajectories (each is a List[int]).
        Returns the plurality voted trajectory.
        """
        # Convert each trajectory to a tuple so it can be hashed and counted
        # A trajectory is only valid for voting if it has length == horizon
        # (Though even partials could be voted on, paper says "whole-trajectory plurality vote")
        # To be strict, we pad or exactly match
        
        valid_trajectories = []
        for traj in parsed_trajectories:
            if len(traj) == horizon:
                valid_trajectories.append(tuple(traj))
            else:
                # If it's partial, we can pad it with None so it's a distinct tuple
                padded = tuple(traj) + (None,) * (horizon - len(traj))
                valid_trajectories.append(padded)
                
        if not valid_trajectories:
            return []
            
        counts = Counter(valid_trajectories)
        best_traj = counts.most_common(1)[0][0]
        
        # Filter out Nones
        result = [x for x in best_traj if x is not None]
        return result
