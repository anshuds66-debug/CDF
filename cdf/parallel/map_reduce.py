from typing import List, Dict, Any

class MapReduce:
    @staticmethod
    def map_trajectory(trajectory_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Maps a single evaluated trajectory to its partial aggregate statistics.
        """
        h = trajectory_result["horizon"]
        success = trajectory_result.get("success", False)
        ff_step = trajectory_result.get("first_failure_step", None)
        
        # Hist is a dict mapping step index -> count, plus 'success' -> count
        ff_hist = {}
        if success:
            ff_hist["success"] = 1
        else:
            ff_hist[ff_step] = 1
            
        return {
            "h": h,
            "n": 1,
            "s": 1 if success else 0,
            "ff_hist": ff_hist
        }

    @staticmethod
    def reduce_partials(p1: Dict[str, Any], p2: Dict[str, Any]) -> Dict[str, Any]:
        """
        Reduces two partial aggregates into one.
        Must be associative and commutative.
        """
        if p1["h"] != p2["h"]:
            raise ValueError("Cannot reduce partials of different horizons")
            
        h = p1["h"]
        n = p1["n"] + p2["n"]
        s = p1["s"] + p2["s"]
        
        ff_hist = {}
        # Merge histograms
        keys = set(p1["ff_hist"].keys()).union(set(p2["ff_hist"].keys()))
        for k in keys:
            ff_hist[k] = p1["ff_hist"].get(k, 0) + p2["ff_hist"].get(k, 0)
            
        return {
            "h": h,
            "n": n,
            "s": s,
            "ff_hist": ff_hist
        }

    @staticmethod
    def reduce_all(partials: List[Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
        """
        Reduces a list of partials, grouped by horizon.
        """
        reduced = {}
        for p in partials:
            h = p["h"]
            if h not in reduced:
                reduced[h] = p
            else:
                reduced[h] = MapReduce.reduce_partials(reduced[h], p)
        return reduced
