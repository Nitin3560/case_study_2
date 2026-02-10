import numpy as np

def remaining_energy_stats(agent_energy):
    e = np.asarray(agent_energy, dtype=float)
    return {
        "mean_energy_j": float(np.mean(e)),
        "min_energy_j": float(np.min(e)),
        "alive_frac": float(np.mean(e > 0.0)),
    }