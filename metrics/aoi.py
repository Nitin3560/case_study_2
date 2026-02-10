import numpy as np

def mean_aoi(tasks, step_i: int) -> float:
    aois = []
    for t in tasks:
        if not t.active:
            continue
        if t.last_received_step is None:
            aois.append(float(step_i - t.appears_step))
        else:
            aois.append(float(step_i - t.last_received_step))
    return float(np.mean(aois)) if aois else 0.0