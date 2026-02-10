import numpy as np

def move_towards(cur_xy: np.ndarray, goal_xy: np.ndarray) -> int:
    """
    Map a desired direction to one of 8 discrete move_dir actions used by env.
    Returns move_dir in [0..7].
    """
    d = goal_xy - cur_xy
    if np.linalg.norm(d) < 1e-6:
        return 0
    x, y = float(d[0]), float(d[1])
    # crude quantization
    if abs(x) > abs(y):
        return 1 if x > 0 else 2
    else:
        return 3 if y > 0 else 4