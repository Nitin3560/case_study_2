import numpy as np


def _collect_aois(tasks, step_i: int, user_active=None, user_last_received=None, user_appears=None, user_delivered=None):
    aois = []
    # If user streams are provided, AoI is measured on users only.
    if user_active is None or user_last_received is None or user_appears is None or user_delivered is None:
        for t in tasks:
            if not t.active:
                continue
            if getattr(t, "delivered", False):
                continue
            if t.last_received_step is None:
                aois.append(float(step_i - t.appears_step))
            else:
                aois.append(float(step_i - t.last_received_step))
    if user_active is not None and user_last_received is not None and user_appears is not None and user_delivered is not None:
        for uid in range(len(user_active)):
            if not user_active[uid]:
                continue
            if user_delivered[uid]:
                continue
            if user_last_received[uid] is None:
                aois.append(float(step_i - user_appears[uid]))
            else:
                aois.append(float(step_i - user_last_received[uid]))
    return aois


def mean_aoi(tasks, step_i: int, user_active=None, user_last_received=None, user_appears=None, user_delivered=None) -> float:
    aois = _collect_aois(tasks, step_i, user_active, user_last_received, user_appears, user_delivered)
    return float(np.mean(aois)) if aois else 0.0


def p95_aoi(tasks, step_i: int, user_active=None, user_last_received=None, user_appears=None, user_delivered=None) -> float:
    aois = _collect_aois(tasks, step_i, user_active, user_last_received, user_appears, user_delivered)
    return float(np.percentile(aois, 95)) if aois else 0.0
