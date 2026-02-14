def task_completion_rate(tasks) -> float:
    total = len(tasks)
    done = sum(1 for t in tasks if t.completed)
    return 100.0 * done / total if total > 0 else 0.0

def task_delivery_rate(tasks) -> float:
    total = len(tasks)
    done = sum(1 for t in tasks if t.delivered)
    return 100.0 * done / total if total > 0 else 0.0

def user_coverage_rate(user_delivered) -> float:
    total = len(user_delivered)
    done = sum(1 for d in user_delivered if d)
    return 100.0 * done / total if total > 0 else 0.0
