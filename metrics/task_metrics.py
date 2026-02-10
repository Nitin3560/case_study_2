def task_completion_rate(tasks) -> float:
    total = len(tasks)
    done = sum(1 for t in tasks if t.completed)
    return 100.0 * done / total if total > 0 else 0.0