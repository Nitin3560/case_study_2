def throughput_bps(total_bytes_delivered: int, dt_s: float, steps: int) -> float:
    if steps <= 0 or dt_s <= 0:
        return 0.0
    total_time = steps * dt_s
    return float(total_bytes_delivered) * 8.0 / total_time