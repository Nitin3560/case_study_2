def throughput_bps(total_bytes_delivered: int, dt_s: float, steps: int) -> float:
    if steps <= 0 or dt_s <= 0:
        return 0.0
    total_time = steps * dt_s
    return float(total_bytes_delivered) * 8.0 / total_time

def throughput_bps_per_agent(bytes_per_agent: list[int], dt_s: float, steps: int) -> list[float]:
    if steps <= 0 or dt_s <= 0:
        return [0.0 for _ in bytes_per_agent]
    total_time = steps * dt_s
    return [float(b) * 8.0 / total_time for b in bytes_per_agent]
