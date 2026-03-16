# -*- coding: utf-8 -*-
# @Organization  : Tongyi Lab, Alibaba
# @Author        : Lingteng Qiu
# @Email         : 220019047@link.cuhk.edu.cn
# @Time          : 2025-10-14 19:09:29
# @Function      : Average Timer
import time
from collections import deque
from typing import Optional


class SlidingAverageTimer:
    """
    A sliding window timer to compute the average of recent time intervals.
    Useful for measuring latency, FPS, or inference time with smooth statistics.
    """

    def __init__(self, window_size: int = 10):
        """
        Args:
            window_size (int): Number of recent time intervals to keep for averaging.
        """
        if window_size <= 0:
            raise ValueError("window_size must be positive")
        self.window_size = window_size
        self.times = deque(maxlen=window_size)  # Only keep the last N values
        self.last_start: Optional[float] = None

    def start(self):
        """Start timing a new interval."""
        self.last_start = time.time()

    def record(self):
        """Stop timing and record the elapsed time."""
        if self.last_start is None:
            raise RuntimeError("Timer not started. Call start() first.")
        elapsed = time.time() - self.last_start
        self.times.append(elapsed)
        self.last_start = None  # Reset to prevent double-record

    def put(self, duration: float):
        """Manually add a time duration (e.g., from external measurement)."""
        self.times.append(duration)

    def avg(self) -> float:
        """Get the average of recorded times. Returns 0 if no data."""
        return sum(self.times) / len(self.times) if self.times else 0.0

    def count(self) -> int:
        """Get the number of recorded times."""
        return len(self.times)

    def reset(self):
        """Clear all recorded times."""
        self.times.clear()
        self.last_start = None

    def __len__(self):
        return self.count()

    def __str__(self):
        avg_time = self.avg()
        return f"SlidingAverageTimer(avg={avg_time*1000:.2f}ms, count={self.count()}, window={self.window_size})"
