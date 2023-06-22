"""
A profiling tool used during bechmarking to record the time taken
"""

import time

class TimeProfiler:
    def __init__(self):
        """ Profiler used to measure time intervals"""
        self._durations = []
        self._current_start_time = 0
        self._current_end_time = 0
        
    def record_start(self):
        self._current_start_time = time.time()
        
    def record_end(self):
        self._current_end_time = time.time()
        
    def save_recorded_time(self):
        self._durations.append(self._current_end_time-self._current_start_time)