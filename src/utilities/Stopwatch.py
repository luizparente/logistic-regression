import time

class Stopwatch:
    def __init__(self):
        self._start_time = None

    def start(self):
        """Starts the stopwatch."""
        self._start_time = time.perf_counter()

    def stop(self):
        """Stops the stopwatch and returns the elapsed time in seconds."""
        elapsed_time = time.perf_counter() - self._start_time
        self._start_time = None
        
        return elapsed_time

    def pause(self):
        """Pauses the stopwatch."""
        elapsed_time = time.perf_counter() - self._start_time
        
        return elapsed_time