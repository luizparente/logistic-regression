import time

class Stopwatch:
    def __init__(self):
        self._start_time = None

    def start(self):
        """Starts the stopwatch."""
        if self._start_time is not None:
            raise Exception("Stopwatch is running. Use .stop() to stop it.")
        
        self._start_time = time.perf_counter()

    def stop(self):
        """Stops the stopwatch and returns the elapsed time in seconds."""
        if self._start_time is None:
            raise Exception("Stopwatch is not running. Use .start() to start it.")
        
        elapsed_time = time.perf_counter() - self._start_time
        self._start_time = None
        
        return elapsed_time
