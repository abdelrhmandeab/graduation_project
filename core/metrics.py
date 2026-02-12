import time

class Metrics:
    def __init__(self):
        self.start_times = {}

    def start(self, key):
        self.start_times[key] = time.time()

    def end(self, key):
        if key not in self.start_times:
            return None
        return time.time() - self.start_times[key]

metrics = Metrics()
