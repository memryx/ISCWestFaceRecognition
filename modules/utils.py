import time
import numpy as np

class Framerate:
    def __init__(self, window=30):
        self.timestamps = [0] * window

    def update(self):
        self.timestamps = self.timestamps[1:] + [time.time()]

    def reset(self):
        self.timestamps = [0] * len(self.timestamps)

    def get(self):
        if not all(self.timestamps):
            return -1
        ts = np.array(self.timestamps)
        return 1 / np.average(ts[1:] - ts[:-1])


