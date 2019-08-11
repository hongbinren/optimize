import numpy as np 


class StepScheduler(object):

    _count = 1

    def __init__(self, drop, drop_step):
        self.drop = drop
        self.drop_step = drop_step

    def __call__(self, lr0):
        while True:
            lr0 *= (self.drop**(np.floor(self._count / self.drop_step)))
            yield lr0
            self._count += 1

