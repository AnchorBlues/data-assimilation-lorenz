
import numpy as np

"""
Lorenz96モデルの基底クラス
"""
# Lorenz96モデルでのdt=1.0が、天気予報(実時間)での5.0日に対応。1日<->dt=0.2, 6時間<->dt=0.05
Lorenz96_TimeScale = 5.0


class Lorenz96RungeKutta4Base(object):
    def __init__(self, F, dt, N):
        self.F = F
        self.dt = dt
        self.N = N
        self.init_x = np.zeros(self.N)
        self.init_x[:] = self.F
        self.init_x[self.N // 2] = self.F * (1 + 1e-3)
