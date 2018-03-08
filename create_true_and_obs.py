# coding:utf-8

# 2017/01/02作成
# モデルの値、観測の値を作成する。

import numpy as np
from Lorenz96_base import Lorenz96_TimeScale
import Lorenz96
import subroutine


def get_F8(IMAX):
    return np.ones(IMAX) * 8.0


def get_F8_9(IMAX):
    return np.r_[np.ones(IMAX / 4 * 3) * 8.0, np.ones(IMAX / 4) * 9.0]


def get_SineCurve(IMAX):
    return 8.0 + np.sin(np.arange(IMAX) * 4.0 * np.pi / IMAX)


class Force(object):
    def __init__(self, name, Func):
        self.name = name
        self.Func = Func

    def draw(self, IMAX):
        import matplotlib.pyplot as plt
        plt.plot(self.get(IMAX))
        return plt

    def get(self, IMAX):
        return self.Func(IMAX)


F = [0] * 10
F[0] = Force('8', get_F8)
F[1] = Force('8-9', get_F8_9)
F[2] = Force('Sine_Curve', get_SineCurve)


class DataCreator(object):
    """
    真値データと観測データを取得する。
    すでに計算を終えていてデータファイルがあるときにはそのファイルをロードし、
    まだ計算していないときには計算・ファイルに保存を行う。
    """

    def __init__(self, N=40, dt=0.05, F_id=0, random_state=0):
        self.N = N
        self.dt = dt
        self.ddays = self.dt * Lorenz96_TimeScale
        self.F_id = F_id
        self.IMAX = int(2 * 365.0 / self.ddays)
        self.F = F[F_id].get(self.IMAX)
        self.fname = '_N' + str(N) + '_dt' + str(dt) + '_F' + F[F_id].name
        self.random_state = random_state

    def make(self):
        """
        dt=0.05<->6時間の時、
        730日間のデータを得るために、4*730回ループを回す必要がある。
        Fが時間変化する場合を想定して、Lorenz96RungeKuttaクラスでなく、
        cal関数を直接呼び出す。
        """
        np.random.seed(self.random_state)
        x, _ = Lorenz96.initial_taskx(self.N)

        # run関数をループさせる回数。730.0日目のデータも得るために、 + 1している。
        Xt = np.zeros((self.IMAX, self.N))
        Xo = np.zeros((self.IMAX, self.N))
        for i in range(self.IMAX):
            if i == 0:
                Xt[i] = x
            else:
                Xt[i] = Lorenz96.cal_oigawa(
                    Xt[i - 1], self.F[i], self.dt)
            Xo[i] = Xt[i] + np.random.randn(self.N)

        return Xt, Xo

    def save(self, Xt, Xo):
        subroutine.save_npz(Xt, 'True' + self.fname)
        subroutine.save_npz(Xo, 'Obs' + self.fname)

    def load(self):
        """
        あったらロード、なかったら作ってセーブ
        """
        try:
            Xt = subroutine.load_npz('True' + self.fname)
            Xo = subroutine.load_npz('Obs' + self.fname)
        except FileNotFoundError:
            Xt, Xo = self.make()
            self.save(Xt, Xo)

        return Xt, Xo
