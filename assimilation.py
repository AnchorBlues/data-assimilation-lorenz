# coding:utf-8

# 2017/01/02作成
# Lorenz96モデルを同化するクラスの定義とか。
# MethodName, ModuleName, FuncNameが要りません。
# 3Dvarとかも、4Dvarと同じ書き方に変えてあげないと。

import numpy as np
import using_jit
from create_true_and_obs import DataCreator
import Lorenz96
import missing_value


def pickup_true_and_obs(Xt, Xo, assim_interval_days=0.25):
    """
    スピンアップから走らせたXt, Xoから、最後の1年間のデータを取り出す。
    assim_interval_days=0.25であれば、Nskip = 1 になるため、
    単純にXt, Xoの後半のデータを取り出すだけになる。
    """
    IMAX = Xt.shape[0]
    N = Xt.shape[1]
    ddays = 2 * 365 / float(IMAX)
    # Nskipとは、Xoから同化に用いる観測を得る際にスキップする要素の数
    Nskip = int(round(assim_interval_days / ddays, 0))
    # Xtが2年間のデータなので、そこから1年間のデータだけを取り出すという意味で2を割る。
    LMAX = int((IMAX / 2) / Nskip)
    Xt_new = np.zeros((LMAX, N))
    Xo_new = np.zeros((LMAX, N))
    for l in range(LMAX):
        Xt_new[l] = Xt[int(IMAX / 2) + Nskip * (l + 1) - 1]
        Xo_new[l] = Xo[int(IMAX / 2) + Nskip * (l + 1) - 1]

    return Xt_new, Xo_new


def pickup_F(F, assim_interval_days=0.25):
    """
    assim_interval_days=0.25であれば、Nskip = 1 になるため、
    単純にFの後半のデータを取り出すだけになる。
    """
    IMAX = F.size
    ddays = 2 * 365 / float(IMAX)
    Nskip = int(round(assim_interval_days / ddays, 0))
    LMAX = int((IMAX / 2) / Nskip)
    F_new = np.zeros(LMAX)
    for l in range(LMAX):
        F_new[l] = F[int(IMAX / 2) + Nskip * (l + 1) - 1]

    return F_new


class Assimilation(object):
    def __init__(self, N=40, dt=0.05, F_id=0,
                 assim_interval_days=0.25,
                 MVid=0, n_of_missn=0,
                 random_state=0):
        np.random.seed(random_state)
        self.N = N
        self.dt = dt
        self.F_id = F_id
        self.assim_interval_days = assim_interval_days
        data_creator = DataCreator(self.N, self.dt, self.F_id, random_state)
        Xt, Xo = data_creator.load()
        self.Xt, self.Xo \
            = pickup_true_and_obs(Xt, Xo,
                                  assim_interval_days=self.assim_interval_days)
        self.F = pickup_F(data_creator.F,
                          assim_interval_days=self.assim_interval_days)
        self.LMAX = self.Xt.shape[0]
        self.l96 = Lorenz96.Lorenz96RungeKutta4(self.F[0], self.dt, self.N)

        # 観測値の取り除きに関して
        self.MVid = MVid
        self.n_of_missn = n_of_missn
        self.p = self.N - self.n_of_missn
        MV = missing_value.MV[self.MVid]
        MS = MV.get_ms(self.n_of_missn, N=self.N, Tmax=self.LMAX,
                       random_state=random_state)
        self.Xo[MS == 0] = np.nan
        self.R = np.identity(self.p)

        self.Xa = np.zeros((self.LMAX, self.N))
        self.RMSE_a = np.zeros(self.LMAX)
        self.RMSE_o = np.zeros(self.LMAX)

    def get_RMSE_o(self):
        self.RMSE_o[:] = using_jit.cal_RMSE_2D(self.Xo, self.Xt)

    def ave_of_RMSE_a(self):
        rmse = self.RMSE_a[400:]
        return np.nanmean(rmse)  # RMSE_aがnanになっている部分はカウントしない。

    def draw_RMSE_a(self):
        import matplotlib.pyplot as plt
        plt.plot(self.RMSE_a)
        plt.xlim(0, )
        plt.ylim(0, )
        return plt


class EnsembleKalmanFilter(Assimilation):
    def __init__(self, m=20, N=40, dt=0.05, F_id=0,
                 assim_interval_days=0.25,
                 MVid=0, n_of_missn=0,
                 random_state=0):
        super().__init__(N=N, dt=dt, F_id=F_id,
                         assim_interval_days=assim_interval_days,
                         MVid=MVid, n_of_missn=n_of_missn,
                         random_state=random_state)

        self.m = m
        self.initial_Enxa = self.l96.get_initial_Xa(m)
        self.Enxa = np.zeros((self.LMAX, self.N, self.m))

    def for_loop(self):
        for l in range(self.LMAX):
            if l == 0:
                self.Enxa[l] = self._next_time_step(self.initial_Enxa,
                                                    self.Xo[l])
            else:
                self.Enxa[l] = self._next_time_step(self.Enxa[l - 1],
                                                    self.Xo[l])

        self.Xa = np.average(self.Enxa, axis=2)
        self.RMSE_a[:] = using_jit.cal_RMSE_2D(self.Xa, self.Xt)


class EnsembleKalmanSmoother(Assimilation):
    def __init__(self, m=20, N=40, dt=0.05, F_id=0,
                 assim_interval_days=0.25,
                 MVid=0, n_of_missn=0, random_state=0):
        super().__init__(N=N, dt=dt, F_id=F_id,
                         assim_interval_days=assim_interval_days,
                         MVid=MVid, n_of_missn=n_of_missn,
                         random_state=random_state)

        self.m = m
        self.initial_Enxa = self.l96.get_initial_Xa(m)
        self.Enxa = np.zeros((self.LMAX, self.N, self.m))

    def for_loop(self):
        for l in range(self.LMAX - 1):
            if l == 0:
                self.Enxa[l] = self._next_time_step(self.initial_Enxa,
                                                    self.Xo[l + 1])
            else:
                self.Enxa[l] = self._next_time_step(self.Enxa[l - 1],
                                                    self.Xo[l + 1])

        # 最後の1時間ステップだけは、解析値をフリーラン。
        self.Enxa[self.LMAX - 1] \
            = self.l96.ensemble_run(self.Enxa[self.LMAX - 2],
                                    days=self.assim_interval_days)
        self.Xa = np.average(self.Enxa, axis=2)
        self.RMSE_a[:] = using_jit.cal_RMSE_2D(self.Xa, self.Xt)


class EnsembleKalmanFilter_4D(Assimilation):
    def __init__(self, m=20, N=40, dt=0.05, F_id=0,
                 assim_interval_days=0.25,
                 assim_window_days=1.00,
                 MVid=0, n_of_missn=0, random_state=0):
        super().__init__(N=N, dt=dt, F_id=F_id,
                         assim_interval_days=assim_interval_days,
                         MVid=MVid, n_of_missn=n_of_missn,
                         random_state=random_state)

        self.m = m
        self.initial_Enxa = self.l96.get_initial_Xa(m)
        self.J = int(round(assim_window_days /
                           assim_interval_days, 0))
        self.LMAX_for_4D = int(self.LMAX / self.J)
        self.Enxa = np.zeros((self.LMAX_for_4D, self.N, self.m))
        self.RMSE_a_AW = np.zeros(self.LMAX_for_4D)
        self.RMSE_rea_AW = np.zeros(self.LMAX_for_4D)

    def for_loop(self):
        for l in range(self.LMAX_for_4D):
            # self.J=5、つまり同化ウィンドウ内で5つ観測を取り込むとすると、
            # 時刻l-1に於ける解析値と時刻l, ..., l+4における観測値をもとに、
            # 時刻l, ..., l+4における解析値(self.Xa[idx])を計算する。
            # xa0というのは、時刻l-1における"再"解析値。
            # RMSE_a_AWは、時刻l+4(l-1)における「解析値」のRMSEで、
            # RMSE_rea_AWは、時刻l+4(l-1)における「再解析値」のRMSE

            # lに対応するインデックス
            idx = range(l * self.J, (l + 1) * self.J)

            if l == 0:
                self.Xa[idx], xa0, self.Enxa[l] \
                    = self._next_time_step(self.initial_Enxa,
                                           self.Xo[idx])
                self.RMSE_rea_AW[l] = np.nan
            else:
                self.Xa[idx], xa0, self.Enxa[l] \
                    = self._next_time_step(self.Enxa[l - 1],
                                           self.Xo[idx])
                self.RMSE_rea_AW[l] = using_jit.cal_RMSE(
                    xa0, self.Xt[l * self.J - 1])

        self.RMSE_a[:] = using_jit.cal_RMSE_2D(self.Xa, self.Xt)
        # for l in range(self.LMAX_for_4D):
        #     self.RMSE_a_AW[l] = self.RMSE_a[(l + 1) * self.J - 1]
        # 上記のコードを最適化すると以下の通り
        self.RMSE_a_AW[:] \
            = self.RMSE_a[(np.arange(self.LMAX_for_4D) + 1) * self.J - 1]
