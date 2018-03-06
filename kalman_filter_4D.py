# coding:utf-8


# 2017/12/18作成。
# 4次元カルマンフィルタークラス。
# assimilation.pyにあったものを移植。

import numpy as np
import func_for_assimilate as ffa
import using_jit
from assimilation import Assimilation


class KalmanFilter_4D(Assimilation):
    def __init__(self, N=40, dt=0.05, F_id=0, delta=0.1,
                 assim_interval_days=0.25,
                 assim_window_days=1.00,
                 MVid=0, n_of_missn=0, random_state=0):
        super().__init__(N=N, dt=dt, F_id=F_id,
                         assim_interval_days=assim_interval_days,
                         MVid=MVid, n_of_missn=n_of_missn,
                         random_state=random_state)

        self.initial_xa = self.l96.get_spin_upped_profile()
        self.initial_Pa = np.identity(self.N) * 1e+1
        self.delta = delta
        self.J = int(round(assim_window_days /
                           assim_interval_days, 0))
        self.LMAX_for_4D = int(self.LMAX / self.J)
        self.Pa = np.zeros((self.LMAX_for_4D, self.N, self.N))
        self.RMSE_a_AW = np.zeros(self.LMAX_for_4D)
        self.RMSE_rea_AW = np.zeros(self.LMAX_for_4D)

    def for_loop(self):
        idx = range(self.J)
        self.Xa[idx], xa0, self.Pa[0] \
            = self._next_time_step(self.initial_xa,
                                   self.initial_Pa,
                                   self.Xo[idx])
        self.RMSE_rea_AW[0] = np.nan
        for l in range(1, self.LMAX_for_4D):
            # self.J=5、つまり同化ウィンドウ内で5つ観測を取り込むとすると、
            # 時刻l-1に於ける解析値と時刻l, ..., l+4における観測値をもとに、
            # 時刻l, ..., l+4における解析値(self.Xa[idx])を計算する。
            # xa0というのは、時刻l-1における"再"解析値。
            # RMSE_a_AWは、時刻l+4(l-1)における「解析値」のRMSEで、
            # RMSE_rea_AWは、時刻l+4(l-1)における「再解析値」のRMSE

            # lに対応するインデックス
            idx = range(l * self.J, (l + 1) * self.J)

            self.Xa[idx], xa0, self.Pa[l] \
                = self._next_time_step(self.Xa[l * self.J - 1],
                                       self.Pa[l - 1],
                                       self.Xo[idx])
            self.RMSE_rea_AW[l] \
                = using_jit.cal_RMSE(xa0, self.Xt[l * self.J - 1])

        self.RMSE_a[:] = using_jit.cal_RMSE_2D(self.Xa, self.Xt)
        # for l in range(self.LMAX_for_4D):
        #     self.RMSE_a_AW[l] = self.RMSE_a[(l + 1) * self.J - 1]
        # 上記のコードを最適化すると以下の通り
        self.RMSE_a_AW[:] \
            = self.RMSE_a[(np.arange(self.LMAX_for_4D) + 1) * self.J - 1]

    def _next_time_step(self, prev_xa, prev_Pa, xo_in_AW):
        N = prev_xa.size
        xa = prev_xa
        Pa = prev_Pa
        for j in range(self.J):
            H = ffa.get_H(xo_in_AW[j])
            the_xo = ffa.left_aligned(xo_in_AW[j])
            the_xf = self.l96.run(
                xa, days=self.assim_interval_days * (j + 1))
            M = self.l96.jacobian(
                xa, days=self.assim_interval_days * (j + 1))
            PaMTHT = Pa @ M.T @ H.T
            K = PaMTHT @ np.linalg.inv(H @ M @ PaMTHT + self.R)
            the_xa = xa + K @ (the_xo - (H @ the_xf))
            the_Pa = (np.identity(N) - K @ H @ M) @ Pa

            # xa, Paをfor loopの中でupdateする
            xa = the_xa
            Pa = the_Pa

        xa_in_AW = self.l96.get_xf_in_AW(
            xa, self.J, days=self.assim_interval_days)
        M = self.l96.jacobian(xa, days=self.assim_interval_days * self.J)
        the_Pa = (1 + self.delta) * M @ Pa @ M.T

        # xaは、時刻t=-1に於ける解析値
        # xa_in_AWは、時刻t=0~J-1に於ける解析値
        # the_Paは、時刻t=J-1に於ける解析誤差共分散行列

        return xa_in_AW, xa, the_Pa
