# coding:utf-8


# 2017/12/18作成。
# カルマンスムーザークラス。
# assimilation.pyにあったものを移植。

import numpy as np
import func_for_assimilate as ffa
import using_jit
from assimilation import Assimilation


class KalmanSmoother(Assimilation):
    # 時刻t=i+1の観測値を同化して、t=iに於ける解析値を計算する。
    def __init__(self, N=40, dt=0.05, F_id=0, delta=0.1,
                 assim_interval_days=0.25,
                 MVid=0, n_of_missn=0, random_state=0):
        super().__init__(N=N, dt=dt, F_id=F_id,
                         assim_interval_days=assim_interval_days,
                         MVid=MVid, n_of_missn=n_of_missn,
                         random_state=random_state)

        self.initial_xa = self.l96.get_spin_upped_profile()
        self.initial_Pa = np.identity(self.N) * 1e+1
        self.Pa = np.zeros((self.LMAX, self.N, self.N))
        self.delta = delta

    def for_loop(self):
        for l in range(self.LMAX - 1):
            if l == 0:
                self.Xa[l], self.Pa[l] = self._next_time_step(self.initial_xa,
                                                              self.initial_Pa,
                                                              self.Xo[l + 1])
            else:
                self.Xa[l], self.Pa[l] = self._next_time_step(self.Xa[l - 1],
                                                              self.Pa[l - 1],
                                                              self.Xo[l + 1])

        self.RMSE_a[:] = using_jit.cal_RMSE_2D(self.Xa, self.Xt)
        # 最後の1時間ステップだけは、解析値をフリーラン。
        LastL = self.LMAX - 1
        self.Xa[LastL] = self.l96.run(self.Xa[LastL - 1],
                                      days=self.assim_interval_days)
        self.RMSE_a[LastL] = using_jit.cal_RMSE(self.Xa[LastL], self.Xt[LastL])

    def _next_time_step(self, prev_xa, prev_Pa, next_xo):
        # 時刻t=Tに於ける観測を同化して、時刻t=0に於ける解析値を計算する。
        # 解析値を算出するために、次のステップにおける予報値も計算する必要がある。
        the_xf = self.l96.run(prev_xa, days=self.assim_interval_days)
        next_xf = self.l96.run(the_xf, days=self.assim_interval_days)
        M = self.l96.jacobian(prev_xa, days=self.assim_interval_days)
        the_Pf = (1 + self.delta) * M @ prev_Pa @ M.T
        H = ffa.get_H(next_xo)
        PfMTHT = the_Pf @ M.T @ H.T
        K = PfMTHT @ np.linalg.inv(H @ M @ PfMTHT + self.R)
        next_xo = ffa.left_aligned(next_xo)
        the_xa = the_xf + K @ (next_xo - H @ next_xf)
        the_Pa = (np.identity(self.N) - K @ H @ M) @ the_Pf

        return the_xa, the_Pa
