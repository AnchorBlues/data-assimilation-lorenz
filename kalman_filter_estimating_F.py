# coding:utf-8


# 2017/12/18作成。
# アーカイブ用のファイル。
# 元々kalman_filter.py内にあったコードを保存しておく。

import numpy as np
import Lorenz96
import func_for_assimilate as ffa
from assimilation import Assimilation
import using_jit


class KF_Estimating_F(Assimilation):
    def __init__(self, N=40, dt=0.05, F_id=0,
                 assim_interval_days=0.25,
                 MVid=0, n_of_missn=0):
        super().__init__(N=N, dt=dt, F_id=F_id,
                         assim_interval_days=assim_interval_days,
                         MVid=MVid, n_of_missn=n_of_missn)

        self.initial_xa = Lorenz96.get_spin_upped_profile(self.N, self.dt)
        self.initial_Pa = np.identity(self.N + 1) * 1e+1
        self.delta = 0.1
        self.delta_of_F = 0.1    # deltaと同じ値にすると発散することが少ないみたい
        self.initial_F = 10.0
        self.Estimating_F = np.empty_like(self.F)

    def for_loop(self):
        prev_xa = self.initial_xa
        prev_Pa = self.initial_Pa
        prev_F = self.initial_F
        self.Xo[np.isnan(self.Xo)] = 1e+5

        for l in range(self.LMAX):
            the_xa, the_Pa, the_F = self.Func(prev_xa, prev_F, prev_Pa, self.Xo[l],
                                              self.R, self.delta, self.delta_of_F,
                                              dt=self.dt, days=self.assim_interval_days,
                                              KSC=1e+5)

            self.Xa[l] = the_xa
            self.RMSE_a[l] = using_jit.cal_RMSE(the_xa, self.Xt[l])
            self.Estimating_F[l] = the_F
            prev_xa = the_xa
            prev_Pa = the_Pa
            prev_F = the_F

        self.Xo[self.Xo == 1e+5] = np.nan


def _next_time_step_estimating_F(prev_xa, prev_F, prev_Pa, the_xo, R, delta, delta_of_F,
                                dt=0.01, days=0.25, KSC=1e+5):
    N = prev_xa.size                # 解析値の地点数
    the_xf = Lorenz96.run(prev_xa, F=prev_F, dt=dt, days=days)
    M = Lorenz96.jacobian_for_s(prev_xa, dt=dt, days=days, F=prev_F)

    # Fの部分だけ別の値でinflationする。
    the_Pf = (1 + delta) * M @ prev_Pa @ M.T
    the_Pf[N, :] = (1 + delta_of_F) * the_Pf[N, :] / (1 + delta)
    # the_Pf[:, N] = (1 + delta_of_F) * the_Pf[:, N] / (1 + delta)
    # the_Pf[N, N] = (1 + delta_of_F) * the_Pf[N, N] / (1 + delta)
    # ↑このコメントアウトを外すと発散してしまう。
    # あくまでクロスタームに対応する部分だけをFに関する部分と見なす。

    H = ffa.get_H(the_xo, KSC=KSC)
    H = ffa.get_H_for_s(H, 1)
    K = the_Pf @ H.T @ np.linalg.inv(H @ the_Pf @ H.T + R)
    the_sf = np.hstack((the_xf, prev_F))
    the_xo = ffa.left_aligned(the_xo, KSC=KSC)
    the_sa = the_sf + K @ (the_xo - H @ the_sf)
    the_xa = the_sa[:N]
    the_F = the_sa[the_sa.size - 1]
    the_Pa = (np.identity(K.shape[0]) - K @ H) @ the_Pf

    return the_xa, the_Pa, the_F


def _next_time_step_estimating_dynamicF(prev_xa, prev_F, prev_Pa, the_xo, R, dt=0.01, days=0.25,
                                       delta=0.0, delta_of_F=1e-2, KSC=1e+5):
    N = prev_xa.size                # 解析値の地点数
    the_xf = Lorenz96.run_dynamic_F(prev_xa, F=prev_F, dt=dt, days=days)
    M = Lorenz96.jacobian_dynamic_F(prev_xa, dt=dt, days=days, F=prev_F)
    the_Pf = (1 + delta) * M @ prev_Pa @ M.T
    the_Pf[N:, :] = (1 + delta_of_F) * the_Pf[N:, :] / (1 + delta)
    H = ffa.get_H(the_xo, KSC=KSC)
    H = ffa.get_H_for_s(H, prev_F.size)
    K = the_Pf @ H.T @ np.linalg.inv(H @ the_Pf @ H.T + R)
    the_sf = np.hstack((the_xf, prev_F))
    the_xo = ffa.left_aligned(the_xo, KSC=KSC)
    the_sa = the_sf + K @ (the_xo - H @ the_sf)
    the_xa = the_sa[:N]
    the_F = the_sa[N:]

    the_Pa = (np.identity(K.shape[0]) - K @ H) @ the_Pf
    return the_xa, the_F, the_Pa, the_xf, the_Pf
