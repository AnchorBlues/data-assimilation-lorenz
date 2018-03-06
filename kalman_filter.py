# coding:utf-8


# 2017/01/03作成。
# Assimilate.py,Lorenz96.pyなどと整合的なコードに。

import numpy as np
import func_for_assimilate as ffa
import using_jit
from assimilation import Assimilation


class KalmanFilter(Assimilation):
    def __init__(self, N=40, dt=0.05, F_id=0, delta=0.1,
                 assim_interval_days=0.25,
                 MVid=0, n_of_missn=0, random_state=0):
        super().__init__(N=N, dt=dt, F_id=F_id,
                         assim_interval_days=assim_interval_days,
                         MVid=MVid, n_of_missn=n_of_missn,
                         random_state=random_state)

        self.initial_xa = self.l96.get_spin_upped_profile()
        self.initial_Pa = np.identity(self.N) * 1e+1
        self.delta = delta
        self.Cov_for_est_Pf = np.zeros((self.LMAX, self.p, self.p))
        self.Cov_for_est_Pf2 = np.zeros((self.LMAX, self.p, self.p))
        self.Cov_for_est_R = np.zeros((self.LMAX, self.p, self.p))
        self.est_R = np.zeros((self.p, self.p))
        self.est_Pf = np.zeros((self.p, self.p))
        self.est_Pf2 = np.zeros((self.p, self.p))
        self.Pa = np.zeros((self.LMAX, self.N, self.N))
        self.Pf = np.zeros((self.LMAX, self.N, self.N))
        self.ave_Pa = np.zeros((self.N, self.N))
        self.ave_Pf = np.zeros((self.N, self.N))
        self.l = None

    def for_loop(self):
        self.l = 0
        self.Xa[0], self.Pa[0] = self._next_time_step(self.initial_xa,
                                                      self.initial_Pa,
                                                      self.Xo[0])
        for l in range(1, self.LMAX):
            self.l = l
            self.Xa[l], self.Pa[l] = self._next_time_step(self.Xa[l - 1],
                                                          self.Pa[l - 1],
                                                          self.Xo[l])

        self.RMSE_a[:] = using_jit.cal_RMSE_2D(self.Xa, self.Xt)
        self.est_R = np.average(self.Cov_for_est_R[200:], axis=0)
        self.est_Pf = np.average(self.Cov_for_est_Pf[200:], axis=0)
        self.est_Pf2 = np.average(self.Cov_for_est_Pf2[200:], axis=0)
        self.ave_Pa = np.average(self.Pa[200:], axis=0)
        self.ave_Pf = np.average(self.Pf[200:], axis=0)

    def _next_time_step(self, prev_xa, prev_Pa, the_xo):
        # 予報
        the_xf = self.l96.run(prev_xa, days=self.assim_interval_days)

        # カルマンゲイン計算
        M = self.l96.jacobian(prev_xa, days=self.assim_interval_days)
        MPM = M @ prev_Pa @ M.T
        the_Pf = (1 + self.delta) * MPM
        H = ffa.get_H(the_xo)
        K = the_Pf @ H.T @ np.linalg.inv(H @ the_Pf @ H.T + self.R)

        # 解析値算出
        the_xo = ffa.left_aligned(the_xo)
        d_ob = the_xo - H @ the_xf
        the_xa = the_xf + K @ d_ob
        the_Pa = (np.identity(self.N) - K @ H) @ the_Pf

        # inflation deltaの推定
        d_ab = the_xa - the_xf
        Cov_for_est_Pf1 = using_jit.cal_covmat(H @ d_ab, d_ob)
        Cov_for_est_Pf2 = using_jit.cal_covmat(d_ob, d_ob) - self.R

        # Rの推定
        d_oa = the_xo - H @ the_xa
        Cov_for_est_R = using_jit.cal_covmat(d_oa, d_ob)

        # 推定値保存
        self.Cov_for_est_Pf[self.l] = Cov_for_est_Pf1
        self.Cov_for_est_Pf2[self.l] = Cov_for_est_Pf2
        self.Cov_for_est_R[self.l] = Cov_for_est_R
        self.Pf[self.l] = the_Pf

        return the_xa, the_Pa
