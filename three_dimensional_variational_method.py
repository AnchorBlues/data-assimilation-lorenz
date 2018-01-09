# coding:utf-8


# 2017/01/06作成。
# Assimilate.py,Lorenz96.pyなどと整合的なコードに。

import numpy as np
import func_for_assimilate as ffa
import using_jit
from assimilation import Assimilation


class ThreeDimensionalVariationalMethod(Assimilation):
    def __init__(self, N=40, dt=0.05, F_id=0, alpha=0.3,
                 assim_interval_days=0.25,
                 MVid=0, n_of_missn=0, random_state=0):
        super().__init__(N=N, dt=dt, F_id=F_id,
                         assim_interval_days=assim_interval_days,
                         MVid=MVid, n_of_missn=n_of_missn,
                         random_state=random_state)

        self.initial_xa = self.l96.get_spin_upped_profile()
        self.B = np.identity(self.N) * alpha
        self.Cov_for_est_B = np.zeros((self.LMAX, self.p, self.p))
        self.Cov_for_est_B2 = np.zeros((self.LMAX, self.p, self.p))
        self.Cov_for_est_R = np.zeros((self.LMAX, self.p, self.p))
        self.est_B = np.zeros((self.p, self.p))
        self.est_B2 = np.zeros((self.p, self.p))
        self.est_R = np.zeros((self.p, self.p))

    def for_loop(self):
        for l in range(self.LMAX):
            if l == 0:
                self.Xa[l], self.Cov_for_est_B[l], \
                    self.Cov_for_est_B2[l], \
                    self.Cov_for_est_R[l] = self._next_time_step(self.initial_xa,
                                                                 self.Xo[l])
            else:
                self.Xa[l], self.Cov_for_est_B[l], \
                    self.Cov_for_est_B2[l], \
                    self.Cov_for_est_R[l] = self._next_time_step(self.Xa[l - 1],
                                                                 self.Xo[l])

        self.RMSE_a[:] = using_jit.cal_RMSE_2D(self.Xa, self.Xt)
        self.est_B = np.average(self.Cov_for_est_B[200:], axis=0)
        self.est_B2 = np.average(self.Cov_for_est_B2[200:], axis=0)
        self.est_R = np.average(self.Cov_for_est_R[200:], axis=0)

    def _next_time_step(self, prev_xa, the_xo):
        the_xf = self.l96.run(prev_xa, days=self.assim_interval_days)
        H = ffa.get_H(the_xo)
        K = self.B @ H.T @ (np.linalg.inv(H @ self.B @ H.T + self.R))
        the_xo = ffa.left_aligned(the_xo)
        d_ob = the_xo - H @ the_xf
        the_xa = the_xf + K @ d_ob

        # Bの推定
        # なお、BそのものではなくHBH.Tを推定する。
        d_ab = the_xa - the_xf
        est_B1 = using_jit.cal_covmat(H @ d_ab, d_ob)
        est_B2 = using_jit.cal_covmat(d_ob, d_ob) - self.R

        # Rの推定
        d_oa = the_xo - H @ the_xa
        Cov_for_est_R = using_jit.cal_covmat(d_oa, d_ob)

        return the_xa, est_B1, est_B2, Cov_for_est_R
