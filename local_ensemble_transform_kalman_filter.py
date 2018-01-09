# coding:utf-8

# LETKFを実装する
import numpy as np
import func_for_assimilate as ffa
from assimilation import EnsembleKalmanFilter


class LocalEnsembleTransformKalmanFilter(EnsembleKalmanFilter):
    def __init__(self, m=20, N=40, dt=0.05, F_id=0,
                 rho=1.1,
                 assim_interval_days=0.25,
                 MVid=0, n_of_missn=0, random_state=0):
        super().__init__(m=m, N=N, dt=dt, F_id=F_id,
                         assim_interval_days=assim_interval_days,
                         MVid=MVid, n_of_missn=n_of_missn,
                         random_state=random_state)

        self.rho = rho
        self.Rinv = np.linalg.inv(self.R)

    def _next_time_step(self, prev_Xa, the_xo):
        H = ffa.get_H(the_xo)

        Xf = self.l96.ensemble_run(prev_Xa, days=self.assim_interval_days)
        Xf_bar = np.average(Xf, axis=1)

        the_xo = ffa.left_aligned(the_xo)

        # broadcasting
        dXf = Xf - Xf_bar[:, None]
        dYf = H @ dXf
        # broadcasting

        A = (self.m - 1) / self.rho * \
            np.identity(self.m) + dYf.T @ self.Rinv @ dYf
        U, D = ffa.Eigenvalue_decomp(A)
        Dinv = np.diag(1. / np.diag(D))  # Dは対角行列より、np.linalg.invよりこっちのほうが高速
        sqrtDinv = np.sqrt(Dinv)  # Dinvは対角行列より、np.sqrtでMatrix square rootが求まる

        K = dXf @ U @ Dinv @ U.T @ dYf.T @ self.Rinv
        T = np.sqrt(self.m - 1) * U @ sqrtDinv @ U.T
        Kxoxf = K @ (the_xo - H @ Xf_bar)

        # すべての列に同じ配列を格納している。(N, m)の行列になる。
        # 以下のコードの最適化。
        # Xf_bar_MAT = np.zeros((self.N, self.m))
        # Kxoxf_MAT = np.zeros((self.N, self.m))
        # for k in range(self.m):
        #     Xf_bar_MAT[:, k] = Xf_bar[:]
        #     Kxoxf_MAT[:, k] = Kxoxf[:]
        Xf_bar_MAT = np.repeat(Xf_bar[:, None], self.m, axis=1)
        Kxoxf_MAT = np.repeat(Kxoxf[:, None], self.m, axis=1)

        Xa = Xf_bar_MAT + Kxoxf_MAT + dXf @ T

        return Xa
