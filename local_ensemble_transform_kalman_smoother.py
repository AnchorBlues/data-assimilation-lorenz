# coding:utf-8

# LETKFスムーザーを実装する
import numpy as np
import func_for_assimilate as ffa
from assimilation import EnsembleKalmanSmoother


class LocalEnsembleTransformKalmanSmoother(EnsembleKalmanSmoother):
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

    def _next_time_step(self, prev_Xa, next_xo):
        H = ffa.get_H(next_xo)
        next_xo = ffa.left_aligned(next_xo)

        the_Xf = self.l96.ensemble_run(prev_Xa, days=self.assim_interval_days)
        the_Xf_bar = np.average(the_Xf, axis=1)
        the_dXf = the_Xf - the_Xf_bar[:, None]

        next_Xf = self.l96.ensemble_run(the_Xf, days=self.assim_interval_days)
        next_Xf_bar = np.average(next_Xf, axis=1)
        next_dXf = next_Xf - next_Xf_bar[:, None]
        next_dYf = H @ next_dXf

        A = (self.m - 1) / self.rho * np.identity(self.m) + next_dYf.T @ self.Rinv @ next_dYf
        U, D = ffa.Eigenvalue_decomp(A)
        Dinv = np.diag(1. / np.diag(D))  # Dは対角行列より、np.linalg.invよりこっちのほうが高速
        sqrtDinv = np.sqrt(Dinv)  # Dinvは対角行列より、np.sqrtでMatrix square rootが求まる

        K = the_dXf @ U @ Dinv @ U.T @ next_dYf.T @ self.Rinv
        T = np.sqrt(self.m - 1) * U @ sqrtDinv @ U.T
        Kxoxf = K @ (next_xo - H @ next_Xf_bar)

        # すべての列に同じ配列を格納している。(N, m)の行列になる。
        # 以下のコードの最適化。
        # the_Xf_bar_MAT = np.zeros((self.N, self.m))
        # Kxoxf_MAT = np.zeros((self.N, self.m))
        # for k in range(self.m):
        #     the_Xf_bar_MAT[:, k] = the_Xf_bar[:]
        #     Kxoxf_MAT[:, k] = Kxoxf[:]
        the_Xf_bar_MAT = np.repeat(the_Xf_bar[:, None], self.m, axis=1)
        Kxoxf_MAT = np.repeat(Kxoxf[:, None], self.m, axis=1)

        Xa = the_Xf_bar_MAT + Kxoxf_MAT + the_dXf @ T

        return Xa
