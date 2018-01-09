# coding:utf-8

# 4次元のLETKFを実装する
import numpy as np
import func_for_assimilate as ffa
from assimilation import EnsembleKalmanFilter_4D


class LocalEnsembleTransformKalmanFilter_4D(EnsembleKalmanFilter_4D):
    def __init__(self, m=20, N=40, dt=0.05, F_id=0,
                 rho=1.1,
                 assim_interval_days=0.25,
                 assim_window_days=1.00,
                 MVid=0, n_of_missn=0, random_state=0):
        super().__init__(m=m, N=N, dt=dt, F_id=F_id,
                         assim_interval_days=assim_interval_days,
                         assim_window_days=assim_window_days,
                         MVid=MVid, n_of_missn=n_of_missn,
                         random_state=random_state)

        self.rho = rho
        self.Rinv = np.linalg.inv(self.R)

    def _next_time_step(self, prev_Xa, xo_in_AW):
        Xa = prev_Xa
        for j in range(self.J):
            # このfor loopの中で同化ウィンドウ内の観測を取り込んで、Xaをアップデートしていく。
            H = ffa.get_H(xo_in_AW[j])
            the_xo = ffa.left_aligned(xo_in_AW[j])

            Xa_bar = np.average(Xa, axis=1)
            dXa = Xa - Xa_bar[:, None]

            Xf = self.l96.ensemble_run(Xa, days=self.assim_interval_days * (j + 1))
            Xf_bar = np.average(Xf, axis=1)
            dXf = Xf - Xf_bar[:, None]
            dYf = H @ dXf

            A = (self.m - 1) / self.rho * np.identity(self.m) + dYf.T @ self.Rinv @ dYf
            U, D = ffa.Eigenvalue_decomp(A)
            Dinv = np.diag(1. / np.diag(D))  # Dは対角行列より、np.linalg.invよりこっちのほうが高速
            sqrtDinv = np.sqrt(Dinv)  # Dinvは対角行列より、np.sqrtでMatrix square rootが求まる

            K = dXa @ U @ Dinv @ U.T @ dYf.T @ self.Rinv
            T = np.sqrt(self.m - 1) * U @ sqrtDinv @ U.T
            Kxoxf = K @ (the_xo - H @ Xf_bar)

            # すべての列に同じ配列を格納している。(N, m)の行列になる。
            # 以下のコードの最適化。
            # Xa_bar_MAT = np.zeros((self.N, self.m))
            # Kxoxf_MAT = np.zeros((self.N, self.m))
            # for k in range(self.m):
            #     Xa_bar_MAT[:, k] = Xa_bar[:]
            #     Kxoxf_MAT[:, k] = Kxoxf[:]
            Xa_bar_MAT = np.repeat(Xa_bar[:, None], self.m, axis=1)
            Kxoxf_MAT = np.repeat(Kxoxf[:, None], self.m, axis=1)

            Xa = Xa_bar_MAT + Kxoxf_MAT + dXa @ T

        xa = np.average(Xa, axis=1)
        xa_in_AW = self.l96.get_xf_in_AW(xa, self.J, days=self.assim_interval_days)
        the_Xa = self.l96.ensemble_run(Xa, days=self.assim_interval_days * self.J)

        return xa_in_AW, xa, the_Xa
