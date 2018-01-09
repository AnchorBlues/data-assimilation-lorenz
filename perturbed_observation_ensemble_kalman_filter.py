# coding:utf-8

# 2017/01/06作成。
import numpy as np
import func_for_assimilate as ffa
from assimilation import EnsembleKalmanFilter


class PerturbedObservationEnsembleKalmanFilter(EnsembleKalmanFilter):
    def __init__(self, m=20, N=40, dt=0.05, F_id=0, alpha=1.0, delta=0.1,
                 assim_interval_days=0.25,
                 MVid=0, n_of_missn=0, random_state=0):
        super().__init__(m=m, N=N, dt=dt, F_id=F_id,
                         assim_interval_days=assim_interval_days,
                         MVid=MVid, n_of_missn=n_of_missn,
                         random_state=random_state)

        self.delta = delta
        self.alpha = alpha

    def _next_time_step(self, prev_Xa, the_xo):
        H = ffa.get_H(the_xo)

        Xf = self.l96.ensemble_run(prev_Xa, days=self.assim_interval_days)

        Xf_bar = np.average(Xf, axis=1)

        # broadcasting
        dXf = np.sqrt(1 + self.delta) * (Xf - Xf_bar[:, None])
        dYf = H.dot(dXf)
        # broadcasting

        K = dXf @ dYf.T @ np.linalg.inv(dYf @ dYf.T + (self.m - 1) * self.R)
        # K = dXf.dot(dYf.T.dot(np.linalg.inv(dYf.dot(dYf.T) + (self.m - 1) * self.R)))

        the_xo = ffa.left_aligned(the_xo)

        # Xa = np.zeros((self.N, self.m))
        # for k in range(self.m):
        #     # Perturbation。理論上はalpha = 1
        #     e = self.alpha * np.random.randn(self.p)
        #     Kxoxf = K @ (the_xo + e - H @ Xf[:, k])
        #     # Kxoxf = K @ (the_xo + e - H @ Xf[:, k].reshape[0]).reshape(K.shape[0])
        #     # Kxoxf = K.dot(
        #     #     the_xo + e - H.dot(Xf[:, k]).reshape(H.shape[0])).reshape(K.shape[0])
        #     Xa[:, k] = Xf[:, k] + Kxoxf
        # 上記のコードを最適化すると以下のとおりとなる
        e_mat = self.alpha * np.random.randn(self.p, self.m)
        Xa = Xf + K @ (the_xo[:, None] + e_mat - H @ Xf)

        return Xa
