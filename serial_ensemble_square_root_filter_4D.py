# coding:utf-8


import numpy as np
import func_for_assimilate as ffa
from assimilation import EnsembleKalmanFilter_4D
from serial_ensemble_square_root_filter import KLocalizer


class SerialEnsembleSquareRootFilter_4D(EnsembleKalmanFilter_4D):
    def __init__(self, m=20, N=40, dt=0.05, F_id=0,
                 delta=0.1, sigma=3.0,
                 assim_interval_days=0.25,
                 assim_window_days=1.00,
                 MVid=0, n_of_missn=0, random_state=0):
        super().__init__(m=m, N=N, dt=dt, F_id=F_id,
                         assim_interval_days=assim_interval_days,
                         assim_window_days=assim_window_days,
                         MVid=MVid, n_of_missn=n_of_missn,
                         random_state=random_state)

        self.delta = delta
        self.sigma = sigma
        # modelを作った後にsigmaを書き換えても、k_localizerを書き換わらないので注意
        self.k_localizer = KLocalizer(self.N, self.sigma)

    def _next_time_step(self, prev_Xa, xo_in_AW):
        Xa = prev_Xa
        for j in range(self.J):
            # このfor loopの中で同化ウィンドウ内の観測を取り込んで、Xaをアップデートしていく。

            # 観測が存在するモデルグリッド
            Not_ms = np.arange(self.N)[~np.isnan(xo_in_AW[j])]
            Xf = self.l96.ensemble_run(Xa, days=self.assim_interval_days * (j + 1))

            Xf_bar = np.average(Xf, axis=1)
            dXf = Xf - Xf_bar[:, None]
            Pf = dXf @ dXf.T / (self.m - 1)

            the_xo = ffa.left_aligned(xo_in_AW[j])

            for i in range(self.p):
                # for loop Obs start
                obs_point = Not_ms[i]
                RHO = self.k_localizer.get_rho(obs_point)
                Xa_bar = np.average(Xa, axis=1)
                if i == 0:
                    dXa = np.sqrt(1 + self.delta) * (Xa - Xa_bar[:, None])
                else:
                    dXa = Xa - Xa_bar[:, None]

                # 観測点1点に対応するHの計算
                x_for_making_H = np.full(self.N, np.nan)
                x_for_making_H[obs_point] = 0.0        # 使う観測点1点だけをKSC以外の値に
                H = ffa.get_H(x_for_making_H)  # 観測1点に対応するH
                # 観測点1点に対応するHの計算

                dYf = H @ dXf
                localR = self.R[i, i]    # スカラー
                localPf = Pf[obs_point, obs_point]  # スカラー
                K = RHO[:, None] * dXa @ dYf.T / ((self.m - 1) * (localPf + localR))

                # アンサンブルアップデート(第一推定値の置き換え)
                Xa_bar_new = Xa_bar + K @ (the_xo[i] - H @ Xf_bar)
                alpha = 1.0 / (1.0 + np.sqrt(localR / (localR + localPf)))
                K_childa = alpha * K
                dXa_new = dXa - K_childa @ H @ dXf
                # アンサンブルアップデート

                Xa = Xa_bar_new[:, None] + dXa_new
                # for loop Obs end

        xa = np.average(Xa, axis=1)
        xa_in_AW = self.l96.get_xf_in_AW(xa, self.J, days=self.assim_interval_days)
        the_Xa = self.l96.ensemble_run(Xa, days=self.assim_interval_days * self.J)

        return xa_in_AW, xa, the_Xa
