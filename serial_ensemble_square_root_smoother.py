# coding:utf-8


# 2017/01/17作成。
# Serial EnSRFの実装。
import numpy as np
import func_for_assimilate as ffa
from assimilation import EnsembleKalmanSmoother
from serial_ensemble_square_root_filter import KLocalizer


class SerialEnsembleSquareRootSmoother(EnsembleKalmanSmoother):
    def __init__(self, m=20, N=40, dt=0.05, F_id=0,
                 delta=0.1, sigma=3.0,
                 assim_interval_days=0.25,
                 MVid=0, n_of_missn=0, random_state=0):
        super().__init__(m=m, N=N, dt=dt, F_id=F_id,
                         assim_interval_days=assim_interval_days,
                         MVid=MVid, n_of_missn=n_of_missn,
                         random_state=random_state)
        self.delta = delta
        self.sigma = sigma
        # modelを作った後にsigmaを書き換えても、k_localizerを書き換わらないので注意
        self.k_localizer = KLocalizer(self.N, self.sigma)

    def _next_time_step(self, prev_Xa, next_xo):
        # 時刻6hの観測データを同化して時刻0hにおける解析値を計算する
        Not_ms = np.arange(self.N)[~np.isnan(next_xo)]  # 観測が存在するモデルグリッド

        the_Xf = self.l96.ensemble_run(prev_Xa, days=self.assim_interval_days)
        next_Xf = self.l96.ensemble_run(the_Xf, days=self.assim_interval_days)

        next_xo = ffa.left_aligned(next_xo)
        next_Xf_bar = np.average(next_Xf, axis=1)
        next_dXf = next_Xf - next_Xf_bar[:, None]
        next_Pf = next_dXf @ next_dXf.T / (self.m - 1)

        for i in range(self.p):
            obs_point = Not_ms[i]
            RHO = self.k_localizer.get_rho(obs_point)
            the_Xf_bar = np.average(the_Xf, axis=1)
            if i == 0:
                the_dXf = np.sqrt(1 + self.delta) * (the_Xf - the_Xf_bar[:, None])
            else:
                the_dXf = the_Xf - the_Xf_bar[:, None]

            # 観測点1点に対応するHの計算
            x_for_making_H = np.full(self.N, np.nan)
            x_for_making_H[obs_point] = 0.0        # 使う観測点1点だけをKSC以外の値に
            H = ffa.get_H(x_for_making_H)  # 観測1点に対応するH
            # 観測点1点に対応するHの計算

            next_dYf = H @ next_dXf
            localR = self.R[i, i]    # スカラー
            localPf = next_Pf[obs_point, obs_point]  # スカラー
            K = RHO[:, None] * the_dXf @ next_dYf.T / ((self.m - 1) * (localPf + localR))

            # アンサンブルアップデート(第一推定値の置き換え)
            Xa_bar = the_Xf_bar + K @ (next_xo[i] - H @ next_Xf_bar)
            alpha = 1.0 / (1.0 + np.sqrt(localR / (localR + localPf)))
            K_childa = alpha * K
            dXa = the_dXf - K_childa @ H @ next_dXf
            # アンサンブルアップデート

            the_Xf = Xa_bar[:, None] + dXa

        Xa = Xa_bar[:, None] + dXa

        return Xa
