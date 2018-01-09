# coding:utf-8


# 2017/01/17作成。
# Serial EnSRFの実装。
import numpy as np
import func_for_assimilate as ffa
from assimilation import EnsembleKalmanFilter


class SerialEnsembleSquareRootFilter(EnsembleKalmanFilter):
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

    def _next_time_step(self, prev_Xa, the_xo):
        Not_ms = np.arange(self.N)[~np.isnan(the_xo)]  # 観測が存在するモデルグリッド

        Xf = self.l96.ensemble_run(prev_Xa, days=self.assim_interval_days)

        the_xo = ffa.left_aligned(the_xo)

        for i in range(self.p):
            obs_point = Not_ms[i]
            RHO = self.k_localizer.get_rho(obs_point)
            Xf_bar = np.average(Xf, axis=1)
            if i == 0:
                dXf = np.sqrt(1 + self.delta) * (Xf - Xf_bar[:, None])
            else:
                dXf = Xf - Xf_bar[:, None]

            Pf = dXf @ dXf.T / (self.m - 1)

            # 観測点1点に対応するHの計算
            x_for_making_H = np.full(self.N, np.nan)
            x_for_making_H[obs_point] = 0.0        # 使う観測点1点だけをKSC以外の値に
            H = ffa.get_H(x_for_making_H)  # 観測1点に対応するH
            # 観測点1点に対応するHの計算

            dYf = H @ dXf
            localR = self.R[i, i]    # スカラー
            localPf = Pf[obs_point, obs_point]  # スカラー
            K = RHO[:, None] * dXf @ dYf.T / \
                ((self.m - 1) * (localPf + localR))

            # アンサンブルアップデート(第一推定値の置き換え)
            Xa_bar = Xf_bar + K @ (the_xo[i] - H @ Xf_bar)
            alpha = 1.0 / (1.0 + np.sqrt(localR / (localR + localPf)))
            K_childa = alpha * K
            dXa = (np.identity(self.N) - K_childa @ H) @ dXf
            # アンサンブルアップデート

            Xf = Xa_bar[:, None] + dXa

        Xa = Xa_bar[:, None] + dXa

        return Xa


def generate_rho_for_Klocalization(sigma, k, N=40):
    # localizationの時に掛け合わせる行列RHOを計算する。
    # sigmaは、RHOをガウス関数で与える際の標準偏差の値。
    # sigmaの範囲は、0.5~10.0とする(そこから大きくずれた場合、RHOの形が崩れる)
    if sigma is None:
        # sigmaがNoneの時はlocalizeしない。
        # RHOはすべての要素が1であるベクトルにする。
        return np.ones(N)

    if sigma > 10.0:
        raise Exception("Error! your sigma is too big!")
    elif sigma < 0.5:
        raise Exception("Error! your sigma is too small!")

    # 2 * Nに特に意味はない。Nよりある程度大きければいい。
    x = np.arange(-2 * N, 2 * N)
    rho_tmp = np.exp(-((k - x) ** 2) / (2.0 * sigma ** 2))
    rho = 0
    for i in range(3):
        # i=1のときが一番メインとなるsummationになる。
        # i=0, i=2は、kが端っこだった時にちゃんとやってくれるようにする。
        rho += rho_tmp[(i + 1) * N:(i + 2) * N]

    return rho


class KLocalizer(object):
    def __init__(self, N, sigma):
        self.N = N
        self.sigma = sigma
        self._dct = {}

    def get_rho(self, k):
        """
        ローカライズする1次元配列rhoを返す
        """
        if self.sigma is None:
            # Noneは辞書に登録できないためすぐreturnする
            return generate_rho_for_Klocalization(sigma=self.sigma,
                                                  k=k,
                                                  N=self.N)

        if k not in self._dct:
            self._dct[k] = generate_rho_for_Klocalization(sigma=self.sigma,
                                                          k=k,
                                                          N=self.N)

        return self._dct[k]
