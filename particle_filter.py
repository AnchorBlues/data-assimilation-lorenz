# coding:utf-8


# 2017/12/20作成。

import numpy as np
import func_for_assimilate as ffa
import using_jit
from assimilation import Assimilation


def get_w(lkh_mat):
    """
    lkh_mat(p, m)に対してget_w
    観測点毎にアンサンブル方向の正規化を行う
    """
    sum = np.sum(lkh_mat, axis=1)
    return (lkh_mat.T / sum).T


def resampling(Xf, w, indices_not_ms):
    """
    Xfから、重みwで「重み付き復元抽出」を行う
    numbaオプションを付けるとLoweringエラーという謎のエラーが出る
    観測があるグリッドの個数のp、アンサンブル数をmとすると、
    w.shape == (p, m)
    Xf.shape == (N, m)
    """
    assert Xf.ndim == w.ndim == 2
    p, m = w.shape
    Xa = np.copy(Xf)
    for i in range(p):
        w_1col = w[i]
        rand = np.random.rand(m)  # 一様乱数をm個生成
        # 以下、ルーレットを回すイメージ。rand[i]が、ルーレットの止まった場所。その場所のw[l]に対応するlを返す
        cum_w = np.cumsum(w_1col)

        # cum_w < v を満たすcum_wの個数
        # indices = np.array(
        #     list(map(lambda v: np.sum(cum_w < v).astype(np.int), rand)))
        # 上記のコードを最適化すると以下の通りになる。
        indices = np.sum(cum_w < rand[:, None], axis=1)

        target_xf = Xf[indices_not_ms[i]]
        Xa[indices_not_ms[i]] = target_xf[indices]
    return Xa


class ParticleFilter(Assimilation):
    """
    観測があるグリッドに関してのみ、SIRによるリサンプリングを行う。
    観測がないグリッドに関しては、予報値を解析値だと思うことにする。
    """

    def __init__(self, m=20, N=40, dt=0.05, F_id=0,
                 assim_interval_days=0.25,
                 MVid=0, n_of_missn=0, delta=0.1, stop_l=None,
                 random_state=0):
        super().__init__(N=N, dt=dt, F_id=F_id,
                         assim_interval_days=assim_interval_days,
                         MVid=MVid,
                         n_of_missn=n_of_missn,
                         random_state=random_state)
        self.m = m
        self.initial_Enxa = self.l96.get_initial_Xa(m)
        self.w_timeseries = np.zeros((self.LMAX, self.p, self.m))
        self.Xf = np.zeros_like(self.Xa)
        self.Enxa = np.zeros((self.LMAX, self.N, self.m))
        self.Enxf = np.zeros_like(self.Enxa)
        self.delta = delta
        self.diagR = np.diag(self.R)
        if stop_l is None:
            self.stop_l = self.LMAX
        else:
            self.stop_l = stop_l

    def for_loop(self):
        for l in range(self.LMAX):
            self.l = l
            if l == 0:
                self.Enxa[l] = self._next_time_step(self.initial_Enxa,
                                                    self.Xo[l])
            else:
                self.Enxa[l] = self._next_time_step(self.Enxa[l - 1],
                                                    self.Xo[l])

            if self.stop_l == l:
                break

        self.Xa = np.average(self.Enxa, axis=2)
        self.RMSE_a[:] = using_jit.cal_RMSE_2D(self.Xa, self.Xt)

    def likelihood(self, Xf, y, H):
        """
        予報値(xf)と観測データ(y)から尤度p(y|xf)を計算する。
        Rは観測誤差分散。
        """
        HXf = H @ Xf  # モデルの結果を観測値のグリッドに落とす(n * m)
        return using_jit.likelihood_(HXf, y, self.diagR, self.p, self.m)

    def _next_time_step(self, prev_Enxa, the_xo):
        # 予報
        Xf = self.l96.ensemble_run(prev_Enxa, days=self.assim_interval_days)
        # 予測にノイズを加える(発散対策)
        Xf += self.delta * np.random.randn(*Xf.shape)
        # 観測行列作成・xoを欠損値抜きにして寄せる。
        H = ffa.get_H(the_xo)
        # 観測があるグリッドのインデックス
        indices_not_ms = np.arange(self.N)[~np.isnan(the_xo)]
        the_xo = ffa.left_aligned(the_xo)

        # 各アンサンブル(粒子)に対して尤度を求める
        lkhs = self.likelihood(Xf, the_xo, H)

        # 重みwの計算
        w = get_w(lkhs)

        # Xf[:, i]から、重みwによって「重み付きサンプリング」することで解析値を求める
        Xa = resampling(Xf, w, indices_not_ms)

        # メンバ変数に保存
        self.w_timeseries[self.l] = w
        self.Enxf[self.l] = Xf
        self.Xf[self.l] = np.average(Xf, axis=1)
        return Xa
