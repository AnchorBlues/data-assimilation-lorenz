# coding:utf-8

# 2017/01/06作成。
# 4次元変分法を実装する
# 小槻先生がやっていたように、一番最初に作った観測データに対してデータ同化する。

# 思ったんですけれども、これ、F_idが0以外の場合にうまく対応しきれていません。

import numpy as np
from scipy.optimize import minimize
import func_for_assimilate as ffa
import using_jit
from assimilation import Assimilation


class FourDimensionalVariationalMethod(Assimilation):
    def __init__(self, N=40, dt=0.05, F_id=0,
                 alpha=0.3,
                 assim_interval_days=0.25,
                 assim_window_days=1.00,
                 MVid=0, n_of_missn=0, random_state=0):
        super().__init__(N=N, dt=dt, F_id=F_id,
                         assim_interval_days=assim_interval_days,
                         MVid=MVid, n_of_missn=n_of_missn,
                         random_state=random_state)
        self.initial_xa = self.l96.get_spin_upped_profile()
        self._B = np.identity(self.N) * alpha
        self.J = int(round(assim_window_days /
                           assim_interval_days, 0))
        # Jは、同化ウィンドウ内に観測が得られる回数。Obs_Coming_Times_in_AW。
        self.LMAX_for_4D = int(self.LMAX / self.J)
        # RMSE of analysis value
        self.RMSE_a_AW = np.zeros(self.LMAX_for_4D)
        # RMSE of reanalysis value
        self.RMSE_rea_AW = np.zeros(self.LMAX_for_4D)
        self.optimization_method = 'l-bfgs'
        self.verbose_opt = False
        self.analysis = 1
        self.Binv = np.linalg.inv(self._B)
        self.Rinv = np.linalg.inv(self.R)

    def set_B(self, B):
        self._B = B
        self.Binv = np.linalg.inv(self._B)

    B = property((lambda self: self._B), set_B)

    def for_loop(self, verbose=False):
        idx = range(self.J)
        self.Xa[idx], xa0 = self._next_time_step(self.initial_xa,
                                                 self.Xo[idx])
        self.RMSE_rea_AW[0] = np.nan
        for l in range(1, self.LMAX_for_4D):
            # self.J=5、つまり同化ウィンドウ内で5つ観測を取り込むとすると、
            # 時刻l-1に於ける解析値と時刻l, ..., l+4における観測値をもとに、
            # 時刻l, ..., l+4における解析値(self.Xa[idx])を計算する。
            # xa0というのは、時刻l-1における"再"解析値。
            # RMSE_a_AWは、時刻l+4(l-1)における「解析値」のRMSEで、
            # RMSE_rea_AWは、時刻l+4(l-1)における「再解析値」のRMSE

            # lに対応するインデックス
            idx = range(l * self.J, (l + 1) * self.J)

            self.Xa[idx], xa0 = self._next_time_step(self.Xa[l * self.J - 1],
                                                     self.Xo[idx])
            self.RMSE_rea_AW[l] = \
                using_jit.cal_RMSE(xa0, self.Xt[l * self.J - 1])

            if verbose and l % 100 == 0:
                print("l=" + str(l) + ', rmse=' + str(self.RMSE_rea_AW[l]))

        self.RMSE_a[:] = using_jit.cal_RMSE_2D(self.Xa, self.Xt)
        # for l in range(self.LMAX_for_4D):
        #     self.RMSE_a_AW[l] = self.RMSE_a[(l + 1) * self.J - 1]
        # 上記のコードを最適化すると以下の通り
        self.RMSE_a_AW[:] \
            = self.RMSE_a[(np.arange(self.LMAX_for_4D) + 1) * self.J - 1]

    def _next_time_step(self, prev_xa, xo_in_AW):
        # jedit_flg = 0の時は何も表示させない。 = 1の時は評価関数Jが小さくなっている様子を表示させる。
        days = self.assim_interval_days
        J = self.J
        N = self.N
        p = self.p

        H_in_AW = np.zeros((J, p, N))
        xo_in_AW_aligned = np.zeros((J, p))

        for j in range(J):
            xo_in_AW_aligned[j] = ffa.left_aligned(xo_in_AW[j])
            H_in_AW[j] = ffa.get_H(xo_in_AW[j])

        # 参考：http://org-technology.com/posts/scipy-unconstrained-minimization-of-multivariate-scalar-functions.html
        def ObjectiveFunction(x):
            xf_in_AW = self.l96.get_xf_in_AW(x, J, days)
            obs_term = 0.0
            for j in range(J):
                d = H_in_AW[j] @ xf_in_AW[j] - xo_in_AW_aligned[j]
                obs_term += 0.5 * d @ self.Rinv @ d

            return 0.5 * (x - prev_xa) @ self.Binv @ (x - prev_xa) + obs_term

        def gradient(x):
            xf_in_AW = self.l96.get_xf_in_AW(x, J, days)
            M_in_AW = self.l96.get_M_in_AW(x, J, days,
                                           analysis=self.analysis)
            obs_term = 0.0
            for j in range(J):
                Mj = M_in_AW[j]
                Hj = H_in_AW[j]
                # 線形アジョイントモデル
                obs_term += Mj.T @ Hj.T @ self.Rinv @ (
                    Hj @ xf_in_AW[j] - xo_in_AW_aligned[j])

            return self.Binv @ (x - prev_xa) + obs_term

        # 初期値はランダム。ただし収束を速くするために、prev_xaとする
        x0 = prev_xa

        if self.optimization_method == 'l-bfgs':
            # scipyを用いて最適化し、ObjectiveFunctionを最小にするxを求める。それがx0になる
            res = minimize(ObjectiveFunction, x0,
                           jac=gradient, method='l-bfgs-b')
            x0 = res.x
        elif self.optimization_method == 'steepest':
            # 最急降下法を自分で実装してやってみた。
            x0 = self.steepest_descent_method(gradient, x0)
        else:
            raise ValueError("your optimization method is not valid!")

        xa_in_AW = self.l96.get_xf_in_AW(x0, J, days)

        if abs(xa_in_AW[J - 1, 0] - self.l96.run(x0, J * days)[0]) > 1e-14:
            raise ValueError('your calculation is not valid!')

        return xa_in_AW, x0

    def steepest_descent_method(self, gradient, initial_x):
        # 最急降下法を生コードから実装します。
        # http://soy-curd.hatenablog.com/entry/2016/05/05/151517
        # http://minus9d.hatenablog.com/entry/2015/01/25/210958
        lr = 0.1
        loop_max = 1000
        eps = 1e-4

        xb = initial_x
        for i in range(loop_max):
            xa = xb - lr / np.sqrt(i + 1) * gradient(xb)
            if i % 10 == 0 and using_jit.cal_RMSE(xa, xb) < eps:
                if self.verbose_opt:
                    print('i=' + str(i))

                break

            xb = xa

        return xa
