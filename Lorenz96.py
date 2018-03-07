# coding:utf-8


# 2017/01/02作成。
import random
from numba import jit, f8, i1
import numpy as np

# Lorenz96モデルでのdt=1.0が、天気予報(実時間)での5.0日に対応。1日<->dt=0.2, 6時間<->dt=0.05
Lorenz96_TimeScale = 5.0


def initial_taskx(N):  # 課題演習に取り組む際に用いる初期値を作成する

    # 初期値を作成する
    F = 8
    x = np.zeros(N)
    x[:] = F
    x[int(N / 2)] = F * (1 + 1e-3)
    # 初期値を作成する
    return x, F


@jit(f8[:](f8[:], f8, f8))
def cal(x, F=8, dt=0.01):
    # 時間ステップを一つ走らせて、xからxaを得る。

    N = len(x)
    xa = np.zeros(N)
    halfdt = dt / 2.0

    for i in range(N):

        if i == 0:
            a = x[N - 2]
            b = x[N - 1]
            c = x[0]
            d = x[1]
        elif i == 1:
            a = x[N - 1]
            b = x[0]
            c = x[1]
            d = x[2]
        elif i == N - 1:
            a = x[N - 3]
            b = x[N - 2]
            c = x[N - 1]
            d = x[0]
        else:
            a = x[i - 2]
            b = x[i - 1]
            c = x[i]
            d = x[i + 1]

        k1 = f(a, b, c, d, F)
        k2 = f(a + halfdt * k1, b + halfdt * k1,
               c + halfdt * k1, d + halfdt * k1, F)
        k3 = f(a + halfdt * k2, b + halfdt * k2,
               c + halfdt * k2, d + halfdt * k2, F)
        k4 = f(a + dt * k3, b + dt * k3, c + dt * k3, d + dt * k3, F)
        xa[i] = c + dt / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    return xa


@jit(f8(f8, f8, f8, f8, f8))
def f(a, b, c, d, F):
    return - a * b + b * d - c + F


@jit(f8[:](f8[:], f8, f8))
def cal_oigawa(x, F=8, dt=0.01):
    k1 = f_oigawa(x, F)
    k2 = f_oigawa(x + 0.5 * dt * k1, F)
    k3 = f_oigawa(x + 0.5 * dt * k2, F)
    k4 = f_oigawa(x + dt * k3, F)
    return x + dt / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


@jit(f8[:](f8[:], f8))
def f_oigawa(x, F):
    N = x.size
    xa = np.zeros(N)
    for i in range(N):
        xa[i] = -x[(i - 2 + N) % N] * x[(i - 1 + N) % N] + \
            x[(i - 1 + N) % N] * x[(i + 1) % N] - x[i] + F

    return xa


@jit(f8[:, :](f8[:]))
def dfdx(x):
    # f(x) = dx/dt = ... の式をxで微分したもの。
    N = x.size
    df = np.zeros((N, N))
    for i in range(N):
        df[i, (i - 2 + N) % N] = -x[(i - 1 + N) % N]
        df[i, (i - 1 + N) % N] = -x[(i - 2 + N) % N] \
            + x[(i + 1 + N) % N]
        df[i, i] = -1
        df[i, (i + 1 + N) % N] = x[(i - 1 + N) % N]

    return df


class Lorenz96_RungeKutta4_Code(object):
    def __init__(self, ID, Func):
        self.ID = ID
        self.Func = Func

    def cal(self, F, dt, IMAX):
        xb, F = initial_taskx(40)
        for i in range(IMAX):
            xa = self.Func(xb, F, dt)
            xb = xa

        return xa


Code = [0] * 2
Code[0] = Lorenz96_RungeKutta4_Code(0, cal)
Code[1] = Lorenz96_RungeKutta4_Code(1, cal_oigawa)


@jit(f8[:](f8[:], f8, f8, f8))
def run(x, F=8, dt=0.01, days=0.25):
    # 日数daysだけ、モデルを走らせる。
    # 0.2時間ステップが1日に対応するので、days = 0.25(0.25日)の場合は0.05時間ステップだけモデルを走らせる。

    t = days / Lorenz96_TimeScale
    imax = int(round(t / dt, 0))
    xb = x
    for i in range(imax):
        xa = cal_oigawa(xb, F=F, dt=dt)
        xb = xa

    return xb


@jit(f8[:](f8[:], f8, f8, f8))
def adjoint_run(x, F=8.0, dt=0.01, days=0.25):
    # アジョイントモデルを走らせる
    t = days / Lorenz96_TimeScale
    imax = int(round(t / dt, 0))
    xb = x
    for i in range(imax):
        xa = cal_oigawa(xb, F=F, dt=- dt)
        xb = xa

    return xb


@jit(f8[:, :](f8[:], f8, f8, f8))
def jacobian(x, dt=0.01, days=0.25, F=8.0):
    # dt = dtで時間tの間モデルを走らせた結果に対応するヤコビアンを求める。
    # 注意！ここでのtの単位は「日」。
    # dt = 0.01の方は数値計算の単位。
    # 0.2時間ステップが1日に対応するので、days = 0.25は0.25日、つまり6時間に対応する。
    # 0.01時間ステップは0.05日、つまり1.2時間に対応する。
    # dt = 0.01、days = 0.25とすると、0.25 / 0.05 = 5回計算を走らせるということになる。

    N = x.size
    d = 1e-4

    if days == 0.0:
        return np.identity(N)

    Jacob = np.zeros((N, N))
    xb = run(x, F=F, dt=dt, days=days)

    for i in range(N):
        x_dash = np.zeros(N)
        x_dash[:] = x[:]

        x_dash[i] = x[i] + d
        xa = run(x_dash, F=F, dt=dt, days=days)

        Jacob[:, i] = (xa[:] - xb[:]) / d

    return Jacob


@jit(f8[:, :](f8[:], f8, f8, f8))
def jacobian_analysis(x, dt=0.01, days=0.25, F=8.0):
    # ヤコビ行列を解析的に計算します
    N = x.size
    if days == 0:
        return np.identity(N)

    t = days / Lorenz96_TimeScale
    imax = int(round(t / dt, 0))
    I = np.identity(N)
    dt05 = 0.5 * dt
    dt10 = dt
    jac = I
    for _ in range(imax):
        DK0 = dfdx(x)
        k0 = f_oigawa(x, F)
        # DK1 = (I + dt05 * DK0) @ dfdx(x + dt05 * k0)
        DK1 = dfdx(x + dt05 * k0) @ (I + dt05 * DK0)
        k1 = f_oigawa(x + dt05 * k0, F)
        # DK2 = (I + dt05 * DK1) @ dfdx(x + dt05 * k1)
        DK2 = dfdx(x + dt05 * k1) @ (I + dt05 * DK1)
        k2 = f_oigawa(x + dt05 * k1, F)
        # DK3 = (I + dt10 * DK2) @ dfdx(x + dt10 * k2)
        DK3 = dfdx(x + dt10 * k2) @ (I + dt10 * DK2)
        k3 = f_oigawa(x + dt10 * k2, F)
        jac_1step = I + dt / 6.0 * (DK0 + 2 * DK1 + 2 * DK2 + DK3)
        jac = jac_1step @ jac
        x = x + dt / 6.0 * (k0 + 2.0 * k1 + 2.0 * k2 + k3)

    return jac


@jit(f8[:, :](f8[:], f8, f8, f8))
def jacobian_for_s(x, dt=0.01, days=0.25, F=8.0):
    # s = (x, F)を時間発展させるためのヤコビアン。
    N = x.size
    d = 1e-4

    if days == 0.0:
        return np.identity(N + 1)
    else:
        Jacob = np.zeros((N + 1, N + 1))
        xb = run(x, F=F, dt=dt, days=days)
        sb = np.hstack((xb, F))

        for i in range(N + 1):
            if i <= N - 1:
                x_dash = np.zeros(N)
                x_dash[:] = x[:]
                x_dash[i] = x[i] + d
                xa = run(x_dash, F=F, dt=dt, days=days)
                sa = np.hstack((xa, F))
            else:
                xa = run(x, F=F + d, dt=dt, days=days)
                sa = np.hstack((xa, F + d))

            Jacob[:, i] = (sa[:] - sb[:]) / d

        return Jacob


@jit(f8[:](f8, f8, i1))
def get_spin_upped_profile(N=40, dt=0.05, random_state=0):
    # 20160430作成。
    # スピンアップされて十分準定常になったプロファイルを得る。
    # random_stateはランダムシード。
    xa, F = initial_taskx(N)
    random.seed(random_state)
    N = random.randint(1000, 3000)    # dt = 0.01の場合は、50日〜150日走らせることに相当。
    for i in range(N):
        xb = cal_oigawa(xa, F=F, dt=dt)
        xa = xb

    return xa


@jit(f8[:, :](f8[:, :], f8[:], f8, f8))
def ensemble_run(prev_Xa, F, dt=0.01, days=0.25):
    # Fはm次元ベクトルで与える。
    N = prev_Xa.shape[0]
    m = prev_Xa.shape[1]
    Xf = np.zeros((N, m))
    for k in range(m):
        Xf[:, k] = run(prev_Xa[:, k], F=F[k], dt=dt, days=days)

    return Xf


@jit(f8[:, :, :](f8[:], i1, f8, f8, f8, i1))
def get_M_in_AW(x, J, F=8.0, dt=0.05, days=0.25, analysis=1):
    N = x.size
    M_in_AW = np.zeros((J, N, N))
    if analysis:
        jac_func = jacobian_analysis
    else:
        jac_func = jacobian

    for j in range(J):
        if j == 0:
            M_in_AW[j] = jac_func(x, dt=dt, days=days, F=F)
        else:
            M_in_AW[j] \
                = jac_func(x, dt=dt, days=days, F=F) @ M_in_AW[j - 1]

        x = run(x, F=F, dt=dt, days=days)

    return M_in_AW


@jit(f8[:, :](f8[:], i1, f8, f8, f8))
def get_xf_in_AW(x, J, F=8.0, dt=0.05, days=0.25):
    N = x.size
    xf_in_AW = np.zeros((J, N))

    for j in range(J):
        if j == 0:
            xf_in_AW[j] = run(x, F=F, dt=dt, days=days)
        else:
            xf_in_AW[j] = run(xf_in_AW[j - 1], F=F, dt=dt, days=days)

    return xf_in_AW


@jit(f8[:, :, :](f8[:, :], i1, f8[:], f8, f8))
def get_ensemble_xf_in_AW(X, J, F, dt=0.05, days=0.25):
    N = X.shape[0]
    m = X.shape[1]
    Xf_in_AW = np.zeros((J, N, m))

    for j in range(J):
        if j == 0:
            Xf_in_AW[j] = ensemble_run(X, F=F, dt=dt, days=days)
        else:
            Xf_in_AW[j] = ensemble_run(
                Xf_in_AW[j - 1], F=F, dt=dt, days=days)

    return Xf_in_AW


class Lorenz96RungeKutta4:
    def __init__(self, F, dt, N):
        self.F = F
        self.dt = dt
        self.N = N
        self.init_x = np.zeros(self.N)
        self.init_x[:] = self.F
        self.init_x[self.N // 2] = self.F * (1 + 1e-3)

    def f(self, x):
        return f_oigawa(x, self.F)

    def cal(self, x):
        return cal_oigawa(x, F=self.F, dt=self.dt)

    def run(self, x, days):
        return run(x, F=self.F, dt=self.dt, days=days)

    def get_spin_upped_profile(self, random_state=0):
        return get_spin_upped_profile(self.N, self.dt, random_state=random_state)

    def get_initial_Xa(self, m, random_state=0):
        initial_Xa = np.zeros((self.N, m))
        for k in range(m):
            initial_Xa[:, k] \
                = self.get_spin_upped_profile(random_state=random_state + k)

        return initial_Xa

    def jacobian(self, x, days):
        return jacobian(x, dt=self.dt, days=days, F=self.F)

    def jacobian_analysis(self, x, days):
        # for loopの中、コメントアウトしている部分は間違いです。
        # この関数の計算は、データ同化ノートの315ページ付近を参照
        return jacobian_analysis(x, dt=self.dt, days=days, F=self.F)

    def ensemble_run(self, prev_Xa, days):
        if not isinstance(self.F, np.ndarray):
            # self.Fがスカラーだと思われるとき、m次元のarrayにする。
            F = np.ones(prev_Xa.shape[1]) * self.F
        else:
            F = self.F

        return ensemble_run(prev_Xa, F=F, dt=self.dt, days=days)

    def get_M_in_AW(self, x, J, days, analysis=1):
        return get_M_in_AW(x, J, F=self.F, dt=self.dt,
                           days=days, analysis=analysis)

    def get_xf_in_AW(self, x, J, days):
        return get_xf_in_AW(x, J, F=self.F, dt=self.dt, days=days)

    def get_ensemble_xf_in_AW(self, X, J, days):
        return get_ensemble_xf_in_AW(X, J, self.F, dt=self.dt, days=days)


if __name__ == '__main__':
    import time
    model = Lorenz96RungeKutta4(F=8.0, dt=0.05, N=40)

    # テスト1
    x2 = model.run(model.init_x, 50)
    print(x2[:5], model.init_x[:5])
    x2 = model.run(x2, 10)
    print(x2[:5])
    print("")

    # テスト2
    np.random.seed(0)
    X2 = (x2 + np.random.randn(20, 40)).T
    start = time.time()
    X2 = model.ensemble_run(X2, 50)
    print("elapsed time:", time.time() - start)
    print(X2[:, 0])
    print("")

    # テスト3
    M = model.jacobian_analysis(x2, 10)
    M2 = model.jacobian(x2, 10)
    diff = np.sum(np.abs(M - M2))
    print(M[0, 0], M2[0, 0], diff, x2[:2])
    print("")

    # テスト4
    for bool_ in [True, False]:
        M = model.get_M_in_AW(x2, J=4, days=1, analysis=bool_)
        print(M[0, 0, 0], x2[:2])
    print("")
