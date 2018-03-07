from ctypes import c_double, c_int
import random
import numpy as np
from numpy.ctypeslib import ndpointer

# Lorenz96モデルでのdt=1.0が、天気予報(実時間)での5.0日に対応。1日<->dt=0.2, 6時間<->dt=0.05
Lorenz96_TimeScale = 5.0

lorenz = np.ctypeslib.load_library("lorenz.so", ".")

lorenz.f.restype = None
lorenz.f.argtypes = [
    ndpointer(c_double),
    ndpointer(c_double),
    c_int,
    c_double
]

lorenz.dfdx.restype = None
lorenz.dfdx.argtypes = [
    ndpointer(c_double),
    ndpointer(c_double),
    c_int
]

lorenz.calc.restype = None
lorenz.calc.argtypes = [
    ndpointer(c_double),
    ndpointer(c_double),
    c_int,
    c_double, c_double
]

lorenz.run.restype = None
lorenz.run.argtypes = [
    ndpointer(c_double),
    ndpointer(c_double),
    c_int,
    c_double, c_double, c_double
]


lorenz.run2.restype = None
lorenz.run2.argtypes = [
    ndpointer(c_double),
    ndpointer(c_double),
    c_int,
    c_double, c_double, c_int
]


class Lorenz96RungeKutta4UsingCtypes(object):
    def __init__(self, F, dt, N, pure_python_flg=False):
        self.F = F
        self.dt = dt
        self.N = N
        self.init_x = np.zeros(self.N)
        self.init_x[:] = self.F
        self.init_x[self.N // 2] = self.F * (1 + 1e-3)
        self.dfdx = self._dfdx_pure_python if pure_python_flg else self._dfdx

    def f(self, x):
        xa = np.empty(self.N)
        lorenz.f(x.astype(np.float64), xa, self.N, self.F)
        return xa

    def cal(self, x):
        xa = np.empty(self.N)
        lorenz.calc(x.astype(np.float64), xa, self.N, self.dt, self.F)
        return xa

    def run(self, x, days):
        if days == 0:
            return x
        xa = np.empty(self.N)
        t = days / Lorenz96_TimeScale
        tmax = int(round(t / self.dt, 0))
        lorenz.run2(x.astype(np.float64), xa, self.N, self.dt, self.F, tmax)
        return xa

    def get_spin_upped_profile(self, random_state=0):
        random.seed(random_state)
        days = random.randint(100, 300)
        return self.run(self.init_x, days)

    def get_initial_Xa(self, m, random_state=0):
        initial_Xa = np.zeros((self.N, m))
        for k in range(m):
            initial_Xa[:, k] \
                = self.get_spin_upped_profile(random_state=random_state + k)

        return initial_Xa

    def jacobian(self, x, days):
        d = 1e-4

        if days == 0.0:
            return np.identity(self.N)

        Jacob = np.zeros((self.N, self.N))
        xb = self.run(x, days)

        for i in range(self.N):
            x_dash = np.zeros(self.N)
            x_dash[:] = x[:]

            x_dash[i] = x[i] + d
            xa = self.run(x_dash, days)

            Jacob[:, i] = (xa[:] - xb[:]) / d

        return Jacob

    def _dfdx(self, x):
        df = np.empty(self.N * self.N)
        lorenz.dfdx(x.astype(np.float64), df, self.N)
        return df.reshape(self.N, self.N)

    def _dfdx_pure_python(self, x):
        # f(x) = dx/dt = ... の式をxで微分したもの。
        N = self.N
        df = np.zeros((N, N))
        for i in range(N):
            df[i, (i - 2 + N) % N] = -x[(i - 1 + N) % N]
            df[i, (i - 1 + N) % N] = -x[(i - 2 + N) % N] \
                + x[(i + 1 + N) % N]
            df[i, i] = -1
            df[i, (i + 1 + N) % N] = x[(i - 1 + N) % N]

        return df

    def jacobian_analysis(self, x, days):
        # for loopの中、コメントアウトしている部分は間違いです。
        # この関数の計算は、データ同化ノートの315ページ付近を参照
        if days == 0:
            return np.identity(self.N)

        t = days / Lorenz96_TimeScale
        imax = int(round(t / self.dt, 0))
        I = np.identity(self.N)
        dt05 = 0.5 * self.dt
        dt10 = self.dt
        jac = I
        for _ in range(imax):
            DK0 = self.dfdx(x)
            k0 = self.f(x)
            # DK1 = (I + dt05 * DK0) @ dfdx(x + dt05 * k0)
            DK1 = self.dfdx(x + dt05 * k0) @ (I + dt05 * DK0)
            k1 = self.f(x + dt05 * k0)
            # DK2 = (I + dt05 * DK1) @ dfdx(x + dt05 * k1)
            DK2 = self.dfdx(x + dt05 * k1) @ (I + dt05 * DK1)
            k2 = self.f(x + dt05 * k1)
            # DK3 = (I + dt10 * DK2) @ dfdx(x + dt10 * k2)
            DK3 = self.dfdx(x + dt10 * k2) @ (I + dt10 * DK2)
            k3 = self.f(x + dt10 * k2)
            jac_1step = I + self.dt / 6.0 * (DK0 + 2 * DK1 + 2 * DK2 + DK3)
            jac = jac_1step @ jac
            x = x + self.dt / 6.0 * (k0 + 2.0 * k1 + 2.0 * k2 + k3)

        return jac

    def ensemble_run(self, prev_Xa, days):
        if days == 0:
            return prev_Xa

        assert prev_Xa.shape[0] == self.N
        m = prev_Xa.shape[1]
        Xf = np.zeros((self.N, m))
        for k in range(m):
            Xf[:, k] = self.run(prev_Xa[:, k], days)

        return Xf

    def get_M_in_AW(self, x, J, days, analysis=True):
        M_in_AW = np.zeros((J, self.N, self.N))
        if days == 0:
            return M_in_AW

        if analysis:
            jac_func = self.jacobian_analysis
        else:
            jac_func = self.jacobian

        M_in_AW[0] = jac_func(x, days)
        x = self.run(x, days)
        for j in range(1, J):
            M_in_AW[j] = jac_func(x, days) @ M_in_AW[j - 1]
            x = self.run(x, days)

        return M_in_AW

    def get_xf_in_AW(self, x, J, days):
        xf_in_AW = np.zeros((J, self.N))
        if days == 0:
            return xf_in_AW

        xf_in_AW[0] = self.run(x, days)
        for j in range(1, J):
            xf_in_AW[j] = self.run(xf_in_AW[j - 1], days)

        return xf_in_AW

    def get_ensemble_xf_in_AW(self, X, J, days):
        assert X.shape[0] == self.N
        m = X.shape[1]
        Xf_in_AW = np.zeros((J, self.N, m))
        if days == 0:
            return Xf_in_AW

        Xf_in_AW[0] = self.ensemble_run(X, days)
        for j in range(1, J):
            Xf_in_AW[j] = self.ensemble_run(Xf_in_AW[j - 1], days=days)

        return Xf_in_AW


if __name__ == '__main__':
    import time
    model = Lorenz96RungeKutta4UsingCtypes(F=8.0, dt=0.05, N=40)

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
        M = model.get_M_in_AW(x2, J=4, days=1.0, analysis=bool_)
        print(M[0, 0, 0], x2[:2])
    print("")
