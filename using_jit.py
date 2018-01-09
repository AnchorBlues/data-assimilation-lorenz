# coding:utf-8

from numba import jit, f8, i1
import numpy as np
import scipy.stats as st


@jit(f8[:](f8[:], f8[:]))
def Resampling(w_tilde, X10):
    M = w_tilde.size
    sum_of_w_tilde = np.zeros(M)
    for i in range(M):
        sum_of_w_tilde[i] = np.sum(w_tilde[:i + 1])

    d = np.random.rand() / M    # 0~1/Mの間の値をとる一様乱数
    Arrow = np.arange(M) / M + d
    X11 = np.array([])
    for i in range(M):
        if i == 0:
            Resample = 0
            for j in range(M):
                if Arrow[j] < sum_of_w_tilde[i]:
                    Resample += 1
                else:
                    break
        else:
            L0 = 0
            for j in range(M):
                if Arrow[j] <= sum_of_w_tilde[i - 1]:
                    L0 += 1
                else:
                    break

            L1 = 0
            for j in range(M):
                if Arrow[j] < sum_of_w_tilde[i]:
                    L1 += 1
                else:
                    break

            Resample = L1 - L0

        if Resample == 0:
            pass
        else:
            X11 = np.r_[X11, np.ones(Resample) * X10[i]]

    if X11.size != M:
        print(X10, X11)
        raise ValueError('your X11\'s size is not valid!')

    return X11


@jit(f8[:, :](f8[:], f8[:, :]))
def Resampling_2D(w_tilde, X10):
    M = w_tilde.size
    sum_of_w_tilde = np.zeros(M)
    for i in range(M):
        sum_of_w_tilde[i] = np.sum(w_tilde[:i + 1])

    d = np.random.rand() / M    # 0~1/Mの間の値をとる一様乱数
    Arrow = np.arange(M) / M + d
    X11 = np.array([])
    for i in range(M):
        if i == 0:
            Resample = 0
            for j in range(M):
                if Arrow[j] < sum_of_w_tilde[i]:
                    Resample += 1
                else:
                    break
        else:
            L0 = 0
            for j in range(M):
                if Arrow[j] <= sum_of_w_tilde[i - 1]:
                    L0 += 1
                else:
                    break

            L1 = 0
            for j in range(M):
                if Arrow[j] < sum_of_w_tilde[i]:
                    L1 += 1
                else:
                    break

            Resample = L1 - L0

        if Resample == 0:
            pass
        else:
            if X11.size == 0:
                X11 = X10[:, i]
                for k in range(Resample - 1):
                    X11 = np.c_[X11, X10[:, i]]
            else:
                for k in range(Resample):
                    X11 = np.c_[X11, X10[:, i]]

    if X11.shape[0] != X10.shape[0]:
        raise ValueError('your X11\'s size is not valid!')

    return X11


@jit(f8(f8[:], f8[:]))
def cal_RMSE(xa, xb):
    """
    xa, xbという2つのプロファイルの間のroot mean square errorを計算する。
    xa, xbの各要素のどれか1つでもnanであれば、nanが返ってくる。
    """
    if xa.size != xb.size:
        raise Exception('error! xa\'s size and xb\'s are not different!')

    if np.sum(xa) == 0 or np.sum(xb) == 0:
        # xaとxbのどちらか一方が全部0であれば、その時刻ではrmseはnanにする。
        return np.nan
    else:
        err = xa - xb
        return np.sqrt(np.sum(err ** 2) / err.size)


@jit(f8[:](f8[:, :], f8[:, :]))
def cal_RMSE_2D(xa, xb):
    """
    # xaが2次元(J, N)の時、j=1, ..., J毎にRMSEを計算して、長さJの配列を返す。
    """
    if xa.shape != xb.shape:
        raise Exception('error! xa\'s size and xb\'s are not different!')

    J = xa.shape[0]
    rmses = np.zeros(J)
    for j in range(J):
        rmses[j] = cal_RMSE(xa[j], xb[j])

    return rmses


@jit(f8[:, :](f8[:], f8[:]))
def cal_covmat(err1, err2):
    """
    誤差から誤差共分散行列を作成する。
    err1.size = 40でテストした結果、
    err1 * err2[:, None]よりもこの実装のほうが僅かに高速でした。
    http://yukara-13.hatenablog.com/entry/2014/01/24/131640
    http://lv4.hateblo.jp/entry/2014/07/23/132849
    http://emoson.hateblo.jp/entry/2014/10/26/133736
    """
    N = err1.size
    Cov = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            Cov[i, j] = err1[i] * err2[j]

    return Cov


@jit(f8[:, :](f8[:, :], f8[:], f8[:], i1, i1))
def likelihood_(HXf, y, diagR, p, m):
    """
    モデルグリッド毎に尤度を計算
    (n(観測があるグリッドの個数), m(アンサンブル数))の形の配列を返す
    HXf.shape == (p, m)
    y.size == p
    """
    lkh_mat = np.zeros((p, m))
    for i in range(p):
        lkh_mat[i] = st.norm.pdf(HXf[i] - y[i], scale=np.sqrt(diagR[i]))
    return lkh_mat
