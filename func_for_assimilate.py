# coding:utf-8

#from numba.decorators import jit, autojit
import random
import numpy as np


def get_H(x, KSC=None):                    # xには、欠損値を含んでいる観測データを入れる。
    # xから、観測行列Hを求める。
    # 観測データxに欠損値がなかった場合には、観測行列Hは単位行列になる。
    if KSC is None:
        return np.identity(x.size)[~np.isnan(x)]

    return np.identity(x.size)[x != KSC]


def get_H_for_s(H, k):
    # Fの推定の際に用いる観測演算子Hの作成。
    # 引数の行列に1列を加える。加えた成分の値は全てゼロ。
    a = np.zeros(H.shape[0] * k).reshape(H.shape[0], k)
    H_d = np.hstack((H, a))
    return H_d


def get_uniform_missn(N, n_of_missn):
    # 均等に観測点を取り除く実験を行いたい。
    # そこで、全観測点数Nと、そこから取り除く観測点数n_of_missnを与えると
    # msのlistを吐き出してくれるような関数を作成する。
    ms = []
    for k in range(n_of_missn):
        ms.append(k * N // n_of_missn)

    return ms


def get_random_missn(N, n_of_missn):
    # ランダムに観測点を取り除く実験を行いたい。
    # そこで、全観測点数Nと、そこから取り除く観測点数n_of_missnを与えると
    # msのlistを吐き出してくれるような関数を作成する。
    ms = list(np.arange(N))
    return random.sample(ms, n_of_missn)


def get_consecutive_missn(N, n_of_missn):
    return list(np.arange(n_of_missn))


def ms_to_method_and_n_of_missn(ms):
    if ms == []:
        method = 'manual'
        n_of_missn = 0
    elif ms[0] == 'random':
        method = 'random'
        n_of_missn = ms[1]
    elif ms[0] == 'uniform':
        method = 'uniform'
        n_of_missn = ms[1]
    elif ms[0] == 'consecutive':
        method = 'consecutive'
        n_of_missn = ms[1]
    else:
        method = 'manual'
        n_of_missn = len(ms)

    return method, n_of_missn


def method_and_n_of_missn_to_ms(method, N, n_of_missn, ms):
    if method == 'random':
        ms = get_random_missn(N, n_of_missn)
    elif method == 'uniform':
        ms = get_uniform_missn(N, n_of_missn)
    elif method == 'consecutive':
        ms = list(np.arange(n_of_missn))
    elif method == 'manual':
        pass
    else:
        raise Exception('your method is not valid!')

    return ms


def left_aligned(x, KSC=None):
    # 欠損値を含むデータに対して、その欠損値の部分を取り除いて左詰めにした配列を返す。
    if KSC is None:
        return x[~np.isnan(x)]

    return x[x != KSC]


def Eigenvalue_decomp(A):
    # 対称行列Aを、直交行列Uを用いて対角化する。
    if not (abs(A - A.T) < 1e-4).all():
        raise Exception("error! A is not symmetric matrix!")

    la, U = np.linalg.eig(A)    # eigen value and eigen vector is resolved.
    U, _ = np.linalg.qr(U)            # Gram - Schmidt orthonormalization

    D = np.diag(la)

    return U.real, D.real        # 時折、UやDが複素数になって計算が止まってしまうことがあるので、実部だけを取り出す


def generate_rho_for_Rlocalization(sigma, k, ms, N=40):
    if sigma > 20.0:
        raise Exception("Error! your sigma is too big!")
    elif sigma < 0.5:
        raise Exception("Error! your sigma is too small!")

    RHO = np.zeros((N, N))
    for i in range(N):
        if abs(k - i) < 20.0:
            d = np.sqrt((i - k) ** 2 + (i - k) ** 2)
        else:
            d = np.sqrt((abs(i - k) - N) ** 2 + (abs(i - k) - N) ** 2)
        RHO[i, i] = np.exp((d ** 2) / (2.0 * (sigma ** 2)))

    RHO = np.delete(RHO, ms, axis=0)
    RHO = np.delete(RHO, ms, axis=1)
    return RHO


def replace_KSC_at_ms(x, ms, KSC=1e+5):
    for j in range(len(ms)):
        x[ms[j]] = KSC

    return x


def get_RMSE_o(obs_data, mdl_data):
    import using_jit
    T = 365 * 4
    RMSE_o = np.zeros(T)
    N = obs_data.shape[1]

    for i in range(T):
        xo = np.zeros(N)
        xo[:] = obs_data[i, :]
        xt = np.zeros(N)
        xt[:] = mdl_data[i, :]
        RMSE_o[i] = using_jit.cal_RMSE(xo, xt)  # 観測値と真値とのRMSE。

    return RMSE_o
