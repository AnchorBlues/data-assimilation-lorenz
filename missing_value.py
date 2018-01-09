# coding:utf-8


# 2016/12/22作成。
# 欠損値クラスを作成する。

import random
import numpy as np
import func_for_assimilate as ffa


class MissingValue(object):
    def __init__(self, FuncName):
        self.FuncName = FuncName

    def get_ms(self, n_of_missn, N=40, Tmax=100, random_state=0):
        random.seed(random_state)
        MS = np.ones((Tmax, N))
        for i in range(Tmax):
            ms = getattr(ffa, self.FuncName)(N, n_of_missn)
            for idx in ms:
                MS[i, idx] = 0

        return MS


MV = {}
MV[0] = MissingValue('get_random_missn')
MV[1] = MissingValue('get_uniform_missn')
MV[2] = MissingValue('get_consecutive_missn')
