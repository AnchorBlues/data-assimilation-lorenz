# coding:utf-8


def save_npz(data, fname, data_dir_flg=True):
    import numpy as np
    import os.path
    if data_dir_flg:
        FNAME = './data_dir/' + fname + '.npz'
    else:
        FNAME = fname + '.npz'

    # 指定したファイル名のファイルが存在した時には、警告文を表示させるだけで保存しない。
    if os.path.exists(FNAME):
        raise Exception('the file exits! save is failured!')
    else:
        np.savez(FNAME, data=data)


def load_npz(fname, data_dir_flg=True):
    import numpy as np
    if data_dir_flg:
        FNAME = './data_dir/' + fname + '.npz'
    else:
        FNAME = fname + '.npz'

    a = np.load(FNAME)
    data = a['data']
    return data
