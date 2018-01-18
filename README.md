# data-assimilation-lorenz
各データ同化アルゴリズムの、lorenz96モデルへの適用


## Lorenz96.py
Lorenz96モデルのコードが書かれているファイル

## assimilation.py
最もベースとなるAssimilationクラスなどを定義しているファイル

## create_true_and_obs.py
観測データ・真値データを作成するためのファイル

## four_dimensional_variational_method.py
4次元変分法のファイル

## func_for_assimilate.py
データ同化のための様々なutil関数が書かれているファイル

## kalman_filter.py
カルマンフィルタのファイル

## kalman_filter_4D.py
4次元カルマンフィルタのファイル

## kalman_smoother.py
カルマンスムーザーのファイル

## local_ensemble_transform_kalman_filter.py
LETKF(Local Ensemble Transform Kalman Filter)のファイル

## local_ensemble_transform_kalman_filter_4D.py
4次元LETKFのファイル

## local_ensemble_transform_kalman_smoother.py
LETKFスムーザーのファイル

## particle_filter.py
粒子フィルタのファイル

## perturbed_observation_ensemble_kalman_filter.py
PO(Perturbed Observation)法のファイル


## serial_ensemble_square_root_filter.py
Serial EnSRF(Ensemble Square Root Filter)のファイル

## serial_ensemble_square_root_filter_4D.py
4次元Serial EnSRFのファイル

## serial_ensemble_square_root_smoother.py
Serial EnSRFスムーザーのファイル

## three_dimensional_variational_method.py
3次元変分法のファイル

## using_jit.py
numbaを用いて高速化させた関数が書かれているファイル
