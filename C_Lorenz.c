
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

double const Lorenz96_TimeScale = 5.0;

void f(double *xb, double *xa, int N, double F){
  int i;
  for(i = 0; i < N; i++ ){
    xa[i] = -xb[(i - 2 + N) % N] * xb[(i - 1 + N) % N] + \
      xb[(i - 1 + N) % N] * xb[(i + 1) % N] - xb[i] + F;
  }
}

void dfdx(double *x, double *df, int N){
  /* f(x) = dx/dt = ... の式をxで微分したもの。  */
  int i;
  /* まずは全て0で埋める */
  for(i = 0; i < N * N; i++ ){
    df[i] = 0.0;
  }
  for(i = 0; i < N; i++ ){
    df[i * N + (i - 2 + N) % N] = -x[(i - 1 + N) % N];
    df[i * N + (i - 1 + N) % N] = -x[(i - 2 + N) % N] + x[(i + 1 + N) % N];
    df[i * N + i] = -1;
    df[i * N + (i + 1 + N) % N] = x[(i - 1 + N) % N];
  }
}

void calc(double *prev_x, double *x,
          int N, double dt, double F){
  double *k1 = (double *)malloc(sizeof(double) *N);
  double *k2 = (double *)malloc(sizeof(double) *N);
  double *k3 = (double *)malloc(sizeof(double) *N);
  double *k4 = (double *)malloc(sizeof(double) *N);
  double *x_tmp = (double *)malloc(sizeof(double) *N);
  int i;
  /* ルンゲクッタ第1ステップ */
  f(prev_x, k1, N, F);
  /* ルンゲクッタ第2ステップ */
  for(i = 0; i < N; i++ ){
    x_tmp[i] = prev_x[i] + 0.5 * dt * k1[i];
  }
  f(x_tmp, k2, N, F);
  /* ルンゲクッタ第3ステップ */
  for(i = 0; i < N; i++ ){
    x_tmp[i] = prev_x[i] + 0.5 * dt * k2[i];
  }
  f(x_tmp, k3, N, F);
  /* ルンゲクッタ第4ステップ */
  for(i = 0; i < N; i++ ){
    x_tmp[i] = prev_x[i] + dt * k3[i];
  }
  f(x_tmp, k4, N, F);
  /* 最終的な、次のタイムステップにおけるxの計算 */
  for(i = 0; i < N; i++ ){
    x[i] = prev_x[i] + dt / 6.0 * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]);
  }
  free(k1);
  free(k2);
  free(k3);
  free(k4);
  free(x_tmp);
}

void double_arr_cpy(double *arr, double *brr, int N){
  /* 長さNの配列arrを、brrにコピーする */
  int i;
  for(i = 0; i < N; i++ ){
    brr[i] = arr[i];
  }
}

void run(double *prev_x, double *x,
         int N, double dt, double F, double days){
  int tmax = (int)round((double)(days / Lorenz96_TimeScale) / dt);
  int i, t;
  double *x_copy = (double *)malloc(sizeof(double) *N);
  /* prev_xのコピーを作成する */
  for(i = 0;i < N;i++ ){
    x_copy[i] = prev_x[i];
  }
  for(t = 0; t < tmax; t++ ){
    calc(x_copy, x, N, dt, F);
    for(i = 0;i < N;i++ ){
      x_copy[i] = x[i];
    }
  }
  free(x_copy);
}


int main(void){
  /*
    run_days = 50.0で走らせて、
    0.782773
    -1.228557
    0.979169
    1.662951
    4.127504
    5.620434
    -2.881063
    2.860022
    2.420567
    4.745853
    6.781690
    -0.197617
    4.404913
    9.476168
    5.760968
    -0.006863
    0.797965
    6.346966
    8.901591
    2.301552
    0.828704
    -5.111792
    -3.511279
    -0.899645
    1.696197
    4.542954
    8.026185
    -0.116704
    1.807124
    2.480270
    4.272240
    6.645838
    -0.586270
    -5.248163
    0.462448
    -0.007327
    0.641410
    6.391053
    9.005648
    -0.757221
    になったら成功。
  */
  int N = 40;
  double F = 8.0;
  double *init_x = (double *)malloc(sizeof(double) *N);
  double *x = (double *)malloc(sizeof(double) *N);
  double run_days = 50.0;
  double dt = 0.05;
  int i;
  for(i = 0; i < N; i++ ){
    init_x[i] = F;
  }
  init_x[N / 2] = F * (1 + 1e-3);
  run(init_x, x, N, dt, F, run_days);
  for(i = 0; i < N; i++ ){
    printf("%lf\n", x[i]);
  }
  free(init_x);
  free(x);
  return 0;
}
