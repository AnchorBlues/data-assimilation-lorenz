#include<iostream>
#include<math.h>
#include<vector>
#include<fstream>
#include<string>

using namespace std;


const int N = 40;
const double dt = 0.005;
const double F = 8.0;
const double err_rate = 1e-3;

double func(double a, double b, double c, double d){
  return  - a * b + b * d - c + F;
}

vector<double> cal(vector<double> x){
  vector<double> xa(N, 0);
  double halfdt = dt * 0.5;
  double  a, b, c, d;
  double k1, k2, k3, k4;
  for (int i = 0; i < N; i++){
    a = x[(i - 2 + N) % N];
    b = x[(i - 1 + N) % N];
    c = x[(i + N) % N];
    d = x[(i + 1 + N) % N];
    k1 = func(a          , b          , c          , d);
    k2 = func(a + halfdt * k1, b + halfdt * k1, c + halfdt * k1, d + halfdt * k1);
    k3 = func(a + halfdt * k2, b + halfdt * k2, c + halfdt * k2, d + halfdt * k2);
    k4 = func(a + dt     * k3, b + dt     * k3, c + dt     * k3, d + dt     * k3);
    xa[i] = c + dt / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4);
  }
  return xa;
}

// Lorenz96 の右辺
vector<double> func_oigawa(vector<double> x){
  vector<double> nx(N, 0);
  for (int i = 0; i < N; i++){
    nx[i] = -x[(i - 2 + N) % N] * x[(i - 1 + N) % N] + x[(i - 1 + N) % N] * x[(i + 1) % N] - x[i] + F;
  }
  return nx;
}

vector<double> sca_dot_vec(double a, vector<double> v){
  vector<double> va(N, 0);
  for (int i = 0; i < N; i++){
    va[i] = a * v[i];
  }
  return va;
}

vector<double> vec_pls_vec(vector<double> a, vector<double> b){
  vector<double> c(N, 0);
  for (int i = 0; i < N; i++){
    c[i] = a[i] + b[i];
  }
  return c;
}

// モデル演算子(ルンゲクッタ法)
vector<double> cal_oigawa(vector<double> x){
  vector<double> k1(N), k2(N), k3(N), k4(N);
  vector<double> tmp1(N), tmp2(N), tmp3(N);
  k1 = func_oigawa(x);
  k2 = func_oigawa(vec_pls_vec(x, sca_dot_vec(0.5 * dt, k1)));
  k3 = func_oigawa(vec_pls_vec(x, sca_dot_vec(0.5 * dt, k2)));
  k4 = func_oigawa(vec_pls_vec(x, sca_dot_vec(dt, k3)));
  tmp1 = vec_pls_vec(k1, sca_dot_vec(2.0, k2));
  tmp2 = vec_pls_vec(tmp1, sca_dot_vec(2.0, k3));
  tmp3 = vec_pls_vec(tmp2, k4);
  return vec_pls_vec(x, sca_dot_vec(dt / 6.0, tmp3));
}

int main(){
  int M = 5000;
  vector<double> x(N, F);     // 要素数N、全ての要素Fで初期化
  vector<double> xa(N, 0);
  vector<double> xb(N, 0);
  x[20] = F * (1 + err_rate);

  ofstream writing_file("Cpp_result.txt");
  xb = x;
  for (int i = 0; i < M; i++){
    xa = cal_oigawa(xb);
    xb = xa;
  }
  // 結果のprint
  for (int i = 0; i < N ; i++){
    cout << xb.at(i) << endl;
    writing_file << xb.at(i) << endl;
  }
  writing_file.close();

  return 0;
}
