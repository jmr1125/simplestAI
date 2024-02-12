#include <OpenCL/OpenCL.h>
#include <cstddef>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <stdexcept>
#include <string>
#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.hpp>
#endif
#include <chrono>

#include "cl-mat.hpp"
#include "clGetErrorString.hpp"
#include <random>

#define right(s)                                                               \
  if (ret != CL_SUCCESS) {                                                     \
    cerr << "ERR: " << s << " : " << clGetErrorString(ret) << "(" << ret       \
         << ")" << endl;                                                       \
  }
using namespace std;

using std::chrono::high_resolution_clock;
// using std::chrono::system_clock;
// #define now() system_clock::now();
#define now() high_resolution_clock::now();
auto t = now();
// auto t1 = t;
//  #define time_get() (-(t - (t1 = system_clock::now())).count(), t = t1)
auto timeget() {
  auto t1 = now();
  auto T = t1 - t;
  t = t1;
  return (double)T.count() / 1000000000;
}
auto rand01() {
  static default_random_engine generator;
  uniform_real_distribution<float> dis(0, 1);
  return dis(generator);
}
int main() {
  init();
  cout << "[" << timeget() << "] init" << endl;
  int m, n, k;
  cin >> m >> n >> k;
  cout << "[" << timeget() << "] input done" << endl;
  cout << "[" << timeget() << "] =====test1=====" << endl;
  matrix a, b, c, c1;
  a.setn(m);
  a.setm(k);
  b.setn(k);
  b.setm(n);
  c.setn(m);
  c.setm(n);
  c1.setn(m);
  c1.setm(n);
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < k; ++j) {
      a(i, j) = rand01();
      // a(i, j) = i + j;
    }
  }
  for (int i = 0; i < k; ++i) {
    for (int j = 0; j < n; ++j) {
      b(i, j) = rand01();
      // b(i, j) = i + j;
    }
  }
  cout << "[" << timeget() << "] randomize" << endl;
  c = mul_mat(a, b);
  cout << "[" << timeget() << "] done" << endl;
  cout << "[" << timeget() << "] run normal way and check" << endl;
  bool different = false;
  for (int x = 0; x < n; ++x) { // a:m*k b:k*n
    if (x * 100 % m == 0) {
      cout << x * 100 / m << "%";
      cout.flush();
    }
    for (int y = 0; y < m; ++y) {
      float res = 0;
      for (unsigned int i = 0; i < k; ++i) {
        res += a(y, i) * b(i, x);
      }
      c1(y, x) = res;
      if (c(y, x) != c1(y, x))
        different = true;
    }
  }
  cout << endl
       << "[" << timeget() << "] done" << endl
       << "different: " << different << endl;
  if (m < 10 && n < 10) {
    cout << "a: " << m << "*" << k << endl;
    cout << "b: " << k << "*" << n << endl;
    cout << "c: " << m << "*" << n << endl;
    cout << "a = " << endl;
    for (int i = 0; i < m; ++i) {
      for (int j = 0; j < k; ++j) {
        cout << a(i, j) << ' ';
      }
      cout << endl;
    }
    cout << "b = " << endl;
    for (int i = 0; i < k; ++i) {
      for (int j = 0; j < n; ++j) {
        cout << b(i, j) << ' ';
      }
      cout << endl;
    }
    cout << "c = " << endl;
    for (int i = 0; i < m; ++i) {
      for (int j = 0; j < n; ++j) {
        cout << c(i, j) << ' ';
      }
      cout << endl;
    }
    cout << "c1 = " << endl;
    for (int i = 0; i < m; ++i) {
      for (int j = 0; j < n; ++j) {
        cout << c1(i, j) << ' ';
      }
      cout << endl;
    }
  }
  cout << "[" << timeget() << "] =====test2=====" << endl;
  VvalT v1, v2;
  v1.resize(n);
  v2.resize(n);
  for (int i = 0; i < n; ++i) {
    v1[i] = rand01();
    v2[i] = rand01();
  }
  auto res1 = add_vec(v1, v2), res2 = mul_vec(v1, v2);
  cout << "[" << timeget() << "] done" << endl;
  if (n < 10) {
    cout << "v1: " << endl;
    for (int i = 0; i < n; ++i) {
      cout << v1[i] << " ";
    }
    cout << "v2: " << endl;
    for (int i = 0; i < n; ++i) {
      cout << v2[i] << " ";
    }
    cout << "res1: " << endl;
    for (int i = 0; i < n; ++i) {
      cout << res1[i] << " ";
    }
    cout << "res2: " << endl;
    for (int i = 0; i < n; ++i) {
      cout << res2[i] << " ";
    }
  }
  cout << "[" << timeget() << "] print vectors" << endl;
  for (int i = 0; i <= n; ++i) {
    if (v1[i] + v2[i] != res1[i] || v1[i] * v2[i] != res2[i]) {
      cout << "different!" << endl;
      break;
    }
  }
  cout << "[" << timeget() << "] normal way" << endl;
  teardown();
}
