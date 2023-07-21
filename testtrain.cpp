#include "main.h"
#include "network.h"
#include "train.h"
#include <cmath>
#include <iostream>
#include <ostream>
using namespace std;

void train(network &net, const VvalT &input, const VvalT &expect) {
  net.setInput(input);
  net.getV();
  // delta d? = delta dv v d? = 2(delta-v)*vd?
  // modify it
  vector<matrix> netVw;
  vector<matrix> netVb;
  netVw.reserve(net.layers.size());
  netVb.reserve(net.layers.size());
  for (int i = 0; i < net.layers.size(); ++i) {
    netVw[i] = net.layers[i].w;
    netVb[i] = net.layers[i].b;
  }
#ifdef USE_OMP
#pragma omp parallel for
#endif
  for (size_t i = 0; i < net.output.size(); ++i) {
    valT v = (net.output[i] - expect[i]) * 2; //(V_i-e)^2 d? = 2(V_i-e)*V_i d ?
                                              // v /= 100;
    // v /= 50000;
    // v /= 1000000000;2 2 3 2 2 2+3+2+2+2*2+2*3+3*2+2*2=29
    // 1000*NumberOfVarible
    v /= 29000;
#ifdef USE_OMP
#pragma omp parallel for
#endif
    for (size_t l = 0; l < net.layers.size(); ++l) {
      const matrix vdb = net.getVdbi(l);
      valT delta_b = vdb(i, 0) * v;
      // valT delta_b = v / vdb(i, 0);
      // delta_b *= fabs(delta_b);
      netVb[l](i, 0) -= delta_b;
      for (int j = 0; j < net.layers[l].w.getm(); ++j) {
        const matrix &&vdw = net.getVdWij(l, j);
        valT delta_w = vdw(i, 0) * v;
        // valT delta_w = v / vdw(i, 0);
        //  delta_w *= fabs(delta_w);
        netVw[l](i, j) -= delta_w;
      }
    }
  }
  for (int i = 0; i < net.layers.size(); ++i) {
    net.layers[i].w = netVw[i];
    net.layers[i].b = netVb[i];
  }
}

void test(network net) {
  cout << "\033[5;1Hrun 10*100 times and calculate " << endl;
  for (int i = 0; i < 10; ++i) {
    int a, b, c;
    a = b = c = 0;
    for (int i = 1; i <= 100; ++i) {
      auto x = genvalT() * 10;
      auto y = genvalT() * 10;
      VvalT in{x, y};
      net.setInput(in);
      net.getV();
      valT delta = (net.output[0] - x * 11.25) * (net.output[0] - x * 11.25) +
                   (net.output[1] - x - y) * (net.output[1] - x - y);
      // cout << delta << ' ';
      if (fabs(delta) < 1e-9) {
        ++a;
      } else if (fabs(delta) < 1e-6) {
        ++b;
      } else {
        ++c;
      }
    }
    cout << "\033[32m<1e-9: " << a << "%\033[0m   ";
    cout << "\033[33m<1e-6: " << b << "%\033[0m   ";
    cout << "\033[31m>1e-6: " << c << "%\033[0m   " << endl;
  }
  // cout << "press Enter" << endl;
  // {
  //   string s;
  //   getline(cin, s);
  // };
}
int main() {
  network net(
      {2, 2, 3, 2, 2}, [](valT v) { return v; },
      [](valT) -> valT { return 1; });
  /*randomize net*/
  for (layer &x : net.layers) {
    for (auto &x : x.w.m) { // rand matrix
      for (auto &x : x) {
        x = (genvalT() - 0.5) * 4;
      }
    }
    for (auto &x : x.b.m) { // rand basis
      for (auto &x : x) {
        x = (genvalT() - 0.5) * 4;
      }
    }
  }
  valT delta = 1;
  long long tot = 0;
  cout << "\033[?1049h";
  int i = 0;
  while ([delta, &i, &net]() {
    if (fabs(delta) > 1e-8) {
      return true;
    }
    if (i <= 100) {
      ++i;
      return true;
    }
    string ans;
    cout << "\033[2;1HAgain?(Y/n/r/t) \033[2;17H   \033[2;17H";
    getline(cin, ans);
    cout << "\033[H";
    if (ans == "n") {
      return false;
    } else if (ans == "r") {
      i = 0;
    } else if (ans == "t") {
      test(net);
      return true;
    }
    return true;
  }()) {
    valT x = genvalT() * 10;
    valT y = genvalT() * 10;
    VvalT in{x, y};
    net.setInput(in);
    net.getV();
    delta = (net.output[0] - x * 11.25) * (net.output[0] - x * 11.25) +
            (net.output[1] - x - y) * (net.output[1] - x - y);
    cout << "\033[Hdelta: ";
    if (delta < 1e-6) {
      cout << "\033[32;1m";
    }
    cout << delta;
    cout.flush();
    if (delta < 0.5) {
      cout << "\033[0m";
    }
    cout << "     \r";
    VvalT expect{x * 11.25, x + y};
    cout << "\033[1;30Htrain: " << tot++ << " times";
    train(net, in, expect);
  }

  cout << "\033[?1049l";
}
