#include "layer.h"
#include "main.h"
#include "matrix.h"
#include "network.h"
#include "train.h"
#include <cmath>
#include <iostream>
#include <ostream>
using namespace std;

void train(network &net, const VvalT &input, const VvalT &expect) {
  net.setInput(input);
  net.getV();
  for (int i = 0; i < net.output.size(); ++i) {
    valT DeltadVi = 2 * (net.output[i] - expect[i]);
    vector<pair<matrix, matrix>> deWAndB;
    VvalT deL1;
    deWAndB.resize(net.layers.size());
    deL1.resize(net.layers.size());
    for (int i = 0; i < net.layers.size(); ++i) {
      deWAndB[i].first.setn(net.layers[i].w.getn());
      deWAndB[i].first.setm(net.layers[i].w.getm());
      deWAndB[i].second.setn(net.layers[i].b.getn());
      deWAndB[i].second.setm(net.layers[i].b.getm());
    }
    for (int l = net.layers.size() - 1; l >= 0; --l) {
      for (int i = 0; i < deWAndB[i].first.getn(); ++i) {
        for (int j = 0; j < deWAndB[j].first.getm(); ++j) {
          deWAndB[l].first(i, j) =
              net.layers[l - 1].getV()[j] *
              (net.layers[l - 1].deFuncv()(net.layers[l - 1].getSum()(i, 0)));
        }
        deWAndB[l].second(i, 0) = net.layers[l - 1].deFuncv()(net.layers[l - 1].getSum()(i, 0);
      }
    }
  }
}

valT genvalT() {
  static std::random_device rd;
  valT v = rd();
  v -= rd.min();
  v /= (rd.max() - rd.min());
  return v;
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
