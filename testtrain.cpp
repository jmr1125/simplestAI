#include "main.h"
#include "network.h"
#include "train.h"
#include <iostream>
#include <ostream>
using namespace std;

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
