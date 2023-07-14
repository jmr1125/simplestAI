#include "layer.h"
#include "main.h"
#include "matrix.h"
#include "network.h"
#include "train.h"
#include <algorithm>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <utility>
#ifdef USE_OMP
#include <omp.h>
#endif
using namespace std;
int main(int argc, char *argv[]) {
  //  if (argc == 1) {
  //    cout << "usage: " << argv[0] << " <1> <2> ...<9>" << endl;
  //    cout << "1 2 .. 9 are files contain data" << endl;
  //    return 1;
  //    // argv[1]="1";
  //  }
  // network net(
  //     {256, 16, 16, 16, 10}, [](valT v) { return max((valT)0, v); },
  //     [](valT v) -> valT { return (v >= 0 ? 1 : 0); });
  network net({}, NULL, NULL);
  {
    ifstream in("handwrite.net");
    funcT func = [](valT v) { return max((valT)0, v); };
    funcT defunc = [](valT v) -> valT { return (v >= 0 ? 1 : 0); };
    if (in) {
      net.load(in);
      for (layer &x : net.layers) {
        x.Funct = func;
        x.deFunct = defunc;
      }
      cout << "read from file" << endl;
    } else {
      net = network({256, 16, 16, 10}, func, defunc);
      /*randomize net*/
      for (layer &x : net.layers) {
        for (auto &x : x.w.m) { // rand matrix
          for (auto &x : x) {
            x = (genvalT() - 0.5) * 5; // 16 8
            // x = genvalT() * 4 - 5;
            // x = 10;
          }
        }
        for (auto &x : x.b.m) { // rand basis
          for (auto &x : x) {
            x = (genvalT() - 0.5) * 5;
            // x = genvalT() * 4 - 5;
            // x = -2;
          }
        }
      }
    }
  }
  // train(net, string(256, '0'), 1);
  vector<pair<string, int>> data;
  for (int i = 1; i < argc; ++i) {
    ifstream ist(argv[i]);
    // for (int i = 1; i <= 3; ++i) {
    //   ifstream ist("/Users/jiang/Desktop/myAI/" + to_string(i));
    // cout << ">>" << i << endl;
    int count = 0;
    string pic;
    while (true) {
      string s;
      if (!(ist >> s)) {
        break;
      }
      // cout << setw(2) << count;
      // cout << ": -=> " << s << endl;
      pic += s;
      count = (count + 1) % 16;
      if (count == 0) {
        // data.push_back(make_pair(pic, stoi(argv[i])));
        data.push_back(make_pair(pic, i));
        // cout << endl;
        // train(net, pic, stoi(argv[i]));
        pic = "";
      }
    }
    // cout << endl;
  }
#ifdef USE_OMP
  auto start = omp_get_wtime();
#else
  auto start = clock();
#endif
  for (int epoch = 0; []() {
         string s{};
         cout << "again?(Y/n):";
         // getline(cin, s);
         if (s == "n") {
           return false;
         } else {
           return true;
         }
       }();
       ++epoch) {
    cout << "epoch: " << epoch << endl;
    {
      random_device rd;
      shuffle(data.begin(), data.end(), rd);
    }
    for (size_t i = 0; i < data.size(); ++i) {
      auto [pic, expect] = data[i];
      VvalT tmp;
      assert(pic.size() == 256);
      tmp.resize(256);
      for (int i = 0; i < 256; ++i) {
        tmp[i] = pic[i] - '0';
      }
      VvalT vexpect(10, 5);
      vexpect[expect] = 10;
      valT delta = 0;
      net.setInput(tmp);
      net.getV();
      assert(net.output.size() == vexpect.size() && vexpect.size() == 10);
      for (int i = 0; i < 10; ++i) {
        delta += (net.output[i] - vexpect[i]) * (net.output[i] - vexpect[i]);
      }
      cout << i << " / " << data.size() << " : ";
      cout << "delta:" << setw(8) << std::fixed;
      if (delta < 325) {
        cout << "\033[32m";
      } else if (delta == 325) {
        cout << "\033[31;1m";
      } else {
        cout << "\033[36;2m";
      }
      cout << delta << "\033[0m";
      cout << setw(9) << "time:";
#ifdef USE_OMP
      cout << omp_get_wtime() - start;
#else
      cout << clock() / CLOCKS_PER_SEC - start;
#endif
      {
        const auto &output = net.output;
        cout << setw(10) << "output:";
        for (const auto &x : output) {
          cout << setw(10) << x;
        }
      }
      cout << endl;
      // cout << "\r";
      train(net, tmp, vexpect);
    }
    {
      ofstream out("handwrite.net");
      net.save(out);
    }
  }
  {
    ofstream out("handwrite.net");
    net.save(out);
  }
}
