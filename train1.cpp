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
#include <omp.h>
#include <string>
#include <utility>
using namespace std;
int main(int argc, char *argv[]) {
  //  if (argc == 1) {
  //    cout << "usage: " << argv[0] << " <1> <2> ...<9>" << endl;
  //    cout << "1 2 .. 9 are files contain data" << endl;
  //    return 1;
  //    // argv[1]="1";
  //  }
  network net(
      {256, 16, 16, 16, 10}, [](valT v) { return max((valT)0, v); },
      [](valT v) -> valT { return (v > 0 ? 1 : 0); });
  /*randomize net*/
  for (layer &x : net.layers) {
    for (auto &x : x.w.m) { // rand matrix
      for (auto &x : x) {
        x = (genvalT() - 0.5) * 4;
        // x = genvalT() * 4 - 5;
        // x = 10;
      }
    }
    for (auto &x : x.b.m) { // rand basis
      for (auto &x : x) {
        x = (genvalT() - 0.5) * 4;
        // x = genvalT() * 4 - 5;
        // x = -2;
      }
    }
  }
  // train(net, string(256, '0'), 1);
  vector<pair<string, int>> data;
  // for (int i = 1; i < argc; ++i) {
  //   ifstream ist(argv[i]);
  for (int i = 1; i <= 3; ++i) {
    ifstream ist("/Users/jiang/Desktop/myAI/" + to_string(i));
    cout << ">>" << i << endl;
    int count = 0;
    string pic;
    while (true) {
      string s;
      if (!(ist >> s)) {
        break;
      }
      cout << setw(2) << count;
      cout << ": -=> " << s << endl;
      pic += s;
      count = (count + 1) % 16;
      if (count == 0) {
        // data.push_back(make_pair(pic, stoi(argv[i])));
        data.push_back(make_pair(pic, i));
        cout << endl;
        // train(net, pic, stoi(argv[i]));
        pic = "";
      }
    }
    cout << endl;
  }
  for (int epoch = 0; []() {
         string s;
         cout << "again?(Y/n):";
         getline(cin, s);
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
    for (auto [pic, expect] : data) {

      VvalT tmp;
      assert(pic.size() == 256);
      tmp.resize(256);
      for (int i = 0; i < 256; ++i) {
        tmp[i] = pic[i] - '0';
      }
      VvalT vexpect(10, 0);
      vexpect[expect] = 10;
      if (net.output.size() != 0) {
        valT delta = 0;
        assert(net.output.size() == vexpect.size() && vexpect.size() == 10);
        for (int i = 0; i < 10; ++i) {
          delta += (net.output[i] - vexpect[i]) * (net.output[i] - vexpect[i]);
        }
        cout << "delta:\t" << std::fixed;
        if (delta < 100) {
          cout << "\033[32m";
        } else if (delta == 100) {
          cout << "\033[31;1m";
        } else {
          cout << "\033[36;2m";
        }
        cout << delta << "\033[0m";
        cout << "time:" << clock() / CLOCKS_PER_SEC;
        {
          const auto &output = net.output;
          cout << setw(10) << "output:";
          for (const auto &x : output) {
            cout << setw(10) << x;
          }
        }
        cout << endl;
      } else {
        cout << "waiting..." << endl;
      }
      train(net, tmp, vexpect);
      // cout << net.getVdVi(i) << endl;
      // net.getVdbi(i);
      // for (int l = 0; l < net.layers.size(); ++l) {
      //   cout << l << " -> " << endl;
      //   cout << "w:" << net.layers[l].w << endl;
      //   cout << "b:" << net.layers[l].b << endl;
      // }
      // {
      //   const auto &output = net.output;
      //   cout << "output: [" << endl;
      //   for (const auto &x : output) {
      //     cout << "[";
      //     cout << x << " ";
      //     cout << "]" << endl;
      //   }
      // }
    }
    // ifstream ist("/Users/jiang/1");
    // int count = 0;
    // string pic;
    // while (true) {
    //   string s;
    //   if (!(ist >> s)) {
    //     break;
    //   }
    //   cout << setw(2) << count;
    //   cout << ": -=> " << s << endl;
    //   pic += s;
    //   count = (count + 1) % 16;
    //   if (count == 0) {
    //     cout << endl;
    //     train(net, pic, 1);
    //     pic = "";
    //   }
    // }
    // cout << endl;
  }
}
