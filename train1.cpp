#include "function.hpp"
#include "layer.hpp"
#include "main.hpp"
#include "matrix.hpp"
#include "network.hpp"
#include <algorithm>
#include <cstdio>
#include <ctime>
#include <curses.h>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <ncurses.h>
#include <string>
#include <thread>
// #include <unistd.h>
#include <utility>
#ifdef USE_OMP
#include <omp.h>
#endif

#ifdef USE_OMP
#define clock() omp_get_wtime()
#else
#define clock() ((double)clock() / CLOCKS_PER_SEC)
#endif

#define getcharcond(ch, cond)                                                  \
  {                                                                            \
    char cccccccccc;                                                           \
    while ((cccccccccc = getch()) == ERR && cond)                              \
      ;                                                                        \
    ch = cccccccccc;                                                           \
  }
#define getchartime(ch, time)                                                  \
  {                                                                            \
    clock_t t = clock();                                                       \
    getcharcond(ch, (clock() - t) <= time);                                    \
  }
#define getchardelay(ch) getcharcond(ch, 1);

using namespace std;
void save(const network &net) {
  ofstream out("handwrite.net");
  net.save(out);
}
int main(int argc, char *argv[]) { // blr?blur?...k
#ifdef USE_OMP
  omp_set_max_active_levels(3);
#endif
  if (argc == 1) {
    cout << "usage: " << argv[0] << " <1> <2> ...<9>" << endl;
    cout << "1 2 .. 9 are files contain data" << endl;
    return 1;
  }
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
            x = (genvalT() - 0.5);
          }
        }
        for (auto &x : x.b.m) { // rand basis
          for (auto &x : x) {
            x = (genvalT() - 0.5);
          }
        }
      }
    }
  }
  vector<pair<matrix, VvalT>> data;
  for (int i = 1; i < argc; ++i) {
    ifstream ist(argv[i]);
    int count = 0;
    string pic;
    while (true) {
      string s;
      if (!(ist >> s)) {
        break;
      }
      pic += s;
      count = (count + 1) % 16;
      if (count == 0) {
        matrix tmp;
        assert(pic.size() == 256);
        tmp.setn(256);
        tmp.setm(1);
        for (int i = 0; i < 256; ++i) {
          tmp(i, 0) = pic[i] - '0';
        }
        VvalT tmp1(10, 0);
        tmp1[stoi(argv[i])] = 2;
        data.push_back(make_pair(tmp, tmp1));
        pic = "";
      }
    }
  }
  auto start = clock();
  cout << "starting..." << start << endl;
  initscr();
  intrflush(stdscr, TRUE), keypad(stdscr, TRUE);
  start_color();
  init_pair(1, COLOR_RED, COLOR_BLACK);
  init_pair(2, COLOR_CYAN, COLOR_BLACK);
  init_pair(3, COLOR_GREEN, COLOR_BLACK);
  nodelay(stdscr, TRUE);
  int t = 0;
  valT scale = 100;
  for (int epoch = 0; [&t]() {
         string s{};
         mvprintw(0, 0, "again?(Y/n/r)");
         clrtoeol();
         if (t) {
           printw(" y");
           clrtoeol();
           t = (t + 1) % 100;
           mvprintw(4, 0, "  r:%d", t);
           clrtoeol();
           return true;
         }
         int c;
         getchartime(c, 5);
         move(4, 0);
         clrtoeol();
         s = " ";
         s[0] = c;
         if (s == "n") {
           return false;
         } else if (s == "r") {
           t = 1;
           return true;
         } else {
           return true;
         }
       }();
       ++epoch) {
    //====EPOCH.FOR.LOOP====
    mvprintw(0, 0, "epoch: %d\n", epoch);
    clrtoeol();
    {
      random_device rd;
      shuffle(data.begin(), data.end(), rd);
    }
    for (const auto &data : data) {
      auto output = net.feed_forward(data.first);
      assert(output.getm() == 1);
      valT delta = 0;
      for (int i = 0; i < output.getn(); ++i) {
        delta +=
            (data.second[i] - output(i, 0)) * (data.second[i] - output(i, 0));
      }
      mvprintw(1, 1, "delta: %f", delta);
      move(2, 1);
      for (int i = 0; i < output.getn(); ++i) {
        printw("%f ", output(i, 0));
      }
      move(3, 1);
      for (int i = 0; i < data.second.size(); ++i) {
        printw("%f ", data.second[i]);
      }
      net.backpropagation(data.first, data.second, 0.0001);
      refresh();
      // char c;
      // getchartime(c, 1);
      // if (c == 'q') {
      //   break;
      // }
    }
    save(net);
  }
  endwin();
}
