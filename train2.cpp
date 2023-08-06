#include "function.h"
#include "layer.h"
#include "main.h"
#include "matrix.h"
#include "network.h"
#include <algorithm>
#include <atomic>
#include <cmath>
#include <ctime>
#include <curses.h>
#include <fstream>
#include <iostream>
#include <limits>
#include <ncurses.h>
#include <stdexcept>
#ifdef USE_OMP
#include <omp.h>
#else
#include <thread>
#endif
using namespace std;
vector<matrix> inputs;
vector<VvalT> outputs;
const valT IT_IS = 2, IT_ISNT = 0.5;
void Readall(istream &in) {
  pair<matrix, VvalT> (*Read)(istream &) =
      [](istream &in) -> pair<matrix, VvalT> {
    VvalT input;
    VvalT output(10, IT_ISNT);
    input.reserve(28 * 28);
    for (int i = 0; i < 28; ++i) {
      for (int j = 0; j < 28; ++j) {
        char c;
        // assert(in);
        if (!in) {
          throw 1;
        }
        in >> c;
        if (c == '1') {
          input.push_back(1);
        } else if (c == '0') {
          input.push_back(0);
        } else {
          char str[] = {'c', ' ', '=', ' ', c, '\0'};
          throw runtime_error(str);
          // cerr << "c = " << (int)c << endl;
          // assert(false && "c!='0'|'1'");
        }
      }
    }
    char expect;
    in >> expect;
    assert('0' <= expect && expect <= '9');
    output[expect - '0'] = IT_IS;
    matrix tmp;
    tmp.setn(28 * 28);
    tmp.setm(1);
    for (int i = 0; i < 28 * 28; ++i) {
      tmp(i, 0) = input[i];
    }
    return make_pair(tmp, output);
  };
  int id = 0;
  while (in) {
    try {
      auto [a, b] = (*Read)(in);
      assert(a.getn() == 28 * 28);
      assert(a.getm() == 1);
      assert(b.size() == 10);
      inputs.push_back(std::move(a));
      outputs.push_back(std::move(b));
      if (id % 6000 == 0) {
        cout << (valT)id / 60000 * 100 << "%" << endl;
      }
      ++id;
    } catch (int a) {
      assert(a == 1);
      break;
    } catch (const runtime_error &a) {
      cerr << "runtime error : " << a.what() << endl;
      if (!(inputs.at(59999).getn() == 28 * 28 &&
            outputs.at(59999).size() == 10)) {
        throw;
      } else {
        cerr << "ignore" << endl;
      }
    }
  }
}
int main() {
#ifdef USE_OMP
  omp_set_dynamic(0);
  // omp_set_nested(1);
  omp_set_max_active_levels(5);
#endif
  network net({}, NULL, NULL);
  int id;
  {
    ifstream in("handwritemnist.net");
    funcT func = funcof(ReLU);
    funcT defunc = defuncof(ReLU);
    if (in) {
      net.load(in);
      for (layer &x : net.layers) {
        x.Funct = func;
        x.deFunct = defunc;
      }
      cout << "read from file" << endl;
      assert(in.good());
      in >> id;
      cout << "id = " << id << endl;
    } else {
      // net = network({28 * 28, 16, 16, 16, 10}, pair_ReLU);
      net = network({28 * 28, 64, 10}, pair_ReLU);
      /*randomize net*/
      for (layer &l : net.layers) {
        for (auto &x : l.w.m) { // rand matrix
          for (auto &x : x) {
            x = (genvalT() - 0.5) * 2 * (1 / sqrt(l.w.getm()));
          }
        }
        for (auto &x : l.b.m) { // rand basis
          for (auto &x : x) {
            x = (genvalT() - 0.5) * 2 * (1 / sqrt(l.b.getn()));
          }
        }
      }
      id = 0;
    }
  }
  ifstream in("traindata.txt");
  Readall(in);
  initscr();
  box(stdscr, ACS_VLINE, ACS_HLINE);
  nodelay(stdscr, TRUE);
  mvprintw(1, 1, "starting..");
  vector<valT> deltas(100, 0);
  int deltasI = 0;
  for (; getch() != 'q'; ++id, id %= 60000) {
    const auto &a = inputs.at(id);
    const auto &b = outputs.at(id);
    const auto o = net.feed_forward(a);
    valT delta = 0;
    for (int i = 0; i < 10; ++i) {
      delta += (b[i] - o(i, 0)) * (b[i] - o(i, 0));
    }
    mvprintw(1, 1, "delta= %f  id= %d", delta, id);
    clrtoeol();
    move(2, 1);
    clrtoeol();
    for (int i = 0; i < 10; ++i) {
      printw("%f ", o(i, 0));
    }
    move(3, 1);
    clrtoeol();
    for (int i = 0; i < 10; ++i) {
      printw("%f ", b[i]);
    }
    {
      deltas[deltasI] = delta;
      ++deltasI;
      deltasI %= deltas.size();
      valT Min = numeric_limits<valT>::max();
      valT Max = numeric_limits<valT>::min();
#ifdef USE_OMP
#pragma parallel for
#endif
      valT sum = 0;
      for (auto x : deltas) {
        Min = min(Min, x);
        Max = max(Max, x);
        sum += x;
      }
      Max = (valT)ceil(Max * 2) / 2;
      Min = (valT)floor(Min * 2) / 2;
      int x1 = 5, y1 = 1, x2 = 25, y2 = y1 + deltas.size();
      vector<vector<char>> ch(x2 - x1 + 1, vector<char>(100, ' '));
#ifdef USE_OMP
#pragma parallel for
#endif
      for (int i = 0; i < deltas.size(); ++i) {
        valT percent = (deltas[i] - Min) / (Max - Min);
        // valT percent = min(valT(1), deltas[i] / 2);
        int x = (x2 - x1) * percent;
        ch.at(x).at(i) = '#';
      }
      mvprintw(x1 - 1, 2, "%f", Min);
      mvprintw(x2 + 1, 2, "%f", Max);
      for (int i = 0; i < x2 - x1; ++i) {
        for (int j = 0; j < 100; ++j) {
          mvaddch(i + x1, 8 + j + y1, ch.at(i).at(j));
        }
      }
      mvprintw(x1 + (x2 - x1) * (sum / deltas.size() - Min) / (Max - Min), 9,
               "==AVG== %f", sum / deltas.size());
    }
    box(stdscr, '|', '-');
    refresh();
    net.backpropagation(a, b, 0.000001);
    // {
    //   clock_t now = clock();
    //   while (clock() - now <= CLOCKS_PER_SEC * 0.0001) {
    //   }
    // }
  }
  endwin();
  {
    ofstream out("handwritemnist.net");
    net.save(out);
    out << id << endl;
  }
}
