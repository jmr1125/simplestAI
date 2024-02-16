#include "function.hpp"
#include "layer.hpp"
#include "main.hpp"
#include "matrix.hpp"
#include "network.hpp"
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
const int OUTPUTSIZE = 47;
const int samplesize = 112800;
const valT IT_IS = 2, IT_ISNT = 0.2;
void Readall(istream &in) {
  pair<matrix, VvalT> (*Read)(istream &) =
      [](istream &in) -> pair<matrix, VvalT> {
    VvalT input;
    VvalT output(OUTPUTSIZE, IT_ISNT);
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
    int expect;
    in >> expect;
    output[expect] = IT_IS;
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
      assert(b.size() == OUTPUTSIZE);
      inputs.push_back(std::move(a));
      outputs.push_back(std::move(b));
      if (id % 6000 == 0) {
        cout << (valT)id / samplesize * 100 << "%" << endl;
      }
      ++id;
    } catch (int a) {
      assert(a == 1);
      break;
    } catch (const runtime_error &a) {
      cerr << "runtime error : " << a.what() << endl;
      if (!(inputs.at(samplesize - 1).getn() == 28 * 28 &&
            outputs.at(samplesize - 1).size() == OUTPUTSIZE)) {
        throw;
      } else {
        cerr << "ignore" << endl;
      }
    }
  }
}
#ifdef USE_OCL
#include "cl-mat.hpp"
#endif
int main() {
#ifdef USE_OCL
  init();
#endif
#ifdef USE_OMP
  omp_set_dynamic(0);
  // omp_set_nested(1);
  omp_set_max_active_levels(5);
#endif
  network net({}, NULL, NULL);
  int id;
  int epoch;
  {
    ifstream in("handwriteemnist.net");
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
      assert(in.good());
      in >> epoch;
      cout << "id = " << id << endl << "epoch = " << epoch << endl;
    } else {
      net = network({28 * 28, 128, 64, OUTPUTSIZE}, pair_ReLU);
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
      epoch = -1;
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
  valT lr;
  for (; getch() != 'q'; ++id, id %= samplesize) {
    if (id == 0) {
      ++epoch;
    }
    if (id == 0 || lr == 0) {
      lr = 0.004 * pow(0.95, epoch);
    }
    const auto &a = inputs.at(id);
    const auto &b = outputs.at(id);
    const auto o = net.feed_forward(a);
    valT delta = 0;
    for (int i = 0; i < OUTPUTSIZE; ++i) {
      delta += (b[i] - o(i, 0)) * (b[i] - o(i, 0));
    }
    mvprintw(1, 1, "delta= %f  id= %d epoch= %d learningRate= %f", delta, id,
             epoch, lr);
    clrtoeol();
    move(2, 1);
    clrtoeol();
    for (int i = 0; i < OUTPUTSIZE; ++i) {
      printw("%.2f ", o(i, 0));
    }
    move(3, 1);
    clrtoeol();
    for (int i = 0; i < OUTPUTSIZE; ++i) {
      printw("%.2f ", b[i]);
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
      // Max = (valT)ceil(Max * 2) / 2;
      // Min = (valT)floor(Min * 2) / 2;
      mvprintw(5, 1, "Min = %f", Min);
      mvprintw(6, 1, "Max = %f", Max);
      mvprintw(7, 1, "Avg = %f", sum / deltas.size());
      box(stdscr, '|', '-');
      refresh();
      net.backpropagation(a, b, 0.00001);
    }
  }
  endwin();
  {
    ofstream out("handwriteemnist.net");
    net.save(out);
    out << id << endl << epoch;
  }
#ifdef USE_OCL
  teardown();
#endif
}
