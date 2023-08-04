#include "layer.h"
#include "main.h"
#include "network.h"
#include "train.h"
#include <algorithm>
#include <atomic>
#include <ctime>
#include <curses.h>
#include <fstream>
#include <iostream>
#include <ncurses.h>
#include <stdexcept>
#ifdef USE_OMP
#include <omp.h>
#else
#include <thread>
#endif
#define gui
using namespace std;
const char loadingstr[]{"/-\\|/-\\|"};
int loadingi = 0;
vector<VvalT> inputs;
vector<VvalT> outputs;
void Readall(istream &in) {
  pair<VvalT, VvalT> (*Read)(istream &) =
      [](istream &in) -> pair<VvalT, VvalT> {
    VvalT input;
    VvalT output(10, -0.5);
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
    output[expect - '0'] = 0.5;
    return make_pair(input, output);
  };
  while (in) {
    try {
      auto [a, b] = (*Read)(in);
      assert(a.size() == 28 * 28);
      assert(b.size() == 10);
      inputs.push_back(std::move(a));
      outputs.push_back(std::move(b));
    } catch (int a) {
      assert(a == 1);
      break;
    } catch (const runtime_error &a) {
      cerr << "runtime error : " << a.what() << endl;
      if (!(inputs.at(59999).size() == 28 * 28 &&
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
    funcT func = funcof(tanh);
    funcT defunc = defuncof(tanh);
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
      net = network({28 * 28, 16, 16, 10}, pair_tanh);
      /*randomize net*/
      for (layer &x : net.layers) {
        for (auto &x : x.w.m) { // rand matrix
          for (auto &x : x) {
            x = (genvalT() - 0.5) * 5;
          }
        }
        for (auto &x : x.b.m) { // rand basis
          for (auto &x : x) {
            x = (genvalT() - 0.5) * 5;
          }
        }
      }
      id = 0;
    }
  }
#ifdef gui
  initscr();
  box(stdscr, ACS_VLINE, ACS_HLINE);
  nodelay(stdscr, TRUE);
#endif
  mvprintw(1, 1, "starting..");
  {
    ifstream in("traindata.txt");
    Readall(in);

#ifdef gui
    for (; getch() != 'q';) {
#else
    for (; printf("(q?)") && getchar() != 'q';) {
#endif
      {

        net.setInput(inputs[id]);
        net.getV();
        valT delta = 0;
#ifdef gui
        move(2, 2);
        printw("%d : output: ", id);
#endif
        for (int x = 0; x < 10; ++x) {
          delta += (net.output[x] - outputs[id][x]) *
                   (net.output[x] - outputs[id][x]);
#ifdef gui
          printw("%Lf ", net.output[x]);
          clrtoeol();
#else
          printf("%Lf ", net.output[x]);
#endif
        }
        mvprintw(3, 2, "%d : expect: ", id);
        for (int x = 0; x < 10; ++x) {
#ifdef gui
          printw("%Lf ", outputs[id][x]);
#else
          printf("%Lf ", outputs[id][x]);
#endif
        }
#ifdef gui
        clrtoeol();
        mvprintw(4, 2, "%d : delta : %f", id, delta);
        clrtoeol();
        mvaddch(5, 2, loadingstr[(loadingi = (loadingi + 1) % 4)]);
        mvprintw(1, 20, "id = %d", id);
        wborder(stdscr, '|', '|', '-', '-', '+', '+', '+', '+');
        refresh();
#else
        printf("\n");
        printf("%d : delta: %Lf\n", i, delta);
        printf("%d : expect: %c\n", i, expect);
        fflush(stdout);
#endif
      }
      valT progress;
      bool ok = false;
      auto a = [&]() {
        train(net, inputs[id], outputs[id],
              (valT)1 / ((16 + 16 + 16 + 10 + 28 * 28 * 16 + 16 * 16 + 16 * 16 +
                          16 * 10) * // number of varibles
                         1000),
              &progress);
        ok = true;
      };
      thread th(a);
      while (!ok) {
        mvprintw(5, 1, "progress: %f%%", progress * 100);
        clrtoeol();
        refresh();
      }
      th.join();
    }
  }
#ifdef gui
  endwin();
#endif
  {
    ofstream out("handwritemnist.net");
    net.save(out);
    out << id << endl;
  }
}
