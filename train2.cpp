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
    int batch_size = 10;

#ifdef gui
    for (; getch() != 'q';) {
#else
    for (; printf("(q?)") && getchar() != 'q';) {
#endif
      vector<VvalT> pics;
      vector<VvalT> exps;
      pics.resize(batch_size);
      exps.resize(batch_size);
      for (int i = 0; i < batch_size; ++i, ++id, id %= inputs.size()) {
        pics[i] = inputs[id];
        exps[i] = outputs[id];

        net.setInput(pics[i]);
        net.getV();
        valT delta = 0;
#ifdef gui
        move(2 + i * 3 + 0, 2);
        printw("%d : output: ", i);
#endif
        for (int x = 0; x < 10; ++x) {
          delta += (net.output[x] - exps[i][x]) * (net.output[x] - exps[i][x]);
#ifdef gui
          printw("%Lf ", net.output[x]);
          clrtoeol();
#else
          printf("%Lf ", net.output[x]);
#endif
        }
        mvprintw(2 + i * 3 + 1, 2, "%d : expect: ", i);
        for (int x = 0; x < 10; ++x) {
#ifdef gui
          printw("%Lf ", exps[i][x]);
#else
          printf("%Lf ", exps[i][x]);
#endif
        }
#ifdef gui
        clrtoeol();
        mvprintw(2 + i * 3 + 2, 2, "%d : delta : %f", i, delta);
        clrtoeol();
        mvaddch(2 + i * 3 + 3, 2, loadingstr[(loadingi = (loadingi + 1) % 4)]);
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
      auto a = [&]() {
        trainn(net, pics, exps,
               (valT)1 / ((16 + 16 + 16 + 10 + 28 * 28 * 16 + 16 * 16 +
                           16 * 16 + 16 * 10) * // number of varibles
                          1000),
               &progress);
      };
#ifdef USE_OMP
      int ok = 0;
#pragma omp parallel num_threads(2) shared(ok, progress)
      {
        if (omp_get_thread_num() == 1) {
          omp_set_num_threads(omp_get_max_threads());
          a();
          ok = 1;
        } else {
          auto t = omp_get_wtime();
          auto T = t;
          while (!ok) {
            while (omp_get_wtime() - t <= 0.3) {
            }
            t = omp_get_wtime();
            mvaddch(2 + (batch_size - 1) * 3 + 3, 2,
                    loadingstr[(loadingi = (loadingi + 1) % 4)]);
            printw(" %Lf ; %Lf%% ", omp_get_wtime() - T, progress * 100);
            refresh();
          }
        }
      }
#else
      clock_t T = clock();
      atomic_int ok = false;
      thread th1([a, &ok]() {
        a();
        ok = true;
      });
      while (!ok) {
        mvaddch(2 + (batch_size - 1) * 3 + 3, 2,
                loadingstr[(loadingi = (loadingi + 1) % 4)]);
        refresh();
        auto t = clock();
        mvprintw(2 + (batch_size - 1) * 3 + 3, 4, " %Lf ; %Lf%% ",
                 valT(t - T) / CLOCKS_PER_SEC, progress * 100);
        while (clock() - t <= CLOCKS_PER_SEC * 0.2) {
        }
      }
      th1.join();
#endif
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
