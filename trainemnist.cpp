#include "NN.hpp"
#include "adam.hpp"
#include "average_layer.hpp"
#include "bias_layer.hpp"
#include "convolution.hpp"
#include "convolution_layer.hpp"
#include "func_layer.hpp"
#include "layers.hpp"
#include "main.hpp"
#include "matrix.hpp"
#include "matrix_layer.hpp"
#include "max_layer.hpp"
#include "ocl.hpp"
#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <curses.h>
#include <fstream>
#include <ios>
#include <iostream>
#include <memory>
#include <ncurses.h>
#include <random>
#include <sstream>
#include <string>
#include <vector>
using namespace std;
int main() {
#ifdef USE_OCL
  init();
#endif
  random_device rd;
  nnet net;
  int total = 0;
  valT lr = .001;
  {
    ifstream netin("emnist.net");
#define if_netin_load(type)                                                    \
  if (netin) {                                                                 \
    net.add_layer(make_shared<type>());                                        \
    net.last_layer()->load(netin);                                             \
  }
#define Init net.last_layer()->init(std::move(rd))
    if_netin_load(convolution_layer) else {
      net.add_convolution_layer({1, 128}, 28, 28, 5, 5, 2);
      Init;
    }

    if_netin_load(bias_layer) else {
      net.add_bias_layer({128, 128}, 28 * 28);
      Init;
    }

    if_netin_load(func_layer) else {
      net.add_func_layer({128, 128}, 28 * 28, Functions::ReLU);
      Init;
    }

    if_netin_load(max_layer) else {
      net.add_max_layer({128, 128}, 28, 28, 2);
      Init;
    }

    if_netin_load(convolution_layer) else {
      net.add_convolution_layer({128, 64}, 14, 14, 5, 5, 2);
      Init;
    }

    if_netin_load(bias_layer) else {
      net.add_bias_layer({64, 64}, 14 * 14);
      Init;
    }

    if_netin_load(func_layer) else {
      net.add_func_layer({64, 64}, 14 * 14, Functions::ReLU);
      Init;
    }

    if_netin_load(max_layer) else {
      net.add_max_layer({64, 64}, 14, 14, 2);
      Init;
    }

    if_netin_load(matrix_layer) else {
      net.add_matrix_layer({64, 1}, 7 * 7, 512);
      Init;
    }

    if_netin_load(bias_layer) else {
      net.add_bias_layer({1, 1}, 512);
      Init;
    }

    if_netin_load(func_layer) else {
      net.add_func_layer({1, 1}, 512, Functions::tanh);
      Init;
    }

    if_netin_load(matrix_layer) else {
      net.add_matrix_layer({1, 1}, 512, 47);
      Init;
    }

    if_netin_load(bias_layer) else {
      net.add_bias_layer({1, 1}, 47);
      Init;
    }

    if_netin_load(func_layer) else {
      net.add_func_layer({1, 1}, 47, Functions::softmax);
      Init;
    }

    netin >> total;
    netin >> lr;
  }

  vector<pair<vector<valT>, vector<valT>>> instance;

  cout << "reading..." << endl;
  {
    ifstream fin("traindata.txt");
    int i;
    for (i = 0; fin; ++i) {
      vector<valT> tmp;
      for (int c = 0; c < 28 * 28; ++c) {
        valT C;
        fin >> C;
        tmp.push_back(C);
      }
      instance.push_back(make_pair<VvalT, VvalT>({}, {}));
      instance.back().first = (std::move(tmp));
      int x;
      fin >> x;
      vector<valT> t1;
      t1.resize(47);
      t1[x] = 1;
      instance.back().second = (std::move(t1));
    }
  }
  // sort(instance.begin(), instance.end(),
  //      [](pair<vector<valT>, vector<valT>> a,
  //         pair<vector<valT>, vector<valT>> b) -> bool {
  //        return a.second < b.second;
  //      });
  // instance = [&]() -> vector<pair<vector<valT>, vector<valT>>> {
  //   vector<pair<vector<valT>, vector<valT>>> tmp;
  //   tmp.resize(instance.size());
  //   for (int i = 0; i < tmp.size(); ++i) {
  //     tmp[i] = instance[(i % 47) * 2400 + i / 47];
  //   }
  //   return tmp;
  // }();
  bool quit = false;
  valT lmin = 0, lmax = 5, scale = 1.0;
  mt19937 g(rd());
  int off = 0;
  // int c = 0;
  vector<int> losses(100);
  int accurate_num = 0;
  int losses_c = 0;
  auto save_net = [&net, &total, &lr]() {
    ofstream of("emnist.net");
    for (auto l : net.layers) {
      l->save(of);
    }
    of << total << endl;
    of << lr;
  };
  const double beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8;
  net.forward(instance[0].first);
  auto grad_size = net.get_varnum();
  adam A(grad_size);
  cout << "total " << grad_size << " varibles" << endl;
  initscr();
  noecho();
  keypad(stdscr, TRUE);
  mousemask(ALL_MOUSE_EVENTS | REPORT_MOUSE_POSITION, NULL);
  nodelay(stdscr, TRUE);
  for (; !quit;) {
    const auto N = instance.size();
    if (off >= N) {
      off = 0;
      shuffle(instance.begin(), instance.end(), g);
    }
    auto out = net.forward(instance[off].first);
#if 0
    {
      for (int x = 0; x < 5; ++x) {
        move(3 + x, 60);
        for (int i = 10 * x; i < min((size_t)10 * x + 10, out.size()); ++i) {
          printw("%.3f ", out[i]);
        }
      }
      for (int x = 0; x < 5; ++x) {
        move(8 + x, 60);
        for (int i = 10 * x;
             i < min((size_t)10 * x + 10, instance[off].second.size()); ++i) {
          printw("%.3f ", instance[off].second[i]);
        }
      }
    }
#endif
#if 0
    {
      for (int i = 0; i < 28; ++i) {
        move(3 + i, 0);
        for (int j = 0; j < 28; ++j) {
          printw("%.0f", instance[off].first.at(i * 28 + j));
        }
      }
    }
#endif
    bool right = false;
    {
      valT loss = 0;
      for (int j = 0; j < instance[off].second.size(); ++j) {
        loss += -instance[off].second[j] * log(out[j]);
      }
      if (loss > lmin && loss < lmax) {
        losses[100 * (loss - lmin) / (lmax - lmin)]++;
      }
      losses_c++;
      mvprintw(0, 30, "%f", loss);
      mvprintw(0, 60, "%d", total++);
      int correct = -1;
      for (int i = 0; i < instance[off].second.size(); ++i) {
        if (instance[off].second[i] > 0.9) {
          correct = i;
          break;
        }
      }
      if (correct != -1) {
        for (int i = 0; i < out.size(); ++i) {
          if (out[i] > out[correct]) {
            right = true;
            break;
          }
        }
        if (right)
          ++accurate_num;
      }
    }
    if (!right) {
      auto g = net.update(instance[off].first, instance[off].second,
                          train_method::loss);
      auto d = A.update(g, lr, total / instance.size() + 1);
      // for (auto &x : g)
      //   x *= -lr;
      // net.update(g);
      net.update(d);
#if 1
      net.randomize_nan(std::move(rd));
#endif
    }
    if (losses_c == 100) {
      for (int i = 0; i < losses.size(); ++i) {
        int h = scale * 1.0 * (LINES - 5) * losses[i] / losses_c;
        for (int j = 0; j <= h; ++j)
          mvaddch(1 + j, i, 'M');
        for (int j = h + 1; j < LINES; ++j)
          mvaddch(1 + j, i, ' ');
      }
      losses_c = 0;
      fill(losses.begin(), losses.end(), 0);
      mvprintw(LINES - 5, 3, "accurate: %f",
               accurate_num * 1.0 / losses.size());
      accurate_num = 0;
    }
#if 0
    for (int x = 0; x < 5; ++x) {
      move(LINES - 5 + x, 60);
      for (int i = 10 * x; i < min((size_t)10 * x + 10, out.size()); ++i) {
        printw("%.3f ", out[i]);
      }
    }
#endif

    mvprintw(0, 0, "%f", lmin);
    mvprintw(0, 10, "%f", lr);
    mvprintw(0, 20, "%d", losses_c);
    mvprintw(0, 90, "%f", lmax);
    ++off;
    refresh();
    auto C = getch();
    stringstream ss;
    auto read_num = [&C, &ss]() {
      clrtoeol();
      while (C != '\n') {
        if (('0' <= C && C <= '9') || C == '.' || C == 'e' || C == '+' ||
            C == '-')
          ss << (char)C;
        C = getch();
        mvprintw(LINES - 4, 5, "> %s", ss.str().c_str());
      }
    };
    if (C == 'S') {
      mvprintw(LINES - 4, 0, "save");
      save_net();
    } else if (C == 'l') {
      mvprintw(LINES - 4, 0, "lr");
      read_num();
      ss >> lr;
    } else if (C == 'm') {
      mvprintw(LINES - 4, 0, "mIn,mAx,Scale,Query");
      while ((C = getch()) == ERR)
        ;
      if (C == 'i') {
        mvprintw(LINES - 4, 0, "min");
        read_num();
        ss >> lmin;
      }
      if (C == 'a') {
        mvprintw(LINES - 4, 0, "max");
        read_num();
        ss >> lmax;
      }
      if (C == 's') {
        mvprintw(LINES - 4, 0, "scle");
        read_num();
        ss >> scale;
      }
      if (C == 'q') {
        while ((C = getch()) != KEY_MOUSE)
          ;
        MEVENT mouseevt;
        if (getmouse(&mouseevt) == OK) {
          mvprintw(mouseevt.y, mouseevt.x, "%f",
                   (1.0 * mouseevt.x / losses.size()) * (lmax - lmin) + lmin);
        }
      }
    } else if (C == 'q')
      break;
  }
  save_net();
#ifdef USE_OCL
  teardown();
#endif
  endwin();
  return 0;
}
