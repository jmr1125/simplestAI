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
  valT grad_size;
  adam A;
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

    if_netin_load(func_layer) else {
      net.add_func_layer({128, 128}, 28 * 28, Functions::ReLU);
      Init;
    }

    if_netin_load(max_layer) else {
      net.add_max_layer({128, 128}, 28, 28, 2);
      Init;
    }

    if_netin_load(convolution_layer) else {
      net.add_convolution_layer({128, 64}, 14, 14, 3, 3, 1);
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
      net.add_matrix_layer({64, 1}, 7 * 7, 128);
      Init;
    }

    if_netin_load(bias_layer) else {
      net.add_bias_layer({1, 1}, 128);
      Init;
    }

    if_netin_load(func_layer) else {
      net.add_func_layer({1, 1}, 128, Functions::tanh);
      Init;
    }

    if_netin_load(matrix_layer) else {
      net.add_matrix_layer({1, 1}, 128, 47);
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
    grad_size = net.get_varnum();
    cout << "total " << grad_size << " varibles" << endl;
    A = adam(grad_size);
    if (netin) {
      for (auto &x : A.m)
        netin >> x;
      for (auto &x : A.mh)
        netin >> x;
      for (auto &x : A.v)
        netin >> x;
      for (auto &x : A.vh)
        netin >> x;
      if (!netin.good()) {
        fill(A.m.begin(), A.m.end(), 0);
        fill(A.mh.begin(), A.mh.end(), 0);
        fill(A.v.begin(), A.v.end(), 0);
        fill(A.vh.begin(), A.vh.end(), 0);
      }
    }
  }

  vector<pair<vector<valT>, pair<vector<valT>, int>>> instance;

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
      instance.push_back(make_pair<VvalT, pair<VvalT, valT>>(
          {}, make_pair<VvalT, valT>({}, 0)));
      instance.back().first = (std::move(tmp));
      int x;
      fin >> x;
      vector<valT> t1;
      t1.resize(47);
      t1[x] = 1;
      instance.back().second.first = (std::move(t1));
      instance.back().second.second = x;
      if (i % 10000 == 0)
        cout << i / 10000 << "0k " << flush;
    }
  }
  sort(instance.begin(), instance.end(),
       [](const pair<vector<valT>, pair<vector<valT>, int>> &a,
          const pair<vector<valT>, pair<vector<valT>, int>> &b) -> bool {
         return a.second.second < b.second.second;
       });
  vector<pair<int, int>> classify(47, {instance.size(), 0});
  for (int i = 0; i < instance.size(); ++i) {
    classify[instance[i].second.second].first =
        min(classify[instance[i].second.second].first, i);
    classify[instance[i].second.second].second++;
  }
  bool quit = false;
  valT lmin = 0, lmax = 5, scale = 1.0;
  mt19937 g(rd());
  // int c = 0;
  vector<int> losses(100);
  int accurate_num = 0;
  int losses_c = 0;
  const double beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8;
  net.forward(instance[0].first);
  auto save_net = [&net, &total, &lr, &A]() {
    ofstream of("emnist.net");
    for (auto l : net.layers) {
      l->save(of);
    }
    of << total << endl;
    of << lr;
    for (auto x : A.m) {
      of << x << ' ';
    }
    for (auto x : A.mh) {
      of << x << ' ';
    }
    for (auto x : A.v) {
      of << x << ' ';
    }
    for (auto x : A.vh) {
      of << x << ' ';
    }
  };
  initscr();
  noecho();
  keypad(stdscr, TRUE);
  mousemask(ALL_MOUSE_EVENTS | REPORT_MOUSE_POSITION, NULL);
  nodelay(stdscr, TRUE);
  for (; !quit;) {
    const auto N = instance.size();
    int off = 0;
    std::uniform_int_distribution<int> r_class(0, 46);
    int classs = r_class(rd);
    std::uniform_int_distribution<int> r_num(0, classify[classs].second - 1);
    off = classify[classs].first + r_num(rd);
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
    bool right = true;
    {
      valT loss = 0;
      for (int j = 0; j < instance[off].second.first.size(); ++j) {
        loss += -instance[off].second.first[j] * log(out[j]);
      }
      if (loss > lmin && loss < lmax) {
        losses[100 * (loss - lmin) / (lmax - lmin)]++;
      }
      if (isnan(loss))
        continue;
      losses_c++;
      mvprintw(0, 30, "%f", loss);
      mvprintw(0, 60, "%d", total++);
      int correct = -1;
      for (int i = 0; i < instance[off].second.first.size(); ++i) {
        if (instance[off].second.first[i] > 0.9) {
          correct = i;
          break;
        }
      }
      if (correct != -1) {
        for (int i = 0; i < out.size(); ++i) {
          if (out[i] > out[correct]) {
            right = false;
            break;
          }
        }
        if (right)
          ++accurate_num;
      }
    }
    // if (!right)
    {
      auto g = net.update(instance[off].first, instance[off].second.first,
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
