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
  {
    ifstream netin("emnist.net");
#define if_netin_load(type)                                                    \
  if (netin) {                                                                 \
    net.add_layer(make_shared<type>());                                        \
    net.last_layer()->load(netin);                                             \
  }
#define Init net.last_layer()->init(std::move(rd))
    if_netin_load(convolution_layer) else {
      net.add_convolution_layer({1, 8}, 28, 28, 3, 3);
      Init;
    }

    if_netin_load(bias_layer) else {
      net.add_bias_layer({8, 8}, 28 * 28);
      Init;
    }

    if_netin_load(func_layer) else {
      net.add_func_layer({8, 8}, 28 * 28, Functions::tanh);
      Init;
    }

    if_netin_load(max_layer) else {
      net.add_max_layer({8, 8}, 28, 28, 2);
      Init;
    }

    if_netin_load(convolution_layer) else {
      net.add_convolution_layer({8, 16}, 14, 14, 3, 3);
      Init;
    }
    if_netin_load(bias_layer) else {
      net.add_bias_layer({16, 16}, 14 * 14);
      Init;
    }

    if_netin_load(func_layer) else {
      net.add_func_layer({16, 16}, 14 * 14, Functions::tanh);
      Init;
    }

    if_netin_load(max_layer) else {
      net.add_max_layer({16, 16}, 14, 14, 2);
      Init;
    }

    if_netin_load(matrix_layer) else {
      net.add_matrix_layer({16, 1}, 7 * 7, 512);
      Init;
    }

    if_netin_load(bias_layer) else {
      net.add_bias_layer({1, 1}, 512);
      Init;
    }

    if_netin_load(func_layer) else {
      net.add_func_layer({1, 1}, 512, Functions::ReLU);
      Init;
    }

    if_netin_load(matrix_layer) else {
      net.add_matrix_layer({1, 1}, 512, 64);
      Init;
    }

    if_netin_load(bias_layer) else {
      net.add_bias_layer({1, 1}, 64);
      Init;
    }

    if_netin_load(func_layer) else {
      net.add_func_layer({1, 1}, 64, Functions::sigmoid);
      Init;
    }

    if_netin_load(matrix_layer) else {
      net.add_matrix_layer({1, 1}, 64, 47);
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
  }

  vector<pair<vector<valT>, vector<valT>>> instance;

  cout << "reading..." << endl;
  {
    ifstream fin("traindata.txt");
    int i;
    for (i = 0; fin; ++i) {
      vector<valT> tmp;
      for (int c = 0; c < 28 * 28; ++c) {
        char C;
        fin >> C;
        tmp.push_back(C - '0');
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
  bool quit = false;
  valT lmin = 0, lmax = 5, scale = 1.0;
  mt19937 g(rd());
  int off = 0;
  // int c = 0;
  vector<int> losses(100);
  int losses_c = 0;
  auto save_net = [&net, &total]() {
    ofstream of("emnist.net");
    for (auto l : net.layers) {
      l->save(of);
    }
    of << total;
  };
  valT lr = .001;
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
    net.forward(instance[off].first);
    constexpr int batch_size = 1; // 6;
    array<nnet, batch_size> nets;
    for (int i = 0; i < batch_size; ++i)
      nets[i] = net;
    array<VvalT, batch_size> g;
    #pragma omp parallel for
    for (int i = 0; i < batch_size; ++i) {
      nets[i].forward(instance[off].first);
      g[i] = nets[i].update(instance[off].first, instance[off].second);
      ++off;
    }
    #pragma omp parallel for
    for (int i = 0; i < grad_size; ++i) {
      for (int j = 1; j < batch_size; ++j) {
        g[0][i] += g[j][i];
      }
      g[0][i] /= batch_size;
    }
    auto d = A.update(g[0], lr, total + 1);
    {
      net.update(d);
      net.forward(instance[off].first);
      valT loss = 0;
      for (int j = 0; j < instance[off].second.size(); ++j) {
        loss += -instance[off].second[j] * log(net.last_layer()->output[j]);
      }
      if (loss > lmin && loss < lmax) {
        losses[100 * (loss - lmin) / (lmax - lmin)]++;
      }
      losses_c++;
      mvprintw(0, 30, "%f", loss);
      mvprintw(0, 60, "%d", total++);
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
    }

    mvprintw(0, 0, "%f", lmin);
    mvprintw(0, 10, "%f", lr);
    mvprintw(0, 20, "%d", losses_c);
    mvprintw(0, 90, "%f", lmax);
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
