#include "NN.hpp"
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
#include <cassert>
#include <cmath>
#include <curses.h>
#include <fstream>
#include <ios>
#include <iostream>
#include <ncurses.h>
#include <random>
#include <sstream>
#include <string>
#include <sys/syslimits.h>
#include <vector>
using namespace std;
int main() {
#ifdef USE_OCL
  init();
#endif
  random_device rd;
  nnet net;
  {
    ifstream netin("emnist.net");
#define if_netin_load                                                          \
  if (netin) {                                                                 \
    net.last_layer()->load(netin);                                             \
  }
#define Init net.last_layer()->init(std::move(rd))
    net.add_layer(new convolution_layer);
    if_netin_load else {
      dynamic_cast<convolution_layer *>(net.last_layer())->n_in = 28;
      dynamic_cast<convolution_layer *>(net.last_layer())->m_in = 28;
      dynamic_cast<convolution_layer *>(net.last_layer())->nK = 5;
      dynamic_cast<convolution_layer *>(net.last_layer())->mK = 5;
      net.last_layer()->Ichannels = 1;
      net.last_layer()->Ochannels = 16;
      net.last_layer()->set_IOsize(28 * 28, 28 * 28 * 16);
      Init;
    }

    net.add_layer(new bias_layer);
    if_netin_load else {
      net.last_layer()->set_IOsize(28 * 28 * 16, 28 * 28 * 16);
    }

    net.add_layer(new func_layer);
    if_netin_load else {
      dynamic_cast<func_layer *>(net.last_layer())->f = Functions::tanh;
      net.last_layer()->set_IOsize(28 * 28 * 16, 28 * 28 * 16);
    }

    net.add_layer(new max_layer);
    if_netin_load else {
      dynamic_cast<max_layer *>(net.last_layer())->i_n =
          dynamic_cast<max_layer *>(net.last_layer())->i_m = 28;
      net.last_layer()->Ichannels = net.last_layer()->Ochannels = 16;
      net.last_layer()->set_IOsize(28 * 28 * 16, 14 * 14 * 16);
    }

    net.add_layer(new convolution_layer);
    if_netin_load else {
      dynamic_cast<convolution_layer *>(net.last_layer())->n_in = 14;
      dynamic_cast<convolution_layer *>(net.last_layer())->m_in = 14;
      dynamic_cast<convolution_layer *>(net.last_layer())->nK = 3;
      dynamic_cast<convolution_layer *>(net.last_layer())->mK = 3;
      net.last_layer()->Ichannels = 16;
      net.last_layer()->Ochannels = 24;
      net.last_layer()->set_IOsize(14 * 14 * 16, 14 * 14 * 24);
      Init;
    }
    net.add_layer(new bias_layer);
    if_netin_load else {
      net.last_layer()->set_IOsize(14 * 14 * 24, 14 * 14 * 24);
    }

    net.add_layer(new func_layer);
    if_netin_load else {
      dynamic_cast<func_layer *>(net.last_layer())->f = Functions::tanh;
      net.last_layer()->set_IOsize(14 * 14 * 24, 14 * 14 * 24);
    }

    net.add_layer(new convolution_layer);
    if_netin_load else {
      dynamic_cast<convolution_layer *>(net.last_layer())->n_in = 14;
      dynamic_cast<convolution_layer *>(net.last_layer())->m_in = 14;
      dynamic_cast<convolution_layer *>(net.last_layer())->nK = 3;
      dynamic_cast<convolution_layer *>(net.last_layer())->mK = 3;
      net.last_layer()->Ichannels = 24;
      net.last_layer()->Ochannels = 12;
      net.last_layer()->set_IOsize(14 * 14 * 24, 14 * 14 * 12);
      Init;
    }
    net.add_layer(new bias_layer);
    if_netin_load else {
      net.last_layer()->set_IOsize(14 * 14 * 12, 14 * 14 * 12);
    }

    net.add_layer(new func_layer);
    if_netin_load else {
      dynamic_cast<func_layer *>(net.last_layer())->f = Functions::tanh;
      net.last_layer()->set_IOsize(14 * 14 * 12, 14 * 14 * 12);
    }

    net.add_layer(new max_layer);
    if_netin_load else {
      dynamic_cast<max_layer *>(net.last_layer())->i_n =
          dynamic_cast<max_layer *>(net.last_layer())->i_m = 14;
      net.last_layer()->Ichannels = net.last_layer()->Ochannels = 12;
      net.last_layer()->set_IOsize(14 * 14 * 12, 7 * 7 * 12);
    }

    net.add_layer(new matrix_layer);
    if_netin_load else {
      net.last_layer()->set_IOsize(7 * 7 * 12, 1024);
      Init;
    }

    net.add_layer(new bias_layer);
    if_netin_load else {
      net.last_layer()->set_IOsize(1024, 1024);
      Init;
    }

    net.add_layer(new func_layer);
    if_netin_load else {
      dynamic_cast<func_layer *>(net.last_layer())->f = Functions::ReLU;
      net.last_layer()->set_IOsize(1024, 1024);
      Init;
    }

    net.add_layer(new matrix_layer);
    if_netin_load else {
      net.last_layer()->set_IOsize(1024, 128);
      Init;
    }

    net.add_layer(new bias_layer);
    if_netin_load else {
      net.last_layer()->set_IOsize(128, 128);
      Init;
    }

    net.add_layer(new func_layer);
    if_netin_load else {
      dynamic_cast<func_layer *>(net.last_layer())->f = Functions::tanh;
      net.last_layer()->set_IOsize(128, 128);
      Init;
    }

    net.add_layer(new matrix_layer);
    if_netin_load else {
      net.last_layer()->set_IOsize(128, 47);
      Init;
    }

    net.add_layer(new bias_layer);
    if_netin_load else {
      net.last_layer()->set_IOsize(47, 47);
      Init;
    }

    net.add_layer(new func_layer);
    if_netin_load else {
      dynamic_cast<func_layer *>(net.last_layer())->f = softmax;
      // dynamic_cast<func_layer *>(net.last_layer())->f =  ReLU;
      net.last_layer()->set_IOsize(47, 47);
      Init;
    }
  }

  vector<vector<valT>> inputs;
  vector<vector<valT>> outputs;

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
      inputs.push_back(std::move(tmp));
      int x;
      fin >> x;
      vector<valT> t1;
      t1.resize(47);
      t1[x] = 1;
      outputs.push_back(std::move(t1));
    }
  }
  assert(inputs.size() == outputs.size());
  int total = 0;
  bool quit = false;
  valT lmin = 0, lmax = 5, scale = 1.0;
  mt19937 g(rd());
  int off = 0;
  // int c = 0;
  vector<int> losses(100);
  int losses_c = 0;
  auto save_net = [&net]() {
    ofstream of("emnist.net");
    for (auto l : net.layers) {
      l->save(of);
    }
  };
  valT lr = .001;
  const double beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8;
  net.forward(inputs[0]);
  auto grad_size = net.get_varnum();
  VvalT m(grad_size, 0), v(grad_size, 0), vh(grad_size, 0), mh(grad_size, 0),
      d(grad_size, 0);
  cout << "total " << grad_size << " varibles" << endl;
  initscr();
  nodelay(stdscr, TRUE);
  keypad(stdscr, TRUE);
  noecho();
  for (; !quit;) {
    const auto N = inputs.size();
    if (off >= N) {
      off = 0;
      shuffle(inputs.begin(), inputs.end(), g);
    }
    net.forward(inputs[off]);
    auto g = net.update(inputs[off], outputs[off]);
    for (int i = 0; i < grad_size; ++i) {
      m[i] = beta1 * m[i] + (1 - beta1) * g[i];
      v[i] = beta2 * v[i] + (1 - beta2) * g[i] * g[i];
      mh[i] = m[i] / (1 - pow(beta1, 1 + total / inputs.size()));
      vh[i] = v[i] / (1 - pow(beta2, 1 + total / inputs.size()));
      d[i] = -lr / (sqrt(vh[i]) + epsilon) * mh[i];
    }
    off++;
    net.update(d);
    {
      valT loss = 0;
      for (int j = 0; j < outputs[off].size(); ++j) {
        loss += -outputs[off][j] * log(net.last_layer()->output[j]);
      }
      if (loss > lmin && loss < lmax) {
        losses[100 * (loss - lmin) / (lmax - lmin)]++;
      }
      losses_c++;
      mvprintw(0, 30, "%f", loss);
      mvprintw(0, 40, "%d", total++);
    }
    if (losses_c == 100) {
      // ++c;
      // c %= 36;
      for (int i = 0; i < losses.size(); ++i) {
        int h = scale * 1.0 * (LINES - 5) * losses[i] / losses_c;
        for (int j = 0; j <= h; ++j)
          mvaddch(1 + j, i, 'M'); //"0123456789abcdefghijklmnopqrstuvwxyz"[c]);
        for (int j = h + 1; j < LINES; ++j)
          mvaddch(1 + j, i, ' ');
      }
      losses_c = 0;
      losses.clear();
      losses.resize(100);
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
      mvprintw(LINES - 4, 0, "mIn,mAx,Scale");
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
