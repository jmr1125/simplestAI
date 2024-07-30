#include "NN.hpp"
#include "average_layer.hpp"
#include "bias_layer.hpp"
#include "cl-mat.hpp"
#include "convolution.hpp"
#include "convolution_layer.hpp"
#include "func_layer.hpp"
#include "layers.hpp"
#include "main.hpp"
#include "matrix.hpp"
#include "matrix_layer.hpp"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <fstream>
#include <ios>
#include <iostream>
#include <random>
#include <string>
#include <sys/signal.h>
#include <vector>
using namespace std;
int times = 0;
void set_times_to0(int) {
  cin.clear();
  times = 0;
}
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
      net.last_layer()->set_IOsize(28 * 28, 32 * 32);
      Init;
    }

    net.add_layer(new func_layer);
    if_netin_load else {
      dynamic_cast<func_layer *>(net.last_layer())->f = sigmoid;
      net.last_layer()->set_IOsize(32 * 32, 32 * 32);
    }

    net.add_layer(new average_layer);
    if_netin_load else {
      dynamic_cast<average_layer *>(net.last_layer())->i_n =
          dynamic_cast<average_layer *>(net.last_layer())->i_m = 32;
      net.last_layer()->set_IOsize(32 * 32, 16 * 16);
    }

    net.add_layer(new bias_layer);
    if_netin_load else {
      net.last_layer()->set_IOsize(16 * 16, 16 * 16);
      Init;
    }

    net.add_layer(new convolution_layer);
    if_netin_load else {
      dynamic_cast<convolution_layer *>(net.last_layer())->n_in = 16;
      dynamic_cast<convolution_layer *>(net.last_layer())->m_in = 16;
      dynamic_cast<convolution_layer *>(net.last_layer())->nK = 5;
      dynamic_cast<convolution_layer *>(net.last_layer())->mK = 5;
      net.last_layer()->set_IOsize(16 * 16, 20 * 20);
      Init;
    }

    net.add_layer(new func_layer);
    if_netin_load else {
      dynamic_cast<func_layer *>(net.last_layer())->f = sigmoid;
      net.last_layer()->set_IOsize(20 * 20, 20 * 20);
    }

    net.add_layer(new average_layer);
    if_netin_load else {
      dynamic_cast<average_layer *>(net.last_layer())->i_n =
          dynamic_cast<average_layer *>(net.last_layer())->i_m = 20;
      net.last_layer()->set_IOsize(20 * 20, 10 * 10);
    }

    net.add_layer(new bias_layer);
    if_netin_load else {
      net.last_layer()->set_IOsize(10 * 10, 10 * 10);
      Init;
    }

    net.add_layer(new matrix_layer);
    if_netin_load else {
      net.last_layer()->set_IOsize(10 * 10, 64);
      Init;
    }

    net.add_layer(new bias_layer);
    if_netin_load else {
      net.last_layer()->set_IOsize(64, 64);
      Init;
    }

    net.add_layer(new func_layer);
    if_netin_load else {
      dynamic_cast<func_layer *>(net.last_layer())->f = ReLU;
      net.last_layer()->set_IOsize(64, 64);
      Init;
    }
    net.add_layer(new matrix_layer);
    if_netin_load else {
      net.last_layer()->set_IOsize(64, 47);
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
  valT lr;
  bool quit = false;
  valT lmin = 0, lmax = 5;
  valT sum = 0, n = 0;
  int batch_size = 64;
  cout << "learning rate: ";
  cin >> lr;
  cout << "start." << endl;
  mt19937 g(rd());
  int off = 0;
  signal(SIGINT, set_times_to0);
  for (; !quit;) {
    const auto N = inputs.size();
    if (off + batch_size >= N) {
      off = 0;
      shuffle(inputs.begin(), inputs.end(), g);
      sum = 0;
      n = 0;
    }
    vector<valT> d;
    for (int i = off; i < batch_size + off; ++i) {
      net.forward(inputs[i]);
      auto delta = net.update(inputs[i], outputs[i], lr);
      if (d.size() == 0) {
        d = delta;
      } else {
        for (int i = 0; i < 47; ++i) {
          d[i] += delta[i];
        }
      }
    }
    off += batch_size;
    for (auto &x : d) {
      x /= batch_size;
      // valT v = (1.0 * (rd() + rd.min()) / (rd.min() + rd.max()));
      // x *= pow(v, 2);
    }
    net.update(d);
    {
      valT loss = 0;
      for (int j = 0; j < outputs[off].size(); ++j) {
        loss += -outputs[off][j] * log(net.last_layer()->output[j]);
      }
      sum += loss;
      ++n;
      cout << times << " : " << loss << "  lr: " << lr << endl;
      int L = 50;
      const valT scale=1.0/3;
      lmin = floor(loss * scale) / scale, lmax = ceil(loss * scale) / scale;
      int v = L * (loss - lmin) / (lmax - lmin);
      cout << lmin << " |" << string(v, '*') << string(max(0, L - v), ' ')
           << "| " << lmax << " avg: " << (n ? sum / n : 0) << endl;
    }
    if (!times) {
      int cmd;
      cin >> cmd;
      if (cmd == 0) {
        cin >> times;
      } else if (cmd == 1) {
        double x;
        cin >> x;
        times = x * N / batch_size;
      } else if (cmd == 2) {
        cin >> lr;
      } else if (cmd == 3) {
        cin >> batch_size;
      } else if (cmd == 4) {
        break;
      } else {
        cout << "?" << endl;
      }
    }
    --times;
    if (times < 0) {
      times = 0;
    }
  }
  cout << "saving..." << endl;
  cout << lr << endl;
  ofstream of("emnist.net");
  for (auto l : net.layers) {
    l->save(of);
  }
#ifdef USE_OCL
  teardown();
#endif
  return 0;
}
