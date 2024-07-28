#include "NN.hpp"
#include "bias_layer.hpp"
#include "cl-mat.hpp"
#include "convolution.hpp"
#include "convolution_layer.hpp"
#include "func_layer.hpp"
#include "layers.hpp"
#include "main.hpp"
#include "matrix_layer.hpp"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <fstream>
#include <ios>
#include <iostream>
#include <random>
#include <string>
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
    net.add_layer(new convolution_layer);
    if (netin) {
      net.last_layer()->load(netin);
    } else {
      dynamic_cast<convolution_layer *>(net.last_layer())->n_in = 28;
      dynamic_cast<convolution_layer *>(net.last_layer())->m_in = 28;
      dynamic_cast<convolution_layer *>(net.last_layer())->nK = 5;
      dynamic_cast<convolution_layer *>(net.last_layer())->mK = 5;
      net.last_layer()->set_IOsize(28 * 28, 32 * 32);
      net.last_layer()->init(std::move(rd));
    }

    net.add_layer(new bias_layer);
    if (netin) {
      net.last_layer()->load(netin);
    } else {
      net.last_layer()->set_IOsize(32 * 32, 32 * 32);
      net.last_layer()->init(std::move(rd));
    }

    net.add_layer(new func_layer);
    if (netin) {
      net.last_layer()->load(netin);
    } else {
      dynamic_cast<func_layer *>(net.last_layer())->f = Functions::tanh;
      net.last_layer()->set_IOsize(32 * 32, 32 * 32);
    }

    net.add_layer(new matrix_layer);
    if (netin) {
      net.last_layer()->load(netin);
    } else {
      net.last_layer()->set_IOsize(32 * 32, 128);
      net.last_layer()->init(std::move(rd));
    }

    net.add_layer(new bias_layer);
    if (netin) {
      net.last_layer()->load(netin);
    } else {
      net.last_layer()->set_IOsize(128, 128);
      net.last_layer()->init(std::move(rd));
    }

    net.add_layer(new func_layer);
    if (netin) {
      net.last_layer()->load(netin);
    } else {
      dynamic_cast<func_layer *>(net.last_layer())->f = ReLU;
      net.last_layer()->set_IOsize(128, 128);
    }

    net.add_layer(new matrix_layer);
    if (netin) {
      net.last_layer()->load(netin);
    } else {
      net.last_layer()->set_IOsize(128, 47);
      net.last_layer()->init(std::move(rd));
    }

    net.add_layer(new bias_layer);
    if (netin) {
      net.last_layer()->load(netin);
    } else {
      net.last_layer()->set_IOsize(47, 47);
      net.last_layer()->init(std::move(rd));
    }

    net.add_layer(new func_layer);
    if (netin) {
      net.last_layer()->load(netin);
    } else {
      dynamic_cast<func_layer *>(net.last_layer())->f = softmax;
      net.last_layer()->set_IOsize(47, 47);
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
  int times = 0;
  valT lmin = 100, lmax = 0;
  cout << "learning rate: ";
  cin >> lr;
  cout << "start." << endl;
  mt19937 g(rd());
  int off = 0;
  for (; !quit;) {
    const auto N = inputs.size();
    const int batch_size = 32;
    if (off + batch_size >= N) {
      off = 0;
      shuffle(inputs.begin(), inputs.end(), g);
      lmin = 100, lmax = 0;
      lr /= 1.01;
      cout << "lr: " << lr << endl;
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
    }
    net.update(d);
    {
      valT loss = 0;
      for (int j = 0; j < outputs[0].size(); ++j) {
        loss += -outputs[0][j] * log(net.last_layer()->output[j]);
      }
      lmin = min(lmin, loss);
      lmax = max(lmax, loss);
      cout << times << " : " << loss << endl;
      int L = 30;
      int v = L * (loss - lmin) / (lmax - lmin);
      cout << lmin << " |" << string(v, '*') << string(L - v, '.') << "| "
           << lmax << endl;
    }
    {
      double x;
      if (!times) {
        cin >> x;
        times = x * N / batch_size;
      } else
        --times;
      if (times <= -1) {
        quit = 1;
        break;
      }
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
