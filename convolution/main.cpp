#include "NN.hpp"
#include "bias_layer.hpp"
#include "convolution.hpp"
#include "convolution_layer.hpp"
#include "func_layer.hpp"
#include "layers.hpp"
#include "matrix_layer.hpp"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <fstream>
#include <ios>
#include <iostream>
#include <random>
#include "../cl-mat.hpp"
using namespace std;
int main() {
#ifdef USE_OCL
  init();
#endif
  random_device rd;
  nnet net;
  int Istart = -1;
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
      net.last_layer()->set_IOsize(32 * 32, 80);
      net.last_layer()->init(std::move(rd));
    }

    net.add_layer(new bias_layer);
    if (netin) {
      net.last_layer()->load(netin);
    } else {
      net.last_layer()->set_IOsize(80, 80);
      net.last_layer()->init(std::move(rd));
    }

    net.add_layer(new func_layer);
    if (netin) {
      net.last_layer()->load(netin);
    } else {
      dynamic_cast<func_layer *>(net.last_layer())->f = ReLU;
      net.last_layer()->set_IOsize(80, 80);
    }

    net.add_layer(new matrix_layer);
    if (netin) {
      net.last_layer()->load(netin);
    } else {
      net.last_layer()->set_IOsize(80, 47);
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
    if (netin) {
      netin >> Istart;
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
  valT sum = 0;
  valT lr;
  bool quit = false;
  int times = 0;
  int i;
  cout << "learning rate: ";
  cin >> lr;
  cout << "start." << endl;
  i = max(0, Istart);
  for (; !quit;) {
    const auto N = inputs.size();
    i %= N;
    for (; i < N; ++i) {
      net.forward(inputs[i]);
      net.update(inputs[i], outputs[i], lr);
      valT loss = 0;
      for (int j = 0; j < outputs[i].size(); ++j) {
        loss += -outputs[i][j] * log(net.last_layer()->output[j]);
      }
      sum += loss;
      if (i % 50 == 0) {
        cout << times << ", " << i << " : " << sum / 50.0 << endl;
        sum = 0;
        if (!times)
          cin >> times;
        else
          --times;
        if (times <= -1) {
          quit = 1;
          Istart = i + 1;
          break;
        }
        // printf("%d : %f\n", i, sum / 50.0);
      }
    }
  }
  cout << "saving..." << endl;
  cout << lr << endl;
  ofstream of("emnist.net");
  for (auto l : net.layers) {
    l->save(of);
  }
  of << Istart;
#ifdef USE_OCL
  teardown();
#endif
  return 0;
}
