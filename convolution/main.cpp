#include "NN.hpp"
#include "bias_layer.hpp"
#include "convolution.hpp"
#include "func_layer.hpp"
#include "matrix_layer.hpp"
#include <cmath>
#include <ios>
#include <iostream>
#include <random>
using namespace std;
int main() {
  random_device rd;
  nnet net;
  net.add_layer(new matrix_layer);
  net.last_layer()->set_IOsize(4, 8);
  net.last_layer()->init(std::move(rd));

  net.add_layer(new bias_layer);
  net.last_layer()->set_IOsize(8, 8);
  net.last_layer()->init(std::move(rd));

  net.add_layer(new func_layer);
  net.last_layer()->set_IOsize(8, 8);
  dynamic_cast<func_layer *>(net.last_layer())->f = ReLU;

  net.add_layer(new matrix_layer);
  net.last_layer()->set_IOsize(8, 4);
  net.last_layer()->init(std::move(rd));

  net.add_layer(new bias_layer);
  net.last_layer()->set_IOsize(4, 4);
  net.last_layer()->init(std::move(rd));

  net.add_layer(new func_layer);
  net.last_layer()->set_IOsize(4, 4);
  dynamic_cast<func_layer *>(net.last_layer())->f = softmax;

  for (int i = 0, c = 0;; ++i, i %= 4, ++c, c %= 1001) {
    auto input = {static_cast<float>(i == 0), static_cast<float>(i == 1),
                  static_cast<float>(i == 2), static_cast<float>(i == 3)};
    auto expect = {static_cast<float>(i == 1), static_cast<float>(i == 2),
                   static_cast<float>(i == 3), static_cast<float>(i == 0)};
    net.forward(input);
    if (c == 0) {
      cout << i << endl;
      cout << "E : ";

      for (auto x : expect) {
        cout << fixed << x << ' ';
      }
      cout << endl;
      cout << "O : ";
      for (auto x : net.last_layer()->output) {
        cout << fixed << x << ' ';
      }
      cout << endl;
      cout << "Loss: " << -log(net.last_layer()->output[(i+3)%4]) << endl;
    }
    net.update(input, expect, 0.00001);
  }
  return 0;
}
