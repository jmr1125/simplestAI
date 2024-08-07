#include "NN.hpp"
#include "bias_layer.hpp"
#include "func_layer.hpp"
#include "layers.hpp"
#include "main.hpp"
#include "matrix_layer.hpp"
#include <iostream>
#include <random>
using namespace std;
int main() {
  std::random_device rd;
  nnet target;
  target.add_layer(new matrix_layer);
  target.last_layer()->set_IOsize(50, 50);
  for (int i = 0; i < 50; ++i)
    for (int j = 0; j < 50; ++j)
      dynamic_cast<matrix_layer *>(target.last_layer())->M(i, j) = sin(i + j);
  target.add_layer(new bias_layer);
  target.last_layer()->set_IOsize(50, 50);
  for (int i = 0; i < 50; ++i)
    dynamic_cast<bias_layer *>(target.last_layer())->bias[i] = 2 * cos(i);
  target.add_layer(new func_layer);
  dynamic_cast<func_layer *>(target.last_layer())->f = Functions::sigmoid;
  target.last_layer()->set_IOsize(50, 50);
  target.add_layer(new matrix_layer);
  target.last_layer()->set_IOsize(50, 50);
  for (int i = 0; i < 50; ++i)
    for (int j = 0; j < 50; ++j)
      dynamic_cast<matrix_layer *>(target.last_layer())->M(i, j) = sin(i * j);
  target.add_layer(new bias_layer);
  target.last_layer()->set_IOsize(50, 50);
  for (int i = 0; i < 50; ++i)
    dynamic_cast<bias_layer *>(target.last_layer())->bias[i] = cos(2 * i) / 2;
  target.add_layer(new func_layer);
  dynamic_cast<func_layer *>(target.last_layer())->f = Functions::softmax;
  target.last_layer()->set_IOsize(50, 50);

  nnet to_be_trained;
  to_be_trained.add_layer(new matrix_layer);
  to_be_trained.last_layer()->set_IOsize(50, 50);
  to_be_trained.last_layer()->init(std::move(rd));
  to_be_trained.add_layer(new bias_layer);
  to_be_trained.last_layer()->set_IOsize(50, 50);
  to_be_trained.last_layer()->init(std::move(rd));
  to_be_trained.add_layer(new func_layer);
  dynamic_cast<func_layer *>(to_be_trained.last_layer())->f =
      Functions::sigmoid;
  to_be_trained.last_layer()->set_IOsize(50, 50);
  to_be_trained.add_layer(new matrix_layer);
  to_be_trained.last_layer()->set_IOsize(50, 50);
  to_be_trained.last_layer()->init(std::move(rd));
  to_be_trained.add_layer(new bias_layer);
  to_be_trained.last_layer()->set_IOsize(50, 50);
  to_be_trained.last_layer()->init(std::move(rd));
  to_be_trained.add_layer(new func_layer);
  dynamic_cast<func_layer *>(to_be_trained.last_layer())->f =
      Functions::softmax;
  to_be_trained.last_layer()->set_IOsize(50, 50);

  const double beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8;
  auto grad_size = to_be_trained.get_varnum();
  VvalT m(grad_size, 0), v(grad_size, 0), mh(grad_size, 0), vh(grad_size, 0);
  valT lr = 0.001;

  for (int total = 0; total < 10000; ++total) {
    VvalT input;
    input.reserve(50);
    for (int c = 0; c < 50; ++c) {
      valT v01 = 1.0 * (rd() - rd.min()) / (rd.max() - rd.min());
      input.push_back((v01 * 2 - 1) * 3);
    }
    target.forward(input);
    const VvalT &output = target.last_layer()->output;
    to_be_trained.forward(input);
    const VvalT &output1 = to_be_trained.last_layer()->output;
    valT loss = 0;
    for (int i = 0; i < 50; ++i) {
      loss += -output[i] * log(output1[i]);
    }
    cout << "loss: " << loss;
    auto g = to_be_trained.update(input, output);
    VvalT d(grad_size, 0);
    for (int i = 0; i < grad_size; ++i) {
      m[i] = beta1 * m[i] + (1 - beta1) * g[i];
      v[i] = beta2 * v[i] + (1 - beta2) * g[i] * g[i];
      mh[i] = m[i] / (1 - pow(beta1, 1 + total / 10));
      vh[i] = v[i] / (1 - pow(beta2, 1 + total / 10));
      d[i] = -lr / (sqrt(vh[i]) + epsilon) * mh[i];
    }
    to_be_trained.update(d);
    to_be_trained.forward(input);
    for (int i = 0; i < 50; ++i) {
      loss += -output[i] * log(output1[i]);
    }
    cout << " -> " << loss << endl;
  }
}
