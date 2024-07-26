#include "bias_layer.hpp"
#include <stdexcept>
#include <string>

bias_layer::~bias_layer() {}
void bias_layer::init(std::random_device &&rd) {
  for (valT &x : bias) {
    x = ((rd() + rd.min()) * 1.0 / rd.max() * 2 - 1) * sqrt(Isize);
  }
}
void bias_layer::set_IOsize(int isize, int osize) {
  if (isize != osize) {
    throw std::runtime_error("init bias_layer: isize != osize " +
                             std::to_string(isize) + " ; " +
                             std::to_string(osize));
  }
  Isize = isize;
  Osize = osize;
  bias.resize(isize);
  output.resize(osize);
}
vector<valT> bias_layer::forward(const vector<valT> &input) {
  for (int i = 0; i < input.size(); i++)
    output[i] = input[i] + bias[i];
  return output;
}
vector<valT> bias_layer::backward(const vector<valT> &grad) { return grad; }
void bias_layer::update(const vector<valT> &grad, const vector<valT> &input,
                        double lr) {
  for (int i = 0; i < grad.size(); i++)
    bias[i] -= lr * grad[i];
}
