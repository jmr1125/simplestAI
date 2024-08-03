#include "bias_layer.hpp"
#include "main.hpp"
#include <istream>
#include <ostream>
#include <stdexcept>
#include <string>
#include <vector>

bias_layer::~bias_layer() {}
void bias_layer::init(std::random_device &&rd) {
  for (valT &x : bias) {
    x = ((rd() + rd.min()) * 1.0 / (rd.max() - rd.min()) * 2 - 1) / sqrt(Isize);
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
  Ichannels = Ochannels = 1;
  bias.resize(isize);
  output.resize(osize);
}
vector<valT> bias_layer::forward(const vector<valT> &input) {
  for (int i = 0; i < input.size(); i++)
    output[i] = input[i] + bias[i];
  return output;
}
vector<valT> bias_layer::backward(const vector<valT> &grad) const {
  return grad;
}
vector<valT> bias_layer::update(const vector<valT> &grad,
                                const vector<valT> &input, double lr) const {
  vector<valT> res;
  res.resize(bias.size());
  for (int i = 0; i < grad.size(); i++) {
    // bias[i] -= lr * grad[i];
    res[i] -= lr * grad[i];
  }
  return std::move(res);
}
void bias_layer::update(vector<valT>::const_iterator &i) {
  for (auto &x : bias) {
    x += (*i);
    ++i;
  }
}
void bias_layer::save(std::ostream &o) const {
  o << Isize << std::endl;
  for (auto x : bias) {
    o << x << ' ';
  }
  o << std::endl;
}
void bias_layer::load(std::istream &i) {
  i >> Isize;
  Osize = Isize;
  set_IOsize(Isize, Osize);
  for (auto &x : bias) {
    i >> x;
  }
}
