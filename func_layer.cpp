#include "func_layer.hpp"
#include "main.hpp"
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <istream>
#include <limits>
#include <ostream>
#include <vector>
functionT funcs[] = {
    f_Identity, f_Binary_step, f_sigmoid, f_tanh, f_ReLU, f_Softplus,
};
functionT dfuncs[] = {
    df_Identity, df_Binary_step, df_sigmoid, df_tanh, df_ReLU, df_Softplus,
};
func_layer::~func_layer() {}
void func_layer::init(std::random_device &&) {}
void func_layer::set_IOsize(int isize, int osize) {
  if (isize != osize) {
    throw std::runtime_error("init bias_layer: isize != osize " +
                             std::to_string(isize) + " ; " +
                             std::to_string(osize));
  }
  Isize = isize;
  Osize = osize;
  Ichannels = Ochannels = 1;
  output.resize(osize);
}
vector<valT> func_layer::forward(const vector<valT> &input) {
  if (f == softmax) {
    valT sum = 0;
    valT maxv = -std::numeric_limits<valT>::max();
    std::for_each(input.begin(), input.end(),
                  [&maxv](valT v) { maxv = std::max(maxv, v); });
    for (auto x : input) {
      sum += exp(x - maxv);
    }
    for (int i = 0; i < input.size(); ++i) {
      output[i] = exp(input[i] - maxv) / sum;
    }
  } else {
    for (int i = 0; i < input.size(); ++i) {
      output[i] = (*funcs[f])(input[i]);
    }
  }
  return output;
}
vector<valT> func_layer::backward(const vector<valT> &grad) const {
  vector<valT> res;
  if (f == softmax) {
    size_t n = grad.size();
    res.resize(n);
    for (int j = 0; j < n; ++j) {
      for (int i = 0; i < n; ++i) {
        double delta = (i == j ? 1 : 0);
        res[j] += output[i] * (delta - output[j]) * grad[i];
      }
    }
  } else {
    for (int i = 0; i < grad.size(); ++i) {
      res.push_back((*dfuncs[f])(output[i]) * grad[i]);
    }
  }
  return std::move(res);
}
vector<valT> func_layer::update(const vector<valT> &,
                                const vector<valT> &) const {
  return {};
}
void func_layer::update(vector<valT>::const_iterator &) { return; }

void func_layer::save(std::ostream &o) const {
  o << Isize << std::endl;
  o << f << std::endl;
}
void func_layer::load(std::istream &i) {
  i >> Isize;
  Osize = Isize;
  int x;
  i >> x;
  f = x == Identity          ? Identity
      : x == Binary_step     ? Binary_step
      : x == sigmoid         ? sigmoid
      : x == Functions::tanh ? Functions::tanh
      : x == ReLU            ? ReLU
      : x == Softplus        ? Softplus
                             : softmax;
  set_IOsize(Isize, Osize);
}

size_t func_layer::get_varnum() const { return 0; }
std::shared_ptr<layer> func_layer::clone() const {
  return std::make_shared<func_layer>(*this);
}
void func_layer::randomize_nan(std::random_device &&) { return; }

valT f_Identity(valT x) { return x; }
valT df_Identity(valT) { return 1; }
valT f_Binary_step(valT x) { return x >= 0 ? 1 : 0; }
valT df_Binary_step(valT) { return 0; }
valT f_sigmoid(valT x) { return 1 / (1 + exp(-x)); }
valT df_sigmoid(valT x) { return f_sigmoid(x) * (1 - f_sigmoid(x)); }
valT f_tanh(valT x) { return std::tanh(x); }
valT df_tanh(valT x) { return 1 - (f_tanh(x) * f_tanh(x)); }
valT f_ReLU(valT x) { return std::max((valT)0, x); }
valT df_ReLU(valT x) {
  return x < 0 ? 0 : 1; // x=0?
}
valT f_Softplus(valT x) { return log(1 + exp(x)); }
valT df_Softplus(valT x) { return 1 / (1 + exp(-x)); }
