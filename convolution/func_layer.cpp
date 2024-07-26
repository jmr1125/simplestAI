#include "func_layer.hpp"
#include <algorithm>
#include <cmath>
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
  output.resize(osize);
}
vector<valT> func_layer::forward(const vector<valT> &input) {
  if (f == softmax) {
    valT sum = 0;
    for (auto x : input) {
      sum += exp(x);
    }
    for (int i = 0; i < input.size(); ++i) {
      output[i] = exp(input[i]) / sum;
    }
  } else {
    for (int i = 0; i < input.size(); ++i) {
      output[i] = (*funcs[f])(input[i]);
    }
  }
  return output;
}
vector<valT> func_layer::backward(const vector<valT> &grad) {
  vector<valT> res;
  if (f == softmax) {
    int n = grad.size();
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
void func_layer::update(const vector<valT> &, const vector<valT> &, double) {
  return;
}

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
