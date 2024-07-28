#include "matrix_layer.hpp"
#include "main.hpp"
#include <istream>
#include <ostream>
#include <random>
#include <vector>

matrix_layer::~matrix_layer() {}
void matrix_layer::init(std::random_device &&rd) {
  for (valT &x : M.m) {
    x = ((rd() + rd.min()) * 1.0 / (rd.max() - rd.min()) * 2 - 1) / sqrt(Isize);
  }
}
void matrix_layer::set_IOsize(int isize, int osize) {
  M.setn(osize);
  M.setm(isize);
  Isize = isize;
  Osize = osize;
}
vector<valT> matrix_layer::forward(const vector<valT> &input) {
  output = M * input;
  return output;
}
vector<valT> matrix_layer::backward(const vector<valT> &grad) const {
  return M.T() * grad;
}
vector<valT> matrix_layer::update(const vector<valT> &grad,
                                  const vector<valT> &input, double lr) const {
  vector<valT> res;
  res.resize(M.getm() * M.getn());
  for (int i = 0; i < M.getn(); i++)
    for (int j = 0; j < M.getm(); j++) {
      // M(i, j) -= lr * grad[i] * input[j];
      res[i * M.getm() + j] -= lr * grad[i] * input[j];
    }
  return std::move(res);
}
void matrix_layer::update(vector<valT>::const_iterator &i) {
  for (auto &x : M.m) {
    x += (*i);
    ++i;
  }
}

void matrix_layer::save(std::ostream &o) const {
  o << Isize << std::endl;
  o << Osize << std::endl;
  for (auto x : M.m) {
    o << x << " ";
  }
}
void matrix_layer::load(std::istream &i) {
  i >> Isize >> Osize;
  set_IOsize(Isize, Osize);
  for (auto &x : M.m) {
    i >> x;
  }
}
