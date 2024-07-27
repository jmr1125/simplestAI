#include "convolution_layer.hpp"
#include "convolution.hpp"
#include "layers.hpp"
#include <istream>
#include <ostream>
#include <random>
#include <string>

convolution_layer::~convolution_layer() {}
void convolution_layer::init(std::random_device &&rd) {
  for (auto &x : K.m) {
    x = ((rd() + rd.min()) * 1.0 / (rd.max() - rd.min()) * 2 - 1) /
        sqrt(n_in * m_in);
  }
}
void convolution_layer::set_IOsize(int isize, int osize) {
  if (nK <= 0 || mK <= 0) {
    throw std::runtime_error(
        "init convolution_layer: nK: " + std::to_string(nK) +
        " ; nK: " + std::to_string(nK));
  }
  K.setn(nK);
  K.setm(mK);
  if (isize != n_in * m_in ||
      osize != (n_in + K.getn() - 1) * (m_in + K.getm() - 1)) {
    throw std::runtime_error("init convolution_layer: isize , osize " +
                             std::to_string(isize) + " ; " +
                             std::to_string(osize) + " ; " +
                             std::to_string(n_in) + std::to_string(m_in));
  }
  return;
}
vector<valT> convolution_layer::forward(const vector<valT> &input) {
  matrix I;
  I.setn(n_in);
  I.setm(m_in);
  I.m = input;
  auto res = convolution(I, K);
  return res.m;
}
vector<valT> convolution_layer::backward(const vector<valT> &grad) {
  matrix D;
  D.setn(n_in + nK - 1);
  D.setm(m_in + mK - 1);
  D.m = grad;
  matrix res;
  res.setn(n_in);
  res.setm(m_in);
  for (int i = 0; i < n_in; ++i) {
    for (int j = 0; j < m_in; ++j) {
      for (int x = i; x < i + nK; ++x)
        for (int y = j; y < j + mK; ++y)
          res(i, j) = K(x - i, y - j) * D(x, y);
    }
  }
  return res.m;
}
void convolution_layer::update(const vector<valT> &grad,
                               const vector<valT> &input, double lr) {
  matrix G;
  G.setn(n_in + nK - 1);
  G.setm(m_in + mK - 1);
  G.m = grad;
  matrix I;
  I.setn(n_in);
  I.setm(m_in);
  for (int i = 0; i < nK; ++i)
    for (int j = 0; j < mK; ++j)
      for (int x = i; x < n_in; ++x)
        for (int y = j; y < m_in; ++y)
          K(i, j) -= I(x - i, y - j) * G(x, y) * lr;
}

void convolution_layer::save(std::ostream &o) {
  o << n_in << ' ' << m_in << ' ' << nK << ' ' << mK << std::endl;
  for (auto x : K.m) {
    o << x << ' ';
  }
}
void convolution_layer::load(std::istream &i) {
  i >> n_in >> m_in >> nK >> mK;
  set_IOsize(n_in * m_in, (n_in + nK - 1) * (m_in + mK - 1));
  for (auto &x : K.m) {
    i >> x;
  }
}
