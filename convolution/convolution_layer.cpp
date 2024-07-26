#include "convolution_layer.hpp"
#include "convolution.hpp"
#include "layers.hpp"
#include <string>

void convolution_layer::set_IOsize(int isize, int osize) {
  if (nK <= 0 || mK <= 0) {
    throw std::runtime_error(
        "init convolution_layer: nK: " + std::to_string(nK) +
        " ; nK: " + std::to_string(nK));
  }
  K.setn(nK);
  K.setm(mK);
  if (isize != n_in * m_in || osize != (n_in + K.getn()) * (m_in + K.getm())) {
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
      for (int x = i; x < n_in + nK - 1; ++x)
        for (int y = j; y < m_in + mK - 1; ++y)
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
