#include "convolution_layer.hpp"
#include "convolution.hpp"
#include "layers.hpp"
#include "main.hpp"
#include "matrix.hpp"
#include <algorithm>
#include <cassert>
#include <istream>
#include <iterator>
#include <ostream>
#include <random>
#include <string>
#include <vector>

convolution_layer::~convolution_layer() {}
void convolution_layer::init(std::random_device &&rd) {
  for (auto &x1 : K)
    for (auto &x2 : x1)
      for (auto &x : x2.m)
        x = ((rd() - rd.min()) * 1.0 / (rd.max() - rd.min()) * 2 - 1) /
            sqrt(n_in * m_in);
}
void convolution_layer::set_IOsize(int isize, int osize) {
  if (nK <= 0 || mK <= 0) {
    throw std::runtime_error(
        "init convolution_layer: nK: " + std::to_string(nK) +
        " ; nK: " + std::to_string(nK));
  }
  if (isize != n_in * m_in * Ichannels || osize != n_in * m_in * Ochannels) {
    throw std::runtime_error("init convolution_layer: isize , osize " +
                             std::to_string(isize) + " ; " +
                             std::to_string(osize) + " ; " +
                             std::to_string(n_in) + " " + std::to_string(m_in));
  }
  {
    // matrix k1;
    // k1.setn(nK);
    // k1.setm(mK);
    // vector<matrix> k2;
    // k2.resize(Ichannels, k1);
    // K.resize(Ochannels, k2);
    K.resize(Ochannels);
    for (auto &x : K) {
      x.resize(Ichannels);
    }
    for (auto &x : K) {
      for (auto &x : x) {
        x.setn(nK);
        x.setm(mK);
      }
    }
  }
  Isize = isize;
  Osize = osize;
  return;
}
vector<valT> convolution_layer::forward(const vector<valT> &input) {
  output.clear();
  output.reserve(Ochannels * n_in * m_in);
  for (int Oc = 0; Oc < Ochannels; ++Oc) {
    matrix tmp_out;
    tmp_out.setn(n_in);
    tmp_out.setm(m_in);
    for (int Ic = 0; Ic < Ichannels; ++Ic) {
      matrix I;
      I.setn(n_in);
      I.setm(m_in);
      I.m = vector(input.begin() + Ic * n_in * m_in,
                   input.begin() + (Ic + 1) * n_in * m_in);
      auto res = convolution(I, K[Oc][Ic]);
      for (int i = 0; i < n_in; ++i)
        for (int j = 0; j < m_in; ++j)
          tmp_out(i, j) += res(i, j);
    }
    for (auto x : tmp_out.m) {
      output.push_back(x);
    }
  }
  return output;
}
vector<valT> convolution_layer::backward(const vector<valT> &grad) const {
  // matrix res;
  // res.setn(n_in);
  // res.setm(m_in);
  // for (int i = 0; i < n_in; ++i) {
  //   for (int j = 0; j < m_in; ++j) {
  //     res(i, j) = 0;
  //     for (int x = i; x < i + nK; ++x)
  //       for (int y = j; y < j + mK; ++y)
  //         res(i, j) += K(x - i, y - j) * D(x, y);
  //   }
  // }
  // return res.m;

  VvalT output;
  output.reserve(Ichannels * n_in * m_in);
  for (int Ic = 0; Ic < Ichannels; ++Ic) {
    matrix t;
    t.setn(n_in);
    t.setm(m_in);
    for (int Oc = 0; Oc < Ochannels; ++Oc) {
      matrix D;
      D.setn(n_in);
      D.setm(m_in);
      D.m = vector(grad.begin() + Oc * n_in * m_in,
                   grad.begin() + (Oc + 1) * n_in * m_in);
      matrix res = convolution(D, rotate(K[Oc][Ic]));
      for (int i = 0; i < n_in; ++i)
        for (int j = 0; j < m_in; ++j)
          t(i, j) += res(nK - 1 + i, mK - 1 + j);
    }
    copy(t.m.begin(), t.m.end(), std::back_inserter(output));
  }
  return std::move(output);
}
vector<valT> convolution_layer::update(const vector<valT> &grad,
                                       const vector<valT> &input) const {
  vector<valT> res;
  res.reserve(Ichannels * Ochannels * nK * mK);
  for (int Oc = 0; Oc < Ochannels; ++Oc) {
#ifdef USE_OMP
#pragma omp for
#endif
    for (int Ic = 0; Ic < Ichannels; ++Ic) {
      matrix G;
      G.setn(n_in);
      G.setm(m_in);

      G.m = vector(grad.begin() + Oc * n_in * m_in,
                   grad.begin() + (Oc + 1) * n_in * m_in);

      matrix I;
      I.setn(n_in);
      I.setm(m_in);

      I.m = vector(input.begin() + Ic * n_in * m_in,
                   input.begin() + (Ic + 1) * n_in * m_in);

      matrix o = convolution(G, rotate(I));

      for (int i = n_in - 1; i < n_in - 1 + nK; ++i)
        for (int j = m_in - 1; j < m_in - 1 + mK; ++j)
          res.push_back(o(i, j));
    }
  }
  return std::move(res);
}
void convolution_layer::update(vector<valT>::const_iterator &i) {
  for (auto &x1 : K)
    for (auto &x2 : x1)
      for (auto &x : x2.m) {
        x += (*i);
        ++i;
      }
}

void convolution_layer::save(std::ostream &o) const {
  o << n_in << ' ' << m_in << ' ' << nK << ' ' << mK << " " << Ichannels << " "
    << Ochannels << std::endl;
  for (auto &x1 : K)
    for (auto &x2 : x1)
      for (auto x : x2.m) {
        o << x << ' ';
      }
  o << std::endl;
}
void convolution_layer::load(std::istream &i) {
  i >> n_in >> m_in >> nK >> mK >> Ichannels >> Ochannels;
  set_IOsize(n_in * m_in * Ichannels, n_in * m_in * Ochannels);
  for (auto &x1 : K)
    for (auto &x2 : x1)
      for (auto &x : x2.m) {
        i >> x;
      }
}

size_t convolution_layer::get_varnum() const {
  return Ichannels * Ochannels * nK * mK;
}
