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

#define o_n (n_in - nK + 1 + pad * 2)
#define o_m (m_in - mK + 1 + pad * 2)
convolution_layer::~convolution_layer() {}
void convolution_layer::init(std::random_device &&rd) {
  for (auto &x1 : K)
    for (auto &x2 : x1)
      for (auto &x : x2.m)
        x = (rand01(rd) * 2 - 1) / sqrt(n_in * m_in * Ichannels);
}
void convolution_layer::set_IOsize(int isize, int osize) {
  if (nK <= 0 || mK <= 0) {
    throw std::runtime_error(
        "init convolution_layer: nK: " + std::to_string(nK) +
        " ; nK: " + std::to_string(nK));
  }
  if (isize != n_in * m_in * Ichannels || osize != o_n * o_m * Ochannels) {
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
// output[Oc][x][y] = sigma(Ic i j , input[Ic][x-i][y-j]*K[Oc][Ic][i][j])
vector<valT> convolution_layer::forward(const vector<valT> &input) {
  output.clear();
  output.reserve(Ochannels * o_n * o_m);
  for (int Oc = 0; Oc < Ochannels; ++Oc) {
    matrix tmp_out;
    tmp_out.setn(o_n);
    tmp_out.setm(o_m);
    for (int Ic = 0; Ic < Ichannels; ++Ic) {
      matrix I;
      I.setn(n_in);
      I.setm(m_in);
      I.m = vector(input.begin() + Ic * n_in * m_in,
                   input.begin() + (Ic + 1) * n_in * m_in);
      auto res = convolution(I, K[Oc][Ic]);
      for (int i = nK / 2 - pad, ti = 0; i < n_in - 1 - (nK / 2 - pad);
           ++i, ++ti)
        for (int j = mK / 2 - pad, tj = 0; j < m_in - 1 - (mK / 2 - pad);
             ++j, ++tj)
          tmp_out(ti, tj) += res(i, j);
    }
    copy(tmp_out.m.begin(), tmp_out.m.end(), std::back_inserter(output));
  }
  return output;
}
#ifdef USE_OCL
#include "ocl.hpp"
#endif
vector<valT> convolution_layer::backward(const vector<valT> &grad) const {
#ifdef USE_OCL
  // #if 0
  return conv_l_backward(*this, grad);
#else
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
      // matrix res = convolution(D, rotate(K[Oc][Ic]));
      // for (int i = 0; i < n_in; ++i)
      //   for (int j = 0; j < m_in; ++j)
      //     t(i, j) += res(nK - 1 + i, mK - 1 + j);
      for (int i = 0; i < nK; ++i) {
        for (int j = 0; j < mK; ++j) {
          for (int x = i; x < n_in; ++x) {
            if (x - i >= n_in)
              continue;
            if (x - i < nK / 2 - pad)
              for (int y = j; y < m_in; ++y) {
                if (y - j >= m_in)
                  continue;
                t(x - i, y - j) += K[Oc][Ic](i, j) * D(x, y);
              }
          }
        }
      }
    }
    copy(t.m.begin(), t.m.end(), std::back_inserter(output));
  }
  return std::move(output);
#endif
}
vector<valT> convolution_layer::update(const vector<valT> &grad,
                                       const vector<valT> &input) const {
  // #ifdef USE_OCL
#if 0
  return conv_l_update(*this, grad, input);
#else
  vector<valT> res;
  res.resize(Ichannels * Ochannels * nK * mK);
  for (int Oc = 0; Oc < Ochannels; ++Oc) {
    for (int Ic = 0; Ic < Ichannels; ++Ic) {
      // matrix G;
      // G.setn(n_in);
      // G.setm(m_in);

      // G.m = vector(grad.begin() + Oc * n_in * m_in,
      //              grad.begin() + (Oc + 1) * n_in * m_in);

      // matrix I;
      // I.setn(n_in);
      // I.setm(m_in);

      // I.m = vector(input.begin() + Ic * n_in * m_in,
      //              input.begin() + (Ic + 1) * n_in * m_in);

      // matrix o = convolution(G, rotate(I));

      // for (int i = n_in - 1; i < n_in - 1 + nK; ++i)
      //   for (int j = m_in - 1; j < m_in - 1 + mK; ++j)
      //     res.push_back(o(i, j));
      for (int i = 0; i < nK; ++i) {
        for (int j = 0; j < mK; ++j) {
          for (int x = i; x < n_in; ++x) {
            if (x - i >= n_in)
              continue;
            for (int y = j; y < m_in; ++y) {
              if (y - j >= n_in)
                continue;
              res.at((Oc * Ichannels + Ic) * nK * mK + i * mK + j) +=
                  // G(x, y) * I((x - i), (y - j));
                  grad[Oc * n_in * m_in + x * m_in + y] *
                  input[Ic * n_in * m_in + (x - i) * m_in + (y - j)];
            }
          }
        }
      }
    }
  }
  return std::move(res);
#endif
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

std::shared_ptr<layer> convolution_layer::clone() const {
  return std::make_shared<convolution_layer>(*this);
}

void convolution_layer::randomize_nan(std::random_device &&rd) {
  for (auto &line : K)
    for (auto &k : line)
      for (auto &x : k.m)
        if (isnan(x))
          x = (rand01(rd) * 2 - 1) / sqrt(n_in * m_in * Ichannels);
}
