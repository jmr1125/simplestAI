#include "train.h"
#include "layer.h"
#include "main.h"
#include "matrix.h"
#include <cassert>
#include <cstddef>
#include <iostream>
#ifdef USE_OMP
// #warning omp
#include <omp.h>
#endif
#include <vector>
void train(network &net, const VvalT &input, const VvalT &expect, valT scale) {
  vector<VvalT> vinput{input};
  vector<VvalT> vexpect{expect};
  trainn(net, vinput, vexpect, scale);
}
void trainn(network &net, const vector<VvalT> &input,
            const vector<VvalT> &expect, valT scale) {
  assert(input.size() == expect.size());
  int n = input.size(); //=expect.size();

  vector<matrix> delVw;
  vector<matrix> delVb;
  delVw.resize(net.layers.size());
  delVb.resize(net.layers.size());
#ifdef USE_OMP
#pragma parallel for
#endif
  for (int i = 0; i < net.layers.size(); ++i) { // init delta
    delVw[i].setn(net.layers[i].w.getn());
    delVw[i].setm(net.layers[i].w.getm());
    delVb[i].setn(net.layers[i].b.getn());
    delVb[i].setm(net.layers[i].b.getm());
    assert(net.layers[i].b.getm() == 1);
  }
  for (int i = 0; i < n; ++i) {
    delta_network d = getdelta_network(net, input[i], expect[i], scale);
#ifdef USE_OMP
#pragma omp parallel for
#endif
    for (int l = 0; l < net.layers.size(); ++l) {
      delVw[l] = delVw[l] + d.first[l] * ((valT)1 / n);
      delVb[l] = delVb[l] + d.second[l] * ((valT)1 / n);
    }
  }

  for (int i = 0; i < net.layers.size(); ++i) {
    net.layers[i].w = delVw[i] + net.layers[i].w;
    net.layers[i].b = delVb[i] + net.layers[i].b;
  }
}
delta_network getdelta_network(network net, const VvalT &input,
                               const VvalT &expect, valT scale) {
  net.setInput(input);
  net.getV();
  // delta d? = delta dv v d? = 2(delta-v)*vd?
  // modify it
  delta_network delta;
  vector<matrix> &netVw = delta.first;
  vector<matrix> &netVb = delta.second;
  netVw.resize(net.layers.size());
  netVb.resize(net.layers.size());
#ifdef USE_OMP
#pragma omp parallel for
#endif
  for (int i = 0; i < net.layers.size(); ++i) { // init delta
    netVw[i].setn(net.layers[i].w.getn());
    netVw[i].setm(net.layers[i].w.getm());
    netVb[i].setn(net.layers[i].b.getn());
    netVb[i].setm(net.layers[i].b.getm());
    assert(net.layers[i].b.getm() == 1);
  }
#ifdef USE_OMP
#pragma omp parallel for
#endif
  for (size_t i = 0; i < net.output.size(); ++i) {
    valT v = (net.output[i] - expect[i]) * 2; //(V_i-e)^2 d? = 2(V_i-e)*V_i d ?
    v *= scale;
#ifdef USE_OMP
#pragma omp parallel for
#endif
    for (size_t l = 0; l < net.layers.size(); ++l) {
      const matrix vdb = net.getVdbi(l);
      valT delta_b = vdb(i, 0) * v;
      netVb[l](i, 0) = -delta_b;
#ifdef USE_OMP
#pragma omp parallel for
#endif
      for (int j = 0; j < net.layers[l].w.getm(); ++j) {
        const matrix &&vdw = net.getVdWij(l, j);
        valT delta_w = vdw(i, 0) * v;
        netVw[l](i, j) = -delta_w;
      }
    }
  }
  return std::move(delta);
};
valT genvalT() {
  static std::random_device rd;
  valT v = rd();
  v -= rd.min();
  v /= (rd.max() - rd.min());
  return v;
}
valT func_x(valT x) { return x; }
valT defunc_x(valT) { return 1; }
valT func_sigma(valT x) { return (valT)1 / (1 + exp(-x)); }
valT defunc_sigma(valT x) { return func_sigma(x) * (1 - func_sigma(x)); }
valT func_tanh(valT x) { return tanh(x); } //(exp(x)-exp(-x))/(exp(x)+exp(-x))
valT defunc_tanh(valT x) { return 1 - func_tanh(x) * func_tanh(x); }
valT func_arctan(valT x) { return atan(x); }
valT defunc_arctan(valT x) { return (valT)1 / (x * x + 1); }
valT func_Softsign(valT x) { return x / (1 + fabs(x)); }
valT defunc_Softsign(valT x) {
  return (valT)1 / ((1 + fabs(x)) * (1 + fabs(x)));
}
valT func_ISRU(valT x, valT alpha) { return x / sqrt(1 + alpha * x * x); }
valT defunc_ISRU(valT x, valT alpha) {
  return (valT)1 / pow(sqrt(1 + alpha * x * x), 3);
}
valT func_ReLU(valT x) { return x < 0 ? 0 : x; }
valT defunc_ReLU(valT x) { return x < 0 ? 0 : 1; }
valT func_Leaky_ReLU(valT x) { return x < 0 ? 0.01 * x : x; }
valT defunc_Leaky_ReLU(valT x) { return x < 0 ? 0.01 : 1; }
valT func_PReLU(valT x, valT alpha) { return x < 0 ? alpha * x : x; }
valT defunc_PReLU(valT x, valT alpha) { return x < 0 ? alpha : 1; }
// RReLU?
valT func_ELU(valT x, valT alpha) { return x < 0 ? alpha * (exp(x) - 1) : x; }
valT defunc_ELU(valT x, valT alpha) {
  return x < 0 ? alpha + func_ELU(x, alpha) : 1;
}
valT func_SELU(valT x) { return 1.0507 * func_ELU(x, 1.67326); }
valT defunc_SELU(valT x) { return 1.0507 * (x < 0 ? 1.67326 * exp(x) : 1); }
valT func_softmax(VvalT x, int i) {
  valT a = exp(x[i]);
  valT b = 0;
  for (int i = 0; i < x.size(); ++i) {
    b += exp(x[i]);
  }
  return a / b;
}
// valT defunc_softmax(VvalT x,int i,int j){}

// reference https://zh.wikipedia.org/zh-cn/%E6%BF%80%E6%B4%BB%E5%87%BD%E6%95%B0
