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
void train(network &net, const VvalT &input, const VvalT &expect,
           valT scale) {
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
  for (int i = 0; i < net.layers.size(); ++i) { // init delta
    delVw[i].setn(net.layers[i].w.getn());
    delVw[i].setm(net.layers[i].w.getm());
    delVb[i].setn(net.layers[i].b.getn());
    delVb[i].setm(net.layers[i].b.getm());
    assert(net.layers[i].b.getm() == 1);
  }
  for (int i = 0; i < n; ++i) {
    delta_network d = getdelta_network(net, input[i], expect[i], scale);
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
delta_network getdelta_network(network &net, const VvalT &input,
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
  // for (int i = 0; i < net.layers.size(); ++i) {
  //   netVw[i] = net.layers[i].w;
  //   netVb[i] = net.layers[i].b;
  // }
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
                                              // v /= 100;
                                              //  v /= 10000;
    v *= scale;
#ifdef USE_OMP
#pragma omp parallel for
#endif
    for (size_t l = 0; l < net.layers.size(); ++l) {
      const matrix vdb = net.getVdbi(l);
      valT delta_b = vdb(i, 0) * v;
      // delta_b *= fabs(delta_b);
      // delta_b *= scale;
      netVb[l](i, 0) = -delta_b;
      for (int j = 0; j < net.layers[l].w.getm(); ++j) {
        const matrix &&vdw = net.getVdWij(l, j);
        valT delta_w = vdw(i, 0) * v;
        // delta_w *= fabs(delta_w);
        // delta_w *= scale;
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
