#include "train.h"
#include "layer.h"
#include "main.h"
#include "matrix.h"
#include <cstddef>
#include <iostream>
#include <vector>
void train(network &net, const VvalT &input, const VvalT &expect) {
  // cout<<tmp<<endl;
  net.setInput(input);
  net.getV();
  // delta d? = delta dv v d? = 2(delta-v)*vd?
  // modify it
  vector<matrix> netVw;
  vector<matrix> netVb;
  netVw.reserve(net.layers.size());
  netVb.reserve(net.layers.size());
  for (int i = 0; i < net.layers.size(); ++i) {
    netVw[i] = net.layers[i].w;
    netVb[i] = net.layers[i].b;
  }
#pragma omp parallel for
  for (size_t i = 0; i < net.output.size(); ++i) {
    valT v = (net.output[i] - expect[i]) * 2; //(V_i-e)^2 d? = 2(V_i-e)*V_i d ?
    v /= 100000;
    for (size_t l = 0; l < net.layers.size(); ++l) {
      const matrix vdb = net.getVdbi(l);
      const valT delta_b = vdb(i, 0) * v;
      netVb[l](i, 0) -= delta_b;
      for (int j = 0; j < net.layers[l].w.getm(); ++j) {
        const matrix &&vdw = net.getVdWij(l, j);
        const valT delta_w = vdw(i, 0) * v;
        netVw[l](i, j) -= delta_w;
      }
    }
  }
  for (int i = 0; i < net.layers.size(); ++i) {
    net.layers[i].w = netVw[i];
    net.layers[i].b = netVb[i];
  }

  // for (int i = 0; i < net.layers.size(); ++i) {
  //   auto &b = net.layers[i].b;
  //   valT v = 2 * delta;
  //   const matrix &&vdb = net.getVdbi(i);
  //   assert(vdb.getm() == 1);
  //   // assert(vdb.getn() == b.getn());
  //   // assert(vdb.getn() == net.output.size());
  //   for (int j = 0; j < net.output.size(); ++j) {
  //     b(j, 0) -= v * vdb(j, 0);
  //   }
  //   auto &w = net.layers[i].w;
  //   for (int j = 0; j < w.getm(); ++j) {
  //     const matrix &&vdw = net.getVdWij(i, j);
  //     for (int k = 0; j < w.getn(); ++j) {
  //       w(k, j) -= v * vdw(k, 0);
  //     }
  //   }
  // }
}

valT genvalT() {
  static std::random_device rd;
  valT v = rd();
  v -= rd.min();
  v /= (rd.max() - rd.min());
  return v;
}
