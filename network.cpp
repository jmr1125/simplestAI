#include "network.h"
#include "layer.h"
#include "matrix.h"
#include <cstddef>

network::network(const vector<int> &sizes, funcT Func, funcT deFunc) {
  layers.resize(sizes.size() - 1);
  for (int i = 0; i < layers.size(); ++i) {
    layers[i].setn(sizes[i + 1]);
    layers[i].setm(sizes[i]);
  }
  for (auto &x : layers) {
    x.Func() = Func;
    x.deFunc() = deFunc;
    // x.basis.setn(x.w.getn());
    // x.basis.setm(1);
    // x.computed = false;
  }
}
matrix network::getVdVi(size_t I) const {
  matrix res = i(layers[layers.size() - 1].w.getn());
  for (size_t x = layers.size() - 1; x > I; --x) {
    res = res * layers[x].getVdV();
  }
  return res;
}
matrix network::getVdWij(size_t l, int j) const {
  matrix tmp;
  tmp.setn(layers[l].w.getn());
  tmp.setm(1);
  for (int i = 0; i < layers[l].w.getn(); ++i) {
    tmp(i, 0) = layers[l].getVdWij(i, j);
  }
  return getVdVi(l) * tmp;
}
matrix network::getVdbi(size_t i) const {
  const matrix &&tmp = getVdVi(i);
  return tmp * layers[i].getVdb();
}
void network::getV() {
  for (int i = 1; i < layers.size(); ++i) {
    layers[i].setInput(layers[i - 1].getV());
  }
  output = layers[layers.size() - 1].getV();
}
void network::setInput(const VvalT &in) { layers[0].setInput(in); }
