#include "network.h"
#include "layer.h"
#include "matrix.h"
#include <cassert>
#include <cstddef>
#include <fstream>

network::network(const vector<int> &sizes, funcT Func, funcT deFunc) {
  if (sizes.size() == 0) {
    return;
  }
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
const matrix &network::getVdVi(size_t I) const {
  assert(computed);
  return VdVi[I];
}
matrix network::getVdWij(size_t l, int j) const {
  assert(computed);
  matrix tmp;
  tmp.setn(layers[l].w.getn());
  tmp.setm(1);
  for (int i = 0; i < layers[l].w.getn(); ++i) {
    tmp(i, 0) = layers[l].getVdWij(i, j);
  }
  return getVdVi(l) * tmp;
}
matrix network::getVdbi(size_t i) const {
  assert(computed);
  const matrix &tmp = getVdVi(i);
  return tmp * layers[i].getVdb();
}
void network::getV() {
  for (int i = 1; i < layers.size(); ++i) {
    layers[i].setInput(layers[i - 1].getV());
  }
  output = layers[layers.size() - 1].getV();

  VdVi.resize(layers.size());
  matrix res = i(layers[layers.size() - 1].w.getn());
  // VdVi[layers.size()-1]=res;
  for (size_t x = layers.size() - 1; /*x >= 0*/ x != -1; --x) {
    VdVi[x] = res;
    res = res * layers[x].getVdV();
  }
  computed = true;
}
void network::setInput(const VvalT &in) {
  layers[0].setInput(in);
  computed = false;
}
void network::save(ostream &fp) const {
  fp << layers.size() << "\n";
  for (const layer &x : layers) {
    fp << x.w.getn() << ' ' << x.w.getm() << "\n";
    for (int i = 0; i < x.w.getn(); ++i) {
      for (int j = 0; j < x.w.getm(); ++j) {
        fp << x.w(i, j) << ' ';
      }
    }
    fp << "\n";
    assert(x.w.getn() == x.b.getn());
    assert(x.b.getm() == 1);
    for (int i = 0; i < x.b.getn(); ++i) {
      fp << x.b(i, 0) << ' ';
    }
    fp << "\n";
  }
}
void network::load(istream &fp) {
  size_t size;
  fp >> size;
  layers.resize(size);
  for (layer &x : layers) {
    size_t n, m;
    fp >> n >> m;
    x.w.setn(n);
    x.w.setm(m);
    x.b.setn(n);
    x.b.setm(1);
    for (int i = 0; i < x.w.getn(); ++i) {
      for (int j = 0; j < x.w.getm(); ++j) {
        fp >> x.w(i, j);
      }
    }
    for (int i = 0; i < x.w.getn(); ++i) {
      fp >> x.b(i, 0);
    }
  }
}
