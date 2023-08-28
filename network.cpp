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
  }
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
matrix network::feed_forward(matrix input) {
  for (auto &x : layers) {
    input = x.activate(input);
  }
  return input;
}
/*
 *\delta^{(L)} = \nabla_a J \odot \sigma'(z^{(L)})
 *\delta^{(h)} = ((W^{(h+1)})^T \cdot \delta^{(h+1)}) \odot \sigma'(z^{(h)})
 *\nabla W^{(h)} = \delta^{(h)} \cdot (a^{(h-1)})^T
 *a^{(h)} = \sigma(z^{(h)})
 *\nabla b^{(h)} = \delta^{(h)}
 */
void network::backpropagation(matrix input, VvalT expect, valT rate) {
  vector<matrix> delta;
  vector<VvalT> dF;
  delta.resize(layers.size());
  dF.resize(layers.size());
  matrix output = feed_forward(input);
  for (int i = layers.size() - 1; i >= 0; --i) {
    if (i == layers.size() - 1) {
      delta.at(i).setn(layers[i].w.getn());
      delta.at(i).setm(1);
      for (int j = 0; j < layers[i].w.getn(); ++j) {
        delta.at(i)(j, 0) = -(expect[j] - output(j, 0));
      }
    } else {
      delta.at(i) = layers[i + 1].w.T() * delta.at(i + 1);
      assert(delta[i].getm() == 1);
    }
  }
#ifdef USE_OMP
#pragma omp parallel for
#endif
  for (int i = 0; i < layers.size(); ++i) {
    dF[i].resize(layers[i].w.getn());
    for (int j = 0; j < layers[i].w.getn(); ++j) {
      dF[i][j] = layers[i].deFuncv()(layers[i].z(j, 0));
    }
  }
  assert(layers.size() == dF.size());
#ifdef USE_OMP
#pragma parallel for
#endif
  for (int i = 0; i < layers.size(); ++i) {
    assert(dF[i].size() == delta[i].getn());
    for (int j = 0; j < layers[i].w.getn(); ++j) {
      delta[i](j, 0) *= dF[i][j];
    }
  }
  for (int i = 0; i < delta.size(); ++i) {
    matrix prev;
    if (i == 0) {
      prev = input;
    } else {
      prev = layers[i - 1].output;
      assert(layers[i - 1].output.getm() == 1);
    }
#ifdef USE_OMP
#pragma paralle for
#endif
    for (int j = 0; j < delta[i].getn(); ++j) {
      for (int k = 0; k < prev.getn(); ++k) {
        layers[i].w(j, k) -= rate * delta[i](j, 0) * prev(k, 0);
      }
      layers[i].b(j, 0) -= rate * delta[i](j, 0);
    }
  }
}
