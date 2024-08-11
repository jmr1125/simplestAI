#include "adam.hpp"
#include "main.hpp"
#include <algorithm>
adam::adam(int size) {
  this->size = size;
  m.resize(size);
  v.resize(size);
  mh.resize(size);
  vh.resize(size);
}
VvalT adam::update(VvalT g, valT alpha, int total) {
  VvalT d(size);
  for (int i = 0; i < size; ++i) {
    m[i] = beta1 * m[i] + (1 - beta1) * g[i];
    v[i] = beta2 * v[i] + (1 - beta2) * g[i] * g[i];
    mh[i] = m[i] / (1 - pow(beta1, 1 + total));
    vh[i] = v[i] / (1 - pow(beta2, 1 + total));
    d[i] = -alpha / (sqrt(vh[i]) + epsilon) * mh[i];
  }
  return std::move(d);
}
