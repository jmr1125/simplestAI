#include "layer.hpp"
#include "matrix.hpp"
#include <algorithm>
#include <cstdio>
void layer::setn(int n) {
  w.setn(n);
  b.setn(n);
  b.setm(1);
}
void layer::setm(int m) { w.setm(m); }
funcT &layer::Func() {
  // computed = false;
  return Funct;
}
funcT layer::Funcv() const { return Funct; }
funcT &layer::deFunc() {
  // computed = false;
  return deFunct;
}
funcT layer::deFuncv() const { return deFunct; }
matrix layer::activate(matrix input) {
  z = w * input + b;
  assert(z.getm() == 1);
  matrix Z = z;
  for (int i = 0; i < Z.getn(); ++i) {
    Z(i, 0) = Funcv()(Z(i, 0));
  }
  return output = Z;
}
