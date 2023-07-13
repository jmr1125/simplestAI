#pragma once
#include "layer.h"
#include "main.h"
#include "matrix.h"
#include <cstddef>
#include <vector>
using std::vector;
struct network {
  network() = delete;
  network(const vector<int> &sizes, funcT Func, funcT deFunc);
  matrix getVdWij(size_t l, int j) const;
  matrix getVdbi(size_t i) const;
  matrix getVdVi(size_t i) const;
  void getV();
  void setInput(const VvalT &in);
  VvalT output;
  vector<layer> layers;
};
