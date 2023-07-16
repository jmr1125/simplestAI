#pragma once
#include "layer.h"
#include "main.h"
#include "matrix.h"
#include <cstddef>
#include <istream>
#include <ostream>
#include <vector>
using std::istream;
using std::ostream;
using std::vector;
struct network {
  network() = delete;
  network(const vector<int> &sizes, funcT Func, funcT deFunc);
  matrix getVdWij(size_t l, int j) const;
  matrix getVdbi(size_t i) const;
  const matrix& getVdVi(size_t i) const;
  void getV();
  void setInput(const VvalT &in);
  void save(ostream &) const;
  void load(istream &);
  VvalT output;
  vector<layer> layers;
  vector<matrix> VdVi;
  bool computed;
};
