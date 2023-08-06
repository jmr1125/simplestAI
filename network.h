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
  void save(ostream &) const;
  void load(istream &);
  matrix feed_forward(matrix input);
  void backpropagation(matrix input, VvalT expected, valT learning_rate);
  vector<layer> layers;
};
