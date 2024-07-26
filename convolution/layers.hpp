#pragma once
#include "../main.hpp"
#include <vector>
#include <random>
using std::vector;
struct layer {
  virtual ~layer() = default;
  virtual void init(std::random_device &&) = 0;
  virtual void set_IOsize(int isize, int osize) = 0;
  virtual vector<valT> forward(const vector<valT> &input) = 0;
  virtual vector<valT> backward(const vector<valT> &grad) = 0;
  virtual void update(const vector<valT> &grad, const vector<valT> &input,
                      double lr) = 0;
  vector<valT> output;
  int Isize, Osize;
};
