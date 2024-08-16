#pragma once
#include "main.hpp"
#include <cstddef>
#include <fstream>
#include <istream>
#include <random>
#include <vector>
using std::vector;
struct layer {
  virtual ~layer() = default;
  virtual void init(std::random_device &&) = 0;
  virtual void set_IOsize(int isize, int osize) = 0;
  virtual vector<valT> forward(const vector<valT> &input) = 0;
  virtual vector<valT> backward(const vector<valT> &grad) const = 0;
  virtual vector<valT> update(const vector<valT> &grad,
                              const vector<valT> &input) const = 0;
  virtual void update(vector<valT>::const_iterator &) = 0;
  virtual void save(std::ostream &) const = 0;
  virtual void load(std::istream &) = 0;
  virtual size_t get_varnum() const = 0;
  virtual std::shared_ptr<layer> clone() const = 0;
  virtual void randomize_nan(std::random_device &&) = 0;
  vector<valT> output;
  int Ichannels, Ochannels;
  int Isize, Osize;
};
