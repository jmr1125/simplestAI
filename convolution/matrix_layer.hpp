#pragma once
#include "../matrix.hpp"
#include "layers.hpp"
struct matrix_layer : public layer {
  virtual ~matrix_layer() override;
  virtual void init(std::random_device&&) override;
  virtual void set_IOsize(int isize, int osize) override;
  virtual vector<valT> forward(const vector<valT> &input) override;
  virtual vector<valT> backward(const vector<valT> &grad) override;
  virtual void update(const vector<valT> &grad, const vector<valT> &input,
                      double lr) override;
  matrix M;
};
