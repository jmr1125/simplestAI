#pragma once
#include "layers.hpp"

struct bias_layer : public layer {
  virtual ~bias_layer() override;
  virtual void init(std::random_device&&) override;
  virtual void set_IOsize(int isize, int osize) override;
  virtual vector<valT> forward(const vector<valT> &input) override;
  virtual vector<valT> backward(const vector<valT> &grad) override;
  virtual void update(const vector<valT> &grad, const vector<valT> &input,
                      double lr) override;
  virtual void save(std::ostream &) override;
  virtual void load(std::istream &) override;
  vector<valT> bias;
};
