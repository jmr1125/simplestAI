#pragma once
#include "layers.hpp"
#include "main.hpp"
#include <cstddef>
#include <random>
#include <vector>

struct bias_layer : public layer {
  virtual ~bias_layer() override;
  virtual void init(std::random_device &&) override;
  virtual void set_IOsize(int isize, int osize) override;
  virtual vector<valT> forward(const vector<valT> &input) override;
  virtual vector<valT> backward(const vector<valT> &grad) const override;
  virtual vector<valT> update(const vector<valT> &grad,
                              const vector<valT> &input) const override;
  virtual void update(vector<valT>::const_iterator &) override;
  virtual void save(std::ostream &) const override;
  virtual void load(std::istream &) override;
  virtual size_t get_varnum() const override;
  virtual std::shared_ptr<layer> clone() const override;
  virtual void randomize_nan(std::random_device &&) override;
  vector<valT> bias;
};
