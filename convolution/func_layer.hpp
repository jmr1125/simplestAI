#pragma once
#include "layers.hpp"
typedef valT (*functionT)(valT);
enum Functions {
  Identity = 0,
  Binary_step,
  sigmoid,
  tanh,
  ReLU,
  Softplus,
  softmax
};

struct func_layer : public layer {
  virtual ~func_layer() override;
  virtual void init(std::random_device &&) override;
  virtual void set_IOsize(int isize, int osize) override;
  virtual vector<valT> forward(const vector<valT> &input) override;
  virtual vector<valT> backward(const vector<valT> &grad) override;
  virtual void update(const vector<valT> &grad, const vector<valT> &input,
                      double lr) override;
  virtual void save(std::ostream &) override;
  virtual void load(std::istream &) override;
  Functions f;
};

valT f_Identity(valT);
valT f_Binary_step(valT);
valT f_sigmoid(valT);
valT f_tanh(valT);
valT f_ReLU(valT);
valT f_Softplus(valT);

valT df_Identity(valT);
valT df_Binary_step(valT);
valT df_sigmoid(valT);
valT df_tanh(valT);
valT df_ReLU(valT);
valT df_Softplus(valT);
