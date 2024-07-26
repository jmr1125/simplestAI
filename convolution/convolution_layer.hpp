#include "../matrix.hpp"
#include "layers.hpp"

struct convolution_layer : public layer {
  virtual void set_IOsize(int isize, int osize) override;
  virtual vector<valT> forward(const vector<valT> &input) override;
  virtual vector<valT> backward(const vector<valT> &grad) override;
  virtual void update(const vector<valT> &grad, const vector<valT> &input,
                      double lr) override;
  matrix K;
  int nK, mK;
  int n_in, m_in;
};
