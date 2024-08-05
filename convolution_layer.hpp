#include "layers.hpp"
#include "matrix.hpp"
#include <istream>
#include <random>

struct convolution_layer : public layer {
  virtual ~convolution_layer() override;
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
  vector<vector<matrix>> K; // K [Output] [Input]
  int nK, mK;
  int n_in, m_in;
};
