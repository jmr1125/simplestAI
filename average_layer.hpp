#include "layers.hpp"
struct average_layer : public layer {
  virtual ~average_layer() override;
  virtual void init(std::random_device &&) override;
  virtual void set_IOsize(int isize, int osize) override;
  virtual vector<valT> forward(const vector<valT> &input) override;
  virtual vector<valT> backward(const vector<valT> &grad) const override;
  virtual vector<valT> update(const vector<valT> &grad,
                              const vector<valT> &input,
                              double lr) const override;
  virtual void update(vector<valT>::const_iterator &) override;
  virtual void save(std::ostream &) const override;
  virtual void load(std::istream &) override;
  int i_n, i_m;
};
