#include "layers.hpp"
#include "main.hpp"
#include <vector>
struct nnet {
  virtual ~nnet();
  void add_layer(layer *);
  layer *last_layer() const;
  vector<valT> forward(vector<valT>);
  vector<valT> update(const vector<valT> &, const vector<valT> &) const;
  void update(vector<valT>);
  size_t get_varnum()const;
  vector<layer *> layers;
};
