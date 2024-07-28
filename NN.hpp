#include "layers.hpp"
#include "main.hpp"
#include <vector>
struct nnet {
  virtual ~nnet();
  void add_layer(layer *);
  layer *last_layer() const;
  vector<valT> forward(vector<valT>);
  vector<valT> update(const vector<valT> &, const vector<valT> &, double) const;
  void update(vector<valT>);
  vector<layer *> layers;
};
