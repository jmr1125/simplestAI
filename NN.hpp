#include "layers.hpp"
struct nnet {
  virtual ~nnet();
  void add_layer(layer *);
  layer *last_layer();
  vector<valT> forward( vector<valT>);
  void update(const vector<valT> &, const vector<valT> &,double);
  vector<layer*>layers;
};
