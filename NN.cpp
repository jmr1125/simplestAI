#include "NN.hpp"
#include "layers.hpp"
#include "main.hpp"
#include <algorithm>
#include <vector>

void nnet::add_layer(layer *l) { layers.push_back(l); }
layer *nnet::last_layer() const { return layers.back(); }
vector<valT> nnet::forward(vector<valT> input) {
  for (auto *l : layers) {
    input = l->forward(input);
  }
  return std::move(input);
}
vector<valT> nnet::update(const vector<valT> &input, const vector<valT> &expect,
                          double lr) const {
  auto n = layers.back()->output.size();
  vector<valT> res;
  vector<valT> delta(n);
  for (int i = 0; i < n; ++i) {
    delta[i] = -expect[i] / layers.back()->output[i];
    // delta[i] = layers.back()->output[i] - expect[i];
  }
  for (int i = layers.size() - 1; i >= 0; --i) {
    auto x =
        layers[i]->update(delta, (i == 0 ? input : layers[i - 1]->output), lr);
    delta = layers[i]->backward(delta);
    for (auto v : x) {
      res.push_back(v);
    }
  }
  return std::move(res);
}
void nnet::update(vector<valT> d) {
  auto i = d.cbegin();
  for (auto l : layers) {
    l->update(i);
  }
}
nnet::~nnet() {
  for (auto *l : layers) {
    delete l;
  }
}
