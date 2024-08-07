#include "NN.hpp"
#include "layers.hpp"
#include "main.hpp"
#include <algorithm>
#include <cstddef>
#include <iterator>
#include <vector>

void nnet::add_layer(layer *l) { layers.push_back({l}); }
layer *nnet::last_layer() const { return layers.back(); }
vector<valT> nnet::forward(vector<valT> input) {
  for (int i = 0; i < layers.size(); ++i) {
    input = layers[i]->forward(input);
  }
  return std::move(input);
}
vector<valT> nnet::update(const vector<valT> &input,
                          const vector<valT> &expect) const {
  auto n = expect.size();
  vector<valT> res;
  res.reserve(get_varnum());
  vector<valT> delta(n);
  {
    for (int i = 0; i < n; ++i) {
      delta[i] =
          -expect[i] /
          last_layer()->output[i]; // the last layer must only has one : softmax
      // delta[i] = layers.back()->output[i] - expect[i];
    }
  }
  for (size_t i = layers.size() - 1; i != -1; --i) {
    auto x = layers[i]->update(delta, (i == 0 ? input : layers[i - 1]->output));
    if (i)
      delta = layers[i]->backward(delta);
    copy(x.begin(), x.end(), std::back_inserter(res));
  }
  return std::move(res);
}
void nnet::update(vector<valT> d) {
  auto I = d.cbegin();
  for (size_t i = layers.size() - 1; i != -1; --i) {
    layers[i]->update(I);
  }
}
nnet::~nnet() {
  for (auto l : layers) {
    delete l;
  }
}
size_t nnet::get_varnum() const {
  size_t res = 0;
  for (auto l : layers) {
    res += l->get_varnum();
  }
  return res;
}
