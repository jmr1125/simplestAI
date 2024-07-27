#include "NN.hpp"
#include "layers.hpp"
#include <algorithm>

void nnet::add_layer(layer *l) { layers.push_back(l); }
layer* nnet::last_layer(){return layers.back();}
vector<valT> nnet::forward(vector<valT> input) {
  for (auto *l : layers) {
    input=l->forward(input);
  }
  return std::move(input);
}
void nnet::update(const vector<valT> &input, const vector<valT> &expect,double lr) {
  auto n = layers.back()->output.size();
  vector<valT> delta(n);
  for (int i = 0; i < n; ++i) {
    delta[i] = -expect[i] / layers.back()->output[i];
    //delta[i] = layers.back()->output[i] - expect[i];
  }
  for (int i=layers.size()-1;i>=0;--i) {
    layers[i]->update(delta, (i == 0 ? input : layers[i - 1]->output), lr);
    delta = layers[i]->backward(delta);
  }
}
nnet::~nnet() {
  for (auto *l : layers) {
    delete l;
  }
}
