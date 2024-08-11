#include "NN.hpp"
#include "layers.hpp"
#include "main.hpp"
#include <algorithm>
#include <cstddef>
#include <iterator>
#include <memory>
#include <stdexcept>
#include <vector>

void nnet::add_layer(std::shared_ptr<layer> l) { layers.push_back({l}); }
std::shared_ptr<layer> nnet::last_layer() const { return layers.back(); }
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
  for (auto x : layers) {
    x.reset();
  }
}
void copy(nnet &a, const nnet &b) {
  a.layers.clear();
  for (auto x : b.layers) {
    a.layers.push_back(x->clone());
  }
}
nnet::nnet(const nnet &x) { copy(*this, x); }
nnet nnet::operator=(const nnet &other) {
  if (this != &other) {
    copy(*this, other);
  }
  return *this;
}
size_t nnet::get_varnum() const {
  size_t res = 0;
  for (auto l : layers) {
    res += l->get_varnum();
  }
  return res;
}
#include "average_layer.hpp"
void nnet::add_average_layer(std::pair<int, int> channel, int i_n, int i_m,
                             int size) {
  if (channel.first != channel.second) {
    throw std::runtime_error("add_average_layer IOchannel");
  }
  add_layer(std::make_shared<average_layer>());
  dynamic_cast<average_layer *>(last_layer().get())->i_n = i_n;
  dynamic_cast<average_layer *>(last_layer().get())->i_m = i_m;
  dynamic_cast<average_layer *>(last_layer().get())->size = size;
  last_layer()->Isize = i_n * i_m * channel.first;
  last_layer()->Osize =
      ceil(1.0 * i_n / size) * ceil(1.0 * i_m / size) * channel.second;
  last_layer()->Ichannels = channel.first;
  last_layer()->Ochannels = channel.second;
  last_layer()->set_IOsize(last_layer()->Isize, last_layer()->Osize);
}
#include "bias_layer.hpp"
void nnet::add_bias_layer(std::pair<int, int> channel, int size) {
  if (channel.first != channel.second) {
    throw std::runtime_error("add_bias_layer IOchannel");
  }
  add_layer(std::make_shared<bias_layer>());
  last_layer()->Isize = last_layer()->Osize = size * channel.first;
  last_layer()->Ichannels = channel.first;
  last_layer()->Ochannels = channel.second;
  last_layer()->set_IOsize(last_layer()->Isize, last_layer()->Osize);
}
#include "convolution_layer.hpp"
void nnet::add_convolution_layer(std::pair<int, int> channel, int i_n, int i_m,
                                 int nK, int mK) {
  add_layer(std::make_shared<convolution_layer>());
  dynamic_cast<convolution_layer *>(last_layer().get())->Ichannels =
      channel.first;
  dynamic_cast<convolution_layer *>(last_layer().get())->Ochannels =
      channel.second;
  dynamic_cast<convolution_layer *>(last_layer().get())->nK = nK;
  dynamic_cast<convolution_layer *>(last_layer().get())->mK = mK;
  dynamic_cast<convolution_layer *>(last_layer().get())->n_in = i_n;
  dynamic_cast<convolution_layer *>(last_layer().get())->m_in = i_m;
  dynamic_cast<convolution_layer *>(last_layer().get())->Isize =
      channel.first * i_n * i_m;

  dynamic_cast<convolution_layer *>(last_layer().get())->Osize =
      channel.second * i_n * i_m;
  last_layer()->set_IOsize(last_layer()->Isize, last_layer()->Osize);
}
#include "func_layer.hpp"
void nnet::add_func_layer(std::pair<int, int> channel, int size, Functions f) {
  if (channel.first != channel.second) {
    throw std::runtime_error("add_func_layer IOchannel");
  }
  add_layer(std::make_shared<func_layer>());
  dynamic_cast<func_layer *>(last_layer().get())->f = f;
  dynamic_cast<func_layer *>(last_layer().get())->Ichannels = channel.first;
  dynamic_cast<func_layer *>(last_layer().get())->Ochannels = channel.second;
  dynamic_cast<func_layer *>(last_layer().get())->Isize = channel.first * size;
  dynamic_cast<func_layer *>(last_layer().get())->Osize = channel.second * size;
  last_layer()->set_IOsize(last_layer()->Isize, last_layer()->Osize);
}
#include "matrix_layer.hpp"
void nnet::add_matrix_layer(std::pair<int, int> channel, int isize, int osize) {
  add_layer(std::make_shared<matrix_layer>());
  dynamic_cast<matrix_layer *>(last_layer().get())->Ichannels = channel.first;
  dynamic_cast<matrix_layer *>(last_layer().get())->Ochannels = channel.second;
  dynamic_cast<matrix_layer *>(last_layer().get())->Isize =
      channel.first * isize;
  dynamic_cast<matrix_layer *>(last_layer().get())->Osize =
      channel.second * osize;
  last_layer()->set_IOsize(last_layer()->Isize, last_layer()->Osize);
}
#include "max_layer.hpp"
void nnet::add_max_layer(std::pair<int, int> channel, int i_n, int i_m,
                         int size) {
  if (channel.first != channel.second) {
    throw std::runtime_error("add_max_layer IOchannel");
  }
  add_layer(std::make_shared<max_layer>());
  dynamic_cast<max_layer *>(last_layer().get())->i_n = i_n;
  dynamic_cast<max_layer *>(last_layer().get())->i_m = i_m;
  dynamic_cast<max_layer *>(last_layer().get())->size = size;
  last_layer()->Isize = i_n * i_m * channel.first;
  last_layer()->Osize =
      ceil(1.0 * i_n / size) * ceil(1.0 * i_m / size) * channel.second;
  last_layer()->Ichannels = channel.first;
  last_layer()->Ochannels = channel.second;
  last_layer()->set_IOsize(last_layer()->Isize, last_layer()->Osize);
}
