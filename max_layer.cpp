#include "max_layer.hpp"
#include "main.hpp"
#include "matrix.hpp"
#include <limits>
#include <ostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

max_layer::~max_layer() {}

void max_layer::init(std::random_device &&) {}
void max_layer::set_IOsize(int isize, int osize) {
  if (i_n * i_m * Ichannels != isize ||
      ceil(1.0 * i_n / size) * ceil(1.0 * i_m / size) * Ochannels != osize) {
    throw std::runtime_error("init max_layer : io: " + std::to_string(isize) +
                             " , " + std::to_string(osize) + " nm: " +
                             std::to_string(i_n) + " , " + std::to_string(i_m));
  }
  output.resize(ceil(1.0 * i_n / size) * ceil(1.0 * i_m / size) * Ichannels);
  Isize = isize;
  Osize = osize;
}
VvalT max_layer::forward(const vector<valT> &in) {
  const int o_n = ceil(1.0 * i_n / size);
  const int o_m = ceil(1.0 * i_m / size);
  input = in;
  // for (int c = 0; c < Ichannels; ++c)
  //   for (int i = 0; i < o_n; ++i) {
  //     for (int j = 0; j < o_m; ++j) {
  //       output[i * o_m + j + c * o_m * o_n] =
  //           (i * size >= i_n || j * size >= i_m)
  //               ? -100
  //               : in[i * size * i_m + j * size + c * i_n * i_m];
  //       for (int dx = 0; dx < size; ++dx)
  //         for (int dy = 0; dy < size; ++dy) {
  //           output[i * o_m + j + c * o_m * o_n] =
  //               std::max(output[i * o_m + j + c * o_m * o_n],
  //                        (i * size + dx >= i_n || j * size + dy >= i_m)
  //                            ? -100
  //                            : in[(i * size + dx) * i_n + j * size + dy +
  //                                 c * i_n * i_m]);
  //         }
  //     }
  //   }
  fill(output.begin(), output.end(), -std::numeric_limits<valT>::max());

  for (int c = 0; c < Ichannels; ++c) {
    for (int i = 0; i < i_n; ++i) {
      for (int j = 0; j < i_m; ++j) {
        output.at((i / size) * o_m + j / size + c * o_n * o_m) =
            std::max(output.at((i / size) * o_m + j / size + c * o_n * o_m),
                     in.at(i * i_m + j + c * i_n * i_m));
      }
    }
  }
  return output;
}
VvalT max_layer::backward(const VvalT &grad) const {
  const int o_n = i_n / size;
  const int o_m = i_m / size;
  VvalT res;
  res.resize(i_n * i_m * Ichannels);
  for (int c = 0; c < Ichannels; ++c)
    for (int i = 0; i < i_n; ++i) {
      for (int j = 0; j < i_m; ++j) {
        if (input[i * i_m + j + c * i_n * i_m] ==
            output[(i / size) * o_m + j / size + c * o_n * o_m])
          res[i * i_m + j + c * i_m * i_n] =
              grad[(i / size) * o_m + j / size + c * o_m * o_n];
        else
          res[i * i_m + j + c * i_m * i_n] = 0;
      }
    }
  return res;
}
VvalT max_layer::update(const VvalT &, const VvalT &) const { return {}; }
void max_layer::update(VvalT::const_iterator &) { return; }

void max_layer::save(ostream &o) const {
  o << i_n << " " << i_m << " " << size << " " << Ichannels << std::endl;
}
void max_layer::load(std::istream &i) {
  i >> i_n >> i_m >> size >> Ichannels;
  Ochannels = Ichannels;
  set_IOsize(i_n * i_m * Ichannels, i_n / 2 * i_m / 2 * Ichannels);
}

size_t max_layer::get_varnum() const { return 0; }

std::shared_ptr<layer> max_layer::clone() const {
  return std::make_shared<max_layer>(*this);
}

void max_layer::randomize_nan(std::random_device &&) { return; }
