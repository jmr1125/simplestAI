#include "func_layer.hpp"
#include "layers.hpp"
#include "main.hpp"
#include <memory>
#include <utility>
#include <vector>
enum train_method {
  loss,l1,l2
};
struct nnet {
  virtual ~nnet();
  nnet() = default;
  nnet(const nnet &);
  nnet operator=(const nnet &other);
  void add_layer(std::shared_ptr<layer>);
  void add_average_layer(std::pair<int, int> channel, int i_n, int i_m,
                         int size);
  void add_bias_layer(std::pair<int, int> channel, int size);
  void add_convolution_layer(std::pair<int, int> channel, int i_n, int i_m,
                             int nK, int mK,int pad);
  void add_func_layer(std::pair<int, int> channel, int size, Functions);
  void add_matrix_layer(std::pair<int, int> channel, int isize, int osize);
  void add_max_layer(std::pair<int, int> channel, int i_n, int i_m, int size);
  std::shared_ptr<layer> last_layer() const;
  vector<valT> forward(vector<valT>);
  vector<valT> update(const vector<valT> &, const vector<valT> &,train_method m=loss) const;
  void update(const vector<valT> &);
  size_t get_varnum() const;
  void randomize_nan(std::random_device &&);
  vector<std::shared_ptr<layer>> layers;
};
