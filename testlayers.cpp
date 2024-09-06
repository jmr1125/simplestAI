#include "NN.hpp"
#include "adam.hpp"
#include "convolution_layer.hpp"
#include "ocl.hpp"
#include <iostream>
#include <random>
using namespace std;
int main() {
  init();
  std::random_device rd;
  nnet target;
  // target.add_func_layer({1, 1}, 10, Functions::softmax);
  //  target.add_bias_layer({1, 1}, 10);
  //   ?target.add_max_layer({2, 2}, 8, 9, 3);
  //   target.add_max_layer({2, 2}, 9, 9, 3);
  //     target.add_max_layer({2, 2}, 8, 8, 2);
  //    target.add_average_layer({2, 2}, 9, 9, 3);
  //     target.add_average_layer({2, 2}, 8, 8, 2);
  // target.add_convolution_layer({2, 3}, 8, 8, 3, 3);
  target.add_convolution_layer({3, 2}, 8, 8, 5, 5, 2);
  // target.add_matrix_layer({1, 1}, 4, 8);
  target.last_layer()->init(std::move(rd));
  nnet test = target;
  test.last_layer()->init(std::move(rd));
#if 1 // test update
  {
    adam A(test.get_varnum());
    valT lr = 0.001;
    for (; lr; cin >> lr) {
      for (int i = 0; i < 1000; ++i) {
        vector<valT> in(target.layers[0]->Isize);
        generate(in.begin(), in.end(), [&rd]() { return rand01(rd) * 2 - 1; });
        auto output = target.forward(in);
        auto predict = test.forward(in);
        {
          valT l2 = 0;
          for (int i = 0; i < predict.size(); ++i) {
            l2 += pow(output[i] - predict[i], 2);
          }
          cout << "l2: " << l2 << endl;
        }
        test.update(A.update(test.update(in, output, train_method::l2), lr, i));
      }
    }
  }
#endif
#if 1 // test backward
  {
    test = target;
    VvalT input_target(target.layers[0]->Isize);
    generate(input_target.begin(), input_target.end(),
             [&rd]() { return 4 * (rand01(rd) * 2 - 1); });
    target.forward(input_target);
    VvalT input_test(target.layers[0]->Isize);
    generate(input_test.begin(), input_test.end(),
             [&rd]() { return 4 * (rand01(rd) * 2 - 1); });
    for (valT lr = 0.01; lr; cin >> lr) {
      for (int i = 0; i < 1000; ++i) {
        test.forward(input_test);
        VvalT delta(test.last_layer()->Osize);
        valT l1 = 0;
        for (int i = 0; i < test.last_layer()->Osize; ++i) {
          delta[i] =
              test.last_layer()->output[i] - target.last_layer()->output[i];
          l1 += delta[i] * delta[i];
          delta[i] = delta[i] < 0 ? -1 : 1;
        }
        VvalT d = test.last_layer()->backward(delta);
        for (int i = 0; i < test.layers[0]->Isize; ++i)
          input_test[i] -= lr * d[i];
        cout << "l1: " << l1 << endl;
      }
    }
  }
#endif
}
