
#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.hpp>
#endif
#include "clGetErrorString.hpp"
#include <string>

#include "matrix.hpp"
void init();
void teardown();

matrix mul_mat(const matrix &a, const matrix &b);
// VvalT mul_vec(const VvalT &a, const VvalT &b);
// VvalT add_vec(const VvalT &a, const VvalT &b);
matrix conv2d(const matrix &a, const matrix &b);

#include "convolution_layer.hpp"
// VvalT conv_l_forward(const convolution_layer &l, const vector<valT> &input);
VvalT conv_l_backward(const convolution_layer &l, const vector<valT> &grad);
VvalT conv_l_update(const convolution_layer &l, const vector<valT> &G,
                    const vector<valT> &input);
VvalT conv_l_forward(const convolution_layer &l, const vector<valT> &input);
