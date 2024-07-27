
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
