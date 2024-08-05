#include "ocl.hpp"
#include "layers.hpp"
#include "matrix.hpp"
#include <OpenCL/OpenCL.h>
#include <cstddef>
#include <cstdio>
#include <mutex>
#include <new>
#include <stdexcept>
#include <string>
#include <vector>
using namespace std;

string Program = R"(
__kernel void add_vec(const unsigned int n,
__global valT *a,
__global valT *b,
__global valT *c){
size_t x=get_global_id(0);
if(x>=n)return;
c[x]=a[x]+b[x];
}
__kernel void mul_vec(const unsigned int n,
__global valT *a,
__global valT *b,
__global valT *c){
size_t x=get_global_id(0);
if(x>=n)return;
c[x]=a[x]*b[x];
}
__kernel void mul_mat(const unsigned int m,
const unsigned int n,
const unsigned int k,
__global valT *a, // m*k
__global valT *b, // k*n
__global valT *c  // m*n
                   ){
size_t x=get_global_id(0),
       y=get_global_id(1);
if(x>=n){return;}
if(y>=m){return;}
valT res=0;
for(unsigned int i=0;i<k;++i){
//res+=a[k*x+i]*b[n*i+y];
res+=a[y*k+i]*b[i*n+x];
}
c[y*n+x]=res;
}

__kernel void conv2d(
__global valT *a,int n_a,int m_a,
__global valT *b,int n_b,int m_b,
__global valT *output){
int x = get_global_id(0);
int y = get_global_id(1);
if(!((x<n_a+n_b-1)&&y<(m_a+m_b-1)))return;
valT res=0.0;

for(int i=0;i<n_b;++i){
  int x1=x-i;
  if(x1<0||x1>=n_a)continue;
  for(int j=0;j<m_b;++j){//O[x][y]=sigma(i,j,A[x-i][y-j]*B[i][j])
    int y1=y-j;
    if(y1<0||y1>=m_a)continue;
    res+=a[x1*m_a+y1]*b[i*m_b+j];
  }
}

output[x*(m_a+m_b-1)+y]=res;
}
)";

#define CHECK_ERROR(err, msg)                                                  \
  if (err != CL_SUCCESS) {                                                     \
    printf("ERR: %s : %s (%d)\n", msg, clGetErrorString(err), err);            \
    throw runtime_error(msg);                                                  \
  }

class OpenCLContext {
public:
  OpenCLContext() {
    cl_int err;
    vector<cl_platform_id> platforms(100);
    cl_uint platform_count;
    err = clGetPlatformIDs(1, platforms.data(), &platform_count);
    CHECK_ERROR(err, "get platform");

    cl_platform_id default_platform = platforms[0];
    err = clGetDeviceIDs(default_platform, CL_DEVICE_TYPE_GPU, 1, &device_id,
                         NULL);
    CHECK_ERROR(err, "get device");

#ifdef valT_double
    Program = R"(
#define valT double
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
)" + Program;
#else
    Program = R"(
#define valT float
)" + Program;
#endif

    context = clCreateContext(0, 1, &device_id, NULL, NULL, &err);
    CHECK_ERROR(err, "create context");

    command_queue = clCreateCommandQueue(context, device_id, 0, &err);
    CHECK_ERROR(err, "create command queue");

    const char *source = Program.c_str();
    size_t length = Program.size();
    program = clCreateProgramWithSource(context, 1, &source, &length, &err);
    CHECK_ERROR(err, "create program");

    err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    if (err != CL_SUCCESS) {
      size_t len;
      clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, 0, NULL,
                            &len);
      vector<char> log(len);
      clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, len,
                            log.data(), NULL);
      printf("Build error:\n%s\n", log.data());
      exit(1);
    }

    k_conv2d = clCreateKernel(program, "conv2d", &err);
    CHECK_ERROR(err, "create kernel conv2d");
    k_mul_mat = clCreateKernel(program, "mul_mat", &err);
    CHECK_ERROR(err, "create kernel mul_mat");
  }

  ~OpenCLContext() {
    clReleaseKernel(k_conv2d);
    clReleaseKernel(k_mul_mat);
    clReleaseProgram(program);
    clReleaseCommandQueue(command_queue);
    clReleaseContext(context);
  }

  cl_context context;
  cl_command_queue command_queue;
  cl_program program;
  cl_kernel k_conv2d, k_mul_mat;
  cl_device_id device_id;
};

matrix conv2d(const matrix &a, const matrix &b) {
  OpenCLContext ocl;

  int n_a = a.getn();
  int m_a = a.getm();
  int n_b = b.getn();
  int m_b = b.getm();
  int n_o = n_a + n_b - 1;
  int m_o = m_a + m_b - 1;

  cl_int err;
  cl_mem in_a = clCreateBuffer(ocl.context, CL_MEM_READ_ONLY,
                               sizeof(valT) * n_a * m_a, NULL, &err);
  CHECK_ERROR(err, "create buffer a");
  cl_mem in_b = clCreateBuffer(ocl.context, CL_MEM_READ_ONLY,
                               sizeof(valT) * n_b * m_b, NULL, &err);
  CHECK_ERROR(err, "create buffer b");
  cl_mem out = clCreateBuffer(ocl.context, CL_MEM_READ_WRITE,
                              sizeof(valT) * n_o * m_o, NULL, &err);
  CHECK_ERROR(err, "create buffer out");

  vector<valT> A(n_a * m_a);
  vector<valT> B(n_b * m_b);
  vector<valT> C(n_o * m_o);

  for (int i = 0; i < n_a; ++i) {
    for (int j = 0; j < m_a; ++j) {
      A[i * m_a + j] = a(i, j);
    }
  }
  for (int i = 0; i < n_b; ++i) {
    for (int j = 0; j < m_b; ++j) {
      B[i * m_b + j] = b(i, j);
    }
  }

  err = clEnqueueWriteBuffer(ocl.command_queue, in_a, CL_TRUE, 0,
                             sizeof(valT) * n_a * m_a, A.data(), 0, NULL, NULL);
  CHECK_ERROR(err, "write buffer a");
  err = clEnqueueWriteBuffer(ocl.command_queue, in_b, CL_TRUE, 0,
                             sizeof(valT) * n_b * m_b, B.data(), 0, NULL, NULL);
  CHECK_ERROR(err, "write buffer b");

  err = clSetKernelArg(ocl.k_conv2d, 0, sizeof(cl_mem), &in_a);
  CHECK_ERROR(err, "set kernel arg 0");
  err = clSetKernelArg(ocl.k_conv2d, 1, sizeof(int), &n_a);
  CHECK_ERROR(err, "set kernel arg 1");
  err = clSetKernelArg(ocl.k_conv2d, 2, sizeof(int), &m_a);
  CHECK_ERROR(err, "set kernel arg 2");
  err = clSetKernelArg(ocl.k_conv2d, 3, sizeof(cl_mem), &in_b);
  CHECK_ERROR(err, "set kernel arg 3");
  err = clSetKernelArg(ocl.k_conv2d, 4, sizeof(int), &n_b);
  CHECK_ERROR(err, "set kernel arg 4");
  err = clSetKernelArg(ocl.k_conv2d, 5, sizeof(int), &m_b);
  CHECK_ERROR(err, "set kernel arg 5");
  err = clSetKernelArg(ocl.k_conv2d, 6, sizeof(cl_mem), &out);
  CHECK_ERROR(err, "set kernel arg 6");

  size_t global[] = {static_cast<size_t>(n_o), static_cast<size_t>(m_o)};
  err = clEnqueueNDRangeKernel(ocl.command_queue, ocl.k_conv2d, 2, NULL, global,
                               NULL, 0, NULL, NULL);
  CHECK_ERROR(err, "enqueue kernel");

  err = clEnqueueReadBuffer(ocl.command_queue, out, CL_TRUE, 0,
                            sizeof(valT) * n_o * m_o, C.data(), 0, NULL, NULL);
  CHECK_ERROR(err, "read buffer");

  matrix res;
  res.setn(n_o);
  res.setm(m_o);
  for (int i = 0; i < n_o; ++i) {
    for (int j = 0; j < m_o; ++j) {
      res(i, j) = C[i * m_o + j];
    }
  }

  clReleaseMemObject(in_a);
  clReleaseMemObject(in_b);
  clReleaseMemObject(out);

  return res;
}

matrix mul_mat(const matrix &a, const matrix &b) {
  if (a.getm() != b.getn()) {
    throw dimension_error("a.m = " + to_string(a.getm()) +
                          "and b.n = " + to_string(b.getn()));
  }

  OpenCLContext ocl;
  auto m = a.getn(), k = a.getm(), n = b.getm();
  cl_mem in_a, in_b, out;
  cl_int ret;
  in_a = clCreateBuffer(ocl.context, CL_MEM_READ_ONLY, sizeof(valT) * m * k,
                        NULL, &ret);
  CHECK_ERROR(ret, "create buf a");
  in_b = clCreateBuffer(ocl.context, CL_MEM_READ_ONLY, sizeof(valT) * k * n,
                        NULL, &ret);
  CHECK_ERROR(ret, "create buf b");
  out = clCreateBuffer(ocl.context, CL_MEM_READ_WRITE, sizeof(valT) * m * n,
                       NULL, &ret);
  CHECK_ERROR(ret, "create buf out");
  auto A = new valT[m * k];
  auto B = new valT[k * n];
  auto C = new valT[m * n];

  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < k; ++j) {
      A[i * k + j] = a(i, j);
    }
  }
  for (int i = 0; i < k; ++i) {
    for (int j = 0; j < n; ++j) {
      B[i * n + j] = b(i, j);
    }
  }

  cl_event wrtA = clCreateUserEvent(ocl.context, &ret);
  CHECK_ERROR(ret, "create event A");
  cl_event wrtB = clCreateUserEvent(ocl.context, &ret);
  CHECK_ERROR(ret, "create event B");
  cl_event computed = clCreateUserEvent(ocl.context, &ret);
  CHECK_ERROR(ret, "create event done");
  ret = clEnqueueWriteBuffer(ocl.command_queue, in_a, CL_FALSE, 0,
                             sizeof(valT) * m * k, A, 0, NULL, &wrtA);
  CHECK_ERROR(ret, "write a");
  ret = clEnqueueWriteBuffer(ocl.command_queue, in_b, CL_FALSE, 0,
                             sizeof(valT) * k * n, B, 0, NULL, &wrtB);
  CHECK_ERROR(ret, "write b");
  ret = clSetKernelArg(ocl.k_mul_mat, 0, sizeof(unsigned int), &m);
  CHECK_ERROR(ret, "set arg 0");
  ret = clSetKernelArg(ocl.k_mul_mat, 1, sizeof(unsigned int), &n);
  CHECK_ERROR(ret, "set arg 1");
  ret = clSetKernelArg(ocl.k_mul_mat, 2, sizeof(unsigned int), &k);
  CHECK_ERROR(ret, "set arg 2");
  ret = clSetKernelArg(ocl.k_mul_mat, 3, sizeof(cl_mem), &in_a);
  CHECK_ERROR(ret, "set arg 3");
  ret = clSetKernelArg(ocl.k_mul_mat, 4, sizeof(cl_mem), &in_b);
  CHECK_ERROR(ret, "set arg 4");
  ret = clSetKernelArg(ocl.k_mul_mat, 5, sizeof(cl_mem), &out);
  CHECK_ERROR(ret, "set arg 5");
  cl_event waitlist[] = {wrtA, wrtB};
  size_t global[] = {n, m};
  ret = clEnqueueNDRangeKernel(ocl.command_queue, ocl.k_mul_mat, 2, NULL,
                               global, NULL, 2, waitlist, &computed);
  CHECK_ERROR(ret, "run");
  waitlist[0] = computed;
  waitlist[1] = NULL;
  ret = clEnqueueReadBuffer(ocl.command_queue, out, CL_TRUE, 0,
                            sizeof(valT) * m * n, C, 1, waitlist, NULL);
  CHECK_ERROR(ret, "read buffer");
  matrix res;
  res.setn(m);
  res.setm(n);
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      res(i, j) = C[i * n + j];
    }
  }
  delete[] A;
  delete[] B;
  delete[] C;
  clReleaseMemObject(in_a);
  clReleaseMemObject(in_b);
  clReleaseMemObject(out);
  return res;
}
