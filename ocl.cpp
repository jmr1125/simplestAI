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

#define right(s)                                                               \
  if (ret != CL_SUCCESS) {                                                     \
    printf("ERR: " #s " : %s (%d)", clGetErrorString(ret), ret);               \
    exit(1);                                                                   \
  }
mutex contextM;
#define Init()                                                                 \
  cl_int ret;                                                                  \
  cl_kernel k_mul_mat, k_mul_vec, k_add_vec, k_conv2d;                         \
  cl_context context;                                                          \
  cl_command_queue command;                                                    \
  cl_program program;                                                          \
  init(ret, k_mul_mat, k_mul_vec, k_add_vec, k_conv2d, context, command,       \
       program)
void init(cl_int &ret, cl_kernel &k_mul_mat, cl_kernel &k_mul_vec,
          cl_kernel &k_add_vec, cl_kernel &k_conv2d, cl_context &context,
          cl_command_queue &command, cl_program program) {
  vector<cl_platform_id> platforms;
  platforms.resize(100);
  cl_uint platform_count;
  ret = clGetPlatformIDs(1, platforms.data(), &platform_count);
  right("get platform");
  cl_platform_id default_platform = platforms[0];
  cl_device_id device_id;
  ret =
      clGetDeviceIDs(default_platform, CL_DEVICE_TYPE_GPU, 1, &device_id, NULL);

#ifdef valT_double
#warning need cl_khr_fp_64
  Program = R"(
#define valT double
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
)" + Program;
  {
    size_t ext_size;
    ret =
        clGetDeviceInfo(device_id, CL_DEVICE_EXTENSIONS, 0, nullptr, &ext_size);
    right("get ext size");
    vector<char> ext_data(ext_size);
    ret = clGetDeviceInfo(device_id, CL_DEVICE_EXTENSIONS, ext_size,
                          ext_data.data(), nullptr);
    right("get ext");
    string extensions(ext_data.begin(), ext_data.end());
    if (extensions.find("cl_khr_fp_64") == string::npos) {

      // throw runtime_error("NO SPPORT cl_khr_fp_64\n");
    }
  }
#else
  Program = R"(
#define valT float
)" + Program;
#endif

  right("get device");
  {
    lock_guard lock(contextM);
    context = clCreateContext(0, 1, &device_id, NULL, NULL, &ret);
    right("create context");
  }
  command = clCreateCommandQueue(context, device_id, 0, &ret);
  right("create command");
  {
    const char *p[2] = {Program.data(), NULL};
    size_t l[2] = {Program.length(), 0};
    program = clCreateProgramWithSource(context, 1, p, l, &ret);
  }
  ret = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
  if (ret != CL_SUCCESS) {
    string msg;
    size_t len;
    clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, 0, NULL,
                          &len);
    msg.resize(len);
    clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, len,
                          msg.data(), NULL);
    printf("error: \n%s", msg.c_str());
    right(msg);
    throw runtime_error(msg);
  }
  // create kernels

  k_mul_mat = clCreateKernel(program, "mul_mat", &ret);
  right("create kernel mul mat");
  k_mul_vec = clCreateKernel(program, "mul_vec", &ret);
  right("create kernel mul vec");
  k_add_vec = clCreateKernel(program, "add_vec", &ret);
  right("create kernel add vec");
  k_conv2d = clCreateKernel(program, "conv2d", &ret);
  right("create kernel conv2d");
}
#define Teardown()                                                             \
  teardown(k_mul_mat, k_mul_vec, k_add_vec, k_conv2d, context, command, program)
void teardown(cl_kernel &k_mul_mat, cl_kernel &k_mul_vec, cl_kernel &k_add_vec,
              cl_kernel &k_conv2d, cl_context &context,
              cl_command_queue &command, cl_program program) {
  clReleaseKernel(k_mul_mat);
  clReleaseKernel(k_mul_vec);
  clReleaseKernel(k_add_vec);
  clReleaseKernel(k_conv2d);
  clReleaseProgram(program);
  clReleaseCommandQueue(command);
  clReleaseContext(context);
}

matrix mul_mat(const matrix &a, const matrix &b) {
  if (a.getm() != b.getn()) {
    throw dimension_error("a.m = " + to_string(a.getm()) +
                          "and b.n = " + to_string(b.getn()));
  }
  Init();
  auto m = a.getn(), k = a.getm(), n = b.getm();
  cl_mem in_a, in_b, out;
  in_a = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(valT) * m * k, NULL,
                        &ret);
  right("create buf a");
  in_b = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(valT) * k * n, NULL,
                        &ret);
  right("create buf b");
  out = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(valT) * m * n, NULL,
                       &ret);
  right("create buf out");
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

  cl_event wrtA = clCreateUserEvent(context, &ret);
  right("create event A");
  cl_event wrtB = clCreateUserEvent(context, &ret);
  right("create event B");
  cl_event computed = clCreateUserEvent(context, &ret);
  right("create event done");
  ret = clEnqueueWriteBuffer(command, in_a, CL_FALSE, 0, sizeof(valT) * m * k,
                             A, 0, NULL, &wrtA);
  right("write a");
  ret = clEnqueueWriteBuffer(command, in_b, CL_FALSE, 0, sizeof(valT) * k * n,
                             B, 0, NULL, &wrtB);
  right("write b");
  ret = clSetKernelArg(k_mul_mat, 0, sizeof(unsigned int), &m);
  right("set arg 0");
  ret = clSetKernelArg(k_mul_mat, 1, sizeof(unsigned int), &n);
  right("set arg 1");
  ret = clSetKernelArg(k_mul_mat, 2, sizeof(unsigned int), &k);
  right("set arg 2");
  ret = clSetKernelArg(k_mul_mat, 3, sizeof(cl_mem), &in_a);
  right("set arg 3");
  ret = clSetKernelArg(k_mul_mat, 4, sizeof(cl_mem), &in_b);
  right("set arg 4");
  ret = clSetKernelArg(k_mul_mat, 5, sizeof(cl_mem), &out);
  right("set arg 5");
  cl_event waitlist[] = {wrtA, wrtB};
  size_t global[] = {n, m};
  ret = clEnqueueNDRangeKernel(command, k_mul_mat, 2, NULL, global, NULL, 2,
                               waitlist, &computed);
  right("run");
  waitlist[0] = computed;
  waitlist[1] = NULL;
  ret = clEnqueueReadBuffer(command, out, CL_TRUE, 0, sizeof(valT) * m * n, C,
                            1, waitlist, NULL);
  right("read buffer");
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
  Teardown();
  return res;
}

// VvalT mul_vec(const VvalT &a, const VvalT &b) {
//   if (a.size() != b.size()) {
//     throw dimension_error("a.m = " + to_string(a.size()) +
//                           "and b.n = " + to_string(b.size()));
//   }
//   auto n = b.size();
//   cl_mem in_a, in_b, out;
//   in_a =
//       clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(valT) * n, NULL,
//       &ret);
//   right("create buf a");
//   in_b =
//       clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(valT) * n, NULL,
//       &ret);
//   right("create buf b");
//   out =
//       clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(valT) * n, NULL,
//       &ret);
//   right("create buf out");

//   cl_event wrtA = clCreateUserEvent(context, &ret);
//   right("create event A");
//   cl_event wrtB = clCreateUserEvent(context, &ret);
//   right("create event B");
//   cl_event computed = clCreateUserEvent(context, &ret);
//   right("create event done");
//   ret = clEnqueueWriteBuffer(command, in_a, CL_FALSE, 0, sizeof(valT) * n,
//                              a.data(), 0, NULL, &wrtA);
//   right("write a");
//   ret = clEnqueueWriteBuffer(command, in_b, CL_FALSE, 0, sizeof(valT) * n,
//                              b.data(), 0, NULL, &wrtB);
//   right("write b");

//   ret = clSetKernelArg(k_mul_vec, 0, sizeof(unsigned int), &n);
//   right("set arg 0");
//   ret = clSetKernelArg(k_mul_vec, 1, sizeof(cl_mem), &in_a);
//   right("set arg 1");
//   ret = clSetKernelArg(k_mul_vec, 2, sizeof(cl_mem), &in_b);
//   right("set arg 2");
//   ret = clSetKernelArg(k_mul_vec, 3, sizeof(cl_mem), &out);
//   right("set arg 3");
//   cl_event waitlist[] = {wrtA, wrtB};
//   size_t global[] = {n};
//   ret = clEnqueueNDRangeKernel(command, k_mul_vec, 1, NULL, global, NULL, 2,
//                                waitlist, &computed);
//   right("run");
//   waitlist[0] = computed;
//   waitlist[1] = NULL;
//   VvalT res;
//   res.resize(n);
//   ret = clEnqueueReadBuffer(command, out, CL_TRUE, 0, sizeof(valT) * n,
//                             res.data(), 1, waitlist, NULL);
//   right("read buffer");
//   clReleaseMemObject(in_a);
//   clReleaseMemObject(in_b);
//   clReleaseMemObject(out);
//   return res;
// }

// VvalT add_vec(const VvalT &a, const VvalT &b) {
//   if (a.size() != b.size()) {
//     throw dimension_error("a.m = " + to_string(a.size()) +
//                           "and b.n = " + to_string(b.size()));
//   }
//   auto n = b.size();
//   cl_mem in_a, in_b, out;
//   in_a =
//       clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(valT) * n, NULL,
//       &ret);
//   right("create buf a");
//   in_b =
//       clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(valT) * n, NULL,
//       &ret);
//   right("create buf b");
//   out =
//       clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(valT) * n, NULL,
//       &ret);
//   right("create buf out");

//   cl_event wrtA = clCreateUserEvent(context, &ret);
//   right("create event A");
//   cl_event wrtB = clCreateUserEvent(context, &ret);
//   right("create event B");
//   cl_event computed = clCreateUserEvent(context, &ret);
//   right("create event done");
//   ret = clEnqueueWriteBuffer(command, in_a, CL_FALSE, 0, sizeof(valT) * n,
//                              a.data(), 0, NULL, &wrtA);
//   right("write a");
//   ret = clEnqueueWriteBuffer(command, in_b, CL_FALSE, 0, sizeof(valT) * n,
//                              b.data(), 0, NULL, &wrtB);
//   right("write b");

//   ret = clSetKernelArg(k_add_vec, 0, sizeof(unsigned int), &n);
//   right("set arg 0");
//   ret = clSetKernelArg(k_add_vec, 1, sizeof(cl_mem), &in_a);
//   right("set arg 1");
//   ret = clSetKernelArg(k_add_vec, 2, sizeof(cl_mem), &in_b);
//   right("set arg 2");
//   ret = clSetKernelArg(k_add_vec, 3, sizeof(cl_mem), &out);
//   right("set arg 3");
//   cl_event waitlist[] = {wrtA, wrtB};
//   size_t global[] = {n};
//   ret = clEnqueueNDRangeKernel(command, k_add_vec, 1, NULL, global, NULL, 2,
//                                waitlist, &computed);
//   right("run");
//   waitlist[0] = computed;
//   waitlist[1] = NULL;
//   VvalT res;
//   res.resize(n);
//   ret = clEnqueueReadBuffer(command, out, CL_TRUE, 0, sizeof(valT) * n,
//                             res.data(), 1, waitlist, NULL);
//   right("read buffer");
//   clReleaseMemObject(in_a);
//   clReleaseMemObject(in_b);
//   clReleaseMemObject(out);
//   return res;
// }

matrix conv2d(const matrix &a, const matrix &b) {
  Init();
  int n_a = a.getn();
  int m_a = a.getm();
  int n_b = b.getn();
  int m_b = b.getm();
  int n_o = n_a + n_b - 1;
  int m_o = m_a + m_b - 1;
  cl_mem in_a, in_b, out;
  in_a = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(valT) * n_a * m_a,
                        NULL, &ret);
  right("create buf a");
  in_b = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(valT) * n_b * m_b,
                        NULL, &ret);
  right("create buf b");
  out = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(valT) * n_o * m_o,
                       NULL, &ret);
  right("create buf out");
  auto A = new valT[n_a * m_a];
  auto B = new valT[n_b * m_b];
  auto C = new valT[n_o * m_o];

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

  cl_event wrtA = clCreateUserEvent(context, &ret);
  right("create event A");
  cl_event wrtB = clCreateUserEvent(context, &ret);
  right("create event B");
  cl_event computed = clCreateUserEvent(context, &ret);
  right("create event done");
  ret = clEnqueueWriteBuffer(command, in_a, CL_FALSE, 0,
                             sizeof(valT) * n_a * m_a, A, 0, NULL, &wrtA);
  right("write a");
  ret = clEnqueueWriteBuffer(command, in_b, CL_FALSE, 0,
                             sizeof(valT) * n_b * m_b, B, 0, NULL, &wrtB);
  right("write b");
  ret = clSetKernelArg(k_conv2d, 0, sizeof(cl_mem), &in_a);
  right("set arg 0");
  ret = clSetKernelArg(k_conv2d, 1, sizeof(int), &n_a);
  right("set arg 1");
  ret = clSetKernelArg(k_conv2d, 2, sizeof(int), &m_a);
  right("set arg 2");
  ret = clSetKernelArg(k_conv2d, 3, sizeof(cl_mem), &in_b);
  right("set arg 3");
  ret = clSetKernelArg(k_conv2d, 4, sizeof(int), &n_b);
  right("set arg 4");
  ret = clSetKernelArg(k_conv2d, 5, sizeof(int), &m_b);
  right("set arg 5");
  ret = clSetKernelArg(k_conv2d, 6, sizeof(cl_mem), &out);
  right("set arg 6");
  cl_event waitlist[] = {wrtA, wrtB};
  size_t global[] = {static_cast<size_t>(n_o), static_cast<size_t>(m_o)};
  ret = clEnqueueNDRangeKernel(command, k_conv2d, 2, NULL, global, NULL, 2,
                               waitlist, &computed);
  right("run");
  waitlist[0] = computed;
  waitlist[1] = NULL;
  ret = clEnqueueReadBuffer(command, out, CL_TRUE, 0, sizeof(valT) * n_o * m_o,
                            C, 1, waitlist, NULL);
  right("read buffer");
  matrix res;
  res.setn(n_o);
  res.setm(m_o);
  for (int i = 0; i < n_o; ++i) {
    for (int j = 0; j < m_o; ++j) {
      res(i, j) = C[i * m_o + j];
    }
  }
  delete[] A;
  delete[] B;
  delete[] C;
  clReleaseMemObject(in_a);
  clReleaseMemObject(in_b);
  clReleaseMemObject(out);
  Teardown();
  return std::move(res);
}
