#include <OpenCL/OpenCL.h>
#include <cstddef>
#include <ctime>
#include <iostream>
#include <stdexcept>
#include <string>
#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.hpp>
#endif
#include <chrono>

#include "clGetErrorString.hpp"

#define right(s)                                                               \
  if (ret != CL_SUCCESS) {                                                     \
    cerr << "ERR: " << s << " : " << clGetErrorString(ret) << "(" << ret       \
         << ")" << endl;                                                       \
  }
using namespace std;

string Program = R"(
__kernel void mul(const unsigned int n,
__global float *a,
__global float *b,
__global float *c){
size_t x=get_global_id(0);
if(x>=n)return;
c[x]=a[x]*b[x];
}
__kernel void mul_mat(const unsigned int m,
const unsigned int n,
const unsigned int k,
__global float *a, // m*k
__global float *b, // k*n
__global float *c  // m*n
                   ){
size_t x=get_global_id(0),
       y=get_global_id(1);
if(x>=n){return;}
if(y>=m){return;}
float res=0;
for(unsigned int i=0;i<k;++i){
res+=a[y*k+i]*b[i*n+x];
}
c[y*m+x]=res;
}
)";
using std::chrono::system_clock;
auto t = system_clock::now();
// auto t1 = t;
//  #define time_get() (-(t - (t1 = system_clock::now())).count(), t = t1)
auto timeget() {
  auto t1 = system_clock::now();
  auto T = t1 - t;
  t = t1;
  return (double)T.count() / 1000000;
}
const int n = 10000, m = 10000, k = 10000;
float a[m][k], b[k][n], c[m][n];
int main() {
  cout << timeget() << " start" << endl;
  cl_int ret;
  vector<cl_platform_id> platforms;
  platforms.resize(100);
  cl_uint platform_count;
  ret = clGetPlatformIDs(1, platforms.data(), &platform_count);
  right("get platform");
  cout << "==platform==" << endl;
  cout << "count: " << platform_count << endl;
  cl_platform_id default_platform = platforms[0];
  string platform_name;
  {
    size_t len;
    clGetPlatformInfo(default_platform, CL_PLATFORM_NAME, 0, NULL, &len);
    platform_name.resize(len);
    clGetPlatformInfo(default_platform, CL_PLATFORM_NAME, len,
                      platform_name.data(), NULL);
  }
  cout << "name: " << platform_name << endl;
  cl_device_id device_id;
  ret =
      clGetDeviceIDs(default_platform, CL_DEVICE_TYPE_GPU, 1, &device_id, NULL);
  right("get device");
  size_t local[3];
  {
    size_t len;
    clGetDeviceInfo(device_id, CL_DEVICE_MAX_WORK_ITEM_SIZES, 0, NULL, &len);
    size_t sizes[len / 8];
    clGetDeviceInfo(device_id, CL_DEVICE_MAX_WORK_ITEM_SIZES, len, sizes, NULL);
    cout << "device: sizes: (x y z)" << len << endl;
    for (int i = 0; i < len / 8; ++i) {
      cout << sizes[i] << ' ';
    }
    local[0] = sizes[0] / 2;
    local[1] = sizes[1] / 2;
    local[2] = sizes[2] / 2;
    cout << dec << endl;
  }
  cl_context context = clCreateContext(0, 1, &device_id, NULL, NULL, &ret);
  right("create context");
  if (!context) {
    cout << "error: create context (" << ret << ")" << endl;
    return 1;
  }
  cl_command_queue command = clCreateCommandQueue(context, device_id, 0, &ret);
  right("create command");
  if (!command) {
    cout << "error: create command (" << ret << ")" << endl;
    return 1;
  }
  cl_program program;
  {
    const char *p[2] = {Program.data(), NULL};
    size_t l[2] = {Program.length(), 0};
    program = clCreateProgramWithSource(context, 1, p, l, &ret);
  }
  right("create program");
  if (!program) {
    cout << "error: create program (" << ret << ")" << endl;
    return 1;
  }

  // build the program
  ret = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
  if (ret != CL_SUCCESS) {
    string msg;
    size_t len;
    cout << "error in program" << endl;
    clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, 0, NULL,
                          &len);
    msg.resize(len);
    clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, len,
                          msg.data(), NULL);
    right(msg);
  }
  cl_kernel kernel = clCreateKernel(program, "mul_mat", &ret);
  right("create kernel");
  if (!kernel) {
    cout << "err in kernel" << endl;
    return 1;
  }
  cout << "[" << timeget() << "] ====init done=====" << endl;
  cl_mem in_a, in_b, out;
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < k; ++j) {
      a[i][j] = (float)rand() / 32768;
    }
  }
  for (int i = 0; i < k; ++i) {
    for (int j = 0; j < n; ++j) {
      b[i][j] = (float)rand() / 32768;
    }
  }
  cout << "[" << timeget() << "] ====randomize done=====" << endl;
  // create args buf
  in_a = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * n * m, NULL,
                        &ret);
  right("create buf a");
  in_b = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * k * m, NULL,
                        &ret);
  right("create buf b");
  out = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * n * k, NULL,
                       &ret);
  right("create buf out");
  // write buf
  ret = clEnqueueWriteBuffer(command, in_a, CL_TRUE, 0, sizeof(float) * m * k,
                             a, 0, NULL, NULL);
  right("write a");
  ret = clEnqueueWriteBuffer(command, in_b, CL_TRUE, 0, sizeof(float) * m * k,
                             b, 0, NULL, NULL);
  right("write b");
  cout << "[" << timeget() << "] create copy buffer" << endl;
  // set args
  ret = clSetKernelArg(kernel, 0, sizeof(unsigned int), &m);
  right("set arg 0");
  ret = clSetKernelArg(kernel, 1, sizeof(unsigned int), &n);
  right("set arg 1");
  ret = clSetKernelArg(kernel, 2, sizeof(unsigned int), &k);
  right("set arg 2");
  ret = clSetKernelArg(kernel, 3, sizeof(cl_mem), &in_a);
  right("set arg 3");
  ret = clSetKernelArg(kernel, 4, sizeof(cl_mem), &in_b);
  right("set arg 4");
  ret = clSetKernelArg(kernel, 5, sizeof(cl_mem), &out);
  right("set arg 5");
  size_t global[2];
  size_t kernelsize;
  ret = clGetKernelWorkGroupInfo(kernel, device_id, CL_KERNEL_WORK_GROUP_SIZE,
                                 sizeof(kernelsize), &kernelsize, NULL);
  right("get group info");
  cout << "group size: " << kernelsize << endl;
  global[0] = n;
  global[1] = m;
  cout << "[" << timeget() << "] set args and get group info" << endl;
  local[0] = local[1] = 16;
  ret = clEnqueueNDRangeKernel(command, kernel, 1, NULL, global, NULL, 0, NULL,
                               NULL);
  right("run");
  clFinish(command);
  cout << "[" << timeget() << "] done" << endl;
  ret = clEnqueueReadBuffer(command, out, CL_TRUE, 0, sizeof(float) * n * m, c,
                            0, NULL, NULL);
  cout << "[" << timeget() << "] copy done" << endl;
  clReleaseMemObject(in_a);
  clReleaseMemObject(in_b);
  clReleaseMemObject(out);
  clReleaseKernel(kernel);
  clReleaseProgram(program);
  clReleaseCommandQueue(command);
  clReleaseContext(context);
}
