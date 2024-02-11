
#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.hpp>
#endif
#include "clGetErrorString.hpp"
#include <string>
using namespace std;

string Program = R"(
__kernel void add_vec(const unsigned int n,
__global float *a,
__global float *b,
__global float *c){
size_t x=get_global_id(0);
if(x>=n)return;
c[x]=a[x]+b[x];
}
__kernel void mul_vec(const unsigned int n,
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
//res+=a[k*x+i]*b[n*i+y];
res+=a[y*k+i]*b[i*n+x];
}
c[y*m+x]=res;
}
)";

#define right(s)                                                               \
  if (ret != CL_SUCCESS) {                                                     \
    printf("ERR: " #s " : %s (%d)", clGetErrorString(ret), ret);               \
  }
cl_kernel k_mul_mat, k_mul_vec;

cl_context context;
cl_command_queue command;
cl_program program;
void init() {
  cl_int ret;
  vector<cl_platform_id> platforms;
  platforms.resize(100);
  cl_uint platform_count;
  ret = clGetPlatformIDs(1, platforms.data(), &platform_count);
  right("get platform");
  cl_platform_id default_platform = platforms[0];
  cl_device_id device_id;
  ret =
      clGetDeviceIDs(default_platform, CL_DEVICE_TYPE_GPU, 1, &device_id, NULL);
  right("get device");
  context = clCreateContext(0, 1, &device_id, NULL, NULL, &ret);
  right("create context");
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
    right(msg);
  }
  // create kernels

  k_mul_mat = clCreateKernel(program, "mul_mat", &ret);
  right("create kernel mat");
  k_mul_vec = clCreateKernel(program, "mul_vec", &ret);
  right("create kernel vec");
}
void teardown() {
  clReleaseKernel(k_mul_mat);
  clReleaseProgram(program);
  clReleaseCommandQueue(command);
  clReleaseContext(context);
}
