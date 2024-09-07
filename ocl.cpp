#include "ocl.hpp"
#include "convolution_layer.hpp"
#include "layers.hpp"
#include "main.hpp"
#include "matrix.hpp"
#include <OpenCL/OpenCL.h>
#include <iostream>
#ifdef __APPLE__
#define CL_HPP_TARGET_OPENCL_VERSION 120
#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#include <OpenCL/opencl.hpp>
#else
#include <CL/cl.hpp>
#endif
#include <cstddef>
#include <cstdio>
#include <new>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
using namespace std;

string Program = R"(
#define valT float

__kernel void mul_mat(const unsigned int m,
const unsigned int n,
const unsigned int k,
__global valT *a, // m*k
__global valT *b, // k*n
__global valT *c  // m*n
                   ){
size_t x=get_global_id(0),
       y=get_global_id(1);
if(x<0||x>=n){return;}
if(y<0||y>=m){return;}
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


#define o_n (n_in - nK + 1 + pad * 2)
#define o_m (m_in - mK + 1 + pad * 2)


__kernel void conv_l_backforward(
int Ich,int Och,int pad,
int nK   ,int mK   ,__global valT *Ks,
int n_in ,int m_in ,__global valT *grad,//grad also input
__global valT *output,int flag) {
if(flag==0){
  //printf("backward\n");
  int xi = get_global_id(0);// of input
  int yi = get_global_id(1);// of input
  int Ic = get_global_id(2);
  if(xi<0||xi>=n_in||yi<0||yi>=m_in||Ic<0||Ic>=Ich)
    return;
  valT res=0;
  for(int Oc=0;Oc<Och;++Oc){
    for(int xk=0;xk<nK;++xk){
      for(int yk=0;yk<mK;++yk){
        int xo=xi-nK+pad+xk+1;
        int yo=yi-mK+pad+yk+1;
        if(xo<0||xo>=o_n)continue;
        if(yo<0||yo>=o_m)continue;
        res+=Ks[(Oc*Ich+Ic)*nK*mK+xk*mK+yk]*grad[Oc*o_n*o_m+xo*o_m+yo];
      }
    }
  }
  // output [x][y][Ic]
  output[Ic*n_in*m_in+xi*m_in+yi]=res;
 }else{
//printf("forward\n");
int xo=get_global_id(0);
int yo=get_global_id(1);
int Oc=get_global_id(2);
// printf("%d %d %d\n",xo,yo,Oc);
if(xo<0||xo>=o_n||yo<0||yo>=o_m)return;
valT res=0;
for(int Ic=0;Ic<Ich;++Ic){
  for(int xk=0;xk<nK;++xk){
    for(int yk=0;yk<mK;++yk){
      int xi=xo+nK-pad-xk-1;
      int yi=yo+mK-pad-yk-1;
      if(xi<0||xi>=n_in)continue;
      if(yi<0||yi>=m_in)continue;
      res+=Ks[(Oc*Ich+Ic)*nK*mK+xk*mK+yk]*grad[Ic*n_in*m_in+xi*m_in+yi];
    }
  }
}
//printf("%d %d %d %d %d %d\n",Oc,o_n,o_m,xo,o_m,yo);
output[Oc*o_n*o_m+xo*o_m+yo]=res;
 }
}

__kernel void conv_l_update(
int Ich,int Och,int pad,
int nK,int mK,__global valT *input,
int n_in,int m_in,__global valT *grad,
__global valT *output
){
//printf("update\n");
int xk = get_global_id(0); // of kernel
int yk = get_global_id(1); // of kernel
int Ic = get_global_id(2);

  if(xk<0||xk>=nK||yk<0||yk>=mK||Ic<0||Ic>=Ich)
    return;

for(int Oc=0;Oc<Och;++Oc){
  valT res=0;
/*
  for(int x=0;x<o_n;++x){
    int xi=x+(nK-1-pad)-i;
    if(xi>=o_n) continue;
    if(xi<0) continue;
    for(int y=0;y<o_m;++y){
      int yi=y+(mK-1-pad)-j;
      if(yi>=o_m) continue;
      if(yi<0) continue;
      res+=
      grad[Oc*n*m+x*m+y]
      *
      input[Ic*n*m+xi*m+yi];
    }
  }*/

  for(int xo=0;xo<o_n;++xo){
    for(int yo=0;yo<o_m;++yo){
      int xi = nK + xo - pad - xk - 1;
      int yi = mK + yo - pad - yk - 1;
      if (xi < 0 || xi >= n_in)
        continue;
      if (yi < 0 || yi >= m_in)
        continue;
      res+=grad[Oc*o_n*o_m+xo*o_m+yo]*input[Ic*n_in*m_in+xi*m_in+yi];
    }
  }
  output[(Oc*Ich+Ic)*nK*mK+xk*mK+yk]=res;
} // for Oc

}

)";
cl::Context context(CL_DEVICE_TYPE_GPU);
cl::Program program(context, Program, true);
cl::CommandQueue queue(context);

cl::compatibility::make_kernel<unsigned int, unsigned int, unsigned int,
                               cl::Buffer, cl::Buffer, cl::Buffer>
    k_mul_mat(program, "mul_mat");
cl::compatibility::make_kernel<cl::Buffer, int, int, cl::Buffer, int, int,
                               cl::Buffer>
    k_conv2d(program, "conv2d");
cl::compatibility::make_kernel<int, int, int, int, int, cl::Buffer, int, int,
                               cl::Buffer, cl::Buffer, int>
    k_conv_l_backforward(program, "conv_l_backforward");
cl::compatibility::make_kernel<int, int, int, int, int, cl::Buffer, int, int,
                               cl::Buffer, cl::Buffer>
    k_conv_l_update(program, "conv_l_update");

void init() {
  vector<cl::Platform> platforms;
  cl::Platform::get(&platforms);
  vector<cl::Device> devices;
  platforms.front().getDevices(CL_DEVICE_TYPE_DEFAULT, &devices);

  std::cout << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(devices.front())
            << endl;
}
void teardown() {}

matrix mul_mat(const matrix &a, const matrix &b) {
  if (a.getm() != b.getn()) {
    throw dimension_error("a.m = " + to_string(a.getm()) +
                          "and b.n = " + to_string(b.getn()));
  }
  auto m = a.getn(), k = a.getm(), n = b.getm();
  cl::Buffer in_a(context, a.m.begin(), a.m.end(), true),
      in_b(context, b.m.begin(), b.m.end(), true),
      out(context, CL_MEM_READ_WRITE, sizeof(valT) * m * n);
  k_mul_mat(cl::EnqueueArgs(queue, cl::NDRange(n, m)), m, n, k, in_a, in_b,
            out);
  queue.finish();
  matrix res;
  res.setn(m);
  res.setm(n);
  cl::copy(queue, out, res.m.begin(), res.m.end());
  return res;
}

matrix conv2d(const matrix &a, const matrix &b) {
  int n_a = a.getn();
  int m_a = a.getm();
  int n_b = b.getn();
  int m_b = b.getm();
  int n_o = n_a + n_b - 1;
  int m_o = m_a + m_b - 1;
  cl::Buffer in_a(context, a.m.begin(), a.m.end(), true),
      in_b(context, b.m.begin(), b.m.end(), true),
      out(context, CL_MEM_READ_WRITE, sizeof(valT) * n_o * m_o);
  k_conv2d(cl::EnqueueArgs(queue, {static_cast<cl::size_type>(n_o),
                                   static_cast<cl::size_type>(m_o)}),
           in_a, n_a, m_a, in_b, n_b, m_b, out);
  queue.finish();
  matrix res;
  res.setn(n_o);
  res.setm(m_o);
  cl::copy(queue, out, res.m.begin(), res.m.end());
  return std::move(res);
}
VvalT conv_l_forward(const convolution_layer &l, const vector<valT> &input) {
#define o_n cl::size_type(l.n_in - l.nK + 1 + l.pad * 2)
#define o_m cl::size_type(l.m_in - l.mK + 1 + l.pad * 2)
  vector<valT> k;
  k.reserve(l.Ichannels * l.Ochannels * l.nK * l.mK);
  for (auto line : l.K) {
    for (auto ker : line) {
      copy(ker.m.begin(), ker.m.end(), back_inserter(k));
    }
  }
  cl::Buffer Ks(context, k.begin(), k.end(), true),
      in(context, input.begin(), input.end(), true),
      out(context, CL_MEM_READ_WRITE, sizeof(valT) * l.Ochannels * o_n * o_m);
  k_conv_l_backforward(
      cl::EnqueueArgs(queue, {static_cast<cl::size_type>(o_n),
                              static_cast<cl::size_type>(o_m),
                              static_cast<cl::size_type>(l.Ochannels)}),
      l.Ichannels, l.Ochannels, l.pad, l.nK, l.mK, Ks, l.n_in, l.m_in, in, out,
      1);
  queue.finish();
  VvalT res;
  res.resize(l.Ochannels * o_n * o_m);
  cl::copy(queue, out, res.begin(), res.end());
  return std::move(res);
}

VvalT conv_l_backward(const convolution_layer &l, const vector<valT> &G) {
  vector<valT> k;
  k.reserve(l.Ichannels * l.Ochannels * l.nK * l.mK);
  for (auto line : l.K) {
    for (auto ker : line) {
      copy(ker.m.begin(), ker.m.end(), back_inserter(k));
    }
  }
  cl::Buffer Ks(context, k.begin(), k.end(), true),
      grad(context, G.begin(), G.end(), true),
      out(context, CL_MEM_READ_WRITE,
          sizeof(valT) * l.Ichannels * l.n_in * l.m_in);
  k_conv_l_backforward(
      cl::EnqueueArgs(queue, {static_cast<cl::size_type>(l.n_in),
                              static_cast<cl::size_type>(l.m_in),
                              static_cast<cl::size_type>(l.Ichannels)}),
      l.Ichannels, l.Ochannels, l.pad, l.nK, l.mK, Ks, l.n_in, l.m_in, grad,
      out, 0);
  queue.finish();
  VvalT res;
  res.resize(l.Ichannels * l.n_in * l.m_in);
  cl::copy(queue, out, res.begin(), res.end());
  return std::move(res);
}

VvalT conv_l_update(const convolution_layer &l, const vector<valT> &G,
                    const vector<valT> &input) {
  cl::Buffer I(context, input.begin(), input.end(), true),
      grad(context, G.begin(), G.end(), true),
      out(context, CL_MEM_READ_WRITE,
          sizeof(valT) * l.Ochannels * l.Ichannels * l.n_in * l.m_in);
  k_conv_l_update(
      cl::EnqueueArgs(queue, {static_cast<cl::size_type>(l.nK),
                              static_cast<cl::size_type>(l.mK),
                              static_cast<cl::size_type>(l.Ichannels)}),
      l.Ichannels, l.Ochannels, l.pad, l.nK, l.mK, I, l.n_in, l.m_in, grad,
      out);
  queue.finish();
  VvalT res;
  res.resize(l.Ochannels * l.Ichannels * l.n_in * l.m_in);
  cl::copy(queue, out, res.begin(), res.end());
  return std::move(res);
}
