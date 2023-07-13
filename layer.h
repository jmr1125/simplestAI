#pragma once
#include "main.h"
#include "matrix.h"
#include <vector>
using std::vector;
using VvalT = vector<valT>;
struct layer {
  void setn(int n);
  void setm(int m);
  VvalT getV(const VvalT &);
  matrix getSum(const VvalT &in)const;
  VvalT getV()const;
  matrix getVdVi(int i)const;
  matrix getVdV()const;
  valT getVdWij(int i,int j)const;
  matrix getVdb()const;
  matrix getSum()const;
  funcT Funcv()const;
  funcT deFuncv()const;
  funcT& Func();
  funcT& deFunc();
  void setInput(const VvalT &);
  // bool computed;
  VvalT input{};
  funcT Funct, deFunct;
  matrix w;
  matrix b;
};
