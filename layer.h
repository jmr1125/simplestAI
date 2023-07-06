#pragma once
#include "main.h"
#include "matrix.h"
#include <vector>
using std::vector;
using VvalT = vector<valT>;
struct layer {
  VvalT getV(const VvalT &);
  vector<VvalT> getVdWi(const VvalT &);
  VvalT getVdb(const VvalT &); 
  matrix getSum(const VvalT &in);
  VvalT getV();
  VvalT getVdb();
  vector<VvalT> getVdV();//output[i]d input[j] = vec[i][j]
  matrix getSum();
  vector<VvalT> getVdWi();//output[i]d W[i][j] = vec[i][j]
  funcT Funcv()const;
  funcT deFuncv()const;
  funcT& Func();
  funcT& deFunc();
  void setInput(const VvalT &);
  bool computed;
  VvalT input;
  funcT Funct, deFunct;
  matrix w;
  matrix basis;
};
