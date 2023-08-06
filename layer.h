#pragma once
#include "main.h"
#include "matrix.h"
#include <vector>
using std::vector;
using VvalT = vector<valT>;
struct layer {
  void setn(int n);
  void setm(int m);
  funcT Funcv() const;
  funcT deFuncv() const;
  funcT &Func();
  funcT &deFunc();
  matrix output;
  matrix z; // z=wa+b
  VvalT delta;
  matrix activate(matrix input);
  funcT Funct, deFunct;
  matrix w;
  matrix b;
};
