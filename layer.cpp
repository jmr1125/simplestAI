#include "layer.h"
#include <cstdio>
matrix getSum(const matrix &w, const VvalT &in,const matrix b) {
  matrix res;
  res=w*in;
  res=res+b;
  return std::move(res);
}
VvalT layer::getV(const VvalT &input) {
  matrix res;
  res = w * input;
  res = res + basis;
  for (int i = 0; i < res.getn(); ++i) {
    res(i, 0) = (*(this->Func))(res(i, 0));
  }

  return res.getvec();
  ;
} // v=f(w*v+basis)

VvalT layer::getVdb(const VvalT &input) { // = f'(w*v+basis)
  matrix res;
  res = w * input;
  res = res + basis;
  for (int i = 0; i < res.getn(); ++i) {
    res(i, 0) = (*(this->deFunc))(res(i, 0));
  }

  return res.getvec();
  ;
}

vector<VvalT>
layer::getVdWi(const VvalT &input) { // = v_i*f'(w*v_i+sigma_other+basis)
  vector<VvalT> res;
  res.resize(w.getn());
  matrix sum;
  sum = w * input;
  sum = sum + basis;
  for (int i = 0; i < w.getn(); ++i) {
    res[i].resize(w.getm());
    for (int j = 0; j < w.getm(); ++j) {
      res[i][j] = input[j] * deFunc(sum(i, 0));
    }
  }
  return res;
}
