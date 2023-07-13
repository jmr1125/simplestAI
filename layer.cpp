#include "layer.h"
#include "matrix.h"
#include <algorithm>
#include <cstdio>
void layer::setn(int n) {
  w.setn(n);
  b.setn(n);
  b.setm(1);
}
void layer::setm(int m) { w.setm(m); }
void layer::setInput(const VvalT &input) {
  if (input == this->input) {
    return;
  }
  this->input = std::move(input);
  // computed = false;
}
matrix layer::getSum() const {
  // static matrix ans;
  // if (!computed) {
  matrix res;
  res = w * input;
  res = res + b;
  return res;
  // ans = std::move(res);
  //   computed = true;
  // }
  // return ans;
}
VvalT layer::getV(const VvalT &input) {
  setInput(input);
  return getV();
} // v=f(w*v+basis)
VvalT layer::getV() const {
  matrix res = getSum();
  for (int i = 0; i < res.getn(); ++i) {
    res(i, 0) = (*(this->Funcv()))(res(i, 0));
  }

  return std::move(res.getvec());
}
/*
 *Vdb=F'(sum)
 *V_idV'_j=W_i,j · f'(W_i,j * V'_j + b_i)
 *V_id
 *VdV_i=W_.,i * f'(W_.,i * V'_i + b)
 *a*b=c
 *c_i=a_i*b_i
 *v_idW_i,j=v'_j * f'(sum_i)
 *vdW_.,j=V'_j * f'(sum)
 */
matrix layer::getVdb() const {
  matrix sum = getSum();
  for (auto &i : sum.m) {
    for (auto &j : i) {
      j = (*deFuncv())(j);
    }
  }
  return std::move(sum);
};
matrix layer::getVdVi(int i) const {
  matrix a = getVdb(); // F'(sum)
  // assert(a.getn() == w.getn());
  for (int x = 0; x < a.getn(); ++x) {
    a(x, 0) *= w(x, i);
  }
  return std::move(a);
}
matrix layer::getVdV() const {
  matrix res;
  res.setm(w.getm());
  res.setn(w.getn());
  for (int i = 0; i < res.getm(); ++i) {
    matrix tmp = getVdVi(i);
    assert(tmp.getm() == 1);
    assert(tmp.getn() == w.getn());
    for (int j = 0; j < res.getn(); ++j) {
      res(j, i) = tmp(j, 0);
    }
  }
  return std::move(res);
}
valT layer::getVdWij(int i, int j) const {
  // matrix sum = getSum();
  // for (int i = 0; i < sum.getn(); ++i) {
  //   sum(i, 0) = (*deFuncv())(sum(i, 0)) * input[j];
  // }
  // return std::move(sum);
  return (*deFuncv())(getSum()(i, 0)) * input[j];
}
funcT &layer::Func() {
  // computed = false;
  return Funct;
}
funcT layer::Funcv() const { return Funct; }
funcT &layer::deFunc() {
  // computed = false;
  return deFunct;
}
funcT layer::deFuncv() const { return deFunct; }
