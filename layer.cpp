#include "layer.h"
#include <algorithm>
#include <cstdio>
// matrix layer::getSum(const matrix &w, const VvalT &in,const matrix b) {
//   matrix res;
//   res=w*in;
//   res=res+b;
//   return std::move(res);
// }
// VvalT layer::getV(const VvalT &input) {
//   matrix res=getSum(w,input,basis);
//   for (int i = 0; i < res.getn(); ++i) {
//     res(i, 0) = (*(this->Func))(res(i, 0));
//   }

//   return res.getvec();
//   ;
// } // v=f(w*v+basis)

// VvalT layer::getVdb(const VvalT &input) { // = f'(w*v+basis)
//   matrix res=getSum(w,input,basis);
//   for (int i = 0; i < res.getn(); ++i) {
//     res(i, 0) = (*(this->deFunc))(res(i, 0));
//   }

//   return res.getvec();
//   ;
// }

// vector<VvalT>
// layer::getVdWi(const VvalT &input) { // = v_i*f'(w*v_i+sigma_other+basis)
//   vector<VvalT> res;
//   res.resize(w.getn());
//   matrix sum=getSum(w,input,basis);
//   for (int i = 0; i < w.getn(); ++i) {
//     res[i].resize(w.getm());
//     for (int j = 0; j < w.getm(); ++j) {
//       res[i][j] = input[j] * deFunc(sum(i, 0));
//     }
//   }
//   return res;
// }
void layer::setInput(const VvalT &input) {
  if (input == this->input) {
    // computed = true;
    return;
  }
  this->input = std::move(input);
  computed = false;
}
// #include <iostream>
matrix layer::getSum() {
  static matrix ans;
  if (!computed) {
    matrix res;
    res = w * input;
    res = res + basis;
    ans = std::move(res);
    // std::clog << __func__ << std::endl;
    computed = true;
  }
  return ans;
}
VvalT layer::getV(const VvalT &input) {
  setInput(input);
  return getV();
} // v=f(w*v+basis)
VvalT layer::getV() {
  matrix res = getSum();
  for (int i = 0; i < res.getn(); ++i) {
    res(i, 0) = (*(this->Funcv()))(res(i, 0));
  }

  return std::move(res.getvec());
}
VvalT layer::getVdb(const VvalT &input) { // = f'(w*v+basis)
  setInput(input);
  return std::move(getVdb());
}
VvalT layer::getVdb() {
  matrix res = getSum();
  for (int i = 0; i < res.getn(); ++i) {
    res(i, 0) = (*(this->deFuncv()))(res(i, 0));
  }

  return std::move(res.getvec());
}
// vector<VvalT>layer::getVdV(){
//   vector<VvalT> res;
//   res.resize(w.getn());
//   matrix sum = getSum();
//   for (int i = 0; i < w.getn(); ++i) {
//     res[i].resize(w.getm());
//     for (int j = 0; j < w.getm(); ++j) {
//       res[i][j] = w(i,j) * (*deFuncv())(sum(i, 0));
//     }
//   }
//   return std::move(res);
// }
// vector<VvalT>
// layer::getVdWi(const VvalT &input) { // = v_i*f'(w*v_i+sigma_other+basis)
//   setInput(input);
//   return std::move(getVdWi());
// }

// vector<VvalT> layer::getVdWi() {
//   vector<VvalT> res;
//   res.resize(w.getn());
//   matrix sum = getSum();
//   for (int i = 0; i < w.getn(); ++i) {
//     res[i].resize(w.getm());
//     for (int j = 0; j < w.getm(); ++j) {
//       res[i][j] = input[j] * (*deFuncv())(sum(i, 0));
//     }
//   }
//   return std::move(res);
// }
#define getVdWiORd(v)                                                          \
  vector<VvalT> res;                                                           \
  res.resize(w.getn());                                                        \
  matrix sum = getSum();                                                       \
  for (int i = 0; i < w.getn(); ++i) {                                         \
    res[i].resize(w.getm());                                                   \
    for (int j = 0; j < w.getm(); ++j) {                                       \
      res[i][j] = (v) * (*deFuncv())(sum(i, 0));                               \
    }                                                                          \
  }                                                                            \
  return std::move(res);
vector<VvalT> layer::getVdWi(const VvalT &in) {
  setInput(in);
  return std::move(getVdWi());
}
vector<VvalT> layer::getVdWi() { getVdWiORd(input[j]); }
vector<VvalT> layer::getVdV() { getVdWiORd(w(i, j)); }
funcT &layer::Func() {
  computed = false;
  return Funct;
}
funcT layer::Funcv() const { return Funct; }
funcT &layer::deFunc() {
  computed = false;
  return deFunct;
}
funcT layer::deFuncv() const { return deFunct; }
