#include "main.h"
#include "matrix.h"
#include <vector>
using std::vector;
using VvalT = vector<valT>;
struct layer {
  VvalT getV(const VvalT &);
  vector<VvalT> getVdWi(const VvalT &);
  VvalT getVdb(const VvalT &);
  funcT Func, deFunc;
  matrix w;
  matrix basis;
};
