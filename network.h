#include "layer.h"
#include "main.h"
#include <vector>
using std::vector;
struct network {
  network() = delete;
  network(const vector<int> &sizes, funcT Func, funcT deFunc);
  VvalT getV(const VvalT &input);
  vector<VvalT> getVdWi();
  vector<VvalT> getVdbi();
  vector<matrix> getvVdV1();
  void getV();
  void setInput(const VvalT &in);
  VvalT output;
  bool computed;
  vector<layer> layers;
  vector<matrix> vVdV;
};
