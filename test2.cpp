#include "layer.h"
#include "main.h"
#include "matrix.h"
#include "network.h"
#include <iostream>
using namespace std;
int main() {
  //======TEST.MATRIX======
  matrix m1;
  m1.setn(2);
  m1.setm(2);
  m1(0, 0) = 0; // fibnacci
  m1(0, 1) = 1; //
  m1(1, 0) = 1; //
  m1(1, 1) = 1; //
  cout << "m1: " << m1 << endl;
  cout << "m1^2: " << m1 * m1 << endl;
  cout << "m1^3: " << m1 * m1 * m1 << endl;
  cout << "m1+2: " << m1 + 2 << endl;
  cout << "m1*3: " << m1 * 3 << endl;
  VvalT vec1{1, 2};
  cout << "vec1 = {1,2}" << endl;
  {
    matrix tmp;
    tmp = m1 * vec1;
    cout << "m1*vec1: " << tmp << endl;
  }
  //======TEST.LAYER======
  layer l1;
  l1.w = m1;
  l1.Func() = [](valT v) { return v * 2; };
  l1.deFunc() = [](valT v) -> valT { return 2; };
  l1.b = {1, 2};
  {
    matrix tmp;
    tmp = l1.getV(vec1);
    cout << "l1(vec1)= " << tmp << endl;
    l1.Func() = [](valT v) -> valT { return v * v; };
    l1.deFunc() = [](valT v) { return v * 2; };
    cout << "change func" << endl;
    tmp = l1.getV(vec1);
    cout << "l1(vec1)= " << tmp << endl;
  }
  //===NO.ARG===
  cout << "call no args" << endl;
  {
    matrix tmp;
    l1.setInput(vec1);
    l1.Func() = [](valT v) -> valT { return v * v; };
    l1.deFunc() = [](valT v) { return v * 2; };
    tmp = l1.getV();
    cout << "V = " << tmp << endl;
    cout << "Vdb = " << l1.getVdb() << endl;
    for (int i = 0; i < l1.w.getm(); ++i) {
      cout << "VdV" << i << " = " << l1.getVdVi(i) << endl;
    }
    cout << "VdV = " << l1.getVdV() << endl;
    for (int i = 0; i < l1.w.getn(); ++i) {
      for (int j = 0; j < l1.w.getm(); ++j) {
        // cout << "VdWij"<<i<<" = " << l1.getVdWij(i) << endl;
        cout << "VdW" << i << ',' << j << " = " << l1.getVdWij(i, j) << endl;
      }
    }
  }
  cout << "====NETWORK====" << endl;
  {
    network n1(
        {2, 3}, [](valT v) { return max((valT)0, v); },
        [](valT v) { return valT(v > 0 ? 1 : 0); });
    n1.setInput({1, 2});
    n1.layers[0].w.m[0] = vector<valT>({0, 1});
    n1.layers[0].w.m[1] = vector<valT>({1, 1});
    n1.layers[0].w.m[2] = vector<valT>({2, 3});
    n1.layers[0].b.m[0][0] = 0.5;
    n1.layers[0].b.m[1][0] = 0.6;
    n1.layers[0].b.m[2][0] = 0.8;
    n1.getV();
    cout << "V = {";
    for (valT x : n1.output) {
      cout << x << ' ';
    }
    cout << "}" << endl;
    cout << "VdVi = " << n1.getVdVi(0) << endl;
    cout << "VdVi = " << n1.layers[0].getVdV() << endl;
    for (int i = 0; i < n1.layers.size(); ++i) {
      cout << "Vdb" << i << " = " << n1.getVdbi(i) << endl;
    }
    for (int j = 0; j < n1.layers[0].w.getm(); ++j) {
      cout << "VdW" << j << " = " << n1.getVdWij(0, j) << endl;
    }
  }
}
