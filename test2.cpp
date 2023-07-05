#include "layer.h"
#include "main.h"
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
  l1.Func = [](valT v) { return v * 2; };
  l1.deFunc = [](valT v) -> valT { return 2; };
  l1.basis = {1, 2};
  {
    matrix tmp;
    tmp = l1.getV(vec1);
    cout << "l1(vec1): " << tmp << endl;
    l1.Func = [](valT v) -> valT { return v * v; };
    l1.deFunc = [](valT v) { return v * 2; };
    tmp = l1.getV(vec1);
    cout << "l1(vec1): " << tmp << endl;
    tmp = l1.getVdb(vec1);
    cout << "l1(vec1) Vdb" << tmp << endl;
  }
  {
    auto ttmp = l1.getVdWi(vec1);
    cout << "l1(vec1) Vdwi [" << endl;
    for(const auto & i : ttmp){
      cout<<"[ ";
      for(const auto & j : i){
	cout<<j<<' ';
      }
      cout<<"]"<<endl;
    }
    cout<<"]"<<endl;
  }
}
