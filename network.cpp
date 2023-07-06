#include "network.h"

network::network(const vector<int> &sizes, funcT Func, funcT deFunc) {
  layers.resize(sizes.size() - 1);
    for(int i=0;i<layers.size();++i){
        layers[i].w.setn(sizes[i+1]);
        layers[i].w.setm(sizes[i]);
    }
  for (auto &x : layers) {
    x.Func() = Func;
    x.deFunc() = deFunc;
    x.basis.setn(x.w.getn());
      x.basis.setm(1);
    x.computed = false;
  }
  computed = false;
}
vector<VvalT> network::getVdWi() {}
vector<VvalT> network::getVdbi() {
  const vector<matrix> &v=getvVdV1();
  vector<VvalT> res;
  res.reserve(layers.size());
  for (int i=0;i<layers.size();++i) {
    res.push_back(v[i]*layers[i].getVdb());
  }
  return res;
}
vector<matrix> network::getvVdV1() {
  static vector<matrix> vVdV1;
  if (!computed) {
    vVdV1.resize(layers.size());
    vVdV1[layers.size() - 1] = i(layers[layers.size() - 1].w.getn());
    for (size_t i = layers.size() - 2; i >= 0; --i) {
      vVdV1[i] = vVdV1[i + 1] * vVdV[i];
    }
  }
  return vVdV1;
}
void network::getV() {
    for(int i=1;i<layers.size();++i){
        layers[i].setInput(layers[i-1].getV());
    }
  output = layers[layers.size() - 1].getV();
  vVdV.resize(layers.size());
  for (int i = 0; i < layers.size(); ++i) {
    matrix &m = vVdV[i];
    m.m = layers[i].getVdV();
  }
}
VvalT network::getV(const VvalT &input) {
  setInput(input);
    getV();
  return output;
}
void network::setInput(const VvalT &in) { layers[0].setInput(in); }
