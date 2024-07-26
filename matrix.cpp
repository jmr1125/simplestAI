#include "matrix.hpp"
#include <cstddef>
#include <ostream>
#include <stdexcept>
#include <string>
using std::endl;
// using std::move;
using std::to_string;
dimension_error::dimension_error(const string &s) : std::runtime_error(s) {}
const char *dimension_error::what() const throw() {
  return std::runtime_error::what();
}

void makemat(vector<valT> &v, size_t n, size_t m) {
  if (n == -1 || m == -1) {
    return;
  }
  v.resize(n * m);
}
void matrix::setn(size_t n) {
  N = n;
  makemat(this->m, N, M);
};
void matrix::setm(size_t m) {
  M = m;
  makemat(this->m, N, M);
}
valT matrix::operator()(size_t x, size_t y) const { return m.at(x * M + y); }
valT &matrix::operator()(size_t x, size_t y) { return m.at(x * M + y); }
size_t matrix::getn() const { return N; }
size_t matrix::getm() const { return M; }
void matrix::swap(matrix &tmp) { m.swap(tmp.m); }
#ifdef USE_OCL
#warning ocl
#include "cl-mat.hpp"
matrix matrix::operator*(const matrix &m1) const {
  if (N * M * m1.getm() < 1000) {
    matrix res;
    res.setn(getn());
    res.setm(m1.getm());
    for (int i = 0; i < getn(); ++i) {
      for (int j = 0; j < m1.getm(); ++j) {
        valT tmp = 0;
        for (int k = 0; k < getm(); ++k) {
          tmp += (*this)(i, k) * m1(k, j);
        }
        res(i, j) = tmp;
      }
    }
    return std::move(res);
  }
  return mul_mat(*this, m1);
}
#else
matrix matrix::operator*(const matrix &m1) const {
  if (getm() != m1.getn()) {
    throw dimension_error(string("m1 n=" + to_string(m1.getn()) +
                                 " this-> m=" + to_string(getm())));
  }
  matrix res;
  res.setn(getn());
  res.setm(m1.getm());
  for (int i = 0; i < getn(); ++i) {
    for (int j = 0; j < m1.getm(); ++j) {
      valT tmp = 0;
      for (int k = 0; k < getm(); ++k) {
        tmp += (*this)(i, k) * m1(k, j);
      }
      res(i, j) = tmp;
    }
  }
  return std::move(res);
};
#endif
vector<valT> matrix::operator*(const vector<valT> &vec) const {
  if (vec.size() != getm()) {
    throw dimension_error((string) "m= " + to_string(getn()) +
                          " ,vec.m= " + to_string(vec.size()));
  }
  vector<valT> res;
  res.resize(getn());
  for (int i = 0; i < getn(); ++i) {
    valT tmp = 0;
    for (int j = 0; j < getm(); ++j) {
      tmp += (*this)(i, j) * vec[j];
    }
    res[i] = tmp;
  }
  return res;
}
// matrix matrix::operator*(const valT v) const {
//   auto m = this->m;
//   for (auto &i : m) {
//       i *= v;
//   }
//   matrix M;
//   M.setm(1);
//   M.setn(this->getn());
//   M.m = m;
//   return M;
// }
matrix matrix::operator+(const matrix &m1) const {
  if (m1.getn() != getn() || m1.getm() != getm()) {
    throw dimension_error(string("m1 n,m=") + to_string(m1.getn()) + "," +
                          to_string(m1.getm()) +
                          "this-> n,m=" + to_string(this->getn()) + "," +
                          to_string(this->getm()));
  }
  matrix res;
  res.setn(getn());
  res.setm(m1.getm());
  for (int i = 0; i < getn(); ++i) {
    for (int j = 0; j < getm(); ++j) {
      res(i, j) = (*this)(i, j) + m1(i, j);
    }
  }
  return std::move(res);
};

matrix matrix::operator+(const valT v) const {
  matrix res(*this);
  for (int i = 0; i < getn(); ++i) {
    for (int j = 0; j < getm(); ++j) {
      res(i, j) += v;
    }
  }
  return res;
}
matrix matrix::operator+(const vector<valT> &vec) const {
  if (getm() != 1) {
    throw dimension_error(string("m= ") + to_string(getm()));
  }
  if (vec.size() != getn()) {
    throw dimension_error(string("this->m= ") + to_string(getm()) +
                          ", vec.m= " + to_string(vec.size()));
  }
  matrix res;
  res.setm(1);
  res.setn(getn());
  for (int i = 0; i < getm(); ++i) {
    res(i, 0) = (*this)(i, 0) + vec[i];
  }
  return res;
}
const matrix &matrix::operator=(const matrix &m1) {
  m = m1.m;
  N = m1.N;
  M = m1.M;
  return std::move(m1);
}
const matrix &matrix::operator=(const vector<valT> &vec) {
  setn(vec.size());
  setm(1);
  for (int i = 0; i < vec.size(); ++i) {
    (*this)(i, 0) = vec[i];
  }
  return *this;
}
vector<valT> matrix::getvec() const {
  if (getm() != 1) {
    throw dimension_error((string) "m= " + to_string(getm()));
  }
  vector<valT> res;
  res.resize(getn());
  for (int i = 0; i < getn(); ++i) {
    res[i] = (*this)(i, 0);
  }
  return res;
}
ostream &operator<<(ostream &ost, const matrix &m) {
  ost << "[" << endl;
  for (int i = 0; i < m.getn(); ++i) {
    ost << "[";
    for (int j = 0; j < m.getm(); ++j) {
      ost << m(i, j) << ' ';
    }
    ost << "]" << endl;
  }
  ost << "]";
  return ost;
}
matrix i(size_t dimensions) {
  matrix res;
  res.setm(dimensions);
  res.setn(dimensions);
  for (int i = 0; i < dimensions; ++i) {
    for (int j = 0; j < dimensions; ++j) {
      res(i, j) = 0;
    }
    res(i, i) = 1;
  }
  return std::move(res);
}
matrix matrix::T() const {
  matrix res;
  res.setm(getn());
  res.setn(getm());
  for (int i = 0; i < this->N; ++i) {
    for (int j = 0; j < this->M; ++j) {
      res(j, i) = this->operator()(i, j);
    }
  }
  return std::move(res);
}
