#include "main.h"
#include <algorithm>
#include <ostream>
#include <stdexcept>
#include <string>
#include <vector>
using std::ostream;
using std::string;
using std::vector;
struct dimension_error : std::runtime_error {
  explicit dimension_error(const string &);
  virtual const char *what() const throw() override;
};
struct matrix {
  matrix() = default;
  matrix(const matrix &) = default;
  matrix(matrix &&) = default;
  // matrix(const matrix &&) = default;
  void setn(int n);
  void setm(int m);
  int getn() const;
  int getm() const;
  void swap(matrix&);
  matrix operator*(const matrix &) const;
  vector<valT> operator*(const vector<valT> &) const;
  matrix operator*(const valT) const;
  matrix operator+(const matrix &) const;
  matrix operator+(const valT) const;
  matrix operator+(const vector<valT> &) const;
  matrix &operator=(matrix &) = default;
  const matrix &operator=(const matrix &);
  const matrix &operator=(const vector<valT> &);
  const matrix &&operator=(matrix &&);
  vector<valT> getvec() const;
  valT operator()(size_t x, size_t y) const;
  valT &operator()(size_t x, size_t y);
  vector<vector<valT>> m;
};
ostream &operator<<(ostream &ost, const matrix &);
