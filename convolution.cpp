#include "convolution.hpp"
#include "main.hpp"
#include <algorithm>

typedef std::complex<valT> cd;
typedef vector<vector<cd>> matcd;

// main for fft and ifft
matcd ft_main(const matcd &in, int flag) {
  const auto n = in.size();
  if (n == 1)
    return in;
  matcd a00(n / 2, vector<cd>(n / 2));
  matcd a10(n / 2, vector<cd>(n / 2));
  matcd a01(n / 2, vector<cd>(n / 2));
  matcd a11(n / 2, vector<cd>(n / 2));
  matcd output(n, vector<cd>(n));
  for (int i = 0; i < n / 2; i++) {
    for (int j = 0; j < n / 2; j++) {
      a00[i][j] = in[i * 2 + 0][j * 2 + 0];
      a01[i][j] = in[i * 2 + 0][j * 2 + 1];
      a10[i][j] = in[i * 2 + 1][j * 2 + 0];
      a11[i][j] = in[i * 2 + 1][j * 2 + 1];
    }
  }
  a00 = ft_main(a00, flag);
  a01 = ft_main(a01, flag);
  a10 = ft_main(a10, flag);
  a11 = ft_main(a11, flag);

  for (int i = 0; i < n / 2; ++i) {
    for (int j = 0; j < n / 2; ++j) {
      // a00[i][j]*=1;
      a10[i][j] *= exp(flag * 2 * (valT)M_PI * cd(0, 1) * ((valT)1.0 * i / n));
      a01[i][j] *= exp(flag * 2 * (valT)M_PI * cd(0, 1) * ((valT)1.0 * j / n));
      a11[i][j] *=
          exp(flag * 2 * (valT)M_PI * cd(0, 1) * ((valT)1.0 * (i + j) / n));
    }
  }

  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      output[i][j] =
          a00[i % (n / 2)][j % (n / 2)] +
          (valT)((i < n / 2) ? 1.0 : -1.0) * a10[i % (n / 2)][j % (n / 2)] +
          (valT)((j < n / 2) ? 1.0 : -1.0) * a01[i % (n / 2)][j % (n / 2)] +
          (valT)((((i < n / 2) && (j < n / 2)) ||
                  ((i >= n / 2) && (j >= n / 2)))
                     ? 1.0
                     : -1.0) *
              a11[i % (n / 2)][j % (n / 2)];
    }
  }
  return output;
}
#include <cmath>
matrix convolution(const matrix &a, const matrix &b) {
  const size_t n =
      std::max(std::max(a.getn(), a.getm()), std::max(b.getn(), b.getm()));
  const int N = (1 << ((int)ceil(log2(n)) + 1));
  matcd A(vector(N, vector<cd>(N)));
  matcd B(vector(N, vector<cd>(N)));
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < N; ++j) {
      A[i][j] = (i < a.getn() && j < a.getm()) ? a(i, j) : 0.0;
      B[i][j] = (i < b.getn() && j < b.getm()) ? b(i, j) : 0.0;
    }
  }
  A = ft_main(A, 1); // before -1 -1 1
  B = ft_main(B, 1);
  matcd C = vector(N, vector<cd>(N));
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < N; ++j) {
      C[i][j] = A[i][j] * B[i][j];
    }
  }
  C = ft_main(C, -1);
  matrix ans;
  ans.setn(a.getn() + b.getn() - 1);
  ans.setm(a.getm() + b.getm() - 1);
  // ans.setn(a.getn());
  // ans.setm(a.getm());
  for (int i = 0; i < ans.getn(); ++i) {
    for (int j = 0; j < ans.getm(); ++j) {
      ans(i, j) = C[i][j].real() / (N * N);
    }
  }
  return ans;
}

matrix rotate(matrix x) {
  // reverse(x.m.begin(), x.m.end());
  // return std::move(x);
  size_t n = x.getn(); // 获取行数
  size_t m = x.getm(); // 获取列数

  // 创建一个临时矩阵用于存储旋转结果
  matrix rotated;
  rotated.setn(n);
  rotated.setm(m);
  rotated.m.resize(n * m);

  // 进行180度旋转
  for (size_t i = 0; i < n; ++i) {
    for (size_t j = 0; j < m; ++j) {
      rotated(i, j) = x(n - 1 - i, m - 1 - j);
    }
  }

  return rotated;
}
