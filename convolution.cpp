#include "convolution.hpp"

typedef std::complex<double> cd;
typedef vector<vector<cd>> matcd;

// main for fft and ifft
matcd ft_main(const matcd &in, int flag) {
  const auto n = in.size();
  if (n == 1)
    return in;
  matcd a00 = vector(n / 2, vector<cd>(n / 2));
  auto a10 = a00, a01 = a00, a11 = a00;
  auto output = vector(n, vector<cd>(n));
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
      a10[i][j] *=
          exp(flag * 2 * M_PI * std::complex<double>(0, 1) * (1.0 * i / n));
      a01[i][j] *=
          exp(flag * 2 * M_PI * std::complex<double>(0, 1) * (1.0 * j / n));
      a11[i][j] *= exp(flag * 2 * M_PI * std::complex<double>(0, 1) *
                       (1.0 * (i + j) / n));
    }
  }

  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      output[i][j] =
          a00[i % (n / 2)][j % (n / 2)] +
          ((i < n / 2) ? 1.0 : -1.0) * a10[i % (n / 2)][j % (n / 2)] +
          ((j < n / 2) ? 1.0 : -1.0) * a01[i % (n / 2)][j % (n / 2)] +
          ((((i < n / 2) && (j < n / 2)) || ((i >= n / 2) && (j >= n / 2)))
               ? 1.0
               : -1.0) *
              a11[i % (n / 2)][j % (n / 2)];
    }
  }
  return output;
}
#include <cmath>
matrix convolution(const matrix &a, const matrix &b) {
  const int n =
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
