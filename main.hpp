#pragma once
#include <vector>
// #define valT double
// #define valT_double
#define valT float
// using valT = double;
//  using valT = long double;
//   using valT = float;
using funcT = valT (*)(valT);
using VvalT = std::vector<valT>;
#define rand01(rd) ((rd() + rd.min()) * 1.0 / (rd.max() - rd.min()))
