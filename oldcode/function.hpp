#include "layer.hpp"
#include "main.hpp"
#include "matrix.hpp"
#include "network.hpp"
#include <random>
#include <utility>

valT genvalT();
valT func_x(valT x);
valT defunc_x(valT);
valT func_sigma(valT x);
valT defunc_sigma(valT x);
valT func_tanh(valT x);
valT defunc_tanh(valT x);
valT func_arctan(valT x);
valT defunc_arctan(valT x);
valT func_Softsign(valT x);
valT defunc_Softsign(valT x);
valT func_ISRU(valT x, valT alpha);
valT defunc_ISRU(valT x, valT alpha);
valT func_ReLU(valT x);
valT defunc_ReLU(valT x);
valT func_Leaky_ReLU(valT x);
valT defunc_Leaky_ReLU(valT x);
valT func_PReLU(valT x, valT alpha);
valT func_ELU(valT x, valT alpha);
valT defunc_ELU(valT x, valT alpha);
valT func_SELU(valT x);
valT defunc_SELU(valT x);
#define funcof(x) func_##x
#define defuncof(x) defunc_##x
#define deffuncpair(x) funcof(x), defuncof(x)
#define pair_x deffuncpair(x)
#define pair_sigma deffuncpair(sigma)
#define pair_tanh deffuncpair(tanh)
#define pair_arctan deffuncpair(arctan)
#define pair_Softsign deffuncpair(Softsign)
#define pair_ISRU deffuncpair(ISRU)
#define pair_ReLU deffuncpair(ReLU)
#define pair_Leaky_ReLU deffuncpair(Leaky_ReLU)
#define pair_PRELU deffuncpair(PRELU)
#define pair_ELU deffuncpair(ELU)
#define pair_SELU deffuncpair(SELU)
