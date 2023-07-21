#include "layer.h"
#include "main.h"
#include "matrix.h"
#include "network.h"
#include <random>
#include <utility>
using std::pair;
using delta_network =
    pair<vector<matrix>, vector<matrix>>; // first = W ; second = b
void train(network &net, const VvalT &input, const VvalT &expect, valT scale);
void trainn(network &net, const vector<VvalT> &input,
            const vector<VvalT> &expect, valT scale);
delta_network getdelta_network(network net, const VvalT &input,
                               const VvalT &expect, valT scale);

valT genvalT();
// #define F_x [](valT x) { return x; }, [](valT x) { return 1; }
// // #define F_01 [](valT x) { return x<0?0:1; }, [](valT x) { return
// x!=0?0:inf;
// // }
// #define F_tanh [](valT x) { return x; }, [](valT x) { return 1; }
// #define F_sigma [](valT x) { return 1/(1+); }, [](valT x) { return 1; }

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
