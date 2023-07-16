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
delta_network getdelta_network(network &net, const VvalT &input,
                               const VvalT &expect, valT scale);

valT genvalT();
