#include <iostream>
#include <DNest4/code/RNG.h>
#include "NoiseModel.h"

using namespace CorrelatedNoise;

int main()
{
    // An RNG
    DNest4::RNG rng(time(0));

    NoiseModel m(100, 100);
    m.log_likelihood(arma::cx_mat(100, 100));

    return 0;
}

