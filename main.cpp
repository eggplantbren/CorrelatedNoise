#include <iostream>
#include <DNest4/code/RNG.h>
#include "NoiseModel.h"

using namespace CorrelatedNoise;

int main()
{
    DNest4::RNG rng(0);

    NoiseModel m(100, 110);
    m.from_prior(rng);

    return 0;
}

