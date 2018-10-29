#include <iostream>
#include <DNest4/code/RNG.h>
#include "NoiseModel.h"

using namespace CorrelatedNoise;

int main()
{
    // An RNG
    DNest4::RNG rng(time(0));

    // Create a noise model
    NoiseModel m(2, 3);
    m.from_prior(rng);
    m.print(std::cout);

    return 0;
}

