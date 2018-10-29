#include <iostream>
#include <DNest4/code/RNG.h>
#include "NoiseModel.h"

using namespace CorrelatedNoise;

int main()
{
    // An RNG
    DNest4::RNG rng(time(0));

    // Create a noise model
    NoiseModel m(100, 100);
    m.from_prior(rng);
    Vector img = m.generate_image(rng);
    for(int i=0; i<img.size(); ++i)
        std::cout << img(i) << std::endl;

    return 0;
}

