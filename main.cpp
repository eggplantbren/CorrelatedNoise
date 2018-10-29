#include <iostream>
#include <DNest4/code/RNG.h>
#include "NoiseModel.h"

using namespace CorrelatedNoise;

int main()
{
    // An RNG
    DNest4::RNG rng(time(0));

    // Generate some data
    Vector data(10000);
    for(int i=0; i<data.size(); ++i)
        data[i] = rng.randn();

    // Log likelihood of true model
    double logl = -0.5*data.size()*log(2*M_PI);
    for(int i=0; i<data.size(); ++i)
        logl += -0.5*data[i]*data[i];
    std::cout << "1 0 " << logl << std::endl;
    
    // Create 1000 noise models
    NoiseModel m(100, 100);
    for(int i=0; i<1000; ++i)
    {
        m.from_prior(rng);
        std::cout << m << ' ' << m.log_likelihood(data) << std::endl;
    }

//    Vector img = m.generate_image(rng);
//    for(int i=0; i<img.size(); ++i)
//        std::cout << img(i) << std::endl;

    return 0;
}

