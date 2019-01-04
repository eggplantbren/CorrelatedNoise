#include <fstream>
#include <iomanip>
#include <iostream>
#include <DNest4/code/RNG.h>
#include "NoiseModel2.h"

using namespace CorrelatedNoise;

int main()
{
    // An RNG
    DNest4::RNG rng(time(0));

    // Load the data
    Eigen::MatrixXd data(100, 101);
    Eigen::MatrixXd sigma_map(100, 101);
    Eigen::MatrixXd model(100, 101);
    std::fstream fin("data.txt", std::ios::in);
    for(int i=0; i<100; ++i)
    {
        for(int j=0; j<101; ++j)
        {
            fin >> data(i, j);
            sigma_map(i, j) = 0.0;
            model(i, j) = 0.0;
        }
    }
    fin.close();

    // Do some MCMC
    double logl;
    NoiseModel2 m(100, 101);
    m.from_prior(rng);
    logl = m.log_likelihood(data, model, sigma_map);

    for(int i=0; i<100000; ++i)
    {
        NoiseModel2 m2 = m;
        double logH = m2.perturb(rng);
        double logl2 = m2.log_likelihood(data, model, sigma_map);
        double logA = logH + logl2 - logl;

        if(rng.rand() <= exp(logA))
        {
            m = m2;
            logl = logl2;
        }

        if((i+1)%1 == 0)
            std::cout << std::setprecision(12) << (i+1) << ' ' << m << ' ' << logl << std::endl;
    }

    return 0;
}

