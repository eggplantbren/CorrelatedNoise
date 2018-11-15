#include <armadillo>
#include <iomanip>
#include <iostream>
#include <DNest4/code/RNG.h>
#include "NoiseModel.h"

using namespace CorrelatedNoise;

int main()
{
    // An RNG
    DNest4::RNG rng(time(0));

//    // Load the data
//    arma::mat data(200, 300);
//    std::fstream fin("data.txt", std::ios::in);
//    for(int i=0; i<200; ++i)
//        for(int j=0; j<300; ++j)
//            fin >> data(i, j);
//    fin.close();
//    arma::cx_mat data_fourier = arma::fft2(data)/sqrt(200*300);

//    arma::mat sigma_map(200, 300);
//    arma::mat model(200, 300);
//    sigma_map.zeros();
//    model.zeros();

//    // Do some MCMC
//    double logl;
//    NoiseModel m(200, 300);
//    m.from_prior(rng);
//    logl = m.log_likelihood(data, model, sigma_map);

//    for(int i=0; i<100000; ++i)
//    {
//        NoiseModel m2 = m;
//        double logH = m2.perturb(rng);
//        double logl2 = m2.log_likelihood(data, model, sigma_map);
//        double logA = logH + logl2 - logl;

//        if(rng.rand() <= exp(logA))
//        {
//            m = m2;
//            logl = logl2;
//        }

//        if((i+1)%10 == 0)
//            std::cout << std::setprecision(12) << (i+1) << ' ' << m << ' ' << logl << std::endl;
//    }

    return 0;
}

