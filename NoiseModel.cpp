#include "NoiseModel.h"
#include <DNest4/code/Distributions/Cauchy.h>
#include <DNest4/code/Utils.h>
#include <iostream>

namespace CorrelatedNoise
{

NoiseModel::NoiseModel(int _ni, int _nj)
:ni(_ni)
,nj(_nj)
,n(ni*nj)
,fft_of_psf(ni, nj)
{

}

void NoiseModel::from_prior(DNest4::RNG& rng)
{
    // Trivial flat priors
    L = 100.0*rng.rand();
    C = 100.0*rng.rand();

    compute_psf();
}

void NoiseModel::compute_psf()
{
    double inv_L_squared = 1.0/(L*L);

    // The noise model kernel
    arma::mat the_model(ni, nj);
    double rsq;
    for(int j=0; j<nj; ++j)
    {
        for(int i=0; i<ni; ++i)
        {
            rsq = (i - ni/2)*(i - ni/2) + (j - nj/2)*(j - nj/2);
            the_model(i, j) = C*exp(-0.5*rsq*inv_L_squared);
        }
    }
    the_model(ni/2, nj/2) += 1E-3*C;

    // FFtshift
    arma::mat fft_shifted(ni, nj);
    int m, n;
    for(int j=0; j<nj; ++j)
    {
        n = DNest4::mod(j - nj/2, nj);
        for(int i=0; i<ni; ++i)
        {
            m = DNest4::mod(i - ni/2, ni);
            fft_shifted(m, n) = the_model(i, j);
        }
    }

    fft_of_psf = arma::fft2(fft_shifted)/sqrt(ni*nj);
}

double NoiseModel::perturb(DNest4::RNG& rng)
{
    double logH = 0.0;

    int which = rng.rand_int(2);

    if(which == 0)
    {
        L += 100.0*rng.randh();
        DNest4::wrap(L, 0.0, 100.0);
    }
    else
    {
        C += 100.0*rng.randh();
        DNest4::wrap(C, 0.0, 100.0);
    }
    compute_psf();

    return logH;
}

double NoiseModel::log_likelihood(const arma::cx_mat& data_fft) const
{
    double logL = 0.0;

    double sd, ratio;
    double inv_root_two = 1.0/sqrt(2.0);
    double C = -0.5*log(2.0*M_PI);

    for(int j=0; j<nj; ++j)
    {
        for(int i=0; i<ni; ++i)
        {
            sd = std::abs(real(fft_of_psf(i, j)))*inv_root_two;
            ratio = real(data_fft(i, j))/sd;
            logL += C - log(sd) - 0.5*ratio*ratio;
        }
    }

//    for(int i=0; i<ni; ++i)
//    {
//        for(int j=0; j<nj; ++j)
//            std::cout << the_model(i, j) << ' ';
//        std::cout << std::endl;
//    }
//    exit(0); 

    return logL;
}

void NoiseModel::print(std::ostream& out) const
{
    out << L << ' ' << C;
}

std::string NoiseModel::description()
{
    return "L, C";
}

std::ostream& operator << (std::ostream& out, const NoiseModel& m)
{
    m.print(out);
    return out;
}

} // namespace CorrelatedNoise

