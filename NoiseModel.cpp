#include "NoiseModel.h"
#include <DNest4/code/Distributions/Cauchy.h>
#include <DNest4/code/Utils.h>
#include <iostream>

namespace CorrelatedNoise
{

NoiseModel::NoiseModel(int _n1, int _n2, double _L)
:n1(_n1)
,n2(_n2)
,n(n1*n2)
,L(_L)
,C(1.0)
,fft_of_psf(n1, n2)
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
    arma::mat the_model(n1, n2);
    double rsq;
    int m, n;
    for(int j=0; j<n1; ++j)
    {
        n = DNest4::mod(j - n2/2, n2);
        for(int i=0; i<n2; ++i)
        {
            m = DNest4::mod(i - n1/2, n1);
            rsq = (i - n1/2)*(i - n1/2) + (j - n2/2)*(j - n2/2);
            the_model(m, n) = C*exp(-0.5*rsq*inv_L_squared);
        }
    }
    the_model(n1/2, n2/2) += 1E-3*C;

    fft_of_psf = arma::fft2(the_model)/sqrt(n1*n2);
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

    for(int j=0; j<n2; ++j)
    {
        for(int i=0; i<n1; ++i)
        {
            sd = std::abs(real(fft_of_psf(i, j)))*inv_root_two;
            ratio = real(data_fft(i, j))/sd;
            logL += C - log(sd) - 0.5*ratio*ratio;
        }
    }

//    for(int i=0; i<n1; ++i)
//    {
//        for(int j=0; j<n2; ++j)
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

