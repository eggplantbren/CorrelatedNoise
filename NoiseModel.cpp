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
    DNest4::Cauchy cauchy(0.0, 5.0);

    // Fairly generic priors
    do
    {
        sigma = cauchy.generate(rng);
    }while(std::abs(sigma) >= 100.0);
    sigma = exp(sigma);

    // Log-uniform(0.1, sqrt(n))
    L = exp(log(0.1) + log(10.0*sqrt(n))*rng.rand());

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
            the_model(i, j) = exp(-0.5*rsq*inv_L_squared);
        }
    }
    the_model(ni/2, nj/2) += 1E-3;

    // Normalise the PSF
    double tot_sq = 0.0;
    for(int j=0; j<nj; ++j)
        for(int i=0; i<ni; ++i)
            tot_sq += pow(the_model(i, j), 2);
    the_model = sigma*the_model/sqrt(tot_sq)*sqrt(ni*nj);

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

    DNest4::Cauchy cauchy(0.0, 5.0);
    int which = rng.rand_int(2);

    if(which == 0)
    {
        sigma = log(sigma);
        logH += cauchy.perturb(sigma, rng);
        if(std::abs(sigma) >= 100.0)
        {
            sigma = 1.0;
            return -1E300;
        }
        sigma = exp(sigma);
    }
    else
    {
        L = log(L);
        L += log(10.0*sqrt(n))*rng.randh();
        DNest4::wrap(L, log(0.1), log(sqrt(n)));
        L = exp(L);
    }

    compute_psf();

    return logH;
}

double NoiseModel::log_likelihood(const arma::cx_mat& data_fft) const
{
    double logL = -0.5*log(2.0*M_PI)*ni*nj;

    arma::cx_mat ratio = data_fft / fft_of_psf;
    for(int j=0; j<nj; ++j)
    {
        for(int i=0; i<ni; ++i)
        {
            logL += -0.5*log(std::real(fft_of_psf(i, j)
                                        *std::conj(fft_of_psf(i, j))));
            logL += -0.5*std::real(ratio(i, j)*std::conj(ratio(i, j)));
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
    out << sigma << ' ' << L;
}

std::string NoiseModel::description()
{
    return "sigma, L";
}

std::ostream& operator << (std::ostream& out, const NoiseModel& m)
{
    m.print(out);
    return out;
}

} // namespace CorrelatedNoise

