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
,f(0.5)
{

}

void NoiseModel::from_prior(DNest4::RNG& rng)
{
    // Trivial flat priors
    L = 100.0*rng.rand();
    C = 100.0*rng.rand();
    f = rng.rand();
}

double NoiseModel::perturb(DNest4::RNG& rng)
{
    double logH = 0.0;

    int which = rng.rand_int(3);

    if(which == 0)
    {
        L += 100.0*rng.randh();
        DNest4::wrap(L, 0.0, 100.0);
    }
    else if(which == 1)
    {
        C += 100.0*rng.randh();
        DNest4::wrap(C, 0.0, 100.0);
    }
    else
    {
        f += rng.randh();
        DNest4::wrap(f, 0.0, 1.0);
    }

    return logH;
}

double NoiseModel::log_likelihood(const arma::cx_mat& image_fft) const
{
    double logL = 0.0;

    double inv_L = 1.0/L;

    // The noise model kernel
    arma::mat the_model(image_fft.n_rows, image_fft.n_cols);
    for(int j=0; j<(int)the_model.n_cols; ++j)
    {
        jj = j % 
        for(int i=0; i<(int)the_model.n_rows; ++i)
        {
            the_model(i, j) = C*exp(-0.5*pow(i*inv_L, 2) - 0.5*pow(j*inv_L, 2));
        }
    }

    for(int i=0; i<the_model.n_rows; ++i)
    {
        for(int j=0; j<the_model.n_cols; ++j)
            std::cout << the_model(i, j) << ' ';
        std::cout << std::endl;
    }
    exit(0); 

    return logL;
}

void NoiseModel::print(std::ostream& out) const
{
    out << L << ' ' << C << ' ' << f;
}

std::string NoiseModel::description()
{
    return "L, C, f";
}

std::ostream& operator << (std::ostream& out, const NoiseModel& m)
{
    m.print(out);
    return out;
}

} // namespace CorrelatedNoise

