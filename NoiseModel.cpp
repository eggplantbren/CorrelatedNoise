#include "NoiseModel.h"
#include <iostream>
#include <DNest4/code/Distributions/Cauchy.h>
#include <DNest4/code/Utils.h>

namespace CorrelatedNoise
{

NoiseModel::NoiseModel(int _ny, int _nx)
:ny(_ny)
,nx(_nx)
,n(ny*nx)
,Cy(ny, ny)
,Cx(nx, nx)
{

}

void NoiseModel::from_prior(DNest4::RNG& rng)
{
    DNest4::Cauchy cauchy(0.0, 5.0);

    // Fairly generic priors
    do
    {
        coeff0 = cauchy.generate(rng);
    }while(std::abs(coeff0) >= 100.0);
    coeff0 = exp(coeff0);

    do
    {
        coeff1 = cauchy.generate(rng);
    }while(std::abs(coeff1) >= 100.0);
    coeff1 = exp(coeff1);

    do
    {
        coeff2 = cauchy.generate(rng);
    }while(std::abs(coeff2) >= 100.0);
    coeff2 = exp(coeff2);

    // Log-uniform(0.1, sqrt(n))
    L = exp(log(0.1) + log(10.0*sqrt(n))*rng.rand());

}

double NoiseModel::perturb(DNest4::RNG& rng)
{
    double logH = 0.0;

    DNest4::Cauchy cauchy(0.0, 5.0);
    int which = rng.rand_int(4);

    if(which == 0)
    {
        coeff0 = log(coeff0);
        logH += cauchy.perturb(coeff0, rng);
        if(std::abs(coeff0) >= 100.0)
        {
            coeff0 = 1.0;
            return -1E300;
        }
        coeff0 = exp(coeff0);
    }
    else if(which == 2)
    {
        coeff1 = log(coeff1);
        logH += cauchy.perturb(coeff1, rng);
        if(std::abs(coeff1) >= 100.0)
        {
            coeff1 = 1.0;
            return -1E300;
        }
        coeff1 = exp(coeff1);
    }
    else if(which == 3)
    {
        coeff2 = log(coeff2);
        logH += cauchy.perturb(coeff2, rng);
        if(std::abs(coeff2) >= 100.0)
        {
            coeff2 = 1.0;
            return -1E300;
        }
        coeff2 = exp(coeff2);
    }
    else
    {
        L = log(L);
        L += log(10.0*sqrt(n))*rng.randh();
        DNest4::wrap(L, log(0.1), log(sqrt(n)));
        L = exp(L);
    }

    return logH;
}

// More complete log likelihood
double NoiseModel::log_likelihood(const Eigen::MatrixXd& data,
                                  const Eigen::MatrixXd& model,
                                  const Eigen::MatrixXd& sigma_map) const
{
    return 0.0;
}

void NoiseModel::print(std::ostream& out) const
{
    out << coeff0 << ' ' << coeff1 << ' ' << coeff2 << ' ' << L;
}

std::string NoiseModel::description()
{
    return "coeff0, coeff1, coeff2, L, ";
}

std::ostream& operator << (std::ostream& out, const NoiseModel& m)
{
    m.print(out);
    return out;
}

} // namespace CorrelatedNoise

