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

