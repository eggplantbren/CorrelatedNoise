#include "NoiseModel.h"
#include <DNest4/code/Distributions/Cauchy.h>
#include <iostream>

namespace CorrelatedNoise
{

NoiseModel::NoiseModel(size_t _ni, size_t _nj)
:ni(_ni)
,nj(_nj)
,n(ni*nj)
,C1(ni, ni)
,C2(nj, nj)
{

}

void NoiseModel::from_prior(DNest4::RNG& rng)
{
    DNest4::Cauchy cauchy(0.0, 5.0);

    do
    {
        sigma0 = cauchy.generate(rng);
    }while(std::abs(sigma0) > 100.0);
    sigma0 = exp(sigma0);

    L = exp(log(1.0) + 0.5*log(n));

    compute_Cs();
}

void NoiseModel::compute_Cs()
{
    // Fill the matrices
    double dist;
    double tau = 1.0/(L*L);
    for(size_t i1=0; i1<ni; ++i1)
    {
        for(size_t i2=i1; i2<ni; ++i2)
        {
            dist = std::abs((double)i1 - (double)i2);
            C1(i1, i2) = sigma0*exp(-dist*dist/tau);
            C1(i2, i1) = C1(i1, i2);
        }
    }
    for(size_t j1=0; j1<nj; ++j1)
    {
        for(size_t j2=j1; j2<nj; ++j2)
        {
            dist = std::abs((double)j1 - (double)j2);
            C2(j1, j2) = sigma0*exp(-dist*dist/tau);
            C2(j2, j1) = C2(j1, j2);
        }
    }

    // Compute decompositions
    L1 = C1.llt().matrixL();
    L2 = C2.llt().matrixL();
}


double NoiseModel::cholesky_element(int i, int j) const
{
    // It's lower triangular
    if(j > i)
        return 0.0;

    return L1(i/nj, j/nj)*L2(i%nj, j%nj);
}

void NoiseModel::print(std::ostream& out) const
{
    for(size_t i1=0; i1<ni; ++i1)
    {
        for(size_t i2=0; i2<ni; ++i2)
            out << C1(i1, i2) << ' '; 
        out << '\n';
    }
    out << "\n\n";


    for(size_t j1=0; j1<nj; ++j1)
    {
        for(size_t j2=0; j2<nj; ++j2)
            out << C2(j1, j2) << ' '; 
        out << '\n';
    }
    out << "\n\n";

    for(size_t i=0; i<n; ++i)
    {
        for(size_t j=0; j<n; ++j)
            out << cholesky_element(i, j) << ' '; 
        out << '\n';
    }

}

} // namespace CorrelatedNoise

