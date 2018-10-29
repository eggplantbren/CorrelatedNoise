#include "NoiseModel.h"
#include <DNest4/code/Distributions/Cauchy.h>
#include <DNest4/code/Utils.h>
#include <iostream>

namespace CorrelatedNoise
{

NoiseModel::NoiseModel(size_t _n1, size_t _n2)
:n1(_n1)
,n2(_n2)
,n(n1*n2)
,C1(n1, n1)
,C2(n2, n2)
{

}

void NoiseModel::from_prior(DNest4::RNG& rng)
{
    DNest4::Cauchy cauchy(0.0, 5.0);

    // Trivial flat priors
    sigma0 = 1.0;//100.0*rng.rand();
    L = 2.0;//100.0*rng.rand();

    compute_Cs();
}

double NoiseModel::perturb(DNest4::RNG& rng)
{
    double logH = 0.0;

    int which = rng.rand_int(2);
    if(which == 0)
    {
        sigma0 += 100.0*rng.randh();
        DNest4::wrap(sigma0, 0.0, 100.0);
    }
    else if(which == 0)
    {
        L += 100.0*rng.randh();
        DNest4::wrap(L, 0.0, 100.0);
    }
    compute_Cs();

    return logH;
}

double NoiseModel::log_det() const
{
    double result = 0.0;
    for(size_t i=0; i<n; ++i)
        result += 2.0*log(cholesky_element(i, i));
    return result;
}

void NoiseModel::compute_Cs()
{
    // Fill the matrices
    double dist;
    double tau = 1.0/(L*L);

    for(size_t i=0; i<n1; ++i)
    {
        for(size_t j=i; j<n1; ++j)
        {
            dist = std::abs((double)i - (double)j);
            C1(i, j) = sigma0*exp(-dist*dist*tau);
            C1(j, i) = C1(i, j);
        }
    }
    for(size_t i=0; i<n2; ++i)
    {
        for(size_t j=i; j<n2; ++j)
        {
            dist = std::abs((double)i - (double)j);
            C2(i, j) = sigma0*exp(-dist*dist*tau);
            C2(j, i) = C2(i, j);
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

    return L1(i/n2, j/n2)*L2(i%n2, j%n2);
}


Vector NoiseModel::generate_image(DNest4::RNG& rng) const
{
    Vector normals(n);
    Vector image(n);
    for(size_t i=0; i<n; ++i)
        normals(i) = rng.randn();

    for(size_t i=0; i<n; ++i)
    {
        image(i) = 0.0;
        for(size_t j=0; j<i; ++j)
            image(i) += cholesky_element(i, j)*normals(j);
        image(i) += cholesky_element(i, i)*normals(i);
    }

    return image;
}

void NoiseModel::print(std::ostream& out) const
{
    out << sigma0 << ' ' << L;
}

std::string NoiseModel::description()
{
    return "sigma0, L, ";
}

} // namespace CorrelatedNoise

