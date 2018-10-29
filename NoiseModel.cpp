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
,C1(n1, n1)
,C2(n2, n2)
{

}

void NoiseModel::from_prior(DNest4::RNG& rng)
{
    // Trivial flat priors
    L = 100.0*rng.rand();

    compute_Cs();
}

double NoiseModel::perturb(DNest4::RNG& rng)
{
    double logH = 0.0;

    L += 100.0*rng.randh();
    DNest4::wrap(L, 0.0, 100.0);

    compute_Cs();

    return logH;
}

double NoiseModel::log_det() const
{
    double result = 0.0;
    for(int i=0; i<n; ++i)
        result += 2.0*log(cholesky_element(i, i));
    return result;
}

double NoiseModel::log_likelihood(const Vector& image) const
{
    return -0.5*n*log(2*M_PI) -0.5*log_det() - 0.5*quadratic_form(image);
}

void NoiseModel::compute_Cs()
{
    // Fill the matrices
    double dist;
    double tau = 1.0/(L*L);

    for(int i=0; i<n1; ++i)
    {
        for(int j=i; j<n1; ++j)
        {
            dist = std::abs(i - j);
            C1(i, j) = exp(-dist*dist*tau);
            C1(j, i) = C1(i, j);
        }

        // To ensure positive definiteness
        C1(i, i) += 1E-5*L;
    }
    for(int i=0; i<n2; ++i)
    {
        for(int j=i; j<n2; ++j)
        {
            dist = std::abs(i - j);
            C2(i, j) = exp(-dist*dist*tau);
            C2(j, i) = C2(i, j);
        }

        // To ensure positive definiteness
        C2(i, i) += 1E-5*L;
    }

    // Compute decompositions
    L1 = C1.llt().matrixL();
    L2 = C2.llt().matrixL();
}


inline double NoiseModel::cholesky_element(int i, int j) const
{
    // It's lower triangular
//    if(j > i)
//        return 0.0;

    return L1(i/n2, j/n2)*L2(i%n2, j%n2);
}


double NoiseModel::quadratic_form(const Vector& ys) const
{
    Vector solution(ys.size());

    // Solve Lx = y
    for(int i=0; i<solution.size(); ++i)
    {
        solution[i] = ys[i];
        for(int j=0; j<i; ++j)
            solution[i] -= cholesky_element(i, j)*solution[j];
        solution[i] /= cholesky_element(i, i);
    }

    return solution.squaredNorm();
}

Vector NoiseModel::generate_image(DNest4::RNG& rng) const
{
    Vector normals(n);
    Vector image(n);
    for(int i=0; i<n; ++i)
        normals(i) = rng.randn();

    for(int i=0; i<n; ++i)
    {
        image(i) = 0.0;
        for(int j=0; j<=i; ++j)
            image(i) += cholesky_element(i, j)*normals(j);
    }

    return image;
}

void NoiseModel::print(std::ostream& out) const
{
    out << L;
}

std::string NoiseModel::description()
{
    return "L, ";
}

std::ostream& operator << (std::ostream& out, const NoiseModel& m)
{
    m.print(out);
    return out;
}

} // namespace CorrelatedNoise

