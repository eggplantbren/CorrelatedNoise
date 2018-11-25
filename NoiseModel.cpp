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

void NoiseModel::compute_Cx()
{
    double inv_L2 = 1.0/(L*L);

    // Loop in column-major order
    for(int j=0; j<nx; ++j)
    {
        for(int i=j; i<nx; ++i)
        {
            Cx(i, j) = exp(-0.5*pow(i - j, 2)*inv_L2);
            if(i != j)
                Cx(j, i) = Cx(i, j);
        }
        Cx(j, j) += 1.0E-6;  // For numerical stability
    }

    // https://eigen.tuxfamily.org/dox/group__TutorialLinearAlgebra.html#title2
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigensolver(Cx);
    Ex = eigensolver.eigenvalues();
    Vx = eigensolver.eigenvectors();
}


void NoiseModel::compute_Cy()
{
    double inv_L2 = 1.0/(L*L);

    // Loop in column-major order
    for(int j=0; j<ny; ++j)
    {
        for(int i=j; i<ny; ++i)
        {
            Cy(i, j) = exp(-0.5*pow(i - j, 2)*inv_L2);
            if(i != j)
                Cy(j, i) = Cy(i, j);
        }
        Cy(j, j) += 1.0E-6;  // For numerical stability
    }

    // https://eigen.tuxfamily.org/dox/group__TutorialLinearAlgebra.html#title2
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigensolver(Cy);
    Ey = eigensolver.eigenvalues();
    Vy = eigensolver.eigenvectors();
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

    // Log-uniform(0.1, 0.1*pixels)
    L = exp(log(0.1) + 0.5*log(n)*rng.rand());

    compute_Cx();
    compute_Cy();
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
    else if(which == 1)
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
    else if(which == 2)
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
        L += 0.5*log(n)*rng.randh();
        DNest4::wrap(L, log(0.1), log(0.1*sqrt(n)));
        L = exp(L);

        compute_Cx();
        compute_Cy();
    }

    return logH;
}

// More complete log likelihood
double NoiseModel::log_likelihood(const Eigen::MatrixXd& data,
                                  const Eigen::MatrixXd& model,
                                  const Eigen::MatrixXd& sigma_map) const
{
    // All eigenvalues of C
    Eigen::MatrixXd Emat = outer(Ey, Ex);
    Eigen::Map<Eigen::VectorXd> E(Emat.data(), n);

    // Flatten data
    Eigen::VectorXd ys(n);
    int k = 0;
    double extra_log_determinant = 0.0;
    double sd;
    for(int i=0; i<ny; ++i)
    {
        for(int j=0; j<nx; ++j)
        {
            sd = sqrt(coeff0*coeff0 + pow(sigma_map(i, j), 2)
                                    + coeff1*std::abs(model(i, j)));
            ys(k++) = (data(i, j) - model(i, j))/sd;
            extra_log_determinant += 2*log(sd);
        }
    }

    // Dot product of the data against eigenvectors of C
    // i.e., the data represented in that basis
    Eigen::VectorXd coeffs(n);
    k = 0;
    for(int i=0; i<ny; ++i)
    {
        for(int j=0; j<nx; ++j)
        {
            Eigen::MatrixXd V = outer(Vy.col(i), Vx.col(j));
            coeffs[k++] = ys.dot(Eigen::Map<Eigen::VectorXd>(V.data(), n));
        }
    }

    double log_det = extra_log_determinant;
    for(int i=0; i<n; ++i)
        log_det += log(E(i));

    Eigen::VectorXd solved = (coeffs.array()/E.array()).matrix();
    double dot_prod = coeffs.dot(solved);

    double logL = -n*0.5*log(2*M_PI) - 0.5*log_det - 0.5*dot_prod;

    return logL;
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

Eigen::MatrixXd outer(const Eigen::VectorXd& x, const Eigen::VectorXd& y)
{
    return y * x.transpose();
}


} // namespace CorrelatedNoise

