#include "NoiseModel2.h"
#include <iostream>
#include <DNest4/code/Distributions/Cauchy.h>
#include <DNest4/code/Utils.h>
#include <Eigen/Sparse>

namespace CorrelatedNoise
{

NoiseModel2::NoiseModel2(int _ny, int _nx)
:ny(_ny)
,nx(_nx)
,n(ny*nx)
{

}


void NoiseModel2::from_prior(DNest4::RNG& rng)
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

    correlation_logit = 5.0*rng.randn();
}

double NoiseModel2::perturb(DNest4::RNG& rng)
{
    double logH = 0.0;

    DNest4::Cauchy cauchy(0.0, 5.0);

    int which = rng.rand_int(3);

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
    else
    {
        logH -= -0.5*pow(correlation_logit/5.0, 2);
        correlation_logit += 5.0*rng.randh();
        logH += -0.5*pow(correlation_logit/5.0, 2);
    }

    return logH;
}

// More complete log likelihood
double NoiseModel2::log_likelihood(const Eigen::MatrixXd& data,
                                   const Eigen::MatrixXd& model,
                                   const Eigen::MatrixXd& sigma_map) const
{
    // Convert from logit
    double alpha = 0.25*exp(correlation_logit)/(1.0 + exp(correlation_logit));

    // Find min of model
    double min = 1E300;
    for(int i=0; i<ny; ++i)
        for(int j=0; j<nx; ++j)
            if(model(i, j) < min)
                min = model(i, j);

    // Flatten data and turn it into standardised residuals
    Eigen::VectorXd ys(n);
    int k = 0;
    double extra_log_determinant = 0.0;
    double sd;
    int num_non_masked = 0;
    for(int i=0; i<ny; ++i)
    {
        for(int j=0; j<nx; ++j)
        {
            sd = sqrt(coeff0*coeff0 + coeff1*(model(i, j) - min)
                                    + pow(sigma_map(i, j), 2));
            if(sigma_map(i, j) < 1E100)
            {
                ys(k++) = (data(i, j) - model(i, j))/sd;
                extra_log_determinant += 2*log(sd);
                ++num_non_masked;
            }
            else
            {
                // Masked pixels
                ys(k++) = 0.0;
            }
        }
    }

    double logL = -0.5*num_non_masked*log(2.0*M_PI) - 0.5*extra_log_determinant;

    std::vector<Eigen::Triplet<double>> triplets;
    int k1, k2;
    for(int i=0; i<ny; ++i)
    {
        for(int j=0; j<nx; ++j)
        {
            // Here
            k1 = j + i*nx;
            triplets.emplace_back(k1, k1, 1.0);

            // Pixel up
            if(i > 0)
            {
                k2 = j + (i-1)*nx;
//                std::cout << k1 << ' ' << k2 << std::endl;
                triplets.emplace_back(k1, k2, -alpha);
            }

            // Pixel down
            if(i < ny - 1)
            {
                k2 = j + (i+1)*nx;
//                std::cout << k1 << ' ' << k2 << std::endl;
                triplets.emplace_back(k1, k2, -alpha);
            }

            // Pixel left
            if(j > 0)
            {
                k2 = (j-1) + i*nx;
//                std::cout << k1 << ' ' << k2 << std::endl;
                triplets.emplace_back(k1, k2, -alpha);
            }

            // Pixel right
            if(j < nx - 1)
            {
                k2 = (j+1) + i*nx;
//                std::cout << k1 << ' ' << k2 << std::endl;
                triplets.emplace_back(k1, k2, -alpha);
            }

        }
    }

    // Make the sparse precision matrix
    Eigen::SparseMatrix<double> sparse_mat(n, n);
    sparse_mat.setFromTriplets(triplets.begin(), triplets.end());

    // Term in the exponential
    logL += -0.5*ys.transpose()*sparse_mat*ys;

    // Use LDLT for log determinant
    Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> ldlt;
    ldlt.compute(sparse_mat);
    Eigen::VectorXd D = ldlt.vectorD();

    // log det of C, not of anything else!!!
    double log_det = 0.0;
    for(int i=0; i<n; ++i)
        log_det += -log(D(i));
    logL += -0.5*log_det;

    if(std::isnan(logL) || std::isinf(logL))
        logL = -1E300;

    return logL;
}

void NoiseModel2::print(std::ostream& out) const
{
    out << coeff0 << ' ' << coeff1 << ' ' << correlation_logit;
}

std::string NoiseModel2::description()
{
    return "coeff0, coeff1, correlation_logit, ";
}

std::ostream& operator << (std::ostream& out, const NoiseModel2& m)
{
    m.print(out);
    return out;
}

} // namespace CorrelatedNoise

