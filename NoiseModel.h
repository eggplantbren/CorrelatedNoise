#ifndef CorrelatedNoise_NoiseModel_h
#define CorrelatedNoise_NoiseModel_h

#include <DNest4/code/RNG.h>
#include <Eigen/Cholesky>
#include <Eigen/Dense>

namespace CorrelatedNoise
{

// Some type aliases similar to Eigen's
using Matrix = Eigen::Matrix<double,
                            Eigen::Dynamic,
                            Eigen::Dynamic,
                            Eigen::RowMajor>;
using Vector = Eigen::Matrix<double, Eigen::Dynamic, 1, Eigen::RowMajor>;
using Cholesky = Eigen::LLT<Matrix>;


// An instance of this class is a point in the parameter space of a
// noise model, and associated functions (such as a likelihood evaluator!)
class NoiseModel
{
    private:

        // Image dimensions
        size_t ni, nj, n;

        // Parameters
        double sigma0;      // Coefficient
        double L;           // Length scales

        // Covariance matrices (1-D)
        Matrix C1, C2;

        // Cholesky decompositions
        Cholesky L1, L2;

        // Compute covariance matrices
        void compute_Cs();

    public:

        // Constructor. Provide image dimensions.
        NoiseModel(size_t _ni, size_t _njn);

        // Generate from prior
        void from_prior(DNest4::RNG& rng);

        // Evaluate log likelihood
        double log_likelihood(const Vector& image) const;
};

}

#endif

