#ifndef CorrelatedNoise_NoiseModel_h
#define CorrelatedNoise_NoiseModel_h

#include <DNest4/code/RNG.h>
#include <Eigen/Cholesky>
#include <Eigen/Dense>
#include <ostream>

namespace CorrelatedNoise
{

// Some type aliases similar to Eigen's
using Matrix = Eigen::Matrix<double,
                            Eigen::Dynamic,
                            Eigen::Dynamic,
                            Eigen::RowMajor>;
using Vector = Eigen::VectorXd;
using Cholesky = Eigen::LLT<Matrix>;


// An instance of this class is a point in the parameter space of a
// noise model, and associated functions (such as a likelihood evaluator!)
class NoiseModel
{
    private:

        // Image dimensions
        size_t n1, n2, n;

        // Parameters
        double sigma0;      // Coefficient
        double L;           // Length scales

        // "Covariance" matrices (factors along the 1-D dimensions)
        Matrix C1, C2;

        // Cholesky decompositions as matrices
        Matrix L1, L2;

        // Compute covariance matrices
        void compute_Cs();

        // Implicit elements of full Cholesky
        // decomposition of C = C1 `kroneckerProduct` C2
        double cholesky_element(int i, int j) const;

        // Log determinant
        double log_det() const;

    public:

        // Constructor. Provide image dimensions.
        NoiseModel(size_t _n1, size_t _n2);

        // Generate from prior
        void from_prior(DNest4::RNG& rng);

        // Perturb
        double perturb(DNest4::RNG& rng);

        // Evaluate log likelihood
        double log_likelihood(const Vector& image) const;

        // Generate an image
        Vector generate_image(DNest4::RNG& rng) const;

        // Print to stream
        void print(std::ostream& out) const;

        // Header string
        static std::string description();
};

}

#endif

