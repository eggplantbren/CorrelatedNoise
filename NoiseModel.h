#ifndef CorrelatedNoise_NoiseModel_h
#define CorrelatedNoise_NoiseModel_h

#include <ostream>
#include <DNest4/code/RNG.h>
#include <Eigen/Dense>

namespace CorrelatedNoise
{

// An instance of this class is a point in the parameter space of a
// noise model, and associated functions (such as a likelihood evaluator!)
class NoiseModel
{
    private:

        // Image dimensions
        int ny, nx, n;

        // Parameters
        double coeff0; // Base sigma
        double coeff1; // Coefficient in front of provided sigma map
        double L;      // Length scale

        // Kronecker factors and their eigenvalues/vectors
        Eigen::MatrixXd Cy, Cx;
        Eigen::VectorXd Ey, Ex;
        Eigen::MatrixXd Vy, Vx;

        // Compute the above matrices
        void compute_Cy();
        void compute_Cx();

    public:

        // Constructor. Provide image dimensions.
        NoiseModel(int _ny, int _nx);

        // Generate from prior
        void from_prior(DNest4::RNG& rng);

        // Perturb
        double perturb(DNest4::RNG& rng);

        // More complete log likelihood
        double log_likelihood(const Eigen::MatrixXd& data,
                              const Eigen::MatrixXd& model,
                              const Eigen::MatrixXd& sigma_map) const;

        // Print to stream
        void print(std::ostream& out) const;

        // Header string
        static std::string description();
};

// Outer product of two vectors
Eigen::MatrixXd outer(const Eigen::VectorXd& x, const Eigen::VectorXd& y);

std::ostream& operator << (std::ostream& out, const NoiseModel& m);


}

#endif

