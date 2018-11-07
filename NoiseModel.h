#ifndef CorrelatedNoise_NoiseModel_h
#define CorrelatedNoise_NoiseModel_h

#include <DNest4/code/RNG.h>
#include <armadillo>
#include <ostream>

namespace CorrelatedNoise
{

// An instance of this class is a point in the parameter space of a
// noise model, and associated functions (such as a likelihood evaluator!)
class NoiseModel
{
    private:

        // Image dimensions
        int ni, nj, n;

        // Parameters
        double sigma, L; // Coefficient and length scale

        // Fourier transform of the PSF
        arma::cx_mat fft_of_psf;
        void compute_psf();

    public:

        // Constructor. Provide image dimensions.
        NoiseModel(int _ni, int _nj);

        // Generate from prior
        void from_prior(DNest4::RNG& rng);

        // Perturb
        double perturb(DNest4::RNG& rng);

        // Evaluate log likelihood
        double log_likelihood(const arma::cx_mat& image_fft) const;

        // Generate an image
        arma::vec generate_image(DNest4::RNG& rng) const;

        // Print to stream
        void print(std::ostream& out) const;

        // Header string
        static std::string description();
};

std::ostream& operator << (std::ostream& out, const NoiseModel& m);


}

#endif

