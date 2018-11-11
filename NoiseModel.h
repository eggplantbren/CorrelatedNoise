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
        double coeff0; // Base sigma
        double coeff1; // Coefficient in front of provided sigma map
        double coeff2; // Coefficient in front of sqrt(abs(model image))
        double L;      // Length scale

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

        // Evaluate log likelihood (pass in FFT of standardised residuals)
        // (Ignores coeff)
        double log_likelihood_flat(const arma::cx_mat& image_fft) const;

        // More complete log likelihood
        double log_likelihood(const arma::mat& data,
                              const arma::mat& model,
                              const arma::mat& sigma_map) const;

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

