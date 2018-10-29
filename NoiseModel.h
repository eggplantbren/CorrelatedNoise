#ifndef CorrelatedNoise_NoiseModel_h
#define CorrelatedNoise_NoiseModel_h

#include <Eigen/Dense>

namespace CorrelatedNoise
{

// An instance of this class is a point in the parameter space of a
// noise model, and associated functions (such as a likelihood evaluator!)

class NoiseModel
{
    private:

        // Parameters



    public:

        // Constructor. Provide image dimensions.
        NoiseModel(size_t ni, size_t nj);

};

}

#endif

