#include "NoiseModel.h"

namespace CorrelatedNoise
{

NoiseModel::NoiseModel(size_t _ni, size_t _nj)
:ni(_ni)
,nj(_nj)
,n(ni*nj)
,C1(ni, ni)
,C2(nj, nj)
{

}

void NoiseModel::compute_Cs()
{
    double dist = 0.0;
    double tau = 1.0/(L*L);
    for(size_t i1=0; i1<ni; ++i1)
    {
        for(size_t i2=i1; i2<ni; ++i2)
        {
            dist = std::abs((double)i1 - (double)i2);
            C1(i1, i2) = sigma0*exp(-dist*dist/tau);
            C1(i2, i1) = C1(i1, i2);
        }
    }
    for(size_t j1=0; j1<nj; ++j1)
    {
        for(size_t j2=j1; j2<nj; ++j2)
        {
            dist = std::abs((double)j1 - (double)j2);
            C2(j1, j2) = sigma0*exp(-dist*dist/tau);
            C2(j2, j1) = C2(j1, j2);
        }
    }
}

} // namespace CorrelatedNoise

