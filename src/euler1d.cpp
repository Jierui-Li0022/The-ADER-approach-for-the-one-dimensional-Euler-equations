// Conserved/primitive conversions and physical flux for the 1D Euler equations.

#include "aderweno/euler1d.hpp"
#include "aderweno/constants.hpp"
#include <cmath>

Primitive conservedToPrimitive(const Conserved& U)
{
    const double rho = (U.rho > RhoMin) ? U.rho : RhoMin;
    const double u   = U.mom / rho;
    const double p   = pressure(rho, U.mom, U.E);
    return Primitive{rho, u, p};
}

Conserved primitiveToConserved(const Primitive& W)
{
    const double mom = W.rho * W.u;
    double E = W.p / (Gamma - 1.0) + 0.5 * W.rho * W.u * W.u;
    if (!std::isfinite(E))
        E = PMin / (Gamma - 1.0) + 0.5 * mom * mom / W.rho;
    return Conserved{W.rho, mom, E};
}

Physical physicalFlux(const Conserved& U)
{
    const Primitive W = conservedToPrimitive(U);
    return Physical{
        U.mom,
        U.mom * W.u + W.p,
        (U.E + W.p) * W.u
    };
}

double maxSignalSpeed(const Conserved& U)
{
    const double rhoSafe = (U.rho > RhoMin) ? U.rho : RhoMin;
    const double uVal    = U.mom / rhoSafe;
    double pVal = (Gamma - 1.0) * (U.E - 0.5 * U.mom * U.mom / rhoSafe);
    if (pVal < PMin) pVal = PMin;
    return std::abs(uVal) + std::sqrt(Gamma * pVal / rhoSafe);
}