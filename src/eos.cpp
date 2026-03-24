// Equation of State implementation: pressure and sound speed for an ideal gas.

#include "aderweno/eos.hpp"
#include "aderweno/constants.hpp"
#include <cmath>

namespace {

inline double clampRho(double rho)
{
    if (!std::isfinite(rho) || rho < RhoMin) return RhoMin;
    return rho;
}

inline double clampP(double p)
{
    if (!std::isfinite(p) || p < PMin) return PMin;
    return p;
}

inline double clampEForPressure(double rho, double mom, double E)
{
    const double eMin = 0.5 * mom * mom / rho + PMin / (Gamma - 1.0);
    if (!std::isfinite(E) || E < eMin) return eMin;
    return E;
}

}

double pressure(double rho, double mom, double E)
{
    rho = clampRho(rho);
    E   = clampEForPressure(rho, mom, E);
    return clampP((Gamma - 1.0) * (E - 0.5 * mom * mom / rho));
}

double soundSpeed(double rho, double mom, double E)
{
    rho = clampRho(rho);
    return std::sqrt(Gamma * pressure(rho, mom, E) / rho);
}