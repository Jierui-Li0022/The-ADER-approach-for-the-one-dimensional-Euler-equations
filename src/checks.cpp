// Validation checks for conserved variables and Newton-Raphson convergence.

#include "aderweno/checks.hpp"
#include "aderweno/eos.hpp"
#include "aderweno/constants.hpp"
#include <cmath>
#include <iostream>

bool isValidRho(double rho)
{
    if (rho <= 0 || !std::isfinite(rho)) {
        std::cerr << "invalid rho: " << rho << "\n";
        return false;
    }
    return true;
}

bool isValidPressure(double p)
{
    if (p <= 0 || !std::isfinite(p)) {
        std::cerr << "invalid pressure: " << p << "\n";
        return false;
    }
    return true;
}

bool checkConserved(const Conserved& U)
{
    if (!isValidRho(U.rho)) return false;
    if (!isValidPressure(pressure(U.rho, U.mom, U.E))) return false;
    return true;
}

bool isConverged(double oldPStar, double newPStar, double tolerance)
{
    return std::abs(newPStar - oldPStar) < tolerance;
}