// Definitions of project-wide constants declared in constants.H.

#include "aderweno/constants.hpp"

const double Gamma        = 1.4;   // Ideal diatomic gas (air at room temperature)
const double RhoMin       = 1e-12; // Effectively vacuum for all test cases here
const double PMin         = 1e-12;
const double PMax         = 1e6;
const int    NewtonMaxIter = 50;   // Typically converges in 4-8 iters for Sod/Lax
const double NewtonTol    = 1e-8;
const double WenoEpsilon  = 1e-6;  // Follows Jiang & Shu (1996) JCP 126:202-228
const int    WenoQPower   = 2;