// Titarev-Toro shock-turbulence interaction test comparing MUSCL and ADER3/4/5.

#include "aderweno/constants.hpp"
#include "aderweno/euler1d.hpp"
#include "aderweno/riemann_solver.hpp"
#include "aderweno/weno.hpp"
#include "aderweno/tt_grp.hpp"
#include "aderweno/muscl_hancock_solver.hpp"
#include "aderweno/ader3_solver.hpp"
#include "aderweno/ader4_solver.hpp"
#include "aderweno/ader5_solver.hpp"

#include <cmath>
#include <vector>
#include <fstream>
#include <iostream>
#include <iomanip>

static constexpr double PI = 3.1415926535897932384626433832795;

static std::vector<Conserved> initTitarevToro(int nx, double x0, double x1)
{
    const double dx = (x1 - x0) / nx;
    const double eps = 0.1;
    const double k   = 20.0 * PI;
    const double x_disc = -4.5;

    std::vector<Conserved> U(nx);
    for (int i = 0; i < nx; ++i) {
        const double xc = x0 + (i + 0.5) * dx;
        double rho, u, p;
        if (xc < x_disc) {
            rho = 1.515695;
            u   = 0.523346;
            p   = 1.805000;
        } else {
            rho = 1.0 + eps * std::sin(k * xc);
            u   = 0.0;
            p   = 1.0;
        }
        U[i] = primitiveToConserved(Primitive{rho, u, p});
    }
    return U;
}

// Interpolate reference solution to target x coordinate
static double interpRef(const std::vector<double>& xref,
                         const std::vector<Conserved>& Uref,
                         double x)
{
    int lo=0, hi=(int)xref.size()-1;
    while (lo < hi-1) {
        int mid=(lo+hi)/2;
        if (xref[mid] < x) lo=mid; else hi=mid;
    }
    double t=(x-xref[lo])/(xref[hi]-xref[lo]);
    return Uref[lo].rho*(1-t) + Uref[hi].rho*t;
}

static void writeCsv(const std::string& filename,
                      const std::vector<double>& xc,
                      const std::vector<Conserved>& muscl,
                      const std::vector<Conserved>& ader3,
                      const std::vector<Conserved>& ader4,
                      const std::vector<Conserved>& ader5,
                      const std::vector<double>& xref,
                      const std::vector<Conserved>& ref)
{
    std::ofstream f(filename);
    f << "x,rho_muscl,rho_ader3,rho_ader4,rho_ader5,rho_ref\n";
    f << std::scientific << std::setprecision(10);
    for (int i = 0; i < (int)xc.size(); ++i) {
        f << xc[i] << ","
          << muscl[i].rho << ","
          << ader3[i].rho << ","
          << ader4[i].rho << ","
          << ader5[i].rho << ","
          << interpRef(xref, ref, xc[i]) << "\n";
    }
    std::cout << "wrote " << filename << "\n";
}

void task_3_3()
{
    const double x0  = -5.0;
    const double x1  =  5.0;
    const double T   =  5.0;
    const int    nx  =  2000;
    const int    nx_ref = 4000;

    // Per-solver CFL: ADER5 needs smaller CFL for stability with CK4
    const double CFL_muscl = 0.9;
    const double CFL_ader3 = 0.9;
    const double CFL_ader4 = 0.9;
    const double CFL_ader5 = 0.45;

    const double dx = (x1 - x0) / nx;
    std::vector<double> xc(nx);
    for (int i = 0; i < nx; ++i)
        xc[i] = x0 + (i + 0.5) * dx;

    const double dx_ref = (x1 - x0) / nx_ref;
    std::vector<double> xc_ref(nx_ref);
    for (int i = 0; i < nx_ref; ++i)
        xc_ref[i] = x0 + (i + 0.5) * dx_ref;

    // Use HLLC for leading term (NOT exact Riemann solver which can fail)
    TTGRPOptions opt{};
    opt.use_exact_q0 = true;
    opt.flux_type    = TTRiemannFluxType::HLLC;

    std::cout << "=== Task 3.3: Titarev-Toro shock-turbulence ===\n";

    std::cout << "[Running MUSCL nx=2000 CFL=" << CFL_muscl << "]\n";
    auto Um = initTitarevToro(nx, x0, x1);
    aderweno::musclHancockRun(Um, x0, x1, T, CFL_muscl, false,
                                aderweno::LimiterType::VanLeer);

    std::cout << "[Running ADER3 nx=2000 CFL=" << CFL_ader3 << "]\n";
    auto U3 = initTitarevToro(nx, x0, x1);
    aderweno::ader3Run(U3, x0, x1, T, CFL_ader3, false, opt);

    std::cout << "[Running ADER4 nx=2000 CFL=" << CFL_ader4 << "]\n";
    auto U4 = initTitarevToro(nx, x0, x1);
    aderweno::ader4Run(U4, x0, x1, T, CFL_ader4, false, opt);

    std::cout << "[Running ADER5 nx=2000 CFL=" << CFL_ader5 << "]\n";
    auto U5 = initTitarevToro(nx, x0, x1);
    aderweno::ader5Run(U5, x0, x1, T, CFL_ader5, false, opt);

    std::cout << "[Running ADER5 nx=4000 (reference) CFL=" << CFL_ader5 << "]\n";
    auto Uref = initTitarevToro(nx_ref, x0, x1);
    aderweno::ader5Run(Uref, x0, x1, T, CFL_ader5, false, opt);

    writeCsv("task_3_3.csv", xc, Um, U3, U4, U5, xc_ref, Uref);
}