// First-order Godunov finite-volume solver using HLLC fluxes.

#include "aderweno/godunov_solver.hpp"
#include "aderweno/riemann_solver.hpp"
#include "aderweno/euler1d.hpp"
#include "aderweno/checks.hpp"
#include <algorithm>
#include <vector>

namespace aderweno {

void godunovAdvance(std::vector<Conserved>& U,
                    double dx, double dt,
                    bool periodic)
{
    const int nx = static_cast<int>(U.size());
    if (nx <= 1) return;

    std::vector<Physical> F(nx + 1);

    if (periodic) {
        F[0] = hllcFlux(U[nx - 1], U[0]);
        for (int i = 0; i < nx - 1; ++i)
            F[i + 1] = hllcFlux(U[i], U[i + 1]);
        F[nx] = F[0];
    } else {
        F[0]  = hllcFlux(U[0],      U[0]);
        F[nx] = hllcFlux(U[nx - 1], U[nx - 1]);
        for (int i = 0; i < nx - 1; ++i)
            F[i + 1] = hllcFlux(U[i], U[i + 1]);
    }

    const double a = dt / dx;
    std::vector<Conserved> Unew(nx);
    for (int i = 0; i < nx; ++i) {
        Unew[i].rho = U[i].rho - a * (F[i+1].fluxRho - F[i].fluxRho);
        Unew[i].mom = U[i].mom - a * (F[i+1].fluxMom - F[i].fluxMom);
        Unew[i].E   = U[i].E   - a * (F[i+1].fluxE   - F[i].fluxE);
        enforcePhysical(Unew[i]);
    }
    U.swap(Unew);
}

void godunovRun(std::vector<Conserved>& U,
                double x0, double x1,
                double Tfinal, double CFL,
                bool periodic)
{
    const int    nx = static_cast<int>(U.size());
    const double dx = (x1 - x0) / nx;
    double t = 0.0;

    while (t < Tfinal - 1e-14) {
        double smax = 0.0;
        for (const auto& Ui : U) smax = std::max(smax, maxSignalSpeed(Ui));
        if (!std::isfinite(smax) || smax <= 0.0) smax = 2.0;

        double dt = CFL * dx / smax;
        if (t + dt > Tfinal) dt = Tfinal - t;

        godunovAdvance(U, dx, dt, periodic);
        t += dt;
    }
}

}