// Third-order ADER solver: WENO5 (M=2) + CK order 2 + TT-GRP + 2-point Gauss.

#include "aderweno/ader3_solver.hpp"
#include "aderweno/weno.hpp"
#include "aderweno/tt_grp.hpp"
#include "aderweno/euler1d.hpp"
#include "aderweno/riemann_solver.hpp"
#include "aderweno/checks.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <chrono>
#include <vector>

namespace aderweno {

static void ader3AdvanceFast(
    std::vector<Conserved>& U, double dx, double dt,
    bool periodic, const TTGRPOptions& opt,
    const WENO1d& weno, WENOWorkspace& ws,
    std::vector<Physical>& Fbar, std::vector<Conserved>& Unew)
{
    const int nx = static_cast<int>(U.size());
    if (nx <= 1) return;

    WenoBCType bc = periodic ? WenoBCType::Periodic : WenoBCType::Outflow;
    weno.reconstruct_all_interfaces(U, bc, ws);

    #pragma omp parallel for schedule(static)
    for (int f = 0; f < nx; ++f) {
        const int iL = (f > 0) ? f - 1 : (periodic ? nx - 1 : 0);
        Physical fHllc = hllcFlux(U[iL], U[f]);
        Physical fAder = ttGrpFluxAder3(ws.PolyL[f], ws.PolyR[f], dt, opt);
        Fbar[f] = safeFlux(fAder, fHllc);
    }

    if (periodic) Fbar[nx] = Fbar[0];
    else          Fbar[nx] = hllcFlux(U[nx-1], U[nx-1]);

    const double coeff = dt / dx;
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < nx; ++i) {
        Unew[i].rho = U[i].rho - coeff * (Fbar[i+1].fluxRho - Fbar[i].fluxRho);
        Unew[i].mom = U[i].mom - coeff * (Fbar[i+1].fluxMom - Fbar[i].fluxMom);
        Unew[i].E   = U[i].E   - coeff * (Fbar[i+1].fluxE   - Fbar[i].fluxE);
        enforcePhysical(Unew[i]);
    }
    U.swap(Unew);
}

void ader3Advance(std::vector<Conserved>& U, double dx, double dt,
                  bool periodic, const TTGRPOptions& opt)
{
    const int nx = static_cast<int>(U.size());
    WENO1d weno(2, dx);
    WENOWorkspace ws = weno.alloc_workspace(nx);
    std::vector<Physical>   Fbar(nx + 1, {0, 0, 0});
    std::vector<Conserved>  Unew(nx);
    ader3AdvanceFast(U, dx, dt, periodic, opt, weno, ws, Fbar, Unew);
}

void ader3Run(std::vector<Conserved>& U,
              double x0, double x1, double Tfinal, double CFL,
              bool periodic, const TTGRPOptions& opt)
{
    const int    nx = static_cast<int>(U.size());
    const double dx = (x1 - x0) / nx;

    WENO1d weno(2, dx);
    WENOWorkspace ws = weno.alloc_workspace(nx);
    std::vector<Physical>  Fbar(nx + 1, {0, 0, 0});
    std::vector<Conserved> Unew(nx);

    double t = 0.0;
    int step = 0;
    std::cout << "[ADER3] nx=" << nx << " dx=" << dx << " Tfinal=" << Tfinal << "\n";

    while (t < Tfinal - 1e-14) {
        double smax = 0.0;
        for (const auto& Ui : U) smax = std::max(smax, maxSignalSpeed(Ui));
        if (!std::isfinite(smax) || smax <= 0.0) smax = 2.0;

        double dt = CFL * dx / smax;
        if (t + dt > Tfinal) dt = Tfinal - t;

        auto t0 = std::chrono::high_resolution_clock::now();
        ader3AdvanceFast(U, dx, dt, periodic, opt, weno, ws, Fbar, Unew);
        double ms = std::chrono::duration<double, std::milli>(
            std::chrono::high_resolution_clock::now() - t0).count();

        t += dt; ++step;
        if (step <= 5 || step % 50 == 0)
            std::cout << "  step " << step << "  t=" << t
                      << "  dt=" << dt << "  smax=" << smax
                      << "  wall=" << ms << " ms\n";
    }
    std::cout << "[ADER3] done: " << step << " steps, t=" << t << "\n";
}

}