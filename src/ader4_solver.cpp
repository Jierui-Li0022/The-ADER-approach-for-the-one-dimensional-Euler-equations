// Fourth-order ADER solver: WENO7 (M=3) + CK order 3 + TT-GRP + 2-point Gauss.

#include "aderweno/ader4_solver.hpp"
#include "aderweno/weno.hpp"
#include "aderweno/tt_grp.hpp"
#include "aderweno/euler1d.hpp"
#include "aderweno/riemann_solver.hpp"
#include "aderweno/checks.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>

namespace aderweno {

// Diagnostic counters reset each ader4Run call
static int    g_fallbackCount    = 0;
static int    g_nanFallbackCount = 0;
static double g_maxRelDiff       = 0.0;

static Physical safeFluxDiag(const Physical& fAder, const Physical& fHllc,
                              double tol = 10.0)
{
    if (!std::isfinite(fAder.fluxRho) ||
        !std::isfinite(fAder.fluxMom) ||
        !std::isfinite(fAder.fluxE)) {
        ++g_nanFallbackCount; ++g_fallbackCount;
        return fHllc;
    }
    auto relDiff = [](double fa, double fh) {
        return std::abs(fa - fh) / std::max({std::abs(fa), std::abs(fh), 1e-6});
    };
    double rd = std::max({relDiff(fAder.fluxRho, fHllc.fluxRho),
                          relDiff(fAder.fluxMom, fHllc.fluxMom),
                          relDiff(fAder.fluxE,   fHllc.fluxE)});
    if (rd > g_maxRelDiff) g_maxRelDiff = rd;

    auto close = [&](double fa, double fh) {
        return std::abs(fa - fh) <= tol * std::max({std::abs(fa), std::abs(fh), 1e-6});
    };
    if (close(fAder.fluxRho, fHllc.fluxRho) &&
        close(fAder.fluxMom, fHllc.fluxMom) &&
        close(fAder.fluxE,   fHllc.fluxE))
        return fAder;
    ++g_fallbackCount;
    return fHllc;
}

static void ader4AdvanceFast(
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
        Physical fAder = ttGrpFluxAder4(ws.PolyL[f], ws.PolyR[f], dt, opt);
        Fbar[f] = safeFluxDiag(fAder, fHllc);
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

void ader4Advance(std::vector<Conserved>& U, double dx, double dt,
                  bool periodic, const TTGRPOptions& opt)
{
    const int nx = static_cast<int>(U.size());
    WENO1d weno(3, dx);
    WENOWorkspace ws = weno.alloc_workspace(nx);
    std::vector<Physical>  Fbar(nx + 1, {0, 0, 0});
    std::vector<Conserved> Unew(nx);
    ader4AdvanceFast(U, dx, dt, periodic, opt, weno, ws, Fbar, Unew);
}

void ader4Run(std::vector<Conserved>& U,
              double x0, double x1, double Tfinal, double CFL,
              bool periodic, const TTGRPOptions& opt)
{
    const int    nx = static_cast<int>(U.size());
    const double dx = (x1 - x0) / nx;

    WENO1d weno(3, dx);
    WENOWorkspace ws = weno.alloc_workspace(nx);
    std::vector<Physical>  Fbar(nx + 1, {0, 0, 0});
    std::vector<Conserved> Unew(nx);

    g_fallbackCount = 0; g_nanFallbackCount = 0; g_maxRelDiff = 0.0;

    double t = 0.0; int step = 0;
    std::cout << "[ADER4] nx=" << nx << " dx=" << dx
              << " Tfinal=" << Tfinal
              << " Nghost=" << weno.requiredGhost() << "\n";

    while (t < Tfinal - 1e-14) {
        double smax = 0.0;
        for (const auto& Ui : U) smax = std::max(smax, maxSignalSpeed(Ui));
        if (!std::isfinite(smax) || smax <= 0.0) smax = 2.0;

        double dt = CFL * dx / smax;
        if (t + dt > Tfinal) dt = Tfinal - t;

        ader4AdvanceFast(U, dx, dt, periodic, opt, weno, ws, Fbar, Unew);
        t += dt; ++step;

        if (step <= 5 || step % 100 == 0)
            std::cout << "  step " << step << "  t=" << t
                      << "  dt=" << dt << "  smax=" << smax << "\n";
    }
    std::cout << "[ADER4] done: " << step << " steps, t=" << t
              << " | fallbacks=" << g_fallbackCount
              << " (NaN=" << g_nanFallbackCount << ")"
              << " maxRelDiff=" << g_maxRelDiff << "\n";
}

}