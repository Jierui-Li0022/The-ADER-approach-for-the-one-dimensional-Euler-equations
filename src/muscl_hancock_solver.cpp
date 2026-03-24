// Second-order MUSCL-Hancock solver with minmod / van Leer / MC slope limiting.

#include "aderweno/muscl_hancock_solver.hpp"
#include "aderweno/riemann_solver.hpp"
#include "aderweno/euler1d.hpp"
#include "aderweno/constants.hpp"
#include "aderweno/checks.hpp"
#include <algorithm>
#include <cmath>
#include <vector>

namespace aderweno {

static inline double sgn(double a) { return (a > 0) - (a < 0); }

static inline double minmod2(double a, double b)
{
    if (a * b <= 0.0) return 0.0;
    return sgn(a) * std::min(std::abs(a), std::abs(b));
}

static inline double vanleer(double a, double b)
{
    if (a * b <= 0.0) return 0.0;
    return 2.0 * a * b / (a + b);
}

static inline double mc(double a, double b)
{
    const double t1 = 0.5 * (a + b);
    return minmod2(t1, minmod2(2.0 * a, 2.0 * b));
}

static inline double limitSlope(double dl, double dr, LimiterType lt)
{
    if (lt == LimiterType::Minmod)  return minmod2(dl, dr);
    if (lt == LimiterType::VanLeer) return vanleer(dl, dr);
    return mc(dl, dr);
}

static inline Primitive clampPrimitive(const Primitive& W)
{
    return Primitive{
        (std::isfinite(W.rho) && W.rho >= RhoMin) ? W.rho : RhoMin,
        std::isfinite(W.u)   ? W.u   : 0.0,
        (std::isfinite(W.p)  && W.p  >= PMin)  ? W.p  : PMin
    };
}

void musclHancockAdvance(std::vector<Conserved>& U,
                         double dx, double dt,
                         bool periodic,
                         LimiterType limiter)
{
    const int nx = static_cast<int>(U.size());
    if (nx <= 1) return;

    std::vector<Primitive> W(nx);
    for (int i = 0; i < nx; ++i) W[i] = conservedToPrimitive(U[i]);

    std::vector<Primitive> slope(nx, Primitive{0.0, 0.0, 0.0});
    for (int i = 0; i < nx; ++i) {
        int im1, ip1;
        if (periodic) {
            im1 = (i - 1 + nx) % nx;
            ip1 = (i + 1) % nx;
        } else {
            if (i == 0 || i == nx - 1) continue;
            im1 = i - 1; ip1 = i + 1;
        }
        slope[i].rho = limitSlope(W[i].rho - W[im1].rho, W[ip1].rho - W[i].rho, limiter);
        slope[i].u   = limitSlope(W[i].u   - W[im1].u,   W[ip1].u   - W[i].u,   limiter);
        slope[i].p   = limitSlope(W[i].p   - W[im1].p,   W[ip1].p   - W[i].p,   limiter);
    }

    // Half-step predictor in primitive variables
    std::vector<Conserved> UL(nx), UR(nx);
    for (int i = 0; i < nx; ++i) {
        Primitive wL = clampPrimitive({W[i].rho - 0.5*slope[i].rho,
                                       W[i].u   - 0.5*slope[i].u,
                                       W[i].p   - 0.5*slope[i].p});
        Primitive wR = clampPrimitive({W[i].rho + 0.5*slope[i].rho,
                                       W[i].u   + 0.5*slope[i].u,
                                       W[i].p   + 0.5*slope[i].p});

        // Inline flux for the half-step update
        const double EL = wL.p/(Gamma-1.0) + 0.5*wL.rho*wL.u*wL.u;
        const double ER = wR.p/(Gamma-1.0) + 0.5*wR.rho*wR.u*wR.u;
        const Physical fL = {wL.rho*wL.u, wL.rho*wL.u*wL.u+wL.p, (EL+wL.p)*wL.u};
        const Physical fR = {wR.rho*wR.u, wR.rho*wR.u*wR.u+wR.p, (ER+wR.p)*wR.u};

        const double half = dt / (2.0 * dx);
        Conserved UL0 = primitiveToConserved(wL);
        Conserved UR0 = primitiveToConserved(wR);

        UL[i].rho = UL0.rho + half * (fL.fluxRho - fR.fluxRho);
        UL[i].mom = UL0.mom + half * (fL.fluxMom - fR.fluxMom);
        UL[i].E   = UL0.E   + half * (fL.fluxE   - fR.fluxE);

        UR[i].rho = UR0.rho + half * (fL.fluxRho - fR.fluxRho);
        UR[i].mom = UR0.mom + half * (fL.fluxMom - fR.fluxMom);
        UR[i].E   = UR0.E   + half * (fL.fluxE   - fR.fluxE);

        enforcePhysical(UL[i]);
        enforcePhysical(UR[i]);
    }

    // Flux computation and conservative update
    std::vector<Physical> F(nx + 1);
    if (periodic) {
        F[0] = hllcFlux(UR[nx-1], UL[0]);
        for (int i = 0; i < nx-1; ++i) F[i+1] = hllcFlux(UR[i], UL[i+1]);
        F[nx] = F[0];
    } else {
        F[0]  = hllcFlux(UL[0],      UL[0]);
        for (int i = 0; i < nx-1; ++i) F[i+1] = hllcFlux(UR[i], UL[i+1]);
        F[nx] = hllcFlux(UR[nx-1], UR[nx-1]);
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

void musclHancockRun(std::vector<Conserved>& U,
                     double x0, double x1,
                     double Tfinal, double CFL,
                     bool periodic,
                     LimiterType limiter)
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

        musclHancockAdvance(U, dx, dt, periodic, limiter);
        t += dt;
    }
}

}