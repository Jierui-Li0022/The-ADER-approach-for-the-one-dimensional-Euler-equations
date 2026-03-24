// Finite volume update steps for the 1D Euler equations.

#include "aderweno/fv_solver.hpp"
#include "aderweno/riemann_solver.hpp"
#include "aderweno/checks.hpp"
#include "aderweno/constants.hpp"
#include <cmath>
#include <iostream>
#include <vector>

void setBoundaryConditions(std::vector<Conserved>& U)
{
    const int nx = static_cast<int>(U.size());
    U[0]      = U[1];
    U[nx - 1] = U[nx - 2];
}

void laxFriedrichsUpdate(std::vector<Conserved>& U, double dx, double dt)
{
    const int nx = static_cast<int>(U.size());
    std::vector<Conserved> Unew = U;

    for (int i = 1; i < nx - 1; ++i) {
        if (!checkConserved(U[i])) {
            std::cerr << "[laxFriedrichs] invalid state at i=" << i << "\n";
            return;
        }
        const Physical fL = laxFriedrichsFlux(U[i-1], U[i]);
        const Physical fR = laxFriedrichsFlux(U[i],   U[i+1]);
        Unew[i].rho = U[i].rho - (dt/dx) * (fR.fluxRho - fL.fluxRho);
        Unew[i].mom = U[i].mom - (dt/dx) * (fR.fluxMom - fL.fluxMom);
        Unew[i].E   = U[i].E   - (dt/dx) * (fR.fluxE   - fL.fluxE);
    }
    U.swap(Unew);
}

void hllUpdate(std::vector<Conserved>& U, double dx, double dt)
{
    const int nx = static_cast<int>(U.size());
    std::vector<Conserved> Unew = U;

    for (int i = 1; i < nx - 1; ++i) {
        if (!checkConserved(U[i])) {
            std::cerr << "[hllUpdate] invalid state at i=" << i << "\n";
            return;
        }
        const Physical fL = hllFlux(U[i-1], U[i]);
        const Physical fR = hllFlux(U[i],   U[i+1]);
        Unew[i].rho = U[i].rho - (dt/dx) * (fR.fluxRho - fL.fluxRho);
        Unew[i].mom = U[i].mom - (dt/dx) * (fR.fluxMom - fL.fluxMom);
        Unew[i].E   = U[i].E   - (dt/dx) * (fR.fluxE   - fL.fluxE);
    }
    U.swap(Unew);
}

void hllcUpdate(std::vector<Conserved>& U, double dx, double dt)
{
    const int nx = static_cast<int>(U.size());
    std::vector<Conserved> Unew = U;

    for (int i = 1; i < nx - 1; ++i) {
        if (!checkConserved(U[i])) {
            std::cerr << "[hllcUpdate] invalid state at i=" << i << "\n";
            return;
        }
        const Physical fL = hllcFlux(U[i-1], U[i]);
        const Physical fR = hllcFlux(U[i],   U[i+1]);
        Unew[i].rho = U[i].rho - (dt/dx) * (fR.fluxRho - fL.fluxRho);
        Unew[i].mom = U[i].mom - (dt/dx) * (fR.fluxMom - fL.fluxMom);
        Unew[i].E   = U[i].E   - (dt/dx) * (fR.fluxE   - fL.fluxE);
    }
    U.swap(Unew);
}

void exactUpdate(std::vector<Conserved>& U, double dx, double dt)
{
    const int nx = static_cast<int>(U.size());
    setBoundaryConditions(U);
    std::vector<Conserved> Unew = U;

    for (int i = 1; i < nx - 1; ++i) {
        if (!checkConserved(U[i-1]) || !checkConserved(U[i]) || !checkConserved(U[i+1])) {
            std::cerr << "[exactUpdate] invalid stencil at i=" << i << "\n";
            std::exit(1);
        }
        const Physical fL = exactFlux(U[i-1], U[i]);
        const Physical fR = exactFlux(U[i],   U[i+1]);

        Unew[i].rho = U[i].rho - (dt/dx) * (fR.fluxRho - fL.fluxRho);
        Unew[i].mom = U[i].mom - (dt/dx) * (fR.fluxMom - fL.fluxMom);
        Unew[i].E   = U[i].E   - (dt/dx) * (fR.fluxE   - fL.fluxE);
        enforcePhysical(Unew[i]);

        if (!checkConserved(Unew[i])) {
            std::cerr << "\n[exactUpdate] UPDATED STATE INVALID at i=" << i << "\n";
            std::cerr << "dt=" << dt << " dx=" << dx << "\n";
            std::cerr << "U[i-1]: rho=" << U[i-1].rho << " mom=" << U[i-1].mom << " E=" << U[i-1].E << "\n";
            std::cerr << "U[i]  : rho=" << U[i].rho   << " mom=" << U[i].mom   << " E=" << U[i].E   << "\n";
            std::cerr << "U[i+1]: rho=" << U[i+1].rho << " mom=" << U[i+1].mom << " E=" << U[i+1].E << "\n";
            std::cerr << "FluxL : fluxRho=" << fL.fluxRho << " fluxMom=" << fL.fluxMom << " fluxE=" << fL.fluxE << "\n";
            std::cerr << "FluxR : fluxRho=" << fR.fluxRho << " fluxMom=" << fR.fluxMom << " fluxE=" << fR.fluxE << "\n";
            std::cerr << "Unew  : rho=" << Unew[i].rho << " mom=" << Unew[i].mom << " E=" << Unew[i].E << "\n\n";
            std::exit(1);
        }
    }
    U.swap(Unew);
    setBoundaryConditions(U);
}