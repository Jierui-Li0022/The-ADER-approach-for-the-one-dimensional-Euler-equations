// Convergence test for the WENO5 solver on the smooth advection problem.

#include "aderweno/test/task_3_1_weno5.hpp"
#include "aderweno/constants.hpp"
#include "aderweno/euler1d.hpp"
#include "aderweno/riemann_solver.hpp"
#include "aderweno/weno.hpp"

#include <cmath>
#include <sstream>
#include <vector>
#include <iostream>
#include <iomanip>
#include <algorithm>

namespace aderweno {

static constexpr double PI = 3.1415926535897932384626433832795;

static double rhoExactPoint(double x, double t) {
    const double s = std::sin(PI * (x - t));
    return 2.0 + s * s * s * s;
}

static double rhoExactCellavg(double xL, double xR, double t) {
    const double xi[5] = {
        -0.9061798459366350, -0.5384693101056831, 0.0,
         0.5384693101056831,  0.9061798459366350
    };
    const double w[5] = {
        0.2369268850561891, 0.4786286704993665, 0.5688888888888889,
        0.4786286704993665, 0.2369268850561891
    };
    const double xc = 0.5 * (xL + xR);
    const double h  = 0.5 * (xR - xL);
    double val = 0.0;
    for (int k = 0; k < 5; ++k)
        val += w[k] * rhoExactPoint(xc + h * xi[k], t);
    return 0.5 * val;
}

static Conserved makeConserved(double rho) {
    const double u = 1.0, p = 1.0;
    Conserved U;
    U.rho = rho;
    U.mom   = rho * u;
    U.E   = p / (Gamma - 1.0) + 0.5 * rho * u * u;
    return U;
}

static std::vector<Conserved> initCellAverages(int nx, double x0, double x1) {
    const double dx = (x1 - x0) / nx;
    std::vector<Conserved> U(nx);
    for (int i = 0; i < nx; ++i) {
        const double xL = x0 + i * dx;
        const double xR = xL + dx;
        U[i] = makeConserved(rhoExactCellavg(xL, xR, 0.0));
    }
    return U;
}

static void enforce_physical(Conserved& U) {
    const double rho_min = 0.5;
    const double p_min   = 0.5;
    if (!std::isfinite(U.rho) || U.rho < rho_min) U.rho = rho_min;
    const double u = U.mom / U.rho;
    if (!std::isfinite(u)) U.mom = U.rho;
    const double Emin = 0.5 * (U.mom * U.mom) / U.rho + p_min / (Gamma - 1.0);
    if (!std::isfinite(U.E) || U.E < Emin) U.E = Emin;
}

static std::vector<Conserved> compute_rhs(const std::vector<Conserved>& U,
                                          double dx, WENO1d& weno) {
    const int nx = (int)U.size();

    std::vector<InterfacePoly> PolyL, PolyR;
    weno.reconstruct_all_interfaces(U, PolyL, PolyR, WenoBCType::Periodic);

    std::vector<Physical> F(nx);
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < nx; ++i) {
        Conserved UL = PolyL[i].Q;
        Conserved UR = PolyR[i].Q;
        enforce_physical(UL);
        enforce_physical(UR);
        F[i] = hllcFlux(UL, UR);
    }

    std::vector<Conserved> L(nx);
    const double coeff = 1.0 / dx;
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < nx; ++i) {
        const int ip1 = (i + 1) % nx;
        L[i].rho = -coeff * (F[ip1].fluxRho - F[i].fluxRho);
        L[i].mom   = -coeff * (F[ip1].fluxMom - F[i].fluxMom);
        L[i].E   = -coeff * (F[ip1].fluxE - F[i].fluxE);
    }
    return L;
}

static void advance_ssprk3(std::vector<Conserved>& U,
                           double dx, double dt, WENO1d& weno) {
    const int nx = (int)U.size();

    auto combine = [&](double a, const std::vector<Conserved>& A,
                       double b, const std::vector<Conserved>& B,
                       const std::vector<Conserved>& LB) -> std::vector<Conserved> {
        std::vector<Conserved> out(nx);
        for (int i = 0; i < nx; ++i) {
            out[i].rho = a * A[i].rho + b * (B[i].rho + dt * LB[i].rho);
            out[i].mom   = a * A[i].mom   + b * (B[i].mom   + dt * LB[i].mom);
            out[i].E   = a * A[i].E   + b * (B[i].E   + dt * LB[i].E);
            enforce_physical(out[i]);
        }
        return out;
    };

    std::vector<Conserved> L0 = compute_rhs(U, dx, weno);
    std::vector<Conserved> U1 = combine(0.0, U, 1.0, U, L0);

    std::vector<Conserved> L1 = compute_rhs(U1, dx, weno);
    std::vector<Conserved> U2 = combine(0.75, U, 0.25, U1, L1);

    std::vector<Conserved> L2 = compute_rhs(U2, dx, weno);
    U = combine(1.0 / 3.0, U, 2.0 / 3.0, U2, L2);
}

static void run_to_time(std::vector<Conserved>& U,
                        double x0, double x1,
                        double Tfinal, double CFL,
                        int weno_degree) {
    const int nx = (int)U.size();
    const double dx = (x1 - x0) / nx;
    WENO1d weno(weno_degree, dx);

    double t = 0.0;
    int n = 0;

    while (t < Tfinal - 1e-14) {
        double smax = 0.0;
        for (const auto& Ui : U)
            smax = std::max(smax, maxSignalSpeed(Ui));
        if (!std::isfinite(smax) || smax <= 0.0) smax = 2.0;

        double dt = CFL * dx / smax;
        if (t + dt > Tfinal) dt = Tfinal - t;

        if (n % 50 == 0)
            std::cout << "[INFO] step=" << n
                      << " t=" << t
                      << " dt=" << dt
                      << " smax=" << smax << "\n";

        advance_ssprk3(U, dx, dt, weno);
        t += dt;
        ++n;
    }
}

static void errorNorms(const std::vector<Conserved>& U,
                        double x0, double x1, double T,
                        double& L1, double& L2, double& Linf) {
    const int nx = (int)U.size();
    const double dx = (x1 - x0) / nx;
    L1 = L2 = Linf = 0.0;
    for (int i = 0; i < nx; ++i) {
        const double e = std::abs(U[i].rho
            - rhoExactCellavg(x0 + i * dx, x0 + (i + 1) * dx, T));
        L1   += e * dx;
        L2   += e * e * dx;
        Linf  = std::max(Linf, e);
    }
    L2 = std::sqrt(L2);
}

void task_3_1_weno5() {
    const double x0 = -1.0, x1 = 1.0, T = 2.0, CFL = 0.95;
    const int weno_degree = 2;  // M=2 -> WENO5

    const std::vector<int> nxs = {20, 40, 80, 160, 320, 640, 1280, 2560};

    struct Row { int nx; double dx, L1, L2, Linf, p1, p2, pinf; };
    std::vector<Row> table;

    for (int nx : nxs) {
        const double dx = (x1 - x0) / nx;
        std::cout << "\n===== nx = " << nx << " =====\n";

        std::vector<Conserved> U = initCellAverages(nx, x0, x1);

        {
            double L1_0, L2_0, Li_0;
            errorNorms(U, x0, x1, 0.0, L1_0, L2_0, Li_0);
            std::cout << "[CHECK T=0] L1=" << L1_0
                      << " L2=" << L2_0
                      << " Linf=" << Li_0 << "\n";
        }

        run_to_time(U, x0, x1, T, CFL, weno_degree);

        double L1, L2, Linf;
        errorNorms(U, x0, x1, T, L1, L2, Linf);
        table.push_back({nx, dx, L1, L2, Linf, 0.0, 0.0, 0.0});
    }

    for (int k = 1; k < (int)table.size(); ++k) {
        const double r = std::log(table[k-1].dx / table[k].dx);
        table[k].p1   = std::log(table[k-1].L1   / table[k].L1)   / r;
        table[k].p2   = std::log(table[k-1].L2   / table[k].L2)   / r;
        table[k].pinf = std::log(table[k-1].Linf / table[k].Linf) / r;
    }

    std::cout << "\nTask 3.1: Empirical convergence (WENO-DK5 + SSP-RK3 + HLLC)\n";
    std::cout << "rho(x,t) = 2+sin^4(pi*(x-t)), u=p=1, periodic, domain [-1,1], T=2\n\n";

    std::cout << std::setw(6)  << "N"
              << std::setw(12) << "dx"
              << std::setw(16) << "L_inf error"
              << std::setw(10) << "order"
              << std::setw(16) << "L1 error"
              << std::setw(10) << "order"
              << std::setw(16) << "L2 error"
              << std::setw(10) << "order"
              << "\n";
    std::cout << std::string(96, '-') << "\n";

    for (int k = 0; k < (int)table.size(); ++k) {
        const auto& r = table[k];
        auto fmt = [&](double p) -> std::string {
            if (k == 0) return "    -";
            std::ostringstream oss;
            oss << std::fixed << std::setprecision(2) << p;
            return oss.str();
        };
        std::cout << std::setw(6)  << r.nx
                  << std::setw(12) << std::scientific << std::setprecision(2) << r.dx
                  << std::setw(16) << std::scientific << std::setprecision(4) << r.Linf
                  << std::setw(10) << fmt(r.pinf)
                  << std::setw(16) << std::scientific << std::setprecision(4) << r.L1
                  << std::setw(10) << fmt(r.p1)
                  << std::setw(16) << std::scientific << std::setprecision(4) << r.L2
                  << std::setw(10) << fmt(r.p2)
                  << "\n";
    }
    std::cout << std::endl;
}

}