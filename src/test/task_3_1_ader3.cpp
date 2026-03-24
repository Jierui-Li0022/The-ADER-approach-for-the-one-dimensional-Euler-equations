// Convergence test for the ADER3 solver on the smooth advection problem.

#include "aderweno/test/task_3_1_ader3.hpp"
#include "aderweno/constants.hpp"
#include "aderweno/euler1d.hpp"
#include "aderweno/ader3_solver.hpp"

#include <cmath>
#include <vector>
#include <iostream>
#include <iomanip>
#include <sstream>
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

static Conserved makeConserved(double rho, double u, double p) {
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
        U[i] = makeConserved(rhoExactCellavg(xL, xR, 0.0), 1.0, 1.0);
    }
    return U;
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

void task_3_1_ader3() {
    const double x0 = -1.0, x1 = 1.0, T = 2.0, CFL = 0.95;

    TTGRPOptions ttopt;
    ttopt.use_exact_q0 = true;
    ttopt.flux_type    = TTRiemannFluxType::HLLC;

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

        aderweno::ader3Run(U, x0, x1, T, CFL, true, ttopt);

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

    std::cout << "\nTask 3.1: Empirical convergence (ADER3 + WENO-DK M=2 + CK order 2 + TT-GRP)\n";
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