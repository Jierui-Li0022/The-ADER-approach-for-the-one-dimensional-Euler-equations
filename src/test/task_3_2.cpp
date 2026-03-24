// Riemann shock-tube tests comparing Godunov, MUSCL, ADER3/4/5 against the exact solution.

#include "aderweno/constants.hpp"
#include "aderweno/euler1d.hpp"
#include "aderweno/riemann_solver.hpp"
#include "aderweno/weno.hpp"
#include "aderweno/tt_grp.hpp"
#include "aderweno/godunov_solver.hpp"
#include "aderweno/muscl_hancock_solver.hpp"
#include "aderweno/ader3_solver.hpp"
#include "aderweno/ader4_solver.hpp"
#include "aderweno/ader5_solver.hpp"

#include <cmath>
#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <algorithm>

struct PrimState { double rho, u, p; };

static PrimState exactRiemannSample(
    double rhoL, double uL, double pL,
    double rhoR, double uR, double pR,
    double xi)
{
    const double g   = Gamma;
    const double gm1 = g - 1.0;
    const double gp1 = g + 1.0;

    double p_star = solvePStar(uL, uR, rhoL, pL, rhoR, pR);
    double u_star = calcVStar(p_star, uL, rhoL, pL, uR, rhoR, pR);

    const double aL  = std::sqrt(g * pL / rhoL);
    const double aR  = std::sqrt(g * pR / rhoR);
    const double aLs = aL * std::pow(p_star / pL, gm1 / (2.0 * g));
    const double aRs = aR * std::pow(p_star / pR, gm1 / (2.0 * g));

    if (xi <= u_star) {
        if (p_star > pL) {
            const double SL = uL - aL * std::sqrt((gp1/(2.0*g)) * (p_star/pL) + gm1/(2.0*g));
            if (xi <= SL) return {rhoL, uL, pL};
            const double rhoLs = rhoL * (p_star/pL + gm1/gp1) / (gm1/gp1 * p_star/pL + 1.0);
            return {rhoLs, u_star, p_star};
        } else {
            const double SLh = uL - aL;
            const double SLt = u_star - aLs;
            if (xi <= SLh) return {rhoL, uL, pL};
            if (xi >= SLt) {
                const double rhoLs = rhoL * std::pow(p_star / pL, 1.0 / g);
                return {rhoLs, u_star, p_star};
            }
            const double coeff   = 2.0/gp1 + gm1/(gp1 * aL) * (uL - xi);
            const double rho_fan = rhoL * std::pow(coeff, 2.0 / gm1);
            const double u_fan   = 2.0/gp1 * (aL + gm1/2.0 * uL + xi);
            const double p_fan   = pL   * std::pow(coeff, 2.0*g / gm1);
            return {rho_fan, u_fan, p_fan};
        }
    } else {
        if (p_star > pR) {
            const double SR = uR + aR * std::sqrt((gp1/(2.0*g)) * (p_star/pR) + gm1/(2.0*g));
            if (xi >= SR) return {rhoR, uR, pR};
            const double rhoRs = rhoR * (p_star/pR + gm1/gp1) / (gm1/gp1 * p_star/pR + 1.0);
            return {rhoRs, u_star, p_star};
        } else {
            const double SRh = uR + aR;
            const double SRt = u_star + aRs;
            if (xi >= SRh) return {rhoR, uR, pR};
            if (xi <= SRt) {
                const double rhoRs = rhoR * std::pow(p_star / pR, 1.0 / g);
                return {rhoRs, u_star, p_star};
            }
            const double coeff   = 2.0/gp1 - gm1/(gp1 * aR) * (uR - xi);
            const double rho_fan = rhoR * std::pow(coeff, 2.0 / gm1);
            const double u_fan   = 2.0/gp1 * (-aR + gm1/2.0 * uR + xi);
            const double p_fan   = pR   * std::pow(coeff, 2.0*g / gm1);
            return {rho_fan, u_fan, p_fan};
        }
    }
}

static std::vector<PrimState> buildExactSolution(
    const std::vector<double>& xc,
    double rhoL, double uL, double pL,
    double rhoR, double uR, double pR,
    double x0_disc, double T)
{
    const int nx = static_cast<int>(xc.size());
    std::vector<PrimState> sol(nx);
    for (int i = 0; i < nx; ++i)
        sol[i] = exactRiemannSample(rhoL, uL, pL, rhoR, uR, pR,
                                      (xc[i] - x0_disc) / T);
    return sol;
}

static std::vector<Conserved> initRiemann(
    int nx, double x_start, double x_end,
    double rhoL, double uL, double pL,
    double rhoR, double uR, double pR,
    double x0_disc)
{
    const double dx = (x_end - x_start) / nx;
    std::vector<Conserved> U(nx);
    for (int i = 0; i < nx; ++i) {
        const double xc = x_start + (i + 0.5) * dx;
        U[i] = (xc <= x0_disc) ? primitiveToConserved(Primitive{rhoL, uL, pL})
                                : primitiveToConserved(Primitive{rhoR, uR, pR});
    }
    return U;
}

static std::vector<PrimState> extractPrim(const std::vector<Conserved>& U)
{
    std::vector<PrimState> W(U.size());
    for (int i = 0; i < static_cast<int>(U.size()); ++i) {
        Primitive p = conservedToPrimitive(U[i]);
        W[i] = {p.rho, p.u, p.p};
    }
    return W;
}

static void printL1Errors(const std::string& label,
                             const std::vector<PrimState>& num,
                             const std::vector<PrimState>& exact,
                             double dx)
{
    double L1_rho = 0, L1_u = 0, L1_p = 0;
    for (int i = 0; i < static_cast<int>(num.size()); ++i) {
        L1_rho += std::abs(num[i].rho - exact[i].rho) * dx;
        L1_u   += std::abs(num[i].u   - exact[i].u  ) * dx;
        L1_p   += std::abs(num[i].p   - exact[i].p  ) * dx;
    }
    std::cout << std::left << std::setw(8) << label
              << "  L1(rho)=" << std::scientific << std::setprecision(4) << L1_rho
              << "  L1(u)="   << L1_u
              << "  L1(p)="   << L1_p << "\n";
}

static void writeCsv(const std::string& filename,
                      const std::vector<double>&    xc,
                      const std::vector<PrimState>& exact,
                      const std::vector<PrimState>& god1,
                      const std::vector<PrimState>& muscl,
                      const std::vector<PrimState>& ader3,
                      const std::vector<PrimState>& ader4,
                      const std::vector<PrimState>& ader5)
{
    std::ofstream f(filename);
    if (!f.is_open()) {
        std::cerr << "[task_3_2] Cannot open " << filename << "\n";
        return;
    }
    f << "x,rho_exact,u_exact,p_exact,"
      << "rho_god1,u_god1,p_god1,"
      << "rho_muscl,u_muscl,p_muscl,"
      << "rho_ader3,u_ader3,p_ader3,"
      << "rho_ader4,u_ader4,p_ader4,"
      << "rho_ader5,u_ader5,p_ader5\n";
    f << std::scientific << std::setprecision(10);
    for (int i = 0; i < static_cast<int>(xc.size()); ++i) {
        f << xc[i]        << ","
          << exact[i].rho << "," << exact[i].u << "," << exact[i].p << ","
          << god1[i].rho  << "," << god1[i].u  << "," << god1[i].p  << ","
          << muscl[i].rho << "," << muscl[i].u << "," << muscl[i].p << ","
          << ader3[i].rho << "," << ader3[i].u << "," << ader3[i].p << ","
          << ader4[i].rho << "," << ader4[i].u << "," << ader4[i].p << ","
          << ader5[i].rho << "," << ader5[i].u << "," << ader5[i].p << "\n";
    }
    std::cout << "  wrote " << filename << "\n";
}

struct RiemannTestCase {
    std::string name;
    double rhoL, uL, pL;
    double rhoR, uR, pR;
    double x0_disc;
    double T_out;
};

static void runTest(const RiemannTestCase& tc, int nx = 355)
{
    std::cout << "\n===== " << tc.name << " =====\n";

    const double x_start = 0.0, x_end = 1.0;
    const double dx = (x_end - x_start) / nx;

    const double CFL_god   = 0.9;
    const double CFL_muscl = 0.9;
    const double CFL_ader3 = 0.9;
    const double CFL_ader4 = 0.9;
    const double CFL_ader5 = 0.9;

    std::vector<double> xc(nx);
    for (int i = 0; i < nx; ++i)
        xc[i] = x_start + (i + 0.5) * dx;

    const auto W_exact = buildExactSolution(xc,
                             tc.rhoL, tc.uL, tc.pL,
                             tc.rhoR, tc.uR, tc.pR,
                             tc.x0_disc, tc.T_out);

    TTGRPOptions opt{};
    opt.use_exact_q0 = false;
    opt.flux_type    = TTRiemannFluxType::HLLC;

    std::cout << "[Godunov] CFL=" << CFL_god << "\n";
    auto Ug = initRiemann(nx, x_start, x_end,
                           tc.rhoL, tc.uL, tc.pL,
                           tc.rhoR, tc.uR, tc.pR, tc.x0_disc);
    aderweno::godunovRun(Ug, x_start, x_end, tc.T_out, CFL_god, false);
    const auto Wg = extractPrim(Ug);

    std::cout << "[MUSCL] CFL=" << CFL_muscl << "\n";
    auto Um = initRiemann(nx, x_start, x_end,
                           tc.rhoL, tc.uL, tc.pL,
                           tc.rhoR, tc.uR, tc.pR, tc.x0_disc);
    aderweno::musclHancockRun(Um, x_start, x_end, tc.T_out, CFL_muscl, false,
                                aderweno::LimiterType::VanLeer);
    const auto Wm = extractPrim(Um);

    auto U3 = initRiemann(nx, x_start, x_end,
                           tc.rhoL, tc.uL, tc.pL,
                           tc.rhoR, tc.uR, tc.pR, tc.x0_disc);
    aderweno::ader3Run(U3, x_start, x_end, tc.T_out, CFL_ader3, false, opt);
    const auto W3 = extractPrim(U3);

    auto U4 = initRiemann(nx, x_start, x_end,
                           tc.rhoL, tc.uL, tc.pL,
                           tc.rhoR, tc.uR, tc.pR, tc.x0_disc);
    aderweno::ader4Run(U4, x_start, x_end, tc.T_out, CFL_ader4, false, opt);
    const auto W4 = extractPrim(U4);

    auto U5 = initRiemann(nx, x_start, x_end,
                           tc.rhoL, tc.uL, tc.pL,
                           tc.rhoR, tc.uR, tc.pR, tc.x0_disc);
    aderweno::ader5Run(U5, x_start, x_end, tc.T_out, CFL_ader5, false, opt);
    const auto W5 = extractPrim(U5);

    std::cout << "\nL1 errors vs exact:\n";
    printL1Errors("Godunov", Wg, W_exact, dx);
    printL1Errors("MUSCL",   Wm, W_exact, dx);
    printL1Errors("ADER3",   W3, W_exact, dx);
    printL1Errors("ADER4",   W4, W_exact, dx);
    printL1Errors("ADER5",   W5, W_exact, dx);

    writeCsv("task_3_2_" + tc.name + ".csv", xc, W_exact, Wg, Wm, W3, W4, W5);
}

void task_3_2()
{
    const RiemannTestCase sod = {
        "sod",
        1.0,   0.0,   1.0,
        0.125, 0.0,   0.1,
        0.5,   0.2
    };

    const RiemannTestCase lax = {
        "lax",
        0.445, 0.698, 3.528,
        0.5,   0.0,   0.571,
        0.6,   0.14
    };

    runTest(sod);
    runTest(lax);
}