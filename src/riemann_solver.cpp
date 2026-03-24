// Approximate (Lax-Friedrichs, HLL, HLLC) and exact Riemann solvers for the 1D Euler equations.


#include "aderweno/riemann_solver.hpp"
#include "aderweno/euler1d.hpp"
#include "aderweno/constants.hpp"
#include "aderweno/checks.hpp"

#include <cmath>
#include <iostream>
#include <algorithm>


// Fast inline helpers — avoid repeated pressure() calls in the hot path.

static inline double fast_pressure(double rho, double m, double E)
{
    const double rho_safe = (rho > 1e-12) ? rho : 1e-12;
    double p = (Gamma - 1.0) * (E - 0.5 * m * m / rho_safe);
    return (p > 1e-12) ? p : 1e-12;
}

static inline double fast_sound_speed(double rho, double p)
{
    return std::sqrt(Gamma * p / rho);
}


/*
Aproximate Riemann solver for the Euler equations


*/

// Lax-Friedrichs numerical flux.
Physical laxFriedrichsFlux(const Conserved& UL, const Conserved& UR) {
    Physical fL = physicalFlux(UL);
    Physical fR = physicalFlux(UR);
    double smax = std::max(maxSignalSpeed(UL), maxSignalSpeed(UR));

    Physical result;
    result.fluxRho = 0.5 * (fL.fluxRho + fR.fluxRho) - 0.5 * smax * (UR.rho - UL.rho);
    result.fluxMom = 0.5 * (fL.fluxMom + fR.fluxMom) - 0.5 * smax * (UR.mom - UL.mom);
    result.fluxE = 0.5 * (fL.fluxE + fR.fluxE) - 0.5 * smax * (UR.E - UL.E);
    return result;
}


// HLL numerical flux.
Physical hllFlux(const Conserved& UL, const Conserved& UR) {

    const double rhoL_s = (UL.rho > 1e-12) ? UL.rho : 1e-12;
    const double rhoR_s = (UR.rho > 1e-12) ? UR.rho : 1e-12;
    const double uL_s = UL.mom / rhoL_s;
    const double uR_s = UR.mom / rhoR_s;
    const double pL_s = fast_pressure(rhoL_s, UL.mom, UL.E);
    const double pR_s = fast_pressure(rhoR_s, UR.mom, UR.E);
    const double aL_s = fast_sound_speed(rhoL_s, pL_s);
    const double aR_s = fast_sound_speed(rhoR_s, pR_s);

    double sL = std::min(uL_s - aL_s, uR_s - aR_s);
    double sR = std::max(uL_s + aL_s, uR_s + aR_s);
    const Physical fL = {UL.mom, UL.mom * uL_s + pL_s, (UL.E + pL_s) * uL_s};
    const Physical fR = {UR.mom, UR.mom * uR_s + pR_s, (UR.E + pR_s) * uR_s};
    
    if (sL >= 0) {
        return fL; // Flux from the left state
    } 

    if (sR <= 0) {
        return fR;
    }

    const double inv_ds = 1.0 / (sR - sL);
    Physical result;
    result.fluxRho = (sR * fL.fluxRho - sL * fR.fluxRho + sL * sR * (UR.rho - UL.rho)) * inv_ds;
    result.fluxMom = (sR * fL.fluxMom - sL * fR.fluxMom + sL * sR * (UR.mom - UL.mom)) * inv_ds;
    result.fluxE   = (sR * fL.fluxE   - sL * fR.fluxE   + sL * sR * (UR.E   - UL.E))   * inv_ds;
    return result;
}


// HLLC numerical flux.
Physical hllcFlux(const Conserved& UL, const Conserved& UR)
{
    const double rhoL_s = (UL.rho > 1e-12) ? UL.rho : 1e-12;
    const double rhoR_s = (UR.rho > 1e-12) ? UR.rho : 1e-12;
    const double uL_s = UL.mom / rhoL_s;
    const double uR_s = UR.mom / rhoR_s;
    const double pL_s = fast_pressure(rhoL_s, UL.mom, UL.E);
    const double pR_s = fast_pressure(rhoR_s, UR.mom, UR.E);
    const double aL_s = fast_sound_speed(rhoL_s, pL_s);
    const double aR_s = fast_sound_speed(rhoR_s, pR_s);

    const double sL = std::min(uL_s - aL_s, uR_s - aR_s);
    const double sR = std::max(uL_s + aL_s, uR_s + aR_s);
    const Physical fL = {UL.mom, UL.mom * uL_s + pL_s, (UL.E + pL_s) * uL_s};
    const Physical fR = {UR.mom, UR.mom * uR_s + pR_s, (UR.E + pR_s) * uR_s};

    if (sL >= 0.0) return fL;
    if (sR <= 0.0) return fR;

    const double denom = rhoL_s * (sL - uL_s) - rhoR_s * (sR - uR_s);
    const double sM = (std::abs(denom) < 1e-14)
        ? 0.5 * (uL_s + uR_s)
        : (pR_s - pL_s + rhoL_s*uL_s*(sL - uL_s) - rhoR_s*uR_s*(sR - uR_s)) / denom;

    if (sM >= 0.0) {
        // Left star state
        const double fac = rhoL_s * (sL - uL_s) / (sL - sM);
        const double Us_rho = fac;
        const double Us_mom   = fac * sM;
        const double Us_E   = fac * (UL.E/rhoL_s + (sM - uL_s) * (sM + pL_s/(rhoL_s*(sL - uL_s))));

        Physical Fs;
        Fs.fluxRho = fL.fluxRho + sL * (Us_rho - UL.rho);
        Fs.fluxMom = fL.fluxMom + sL * (Us_mom   - UL.mom);
        Fs.fluxE = fL.fluxE + sL * (Us_E   - UL.E);
        return Fs;
    } else {
        // Right star state
        const double fac = rhoR_s * (sR - uR_s) / (sR - sM);
        const double Us_rho = fac;
        const double Us_mom   = fac * sM;
        const double Us_E   = fac * (UR.E/rhoR_s + (sM - uR_s) * (sM + pR_s/(rhoR_s*(sR - uR_s))));

        Physical Fs;
        Fs.fluxRho = fR.fluxRho + sR * (Us_rho - UR.rho);
        Fs.fluxMom = fR.fluxMom + sR * (Us_mom   - UR.mom);
        Fs.fluxE = fR.fluxE + sR * (Us_E   - UR.E);
        return Fs;
    }
}

enum class ApproximateSolverType {
    LaxFriedrichs,
    HLL,
    HLLC
};


// Dispatch to the selected approximate Riemann solver.
Physical approximate_riemann_flux(const Conserved& UL, const Conserved& UR, ApproximateSolverType solver_type) {
    switch (solver_type) {
        case ApproximateSolverType::LaxFriedrichs:
            return laxFriedrichsFlux(UL, UR);
        case ApproximateSolverType::HLL:
            return hllFlux(UL, UR);
        case ApproximateSolverType::HLLC:
            return hllcFlux(UL, UR);
    }
}


// Exact Riemann solver


double fk(double p_star, double v_k, double rho_k, double p_k) {
    if (!std::isfinite(p_star)) p_star = PMin;
    if (!std::isfinite(p_k) || p_k < PMin) p_k = PMin;
    if (!std::isfinite(rho_k) || rho_k < 1e-12) rho_k = 1e-12;

    const double c_k = std::sqrt(Gamma * p_k / rho_k);

    const double A_k = 2.0 / ((Gamma + 1.0) * rho_k);
    const double B_k = (Gamma - 1.0) / (Gamma + 1.0) * p_k;

    if (p_star >= p_k) {
        // shock
        return (p_star - p_k) * std::sqrt(A_k / (p_star + B_k));
    } else {
        // rarefaction
        return 2.0 * c_k / (Gamma - 1.0) *
               (std::pow(p_star / p_k, (Gamma - 1.0) / (2.0 * Gamma)) - 1.0);
    }
}


double dfkDp(double p_star, double v_k, double rho_k, double p_k) {
    if (!std::isfinite(p_star)) p_star = PMin;
    if (!std::isfinite(p_k) || p_k < PMin) p_k = PMin;
    if (!std::isfinite(rho_k) || rho_k < 1e-12) rho_k = 1e-12;

    const double c_k = std::sqrt(Gamma * p_k / rho_k);

    const double A_k = 2.0 / ((Gamma + 1.0) * rho_k);
    const double B_k = (Gamma - 1.0) / (Gamma + 1.0) * p_k;

    if (p_star >= p_k) {
        // shock
        return std::sqrt(A_k / (p_star + B_k)) *
               (1.0 - 0.5 * (p_star - p_k) / (p_star + B_k));
    } else {
        // rarefaction
        return 1.0 / (rho_k * c_k) *
               std::pow(p_star / p_k, -(Gamma + 1.0) / (2.0 * Gamma));
    }
}


// Solve for star-region pressure p* using Newton-Raphson iteration (Toro 2009).
double solvePStar(double vL, double vR, double rhoL, double pL, double rhoR, double pR) {
    const double p_max = 1e6;

    if (!std::isfinite(pL) || pL < PMin) pL = PMin;
    if (!std::isfinite(pR) || pR < PMin) pR = PMin;
    if (!std::isfinite(rhoL) || rhoL < 1e-12) rhoL = 1e-12;
    if (!std::isfinite(rhoR) || rhoR < 1e-12) rhoR = 1e-12;
    const double aL = std::sqrt(Gamma * pL / rhoL);
    const double aR = std::sqrt(Gamma * pR / rhoR);
    double p0 = 0.5*(pL + pR) - 0.125*(vR - vL)*(rhoL + rhoR)*(aL + aR);
    double p_star = std::clamp(std::max(p0, PMin), PMin, p_max);

    const int max_iter = 50;
    for (int iter = 0; iter < max_iter; ++iter) {
        const double f  = fk(p_star, vL, rhoL, pL) + fk(p_star, vR, rhoR, pR) + (vR - vL);
        const double df = dfkDp(p_star, vL, rhoL, pL) + dfkDp(p_star, vR, rhoR, pR);

        if (!std::isfinite(f) || !std::isfinite(df) || std::abs(df) < 1e-14) {
            std::cerr << "[Newton] invalid f/df at iter=" << iter
                      << " p*=" << p_star << " f=" << f << " df=" << df << "\n";
            return p_star;
        }

        const double p_old = p_star;
        double p_new = p_old - f / df;

        if (!std::isfinite(p_new)) return p_old;

        p_new = std::max(p_new, PMin);
        p_new = std::clamp(p_new, 0.1 * p_old, 10.0 * p_old);
        p_new = std::min(p_new, p_max);

        if (std::abs(p_new - p_old) <= 1e-8 * std::max(PMin, p_new)) {
            return p_new;
        }

        p_star = p_new;
    }

    return p_star;
}


double calcVStar(double p_star,
                        double vL, double rhoL, double pL,
                        double vR, double rhoR, double pR) {
    return 0.5 * (vL + vR)
         + 0.5 * (fk(p_star, vR, rhoR, pR) - fk(p_star, vL, rhoL, pL));
}


// Star-region density for a rarefaction wave.
double calculate_rarefaction_rho_star(double p_star, double rho_k, double p_k) {
    return rho_k * std::pow(p_star / p_k, 1 / Gamma);    
}

// Star-region density for a shock wave.
double calculate_shock_rho_star(double p_star, double rho_k, double p_k) {
    return rho_k * ((p_star / p_k + (Gamma - 1) / (Gamma + 1)) / ((Gamma - 1) / (Gamma + 1) * p_star / p_k + 1));
}


double rho_left_star(double p_star, double vL, double rhoL, double pL){
    if (p_star > pL) {
        return calculate_shock_rho_star(p_star, rhoL, pL);
    }
    if (p_star < pL)
    {
        return calculate_rarefaction_rho_star(p_star, rhoL, pL);
    }

    return rhoL;
}

double rho_right_star(double p_star, double vR, double rhoR, double pR){
    if (p_star > pR) {
        return calculate_shock_rho_star(p_star, rhoR, pR);
    }
    if (p_star < pR)
    {
        return calculate_rarefaction_rho_star(p_star, rhoR, pR);
    }

    return rhoR;
}


bool is_interface_on_right(double v_star) {
    return v_star > 0;
}

bool is_interface_on_left(double v_star) {
    return v_star < 0;
}

bool is_interface_at_center(double v_star) {
    return v_star == 0;
}


bool is_left_wave_shock(double p_star, double pL) {
    return p_star > pL;
}

bool is_left_wave_rarefaction(double p_star, double pL) {
    return p_star < pL;
}

bool is_right_wave_shock(double p_star, double pR) {
    return p_star > pR;
}

bool is_right_wave_rarefaction(double p_star, double pR) {
    return p_star < pR;
}


bool is_interface_in_left_constant_region(double SL_head) {
    return SL_head >= 0;
}

bool is_interface_in_left_star_region(double SL_tail) {
    return SL_tail < 0;
}

bool is_interface_in_left_rarefaction_fan(double SL_head, double SL_tail) {
    return SL_head < 0 && SL_tail >= 0;
}

bool is_interface_in_right_constant_region(double SR_head) {
    return SR_head <= 0;
}

bool is_interface_in_right_star_region(double SR_tail) {
    return SR_tail > 0;
}

bool is_interface_in_right_rarefaction_fan(double SR_head, double SR_tail) {
    return SR_head > 0 && SR_tail <= 0;
}


/*
    struct ExactRiemannState
    {
        double rho_k_star;
        double v_k_star;
        double p_star;
    };
*/


// Shock speed from Rankine-Hugoniot conditions.
double calculate_shock_speed(double p_star, double v_k, double rho_k, double p_k) {
    double c_s_k = std::sqrt(Gamma * p_k / rho_k);
    return v_k + c_s_k * std::sqrt((Gamma + 1) / (2 * Gamma) * (p_star / p_k ) + (Gamma - 1) / 2 * Gamma);
}

// Star-region velocity for a rarefaction wave.
double calculate_rarefaction_v_star(double p_star, double v_k, double rho_k, double p_k) {
    double c_s_k = std::sqrt(Gamma * p_k / rho_k);
    return v_k + 2 * c_s_k / (Gamma - 1) * (std::pow(p_star / p_k, (Gamma - 1) / (2 * Gamma)) - 1);
}

// Star-region velocity for a shock wave.
double calculate_shock_v_star(double p_star, double v_k, double rho_k, double p_k) {
    const double SK_local       = calculate_shock_speed(p_star, v_k, rho_k, p_k);
    const double rho_k_star_local = calculate_shock_rho_star(p_star, rho_k, p_k);
    return (1 - rho_k / rho_k_star_local) * SK_local + v_k * rho_k / rho_k_star_local;
}


// Sound speed for the left state.
double calculate_cL(double rhoL, double pL) {
    return std::sqrt(Gamma * pL / rhoL);
}

// Sound speed for the right state.
double calculate_cR(double rhoR, double pR) {
    return std::sqrt(Gamma * pR / rhoR);
}


// Head speed of the left rarefaction fan.
double calculate_SL_head(double vL, double cL) {
    return vL - cL;
}

// Tail speed of the left rarefaction fan.
double calculate_SL_tail(double v_star, double c_star) {
    return v_star - c_star;
}

// Head speed of the right rarefaction fan.
double calculate_SR_head(double vR, double cR) {
    return vR + cR;
}

// Tail speed of the right rarefaction fan.
double calculate_SR_tail(double v_star, double c_star) {
    return v_star + c_star;
}   


ExactRiemannState calculate_shock_state(double p_star, double v_k, double rho_k, double p_k) {
    double SK = calculate_shock_speed(p_star, v_k, rho_k, p_k);
    double rho_k_star = calculate_shock_rho_star(p_star, rho_k, p_k);
    double v_k_star = calculate_shock_v_star(p_star, v_k, rho_k, p_k);
    return ExactRiemannState{rho_k_star, v_k_star, p_star};
}

ExactRiemannState calculate_rarefaction_state(double p_star, double v_k, double v_star, double rho_k, double p_k) {
    double c_s_k = std::sqrt(Gamma * p_k / rho_k);

    if (is_interface_on_left(v_star)) {
        double v_k_star = calculate_rarefaction_v_star(p_star, v_k, rho_k, p_k);
        double rho_k_star = calculate_rarefaction_rho_star(p_star, rho_k, p_k);
        return ExactRiemannState{rho_k_star, v_k_star, p_star};
    }

    if (is_interface_on_right(v_star))
    {
        double v_k_star = calculate_rarefaction_v_star(p_star, v_k, rho_k, p_k);
        double rho_k_star = calculate_rarefaction_rho_star(p_star, rho_k, p_k);
        return ExactRiemannState{rho_k_star, v_k_star, p_star};
    }
    return ExactRiemannState{rho_k, v_k, p_k};
}


ExactRiemannState calculate_constant_state(double p_star, double v_k, double rho_k, double p_k) {
    return ExactRiemannState{rho_k, v_k, p_k};
}


// Determine the exact Riemann state at the interface given p* and u*.
ExactRiemannState determine_interface_state(
    double p_star, double v_star,
    double vL, double vR,
    double rhoL, double pL,
    double rhoR, double pR)
{
    const double aL    = std::sqrt(Gamma * pL / rhoL);
    const double aR    = std::sqrt(Gamma * pR / rhoR);
    const double aLs   = aL * std::pow(p_star / pL, (Gamma-1.0)/(2.0*Gamma));
    const double aRs   = aR * std::pow(p_star / pR, (Gamma-1.0)/(2.0*Gamma));

    // Left wave
    if (p_star > pL) {
        // left shock
        const double SL = vL - aL * std::sqrt((Gamma+1)/(2*Gamma) * p_star/pL + (Gamma-1)/(2*Gamma));
        if (SL >= 0.0) {
            return ExactRiemannState{rhoL, vL, pL};
        } else {
            const double rhoLs = calculate_shock_rho_star(p_star, rhoL, pL);
            return ExactRiemannState{rhoLs, v_star, p_star};
        }
    } else {
        // left rarefaction
        const double SLh = vL - aL;   // head
        const double SLt = v_star - aLs; // tail
        if (SLh >= 0.0) {
            return ExactRiemannState{rhoL, vL, pL};
        } else if (SLt <= 0.0) {
            // inside rarefaction fan
            const double coeff = 2.0/(Gamma+1) + (Gamma-1)/((Gamma+1)*aL) * vL;
            const double rho_fan = rhoL * std::pow(coeff, 2.0/(Gamma-1));
            const double v_fan   = 2.0/(Gamma+1) * (aL + (Gamma-1)/2.0 * vL);
            const double p_fan   = pL * std::pow(coeff, 2.0*Gamma/(Gamma-1));
            return ExactRiemannState{rho_fan, v_fan, p_fan};
        } else {
            const double rhoLs = calculate_rarefaction_rho_star(p_star, rhoL, pL);
            return ExactRiemannState{rhoLs, v_star, p_star};
        }
    }

    // Right wave (v_star < 0)
    if (p_star > pR) {
        // right shock
        const double SR = vR + aR * std::sqrt((Gamma+1)/(2*Gamma) * p_star/pR + (Gamma-1)/(2*Gamma));
        if (SR <= 0.0) {
            return ExactRiemannState{rhoR, vR, pR};
        } else {
            const double rhoRs = calculate_shock_rho_star(p_star, rhoR, pR);
            return ExactRiemannState{rhoRs, v_star, p_star};
        }
    } else {
        // right rarefaction
        const double SRh = vR + aR;
        const double SRt = v_star + aRs;
        if (SRh <= 0.0) {
            return ExactRiemannState{rhoR, vR, pR};
        } else if (SRt >= 0.0) {
            // inside rarefaction fan
            const double coeff = 2.0/(Gamma+1) - (Gamma-1)/((Gamma+1)*aR) * vR;
            const double rho_fan = rhoR * std::pow(coeff, 2.0/(Gamma-1));
            const double v_fan   = 2.0/(Gamma+1) * (-aR + (Gamma-1)/2.0 * vR);
            const double p_fan   = pR * std::pow(coeff, 2.0*Gamma/(Gamma-1));
            return ExactRiemannState{rho_fan, v_fan, p_fan};
        } else {
            const double rhoRs = calculate_rarefaction_rho_star(p_star, rhoR, pR);
            return ExactRiemannState{rhoRs, v_star, p_star};
        }
    }
}


// Compute the physical flux from the star-region state.

Physical exact_riemann_flux(double p_star, double v_star, double rho_star) {
    const double rho = rho_star;
    const double u   = v_star;
    const double p   = p_star;
    const double E = p / (Gamma - 1.0) + 0.5 * rho * u * u;

    const double F1 = rho * u;
    const double F2 = rho * u * u + p;
    const double F3 = (E + p) * u;   // ★ 正确的能量通量

    return Physical{F1, F2, F3};
}


// Compute the exact Riemann flux given left and right conserved states.
Physical exactFlux(const Conserved& UL, const Conserved& UR) {
    Primitive WL = conservedToPrimitive(UL);
    Primitive WR = conservedToPrimitive(UR);

    auto bad = [](const Primitive& W){
        return (!std::isfinite(W.rho) || !std::isfinite(W.u) || !std::isfinite(W.p)
                || W.rho <= 1e-12 || W.p <= 1e-12);
    };

    if (bad(WL) || bad(WR)) {
        return hllcFlux(UL, UR);
    }

    double p_star = solvePStar(WL.u, WR.u, WL.rho, WL.p, WR.rho, WR.p);

    if (!std::isfinite(p_star) || p_star <= 1e-12 || p_star > 1e6) {
        return hllcFlux(UL, UR);
    }

    double u_star = calcVStar(p_star, WL.u, WL.rho, WL.p, WR.u, WR.rho, WR.p);
    if (!std::isfinite(u_star)) {
        return hllcFlux(UL, UR);
    }

    auto rho_star_side = [&](double pS, double rho, double p) {
        const double pr = pS / p;
        if (pS > p) {
            // shock
            const double gm1 = Gamma - 1.0;
            const double gp1 = Gamma + 1.0;
            return rho * ( (pr + gm1/gp1) / ( (gm1/gp1)*pr + 1.0 ) );
        } else {
            // rarefaction
            return rho * std::pow(pr, 1.0 / Gamma);
        }
    };

    double rhoL_star = rho_star_side(p_star, WL.rho, WL.p);
    double rhoR_star = rho_star_side(p_star, WR.rho, WR.p);

    if (!std::isfinite(rhoL_star) || !std::isfinite(rhoR_star) ||
        rhoL_star <= 1e-12 || rhoR_star <= 1e-12) {
        return hllcFlux(UL, UR);
    }

    double rho_star = (u_star >= 0.0) ? rhoL_star : rhoR_star;

    return exact_riemann_flux(p_star, u_star, rho_star);
}