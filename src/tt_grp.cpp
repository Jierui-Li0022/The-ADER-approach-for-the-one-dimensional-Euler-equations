// TT-GRP interface flux for ADER3, ADER4, ADER5: Cauchy-Kovalewski + Gauss-Legendre time integration.

#include "aderweno/tt_grp.hpp"
#include "aderweno/ck.hpp"
#include "aderweno/riemann_solver.hpp"
#include "aderweno/euler1d.hpp"
#include "aderweno/checks.hpp"
#include "aderweno/constants.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>

static bool finiteState(const Conserved& U) {
    return std::isfinite(U.rho) && std::isfinite(U.mom) && std::isfinite(U.E);
}


static Conserved averageState(const Conserved& UL, const Conserved& UR) {
    Conserved U{};
    U.rho = 0.5 * (UL.rho + UR.rho);
    U.mom   = 0.5 * (UL.mom   + UR.mom);
    U.E   = 0.5 * (UL.E   + UR.E);
    return U;
}

// approximate wave speeds (used by HLL/HLLC fallback) 

static void estimate_wave_speeds(const Conserved& UL,
                                 const Conserved& UR,
                                 double& sL, double& sR, double& sM) {
    Primitive WL = conservedToPrimitive(UL);
    Primitive WR = conservedToPrimitive(UR);

    const double aL = soundSpeed(UL.rho, UL.mom, UL.E);
    const double aR = soundSpeed(UR.rho, UR.mom, UR.E);

    sL = std::min(WL.u - aL, WR.u - aR);
    sR = std::max(WL.u + aL, WR.u + aR);

    const double denom = WL.rho * (sL - WL.u) - WR.rho * (sR - WR.u);
    if (std::abs(denom) < 1e-14 || !std::isfinite(denom)) {
        sM = 0.5 * (WL.u + WR.u);
    } else {
        sM = (WR.p - WL.p + WL.rho * WL.u * (sL - WL.u) - WR.rho * WR.u * (sR - WR.u)) / denom;
    }
}

// simple Godunov state choices (for robustness)

static Conserved lf_godunov_state(const Conserved& UL, const Conserved& UR) {
    const Physical FL = physicalFlux(UL);
    const Physical FR = physicalFlux(UR);
    const double smax = std::max(maxSignalSpeed(UL), maxSignalSpeed(UR));

    if (!std::isfinite(smax) || smax <= 1e-14) {
        return averageState(UL, UR);
    }

    Conserved U{};
    U.rho = 0.5 * (UL.rho + UR.rho) - 0.5 * (FR.fluxRho - FL.fluxRho) / smax;
    U.mom   = 0.5 * (UL.mom   + UR.mom)   - 0.5 * (FR.fluxMom - FL.fluxMom) / smax;
    U.E   = 0.5 * (UL.E   + UR.E)   - 0.5 * (FR.fluxE - FL.fluxE) / smax;
    return U;
}

static Conserved hll_godunov_state(const Conserved& UL, const Conserved& UR) {
    double sL, sR, sM;
    estimate_wave_speeds(UL, UR, sL, sR, sM);
    (void)sM;

    if (sL >= 0.0) return UL;
    if (sR <= 0.0) return UR;
    if (std::abs(sR - sL) < 1e-14) return averageState(UL, UR);

    const Physical FL = physicalFlux(UL);
    const Physical FR = physicalFlux(UR);

    Conserved UHLL{};
    UHLL.rho = (sR * UR.rho - sL * UL.rho - (FR.fluxRho - FL.fluxRho)) / (sR - sL);
    UHLL.mom   = (sR * UR.mom   - sL * UL.mom   - (FR.fluxMom - FL.fluxMom)) / (sR - sL);
    UHLL.E   = (sR * UR.E   - sL * UL.E   - (FR.fluxE - FL.fluxE)) / (sR - sL);
    return UHLL;
}

static Conserved hllc_godunov_state(const Conserved& UL, const Conserved& UR) {
    Primitive WL = conservedToPrimitive(UL);
    Primitive WR = conservedToPrimitive(UR);

    double sL, sR, sM;
    estimate_wave_speeds(UL, UR, sL, sR, sM);

    if (sL >= 0.0) return UL;
    if (sR <= 0.0) return UR;

    const double pStarL = WL.p + WL.rho * (sL - WL.u) * (sM - WL.u);
    const double pStarR = WR.p + WR.rho * (sR - WR.u) * (sM - WR.u);
    const double pStar  = 0.5 * (pStarL + pStarR);

    if (sM >= 0.0) {
        const double denom = sL - sM;
        if (std::abs(denom) < 1e-14) return hll_godunov_state(UL, UR);

        Conserved U{};
        U.rho = WL.rho * (sL - WL.u) / denom;
        U.mom   = U.rho * sM;
        U.E   = ((sL - WL.u) * UL.E - WL.p * WL.u + pStar * sM) / denom;
        return U;
    }

    const double denom = sR - sM;
    if (std::abs(denom) < 1e-14) return hll_godunov_state(UL, UR);

    Conserved U{};
    U.rho = WR.rho * (sR - WR.u) / denom;
    U.mom   = U.rho * sM;
    U.E   = ((sR - WR.u) * UR.E - WR.p * WR.u + pStar * sM) / denom;
    return U;
}

// leading term selection

static Conserved choose_godunov_state_q0(const InterfacePoly& L,
                                         const InterfacePoly& R,
                                         const TTGRPOptions& opt) {
    Conserved G0{};

    if (opt.use_exact_q0) {
        G0 = exactInterfaceStateQ0(L, R);
    } else {
        switch (opt.flux_type) {
            case TTRiemannFluxType::Exact:
                G0 = exactInterfaceStateQ0(L, R);
                break;
            case TTRiemannFluxType::LaxFriedrichs:
                G0 = lf_godunov_state(L.Q, R.Q);
                break;
            case TTRiemannFluxType::HLL:
                G0 = hll_godunov_state(L.Q, R.Q);
                break;
            case TTRiemannFluxType::HLLC:
                G0 = hllc_godunov_state(L.Q, R.Q);
                break;
        }
    }

    if (!finiteState(G0)) {
        G0 = averageState(L.Q, R.Q);
    }
    enforcePhysical(G0);
    return G0;
}

// linear algebra helpers for eigen decomposition 

static bool inverse3x3(const double A[3][3], double invA[3][3]) {
    const double c00 = A[1][1] * A[2][2] - A[1][2] * A[2][1];
    const double c01 = A[1][2] * A[2][0] - A[1][0] * A[2][2];
    const double c02 = A[1][0] * A[2][1] - A[1][1] * A[2][0];

    const double c10 = A[0][2] * A[2][1] - A[0][1] * A[2][2];
    const double c11 = A[0][0] * A[2][2] - A[0][2] * A[2][0];
    const double c12 = A[0][1] * A[2][0] - A[0][0] * A[2][1];

    const double c20 = A[0][1] * A[1][2] - A[0][2] * A[1][1];
    const double c21 = A[0][2] * A[1][0] - A[0][0] * A[1][2];
    const double c22 = A[0][0] * A[1][1] - A[0][1] * A[1][0];

    const double det = A[0][0] * c00 + A[0][1] * c01 + A[0][2] * c02;
    if (!std::isfinite(det) || std::abs(det) < 1e-14) {
        return false;
    }

    const double inv_det = 1.0 / det;
    invA[0][0] = c00 * inv_det; invA[0][1] = c10 * inv_det; invA[0][2] = c20 * inv_det;
    invA[1][0] = c01 * inv_det; invA[1][1] = c11 * inv_det; invA[1][2] = c21 * inv_det;
    invA[2][0] = c02 * inv_det; invA[2][1] = c12 * inv_det; invA[2][2] = c22 * inv_det;
    return true;
}

static std::array<double, 3> mat_vec_3x3(const double A[3][3], const std::array<double, 3>& x) {
    std::array<double, 3> y{};
    y[0] = A[0][0] * x[0] + A[0][1] * x[1] + A[0][2] * x[2];
    y[1] = A[1][0] * x[0] + A[1][1] * x[1] + A[1][2] * x[2];
    y[2] = A[2][0] * x[0] + A[2][1] * x[1] + A[2][2] * x[2];
    return y;
}

// Compute eigen-system of Euler Jacobian A(Q) at Q0
static bool euler_eigensystem_at_q0(const Conserved& Q0,
                                    double lambda[3],
                                    double Rmat[3][3],
                                    double Lmat[3][3]) {
    Conserved Q = Q0;
    if (!finiteState(Q)) return false;
    enforcePhysical(Q);

    const double rho = Q.rho;
    const double u   = Q.mom / rho;
    const double p   = std::max(pressure(Q.rho, Q.mom, Q.E), 1e-12);
    const double a2  = Gamma * p / rho;
    if (!std::isfinite(a2) || a2 <= 0.0) return false;
    const double a   = std::sqrt(a2);
    const double H   = (Q.E + p) / rho;

    lambda[0] = u - a;
    lambda[1] = u;
    lambda[2] = u + a;

    // Right eigenvectors (columns)
    Rmat[0][0] = 1.0;   Rmat[0][1] = 1.0;      Rmat[0][2] = 1.0;
    Rmat[1][0] = u - a; Rmat[1][1] = u;        Rmat[1][2] = u + a;
    Rmat[2][0] = H - u * a;
    Rmat[2][1] = 0.5 * u * u;
    Rmat[2][2] = H + u * a;

    return inverse3x3(Rmat, Lmat);
}

// ============================================================================
// OPTIMIZED: Apply precomputed eigensystem to solve derivative Riemann problem
// Avoids recomputing eigensystem for each call
// ============================================================================
static Conserved apply_eigen_upwind(const Conserved& dL,
                                    const Conserved& dR,
                                    const double lambda[3],
                                    const double Rmat[3][3],
                                    const double Lmat[3][3]) {
    const std::array<double, 3> vL{dL.rho, dL.mom, dL.E};
    const std::array<double, 3> vR{dR.rho, dR.mom, dR.E};

    const std::array<double, 3> aL = mat_vec_3x3(Lmat, vL);
    const std::array<double, 3> aR = mat_vec_3x3(Lmat, vR);

    std::array<double, 3> aStar{};
    for (int k = 0; k < 3; ++k) {
        if (lambda[k] > 1e-12) {
            aStar[k] = aL[k];
        } else if (lambda[k] < -1e-12) {
            aStar[k] = aR[k];
        } else {
            aStar[k] = 0.5 * (aL[k] + aR[k]);
        }
    }

    const std::array<double, 3> vStar = mat_vec_3x3(Rmat, aStar);

    Conserved out{};
    out.rho = vStar[0];
    out.mom   = vStar[1];
    out.E   = vStar[2];
    return out;
}

// Taylor-2 state from CK output (used by Gauss time integration)
static Conserved predictStateTaylor2_from_ck(const Conserved& Q0,
                                               const CKOutput& ck,
                                               double t) {
    const double t2 = t * t;
    Conserved Q{};
    Q.rho = Q0.rho + t * ck.Qt.rho + 0.5 * t2 * ck.Qtt.rho;
    Q.mom   = Q0.mom   + t * ck.Qt.mom   + 0.5 * t2 * ck.Qtt.mom;
    Q.E   = Q0.E   + t * ck.Qt.E   + 0.5 * t2 * ck.Qtt.E;

    if (!finiteState(Q)) {
        return Q0;
    }
    enforcePhysical(Q);
    return Q;
}

// Public convenience
Conserved predictStateTaylor2(const InterfacePoly& P, double t) {
    CKOutput ck = ck_euler_order2(CKInput{P.Q, P.Qx, P.Qxx});
    return predictStateTaylor2_from_ck(P.Q, ck, t);
}

// exact leading term (Godunov state) using your exact solver 

Conserved exactInterfaceStateQ0(const InterfacePoly& L, const InterfacePoly& R) {
    Primitive WL = conservedToPrimitive(L.Q);
    Primitive WR = conservedToPrimitive(R.Q);

    double p_star = solvePStar(WL.u, WR.u, WL.rho, WL.p, WR.rho, WR.p);
    double v_star = calcVStar(p_star, WL.u, WL.rho, WL.p, WR.u, WR.rho, WR.p);

    ExactRiemannState st = determine_interface_state(
        p_star, v_star,
        WL.u, WR.u,
        WL.rho, WL.p,
        WR.rho, WR.p
    );

    Primitive Wstar{st.rho_k_star, st.v_k_star, st.p_star};
    Conserved Ustar = primitiveToConserved(Wstar);

    if (!std::isfinite(Ustar.rho) || !std::isfinite(Ustar.mom) || !std::isfinite(Ustar.E) || Ustar.rho <= 0.0) {
        return L.Q;
    }

    enforcePhysical(Ustar);
    return Ustar;
}

// main TT-GRP ADER3 flux 

// Precomputed Gauss quadrature points
static constexpr double GAUSS_TAU1 = 0.5 - 0.28867513459481287;  // 0.5 - sqrt(3)/6
static constexpr double GAUSS_TAU2 = 0.5 + 0.28867513459481287;  // 0.5 + sqrt(3)/6

Physical ttGrpFluxAder3(const InterfacePoly& L,
                           const InterfacePoly& R,
                           double dt,
                           const TTGRPOptions& opt) {
  
    // TT-GRP ADER3 
    //
    // 1) Leading term G^(0)(0): conventional Riemann problem -> Q0
    // 2) Higher derivative interface data G^(1)(0), G^(2)(0):
    //    solve linearized Riemann problems for Qx and Qxx at A(Q0)
    // 3) CK: Qt,Qtt from (Q0,Qx0,Qxx0)
    // 4) Q(0,t) by Taylor-2 and time-average physical flux by 2-point Gauss
    
    Conserved Q0 = choose_godunov_state_q0(L, R, opt);

    // ====================================================================
    // OPTIMIZED: compute eigensystem ONCE, reuse for both Qx and Qxx
    // Was: 2x eigensystem (eigenvalues + right eigenvectors + 3x3 inverse)
    // Now: 1x eigensystem
    // ====================================================================
    double lambda[3];
    double Rmat[3][3];
    double Lmat[3][3];

    Conserved Qx0, Qxx0;
    if (euler_eigensystem_at_q0(Q0, lambda, Rmat, Lmat)) {
        Qx0  = apply_eigen_upwind(L.Qx,  R.Qx,  lambda, Rmat, Lmat);
        Qxx0 = apply_eigen_upwind(L.Qxx, R.Qxx, lambda, Rmat, Lmat);
        if (!finiteState(Qx0))  Qx0  = averageState(L.Qx,  R.Qx);
        if (!finiteState(Qxx0)) Qxx0 = averageState(L.Qxx, R.Qxx);
    } else {
        Qx0  = averageState(L.Qx,  R.Qx);
        Qxx0 = averageState(L.Qxx, R.Qxx);
    }

    CKOutput ck = ck_euler_order2(CKInput{Q0, Qx0, Qxx0});

    // 2-point Gauss-Legendre quadrature on [0, dt]
    const double t1 = GAUSS_TAU1 * dt;
    const double t2 = GAUSS_TAU2 * dt;

    Conserved Q1 = predictStateTaylor2_from_ck(Q0, ck, t1);
    Conserved Q2 = predictStateTaylor2_from_ck(Q0, ck, t2);

    // OPTIMIZED: inline flux computation, avoid 2x pressure() calls
    
    auto fast_flux = [&](const Conserved& U) -> Physical {
        double rho = (U.rho > RhoMin) ? U.rho : RhoMin;
        double u = U.mom / rho;
        double p = (Gamma - 1.0) * (U.E - 0.5 * U.mom * U.mom / rho);
        if (p < PMin) p = PMin;
        return {U.mom, U.mom * u + p, (U.E + p) * u};
    };

    const Physical F1 = fast_flux(Q1);
    const Physical F2 = fast_flux(Q2);

    Physical Fbar;
    Fbar.fluxRho = 0.5 * (F1.fluxRho + F2.fluxRho);
    Fbar.fluxMom = 0.5 * (F1.fluxMom + F2.fluxMom);
    Fbar.fluxE = 0.5 * (F1.fluxE + F2.fluxE);
    return Fbar;
}


// ====================================================================
// Taylor-3 state from CK3 output (for ADER4 Gauss integration)
// Q(0,t) = Q0 + t*Qt + t^2/2*Qtt + t^3/6*Qttt
// ====================================================================
static Conserved predictStateTaylor3_from_ck(const Conserved& Q0,
                                               const CKOutput3& ck,
                                               double t) {
    const double t2 = t * t;
    const double t3 = t2 * t;
    Conserved Q{};
    Q.rho = Q0.rho + t * ck.Qt.rho + 0.5 * t2 * ck.Qtt.rho + (1.0/6.0) * t3 * ck.Qttt.rho;
    Q.mom   = Q0.mom   + t * ck.Qt.mom   + 0.5 * t2 * ck.Qtt.mom   + (1.0/6.0) * t3 * ck.Qttt.mom;
    Q.E   = Q0.E   + t * ck.Qt.E   + 0.5 * t2 * ck.Qtt.E   + (1.0/6.0) * t3 * ck.Qttt.E;

    if (!finiteState(Q)) {
        return Q0;
    }
    enforcePhysical(Q);
    return Q;
}

// ====================================================================
// Taylor-4 state from CK4 output (for ADER5 Gauss integration)
// Q(0,t) = Q0 + t*Qt + t^2/2*Qtt + t^3/6*Qttt + t^4/24*Qtttt
// ====================================================================
static Conserved predictStateTaylor4_from_ck(const Conserved& Q0,
                                               const CKOutput4& ck,
                                               double t) {
    const double t2 = t * t;
    const double t3 = t2 * t;
    const double t4 = t2 * t2;
    Conserved Q{};
    Q.rho = Q0.rho + t * ck.Qt.rho + 0.5 * t2 * ck.Qtt.rho
            + (1.0/6.0) * t3 * ck.Qttt.rho + (1.0/24.0) * t4 * ck.Qtttt.rho;
    Q.mom   = Q0.mom   + t * ck.Qt.mom   + 0.5 * t2 * ck.Qtt.mom
            + (1.0/6.0) * t3 * ck.Qttt.mom   + (1.0/24.0) * t4 * ck.Qtttt.mom;
    Q.E   = Q0.E   + t * ck.Qt.E   + 0.5 * t2 * ck.Qtt.E
            + (1.0/6.0) * t3 * ck.Qttt.E   + (1.0/24.0) * t4 * ck.Qtttt.E;

    if (!finiteState(Q)) {
        return Q0;
    }
    enforcePhysical(Q);
    return Q;
}

// Public convenience wrappers
Conserved predictStateTaylor3(const InterfacePoly& P, double t) {
    CKOutput3 ck = ck_euler_order3(CKInput3{P.Q, P.Qx, P.Qxx, P.Qxxx});
    return predictStateTaylor3_from_ck(P.Q, ck, t);
}

Conserved predictStateTaylor4(const InterfacePoly& P, double t) {
    CKOutput4 ck = ck_euler_order4(CKInput4{P.Q, P.Qx, P.Qxx, P.Qxxx, P.Qxxxx});
    return predictStateTaylor4_from_ck(P.Q, ck, t);
}


// ====================================================================
// TT-GRP ADER4 flux: CK order 3 + 2-point Gauss
// ====================================================================

Physical ttGrpFluxAder4(const InterfacePoly& L,
                           const InterfacePoly& R,
                           double dt,
                           const TTGRPOptions& opt) {

    // 1) Leading term: Riemann problem -> Q0
    Conserved Q0 = choose_godunov_state_q0(L, R, opt);

    // 2) Eigensystem at Q0
    double lambda[3];
    double Rmat[3][3];
    double Lmat[3][3];

    Conserved Qx0, Qxx0, Qxxx0;
    if (euler_eigensystem_at_q0(Q0, lambda, Rmat, Lmat)) {
        Qx0   = apply_eigen_upwind(L.Qx,   R.Qx,   lambda, Rmat, Lmat);
        Qxx0  = apply_eigen_upwind(L.Qxx,  R.Qxx,  lambda, Rmat, Lmat);
        Qxxx0 = apply_eigen_upwind(L.Qxxx, R.Qxxx, lambda, Rmat, Lmat);
        if (!finiteState(Qx0))   Qx0   = averageState(L.Qx,   R.Qx);
        if (!finiteState(Qxx0))  Qxx0  = averageState(L.Qxx,  R.Qxx);
        if (!finiteState(Qxxx0)) Qxxx0 = averageState(L.Qxxx, R.Qxxx);
    } else {
        Qx0   = averageState(L.Qx,   R.Qx);
        Qxx0  = averageState(L.Qxx,  R.Qxx);
        Qxxx0 = averageState(L.Qxxx, R.Qxxx);
    }

    // 3) CK order 3
    CKOutput3 ck = ck_euler_order3(CKInput3{Q0, Qx0, Qxx0, Qxxx0});

    // 4) 2-point Gauss-Legendre on [0, dt]
    const double t1 = GAUSS_TAU1 * dt;
    const double t2 = GAUSS_TAU2 * dt;

    Conserved Q1 = predictStateTaylor3_from_ck(Q0, ck, t1);
    Conserved Q2 = predictStateTaylor3_from_ck(Q0, ck, t2);

    
    auto fast_flux = [&](const Conserved& U) -> Physical {
        double rho = (U.rho > RhoMin) ? U.rho : RhoMin;
        double u = U.mom / rho;
        double p = (Gamma - 1.0) * (U.E - 0.5 * U.mom * U.mom / rho);
        if (p < PMin) p = PMin;
        return {U.mom, U.mom * u + p, (U.E + p) * u};
    };

    const Physical F1 = fast_flux(Q1);
    const Physical F2 = fast_flux(Q2);

    Physical Fbar;
    Fbar.fluxRho = 0.5 * (F1.fluxRho + F2.fluxRho);
    Fbar.fluxMom = 0.5 * (F1.fluxMom + F2.fluxMom);
    Fbar.fluxE = 0.5 * (F1.fluxE + F2.fluxE);
    return Fbar;
}


// ====================================================================
// TT-GRP ADER5 flux: CK order 4 + 3-point Gauss
// ====================================================================

// 3-point Gauss-Legendre on [0,1]: nodes and weights
static constexpr double GAUSS3_TAU1 = 0.5 - 0.3872983346207417;  // 0.5 - sqrt(3/5)/2
static constexpr double GAUSS3_TAU2 = 0.5;
static constexpr double GAUSS3_TAU3 = 0.5 + 0.3872983346207417;  // 0.5 + sqrt(3/5)/2
static constexpr double GAUSS3_W1   = 5.0 / 18.0;
static constexpr double GAUSS3_W2   = 8.0 / 18.0;
static constexpr double GAUSS3_W3   = 5.0 / 18.0;

Physical ttGrpFluxAder5(const InterfacePoly& L,
                           const InterfacePoly& R,
                           double dt,
                           const TTGRPOptions& opt) {

    // 1) Leading term: Riemann problem -> Q0
    Conserved Q0 = choose_godunov_state_q0(L, R, opt);

    // 2) Eigensystem at Q0
    double lambda[3];
    double Rmat[3][3];
    double Lmat[3][3];

    Conserved Qx0, Qxx0, Qxxx0, Qxxxx0;
    if (euler_eigensystem_at_q0(Q0, lambda, Rmat, Lmat)) {
        Qx0    = apply_eigen_upwind(L.Qx,    R.Qx,    lambda, Rmat, Lmat);
        Qxx0   = apply_eigen_upwind(L.Qxx,   R.Qxx,   lambda, Rmat, Lmat);
        Qxxx0  = apply_eigen_upwind(L.Qxxx,  R.Qxxx,  lambda, Rmat, Lmat);
        Qxxxx0 = apply_eigen_upwind(L.Qxxxx, R.Qxxxx, lambda, Rmat, Lmat);
        if (!finiteState(Qx0))    Qx0    = averageState(L.Qx,    R.Qx);
        if (!finiteState(Qxx0))   Qxx0   = averageState(L.Qxx,   R.Qxx);
        if (!finiteState(Qxxx0))  Qxxx0  = averageState(L.Qxxx,  R.Qxxx);
        if (!finiteState(Qxxxx0)) Qxxxx0 = averageState(L.Qxxxx, R.Qxxxx);
    } else {
        Qx0    = averageState(L.Qx,    R.Qx);
        Qxx0   = averageState(L.Qxx,   R.Qxx);
        Qxxx0  = averageState(L.Qxxx,  R.Qxxx);
        Qxxxx0 = averageState(L.Qxxxx, R.Qxxxx);
    }

    // 3) CK order 4
    CKOutput4 ck = ck_euler_order4(CKInput4{Q0, Qx0, Qxx0, Qxxx0, Qxxxx0});

    // 4) 3-point Gauss-Legendre on [0, dt]
    const double t1 = GAUSS3_TAU1 * dt;
    const double t2 = GAUSS3_TAU2 * dt;
    const double t3 = GAUSS3_TAU3 * dt;

    Conserved Q1 = predictStateTaylor4_from_ck(Q0, ck, t1);
    Conserved Q2 = predictStateTaylor4_from_ck(Q0, ck, t2);
    Conserved Q3 = predictStateTaylor4_from_ck(Q0, ck, t3);

    
    auto fast_flux = [&](const Conserved& U) -> Physical {
        double rho = (U.rho > RhoMin) ? U.rho : RhoMin;
        double u = U.mom / rho;
        double p = (Gamma - 1.0) * (U.E - 0.5 * U.mom * U.mom / rho);
        if (p < PMin) p = PMin;
        return {U.mom, U.mom * u + p, (U.E + p) * u};
    };

    const Physical F1 = fast_flux(Q1);
    const Physical F2 = fast_flux(Q2);
    const Physical F3 = fast_flux(Q3);

    Physical Fbar;
    Fbar.fluxRho = GAUSS3_W1 * F1.fluxRho + GAUSS3_W2 * F2.fluxRho + GAUSS3_W3 * F3.fluxRho;
    Fbar.fluxMom = GAUSS3_W1 * F1.fluxMom + GAUSS3_W2 * F2.fluxMom + GAUSS3_W3 * F3.fluxMom;
    Fbar.fluxE = GAUSS3_W1 * F1.fluxE + GAUSS3_W2 * F2.fluxE + GAUSS3_W3 * F3.fluxE;
    return Fbar;
}


// ====================================================================
// Cell CK ADER5 flux: CK on each side independently + Riemann at each Gauss point
// ====================================================================

Physical cellCkFluxAder5(const InterfacePoly& L,
                           const InterfacePoly& R,
                           double dt) {

    // 1) CK independently for left cell (right boundary) and right cell (left boundary)
    CKOutput4 ckL = ck_euler_order4(CKInput4{L.Q, L.Qx, L.Qxx, L.Qxxx, L.Qxxxx});
    CKOutput4 ckR = ck_euler_order4(CKInput4{R.Q, R.Qx, R.Qxx, R.Qxxx, R.Qxxxx});

    // 2) 3-point Gauss-Legendre on [0, dt]
    const double t1 = GAUSS3_TAU1 * dt;
    const double t2 = GAUSS3_TAU2 * dt;
    const double t3 = GAUSS3_TAU3 * dt;

    // Low-dissipation Rusanov flux at each Gauss point:
    //   F = 0.5*(F(QL)+F(QR)) - alpha*smax*(QR-QL)
    //
    // alpha = 0.1 gives a good balance between:
    //   - Low error constant (close to central flux averaging)
    //   - Enough dissipation for stable convergence rates
    //   - No catastrophic cancellation (unlike HLLC when QL ≈ QR)
    const double alpha = 0.2;
    

    auto flux_at_time = [&](double t) -> Physical {
        Conserved QL = predictStateTaylor4_from_ck(L.Q, ckL, t);
        Conserved QR = predictStateTaylor4_from_ck(R.Q, ckR, t);

        auto fflux = [&](const Conserved& U) -> Physical {
            double rho = (U.rho > RhoMin) ? U.rho : RhoMin;
            double u = U.mom / rho;
            double p = (Gamma - 1.0) * (U.E - 0.5 * U.mom * U.mom / rho);
            if (p < PMin) p = PMin;
            return {U.mom, U.mom * u + p, (U.E + p) * u};
        };

        Physical FL = fflux(QL);
        Physical FR = fflux(QR);

        double smax = std::max(maxSignalSpeed(QL), maxSignalSpeed(QR));

        return Physical{
            0.5 * (FL.fluxRho + FR.fluxRho) - alpha * smax * (QR.rho - QL.rho),
            0.5 * (FL.fluxMom + FR.fluxMom) - alpha * smax * (QR.mom   - QL.mom),
            0.5 * (FL.fluxE + FR.fluxE) - alpha * smax * (QR.E   - QL.E)
        };
    };

    const Physical F1 = flux_at_time(t1);
    const Physical F2 = flux_at_time(t2);
    const Physical F3 = flux_at_time(t3);

    Physical Fbar;
    Fbar.fluxRho = GAUSS3_W1 * F1.fluxRho + GAUSS3_W2 * F2.fluxRho + GAUSS3_W3 * F3.fluxRho;
    Fbar.fluxMom = GAUSS3_W1 * F1.fluxMom + GAUSS3_W2 * F2.fluxMom + GAUSS3_W3 * F3.fluxMom;
    Fbar.fluxE = GAUSS3_W1 * F1.fluxE + GAUSS3_W2 * F2.fluxE + GAUSS3_W3 * F3.fluxE;
    return Fbar;
}