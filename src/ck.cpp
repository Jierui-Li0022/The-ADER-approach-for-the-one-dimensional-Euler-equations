// Cauchy-Kovalewski procedure: converts spatial derivatives from WENO into time derivatives.
// ck.cpp — Cauchy-Kovalewski procedure for 1D Euler equations

#include "aderweno/ck.hpp"
#include "aderweno/constants.hpp"

#include <cmath>
#include <cstring>

// Maximum supported ADER order (ADER5 => N_TIME=4, NSPACE=5)
static constexpr int MAXORD = 5;
static constexpr int NVAR   = 3;  // ρ, ρu, E

//  Internal 2D Taylor coefficient array
struct Tay {
    double c[MAXORD][MAXORD];
    Tay() { std::memset(c, 0, sizeof(c)); }
    double& operator()(int m, int n)       { return c[m][n]; }
    double  operator()(int m, int n) const { return c[m][n]; }
};

//  Cauchy product
static inline double cmul(const Tay& a, const Tay& b, int m, int n)
{
    double s = 0.0;
    for (int i = 0; i <= m; ++i)
        for (int j = 0; j <= n; ++j)
            s += a(i, j) * b(m - i, n - j);
    return s;
}

//  Cauchy division:  c = a / b   (b(0,0) ≠ 0)
static inline double cdiv(const Tay& a, const Tay& b, const Tay& c, int m, int n)
{
    double s = a(m, n);
    for (int i = 0; i <= m; ++i)
        for (int j = 0; j <= n; ++j)
            if (i != 0 || j != 0)
                s -= b(i, j) * c(m - i, n - j);
    return s / b(0, 0);
}

//  Generic CK engine
static void ck_engine(int M_order,
                      const double qx_raw[NVAR][MAXORD],  
                      double qt_raw[NVAR][MAXORD])        
{
    const int N_time = M_order - 1;

    // Conserved variable Taylor arrays
    Tay rho, mom, ene;
    Tay* qt[NVAR] = {&rho, &mom, &ene};

    // Convert input: q̂(0,n) = Q^(n) / n!
    {
        double fact = 1.0;
        for (int n = 0; n < M_order; ++n) {
            rho(0, n) = qx_raw[0][n] / fact;
            mom(0, n) = qx_raw[1][n] / fact;
            ene(0, n) = qx_raw[2][n] / fact;
            fact *= (n + 1);
        }
    }

    // Derived quantity arrays
    Tay vel;       // u = mom/rho
    Tay vel2;      // u²
    Tay rho_vel2;  // ρu²
    Tay pres;      // p = (γ-1)(E - ½ρu²)
    Tay e_plus_p;  // E + p
    Tay flux[NVAR]; // F₁, F₂, F₃

    const double gm1 = Gamma - 1.0;

    // Process each time level
    for (int m = 0; m < N_time; ++m) {
        int max_n = M_order - 1 - m;

            for (int n = 0; n <= max_n; ++n) {
            vel(m, n)      = cdiv(mom, rho, vel, m, n);
            vel2(m, n)     = cmul(vel, vel, m, n);
            rho_vel2(m, n) = cmul(rho, vel2, m, n);
            pres(m, n)     = gm1 * (ene(m, n) - 0.5 * rho_vel2(m, n));
            e_plus_p(m, n) = ene(m, n) + pres(m, n);
        }

            for (int n = 0; n <= max_n; ++n) {
            flux[0](m, n) = mom(m, n);
            flux[1](m, n) = cmul(vel, mom, m, n) + pres(m, n);
            flux[2](m, n) = cmul(vel, e_plus_p, m, n);
        }

            int max_n_next = M_order - 1 - (m + 1);
        for (int k = 0; k < NVAR; ++k) {
            for (int n = 0; n <= max_n_next; ++n) {
                (*qt[k])(m + 1, n) = -double(n + 1) / double(m + 1)
                                     * flux[k](m, n + 1);
            }
        }
    }

    // Convert output: Q_t^(m) = m! · q̂(m,0)
    {
        double fact = 1.0;
        for (int m = 1; m <= N_time; ++m) {
            fact *= m;
            qt_raw[0][m] = rho(m, 0) * fact;
            qt_raw[1][m] = mom(m, 0) * fact;
            qt_raw[2][m] = ene(m, 0) * fact;
        }
    }
}

//  Helper: pack Conserved fields into raw array
static inline void pack(double dst[NVAR][MAXORD], int n, const Conserved& C)
{
    dst[0][n] = C.rho;
    dst[1][n] = C.mom;
    dst[2][n] = C.E;
}

static inline Conserved unpack(const double src[NVAR][MAXORD], int m)
{
    return Conserved{src[0][m], src[1][m], src[2][m]};
}

//  Order 2 (ADER3):  (Q, Qx, Qxx) → (Qt, Qtt)
CKOutput ck_euler_order2(const CKInput& in)
{
    double qx[NVAR][MAXORD] = {};
    double qt[NVAR][MAXORD] = {};

    pack(qx, 0, in.Q);
    pack(qx, 1, in.Qx);
    pack(qx, 2, in.Qxx);

    ck_engine(3, qx, qt);

    CKOutput out;
    out.Qt  = unpack(qt, 1);
    out.Qtt = unpack(qt, 2);
    return out;
}

//  Order 3 (ADER4):  (Q, Qx, Qxx, Qxxx) → (Qt, Qtt, Qttt)
CKOutput3 ck_euler_order3(const CKInput3& in)
{
    double qx[NVAR][MAXORD] = {};
    double qt[NVAR][MAXORD] = {};

    pack(qx, 0, in.Q);
    pack(qx, 1, in.Qx);
    pack(qx, 2, in.Qxx);
    pack(qx, 3, in.Qxxx);

    ck_engine(4, qx, qt);

    CKOutput3 out;
    out.Qt   = unpack(qt, 1);
    out.Qtt  = unpack(qt, 2);
    out.Qttt = unpack(qt, 3);
    return out;
}

//  Order 4 (ADER5):  (Q, Qx, Qxx, Qxxx, Qxxxx) → (Qt, Qtt, Qttt, Qtttt)
CKOutput4 ck_euler_order4(const CKInput4& in)
{
    double qx[NVAR][MAXORD] = {};
    double qt[NVAR][MAXORD] = {};

    pack(qx, 0, in.Q);
    pack(qx, 1, in.Qx);
    pack(qx, 2, in.Qxx);
    pack(qx, 3, in.Qxxx);
    pack(qx, 4, in.Qxxxx);

    ck_engine(5, qx, qt);

    CKOutput4 out;
    out.Qt    = unpack(qt, 1);
    out.Qtt   = unpack(qt, 2);
    out.Qttt  = unpack(qt, 3);
    out.Qtttt = unpack(qt, 4);
    return out;
}