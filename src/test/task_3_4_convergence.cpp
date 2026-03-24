// Long-time convergence test: all solvers run to T=1000 on smooth advection.

#include "aderweno/test/task_3_4_convergence.hpp"

#include "aderweno/constants.hpp"
#include "aderweno/euler1d.hpp"
#include "aderweno/riemann_solver.hpp"
#include "aderweno/tt_grp.hpp"
#include "aderweno/muscl_hancock_solver.hpp"
#include "aderweno/ader3_solver.hpp"
#include "aderweno/ader4_solver.hpp"
#include "aderweno/ader5_solver.hpp"

#include <cmath>
#include <vector>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <algorithm>

static constexpr double PI = 3.1415926535897932384626433832795;

static double rhoExactPoint(double x, double t) {
    const double s = std::sin(PI * (x - t));
    return 2.0 + s*s*s*s;
}

static double rhoExactCellavg(double xL, double xR, double t) {
    const double xi[5] = {
        -0.9061798459366350,-0.5384693101056831,0.0,
         0.5384693101056831, 0.9061798459366350
    };
    const double w[5] = {
        0.2369268850561891,0.4786286704993665,0.5688888888888889,
        0.4786286704993665,0.2369268850561891
    };
    const double xc=0.5*(xL+xR), h=0.5*(xR-xL);
    double val=0.0;
    for(int k=0;k<5;++k) val+=w[k]*rhoExactPoint(xc+h*xi[k],t);
    return 0.5*val;
}

static std::vector<Conserved> initSmooth(int nx, double x0, double x1) {
    const double dx=(x1-x0)/nx;
    std::vector<Conserved> U(nx);
    for(int i=0;i<nx;++i){
        double xL=x0+i*dx, xR=xL+dx;
        double rho=rhoExactCellavg(xL,xR,0.0);
        U[i].rho=rho;
        U[i].mom  =rho*1.0;
        U[i].E  =1.0/(Gamma-1.0)+0.5*rho;
    }
    return U;
}

void task_3_4_convergence() {
    const double x0=-1.0, x1=1.0, CFL=0.95;
    const double T=1000.0;
    const int nx=100;
    const double dx=(x1-x0)/nx;

    TTGRPOptions opt{};
    opt.use_exact_q0 = true;
    opt.flux_type    = TTRiemannFluxType::HLLC;

    auto t_start = std::chrono::high_resolution_clock::now();
    std::cerr<<"=== task_3_4_convergence: T="<<T<<", nx="<<nx<<" ===\n";

    auto Um=initSmooth(nx,x0,x1);
    aderweno::musclHancockRun(Um,x0,x1,T,CFL,true,aderweno::LimiterType::VanLeer);
    std::cerr<<"[MUSCL] done\n";

    auto U3=initSmooth(nx,x0,x1);
    aderweno::ader3Run(U3,x0,x1,T,CFL,true,opt);
    std::cerr<<"[ADER3] done\n";

    auto U4=initSmooth(nx,x0,x1);
    aderweno::ader4Run(U4,x0,x1,T,CFL,true,opt);
    std::cerr<<"[ADER4] done\n";

    auto U5=initSmooth(nx,x0,x1);
    aderweno::ader5Run(U5,x0,x1,T,CFL,true,opt);
    std::cerr<<"[ADER5] done\n";

    std::ofstream f("task_3_4_convergence.csv");
    f<<"x,rho_exact,rho_muscl,rho_ader3,rho_ader4,rho_ader5\n";
    f<<std::scientific<<std::setprecision(10);
    for(int i=0;i<nx;++i){
        double xc=x0+(i+0.5)*dx;
        f<<xc<<","<<rhoExactPoint(xc,T)<<","
         <<Um[i].rho<<","<<U3[i].rho<<","<<U4[i].rho<<","<<U5[i].rho<<"\n";
    }
    std::cerr<<"wrote task_3_4_convergence.csv\n";

    auto t_end = std::chrono::high_resolution_clock::now();
    double total_sec = std::chrono::duration<double>(t_end - t_start).count();
    int hours   = (int)(total_sec / 3600);
    int minutes = (int)((total_sec - hours*3600) / 60);
    int seconds = (int)(total_sec - hours*3600 - minutes*60);
    std::cerr<<"=== Total time: "<<hours<<" h "<<minutes<<" min "<<seconds<<" sec ===\n";
}