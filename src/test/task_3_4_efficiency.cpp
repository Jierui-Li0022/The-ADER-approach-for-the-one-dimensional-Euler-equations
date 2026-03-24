// Efficiency test: CPU time vs L2 error for MUSCL and ADER3/4/5 over multiple grid sizes.

#include "aderweno/test/task_3_4_efficiency.hpp"

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

static double l2Error(const std::vector<Conserved>& U, double x0, double x1, double T) {
    const int nx=(int)U.size();
    const double dx=(x1-x0)/nx;
    double err=0.0;
    for(int i=0;i<nx;++i){
        double e=U[i].rho-rhoExactCellavg(x0+i*dx,x0+(i+1)*dx,T);
        err+=e*e*dx;
    }
    return std::sqrt(err);
}

struct EffRow { std::string scheme; int nx; double cpu_sec; double L2; };

static EffRow runTimedEff(const std::string& name, int nx,
                            double x0, double x1, double T, double CFL,
                            const TTGRPOptions& opt) {
    auto U=initSmooth(nx,x0,x1);
    auto t0=std::chrono::high_resolution_clock::now();

    if      (name=="MUSCL") aderweno::musclHancockRun(U,x0,x1,T,CFL,true,aderweno::LimiterType::VanLeer);
    else if (name=="ADER3") aderweno::ader3Run(U,x0,x1,T,CFL,true,opt);
    else if (name=="ADER4") aderweno::ader4Run(U,x0,x1,T,CFL,true,opt);
    else if (name=="ADER5") aderweno::ader5Run(U,x0,x1,T,CFL,true,opt);

    auto t1=std::chrono::high_resolution_clock::now();
    double cpu=std::chrono::duration<double>(t1-t0).count();
    double L2=l2Error(U,x0,x1,T);
    std::cerr<<name<<" nx="<<nx<<" cpu="<<cpu<<"s L2="<<L2<<"\n";
    return {name,nx,cpu,L2};
}

void task_3_4_efficiency() {
    const double x0=-1.0, x1=1.0, CFL=0.95;

    TTGRPOptions opt{};
    opt.use_exact_q0 = true;
    opt.flux_type    = TTRiemannFluxType::HLLC;

    auto t_start = std::chrono::high_resolution_clock::now();

    const double T = 100.0;
    const std::vector<int> nx_list = {20, 40, 80, 160, 320, 640, 1280, 2560};
    const std::vector<std::string> schemes = {"MUSCL", "ADER3", "ADER4", "ADER5"};

    std::vector<EffRow> rows;

    std::cerr << "=== task_3_4_efficiency: T=" << T << " ===\n";
    for (const auto& scheme : schemes) {
        for (int nx : nx_list) {
            rows.push_back(runTimedEff(scheme, nx, x0, x1, T, CFL, opt));
        }
    }

    std::ofstream f("task_3_4_efficiency.csv");
    f << "scheme,nx,cpu_sec,L2\n";
    f << std::scientific << std::setprecision(10);
    for (const auto& r : rows)
        f << r.scheme << "," << r.nx << "," << r.cpu_sec << "," << r.L2 << "\n";
    std::cerr << "wrote task_3_4_efficiency.csv\n";

    auto t_end = std::chrono::high_resolution_clock::now();
    double total_sec = std::chrono::duration<double>(t_end - t_start).count();
    int hours   = (int)(total_sec / 3600);
    int minutes = (int)((total_sec - hours*3600) / 60);
    int seconds = (int)(total_sec - hours*3600 - minutes*60);
    std::cerr << "=== Total time: " << hours << " h " << minutes << " min " << seconds << " sec ===\n";
}