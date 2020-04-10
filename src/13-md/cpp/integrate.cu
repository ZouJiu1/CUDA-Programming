#include "integrate.cuh"
#include "force.cuh"
#include "error.cuh"
#include <stdio.h>
#include <math.h>
#include <time.h>

static real sum(int N, real *x)
{
    real s = 0.0;
    for (int n = 0; n < N; ++n) 
    {
        s += x[n];
    }
    return s;
}

static void scale_velocity(int N, real T_0, Atom *atom)
{
    real temperature = sum(N, atom->ke) / (1.5 * K_B * N);
    real scale_factor = sqrt(T_0 / temperature);
    for (int n = 0; n < N; ++n)
    { 
        atom->vx[n] *= scale_factor;
        atom->vy[n] *= scale_factor;
        atom->vz[n] *= scale_factor;
    }
}

static void integrate
(int N, real time_step, Atom *atom, int flag)
{
    real *m = atom->m;
    real *x = atom->x;
    real *y = atom->y;
    real *z = atom->z;
    real *vx = atom->vx;
    real *vy = atom->vy;
    real *vz = atom->vz;
    real *fx = atom->fx;
    real *fy = atom->fy;
    real *fz = atom->fz;
    real *ke = atom->ke;
    real time_step_half = time_step * 0.5;
    for (int n = 0; n < N; ++n)
    {
        real mass_inv = 1.0 / m[n];
        real ax = fx[n] * mass_inv;
        real ay = fy[n] * mass_inv;
        real az = fz[n] * mass_inv;
        vx[n] += ax * time_step_half;
        vy[n] += ay * time_step_half;
        vz[n] += az * time_step_half;
        if (flag == 1) 
        { 
            x[n] += vx[n] * time_step; 
            y[n] += vy[n] * time_step; 
            z[n] += vz[n] * time_step; 
        }
        else
        {
            real v2 = vx[n]*vx[n] + vy[n]*vy[n] + vz[n]*vz[n];
            ke[n] = m[n] * v2 * 0.5;
        }
    }
}

void equilibration
(
    int Ne, int N, int MN, real T_0, 
    real time_step, Atom *atom
)
{
    find_force(N, MN, atom);
    for (int step = 0; step < Ne; ++step)
    { 
        integrate(N, time_step, atom, 1);
        find_force(N, MN, atom);
        integrate(N, time_step, atom, 2);
        scale_velocity(N, T_0, atom);
    }
}

void production
(
    int Np, int Ns, int N, int MN, real T_0, 
    real time_step, Atom *atom
)
{
    float t_force = 0.0f;
    cudaEvent_t start_total, stop_total, start_force, stop_force;
    CHECK(cudaEventCreate(&start_total));
    CHECK(cudaEventCreate(&stop_total));
    CHECK(cudaEventCreate(&start_force));
    CHECK(cudaEventCreate(&stop_force));

    CHECK(cudaEventRecord(start_total));
    cudaEventQuery(start_total);

    FILE *fid = fopen("energy.txt", "w");
    for (int step = 0; step < Np; ++step)
    {  
        integrate(N, time_step, atom, 1);

        CHECK(cudaEventRecord(start_force));
        cudaEventQuery(start_force);
        find_force(N, MN, atom);
        CHECK(cudaEventRecord(stop_force));
        CHECK(cudaEventSynchronize(stop_force));
        float t_tmp;
        CHECK(cudaEventElapsedTime(&t_tmp, start_force, stop_force));
        t_force += t_tmp;

        integrate(N, time_step, atom, 2);

        if (0 == step % Ns)
        {
            fprintf(fid, "%g %g\n", sum(N, atom->ke), sum(N, atom->pe));
        }
    }
    fclose(fid);

    CHECK(cudaEventRecord(stop_total));
    CHECK(cudaEventSynchronize(stop_total));
    float t_total;
    CHECK(cudaEventElapsedTime(&t_total, start_total, stop_total));
    printf("Time used for production = %g s\n", t_total * 0.001);
    printf("Time used for force part = %g s\n", t_force * 0.001);

    CHECK(cudaEventDestroy(start_total));
    CHECK(cudaEventDestroy(stop_total));
    CHECK(cudaEventDestroy(start_force));
    CHECK(cudaEventDestroy(stop_force));
}


