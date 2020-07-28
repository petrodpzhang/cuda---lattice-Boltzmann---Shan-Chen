
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>
#include <cmath>
#include <time.h>
#include <fstream>
#include <sstream>
#include <string>
#include <cstdlib>
#include <algorithm>

using namespace std;

// 1. constant varibles

const int Nx = 16;
const int Ny = 16;
const int Nz = 16;
int Nlattice = Nx * Ny * Nz;
const double tauA = 1.0;
const double tauB = 1.0;

// 2. GPU and CPU arrays

double* d_ux, * h_ux;
double* d_uy, * h_uy;
double* d_uz, * h_uz;
double* d_rho_1, * h_rho_1;
double* d_rho_2, * h_rho_2;
double* d_psi_1, * h_psi_1;
double* d_psi_2, * h_psi_2;
double* d_f_1, * h_f_1;
double* d_f_2, * h_f_2;
double* d_f_post_1, * h_f_post_1;
double* d_f_post_2, * h_f_post_2;
double* d_Fx_1;
double* d_Fx_2;
double* d_Fy_1;
double* d_Fy_2;
double* d_Fz_1;
double* d_Fz_2;
double* press, * test, * testq;

// 3.D3Q19

const int Q = 19;
const int cx[19] = { 0,1,-1,0,0,0,0,1,-1,1,-1,0,0,1,-1,1,-1,0,0 };
const int cy[19] = { 0,0,0,1,-1,0,0,1,-1,0,0,1,-1,-1,1,0,0,1,-1 };
const int cz[19] = { 0,0,0,0,0,1,-1,0,0,1,-1,1,-1,0,0,-1,1,-1,1 };
const double w[19] = { 1. / 3,1. / 18 ,1. / 18 ,1. / 18 ,1. / 18 ,1. / 18 ,1. / 18 ,1. / 36,1. / 36, 1. / 36, 1. / 36, 1. / 36, 1. / 36, 1. / 36, 1. / 36, 1. / 36, 1. / 36, 1. / 36, 1. / 36 };

// 4. Assign a 3D distribution of CUDA

int threadsAlongX = 8;
int threadsAlongY = 8;
int threadsAlongZ = 8;

dim3 block(threadsAlongX, threadsAlongY, threadsAlongZ);
dim3 grid(1 + (Nx - 1) / threadsAlongX, 1 + (Ny - 1) / threadsAlongY, 1 + (Nz - 1) / threadsAlongZ);

// 5. initialize

void Initialization()
{
    double feq_1[19], feq_2[19];
    double tmp_rho_1, tmp_rho_2, tmp_ux, tmp_uy, tmp_uz, ux2, uy2, uz2, uxyz2, uxy2, uxz2, uyz2, uxy, uxz, uyz;

    for (int z = 0; z < Nz; z++)
    {
        for (int y = 0; y < Ny; y++)
        {
            for (int x = 0; x < Nx; x++)
            {
                int index = z * Nx * Ny + y * Nx + x;
                h_ux[index] = 0.0;
                h_uy[index] = 0.0;
                h_uz[index] = 0.0;
                h_psi_1[index] = 0.0;
                h_psi_2[index] = 0.0;
                if (z < Nz / 2.0)
                {
                    h_rho_1[index] = 0.1;
                    h_rho_2[index] = 0.9;
                }
                else
                {
                    h_rho_1[index] = 0.9;
                    h_rho_2[index] = 0.1;
                }
            }
        }
    }

    for (int z = 0; z < Nz; z++)
    {
        for (int y = 0; y < Ny; y++)
        {
            for (int x = 0; x < Nx; x++)
            {
                int index = z * Nx * Ny + y * Nx + x;

                tmp_rho_1 = h_rho_1[index];
                tmp_rho_2 = h_rho_2[index];
                tmp_ux = h_ux[index];
                tmp_uy = h_uy[index];
                tmp_uz = h_uz[index];
                ux2 = tmp_ux * tmp_ux;
                uy2 = tmp_uy * tmp_uy;
                uz2 = tmp_uz * tmp_uz;
                uxyz2 = ux2 + uy2 + uz2;
                uxy2 = ux2 + uy2;
                uxz2 = ux2 + uz2;
                uyz2 = uy2 + uz2;
                uxy = 2.0f * tmp_ux * tmp_uy;
                uxz = 2.0f * tmp_ux * tmp_uz;
                uyz = 2.0f * tmp_uy * tmp_uz;

                feq_1[0] = tmp_rho_1 * w[0] * (1.0f - 1.5f * uxyz2);
                feq_1[1] = tmp_rho_1 * w[1] * (1.0f + 3.0f * tmp_ux + 4.5f * ux2 - 1.5f * uxyz2);
                feq_1[2] = tmp_rho_1 * w[2] * (1.0f - 3.0f * tmp_ux + 4.5f * ux2 - 1.5f * uxyz2);
                feq_1[3] = tmp_rho_1 * w[3] * (1.0f + 3.0f * tmp_uy + 4.5f * uy2 - 1.5f * uxyz2);
                feq_1[4] = tmp_rho_1 * w[4] * (1.0f - 3.0f * tmp_uy + 4.5f * uy2 - 1.5f * uxyz2);
                feq_1[5] = tmp_rho_1 * w[5] * (1.0f + 3.0f * tmp_uz + 4.5f * uz2 - 1.5f * uxyz2);
                feq_1[6] = tmp_rho_1 * w[6] * (1.0f - 3.0f * tmp_uz + 4.5f * uz2 - 1.5f * uxyz2);
                feq_1[7] = tmp_rho_1 * w[7] * (1.0f + 3.0f * (tmp_ux + tmp_uy) + 4.5f * (uxy2 + uxy) - 1.5f * uxyz2);
                feq_1[8] = tmp_rho_1 * w[8] * (1.0f + 3.0f * (tmp_ux - tmp_uy) + 4.5f * (uxy2 - uxy) - 1.5f * uxyz2);
                feq_1[9] = tmp_rho_1 * w[9] * (1.0f + 3.0f * (tmp_uy - tmp_ux) + 4.5f * (uxy2 - uxy) - 1.5f * uxyz2);
                feq_1[10] = tmp_rho_1 * w[10] * (1.0f - 3.0f * (tmp_ux + tmp_uy) + 4.5f * (uxy2 + uxy) - 1.5f * uxyz2);
                feq_1[11] = tmp_rho_1 * w[11] * (1.0f + 3.0f * (tmp_ux + tmp_uz) + 4.5f * (uxz2 + uxz) - 1.5f * uxyz2);
                feq_1[12] = tmp_rho_1 * w[12] * (1.0f + 3.0f * (tmp_ux - tmp_uz) + 4.5f * (uxz2 - uxz) - 1.5f * uxyz2);
                feq_1[13] = tmp_rho_1 * w[13] * (1.0f + 3.0f * (tmp_uz - tmp_ux) + 4.5f * (uxz2 - uxz) - 1.5f * uxyz2);
                feq_1[14] = tmp_rho_1 * w[14] * (1.0f - 3.0f * (tmp_ux + tmp_uz) + 4.5f * (uxz2 + uxz) - 1.5f * uxyz2);
                feq_1[15] = tmp_rho_1 * w[15] * (1.0f + 3.0f * (tmp_uy + tmp_uz) + 4.5f * (uyz2 + uyz) - 1.5f * uxyz2);
                feq_1[16] = tmp_rho_1 * w[16] * (1.0f + 3.0f * (tmp_uz - tmp_uy) + 4.5f * (uyz2 - uyz) - 1.5f * uxyz2);
                feq_1[17] = tmp_rho_1 * w[17] * (1.0f + 3.0f * (tmp_uy - tmp_uz) + 4.5f * (uyz2 - uyz) - 1.5f * uxyz2);
                feq_1[18] = tmp_rho_1 * w[18] * (1.0f - 3.0f * (tmp_uy + tmp_uz) + 4.5f * (uyz2 + uyz) - 1.5f * uxyz2);

                feq_2[0] = tmp_rho_2 * w[0] * (1.0f - 1.5f * uxyz2);
                feq_2[1] = tmp_rho_2 * w[1] * (1.0f + 3.0f * tmp_ux + 4.5f * ux2 - 1.5f * uxyz2);
                feq_2[2] = tmp_rho_2 * w[2] * (1.0f - 3.0f * tmp_ux + 4.5f * ux2 - 1.5f * uxyz2);
                feq_2[3] = tmp_rho_2 * w[3] * (1.0f + 3.0f * tmp_uy + 4.5f * uy2 - 1.5f * uxyz2);
                feq_2[4] = tmp_rho_2 * w[4] * (1.0f - 3.0f * tmp_uy + 4.5f * uy2 - 1.5f * uxyz2);
                feq_2[5] = tmp_rho_2 * w[5] * (1.0f + 3.0f * tmp_uz + 4.5f * uz2 - 1.5f * uxyz2);
                feq_2[6] = tmp_rho_2 * w[6] * (1.0f - 3.0f * tmp_uz + 4.5f * uz2 - 1.5f * uxyz2);
                feq_2[7] = tmp_rho_2 * w[7] * (1.0f + 3.0f * (tmp_ux + tmp_uy) + 4.5f * (uxy2 + uxy) - 1.5f * uxyz2);
                feq_2[8] = tmp_rho_2 * w[8] * (1.0f + 3.0f * (tmp_ux - tmp_uy) + 4.5f * (uxy2 - uxy) - 1.5f * uxyz2);
                feq_2[9] = tmp_rho_2 * w[9] * (1.0f + 3.0f * (tmp_uy - tmp_ux) + 4.5f * (uxy2 - uxy) - 1.5f * uxyz2);
                feq_2[10] = tmp_rho_2 * w[10] * (1.0f - 3.0f * (tmp_ux + tmp_uy) + 4.5f * (uxy2 + uxy) - 1.5f * uxyz2);
                feq_2[11] = tmp_rho_2 * w[11] * (1.0f + 3.0f * (tmp_ux + tmp_uz) + 4.5f * (uxz2 + uxz) - 1.5f * uxyz2);
                feq_2[12] = tmp_rho_2 * w[12] * (1.0f + 3.0f * (tmp_ux - tmp_uz) + 4.5f * (uxz2 - uxz) - 1.5f * uxyz2);
                feq_2[13] = tmp_rho_2 * w[13] * (1.0f + 3.0f * (tmp_uz - tmp_ux) + 4.5f * (uxz2 - uxz) - 1.5f * uxyz2);
                feq_2[14] = tmp_rho_2 * w[14] * (1.0f - 3.0f * (tmp_ux + tmp_uz) + 4.5f * (uxz2 + uxz) - 1.5f * uxyz2);
                feq_2[15] = tmp_rho_2 * w[15] * (1.0f + 3.0f * (tmp_uy + tmp_uz) + 4.5f * (uyz2 + uyz) - 1.5f * uxyz2);
                feq_2[16] = tmp_rho_2 * w[16] * (1.0f + 3.0f * (tmp_uz - tmp_uy) + 4.5f * (uyz2 - uyz) - 1.5f * uxyz2);
                feq_2[17] = tmp_rho_2 * w[17] * (1.0f + 3.0f * (tmp_uy - tmp_uz) + 4.5f * (uyz2 - uyz) - 1.5f * uxyz2);
                feq_2[18] = tmp_rho_2 * w[18] * (1.0f - 3.0f * (tmp_uy + tmp_uz) + 4.5f * (uyz2 + uyz) - 1.5f * uxyz2);

                for (int k = 0; k < Q; k++)
                {
                    h_f_1[Nlattice * k + index] = feq_1[k];
                    h_f_post_1[Nlattice * k + index] = feq_1[k];
                    h_f_2[Nlattice * k + index] = feq_2[k];
                    h_f_post_2[Nlattice * k + index] = feq_2[k];
                }
            }
        }
    }
}

// 6. Density

__global__ void computeDensity(int Nlattice, double* __restrict__ d_f_1, double* __restrict__ d_f_2, double* __restrict__ d_rho_1, double* __restrict__ d_rho_2, double* __restrict__ d_psi_1, double* __restrict__ d_psi_2)
{
    
    const int rho0 = 1.0;
    const double gA = -4.7;
    const double gB = 0.0;
    const double gAB = 6.0;
    const int cx[19] = { 0,1,-1,0,0,0,0,1,-1,1,-1,0,0,1,-1,1,-1,0,0 };
    const int cy[19] = { 0,0,0,1,-1,0,0,1,-1,0,0,1,-1,-1,1,0,0,1,-1 };
    const int cz[19] = { 0,0,0,0,0,1,-1,0,0,1,-1,1,-1,0,0,-1,1,-1,1 };
    const double w[19] = { 1. / 3,1. / 18 ,1. / 18 ,1. / 18 ,1. / 18 ,1. / 18 ,1. / 18 ,1. / 36,1. / 36, 1. / 36, 1. / 36, 1. / 36, 1. / 36, 1. / 36, 1. / 36, 1. / 36, 1. / 36, 1. / 36, 1. / 36 };

    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    int k = threadIdx.z + blockIdx.z * blockDim.z;
    int index = k * Nx * Ny + j * Nx + i;
        
    d_rho_1[index] = d_f_1[index] + d_f_1[index + Nlattice] + d_f_1[index + Nlattice * 2] + d_f_1[index + Nlattice * 3] + d_f_1[index + Nlattice * 4] + d_f_1[index + Nlattice * 5] + d_f_1[index + Nlattice * 6] +
        d_f_1[index + Nlattice * 7] + d_f_1[index + Nlattice * 8] + d_f_1[index + Nlattice * 9] + d_f_1[index + Nlattice * 10] + d_f_1[index + Nlattice * 11] + d_f_1[index + Nlattice * 12] +
        d_f_1[index + Nlattice * 13] + d_f_1[index + Nlattice * 14] + d_f_1[index + Nlattice * 15] + d_f_1[index + Nlattice * 16] + d_f_1[index + Nlattice * 17] + d_f_1[index + Nlattice * 18];

    d_rho_2[index] = d_f_2[index] + d_f_2[index + Nlattice] + d_f_2[index + Nlattice * 2] + d_f_2[index + Nlattice * 3] + d_f_2[index + Nlattice * 4] + d_f_2[index + Nlattice * 5] + d_f_2[index + Nlattice * 6] +
        d_f_2[index + Nlattice * 7] + d_f_2[index + Nlattice * 8] + d_f_2[index + Nlattice * 9] + d_f_2[index + Nlattice * 10] + d_f_2[index + Nlattice * 11] + d_f_2[index + Nlattice * 12] +
        d_f_2[index + Nlattice * 13] + d_f_2[index + Nlattice * 14] + d_f_2[index + Nlattice * 15] + d_f_2[index + Nlattice * 16] + d_f_2[index + Nlattice * 17] + d_f_2[index + Nlattice * 18];

    d_psi_1[index] = rho0 * (1.0 - exp(-d_rho_1[index] / rho0));
    d_psi_2[index] = rho0 * (1.0 - exp(-d_rho_2[index] / rho0));
    
}

// 7. force

__global__ void computeSCForces(double* __restrict__ d_psi_1, double* __restrict__ d_psi_2, double* __restrict__ d_Fx_1, double* __restrict__ d_Fy_1, double* __restrict__ d_Fz_1, double* __restrict__ d_Fx_2, double* __restrict__ d_Fy_2, double* __restrict__ d_Fz_2)
{
    const int rho0 = 1.0;
    const double gA = -4.7;
    const double gB = 0.0;
    const double gAB = 6.0;
    
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    int k = threadIdx.z + blockIdx.z * blockDim.z;
    int index = k * Nx * Ny + j * Nx + i;

    int i_1 = (i + 1 + Nx) % Nx;
    int j_1 = j;
    int k_1 = k;
    int index_1 = k_1 * Nx * Ny + j_1 * Nx + i_1;

    int i_2 = (i - 1 + Nx) % Nx;
    int j_2 = j;
    int k_2 = k;
    int index_2 = k_2 * Nx * Ny + j_2 * Nx + i_2;

    int i_3 = i;
    int j_3 = (j + 1 + Ny) % Ny;
    int k_3 = k;
    int index_3 = k_3 * Nx * Ny + j_3 * Nx + i_3;

    int i_4 = i;
    int j_4 = (j - 1 + Ny) % Ny;
    int k_4 = k;
    int index_4 = k_4 * Nx * Ny + j_4 * Nx + i_4;

    int i_5 = i;
    int j_5 = j;
    int k_5 = (k + 1 + Nz) % Nz;
    int index_5 = k_5 * Nx * Ny + j_5 * Nx + i_5;

    int i_6 = i;
    int j_6 = j;
    int k_6 = (k - 1 + Nz) % Nz;
    int index_6 = k_6 * Nx * Ny + j_6 * Nx + i_6;

    int i_7 = (i + 1 + Nx) % Nx;
    int j_7 = (j + 1 + Ny) % Ny;
    int k_7 = k;
    int index_7 = k_7 * Nx * Ny + j_7 * Nx + i_7;

    int i_8 = (i - 1 + Nx) % Nx;
    int j_8 = (j - 1 + Ny) % Ny;
    int k_8 = k;
    int index_8 = k_8 * Nx * Ny + j_8 * Nx + i_8;

    int i_9 = (i + 1 + Nx) % Nx;
    int j_9 = j;
    int k_9 = (k + 1 + Nz) % Nz;
    int index_9 = k_9 * Nx * Ny + j_9 * Nx + i_9;

    int i_10 = (i - 1 + Nx) % Nx;
    int j_10 = j;
    int k_10 = (k - 1 + Nz) % Nz;
    int index_10 = k_10 * Nx * Ny + j_10 * Nx + i_10;

    int i_11 = i;
    int j_11 = (j + 1 + Ny) % Ny;
    int k_11 = (k + 1 + Nz) % Nz;
    int index_11 = k_11 * Nx * Ny + j_11 * Nx + i_11;

    int i_12 = i;
    int j_12 = (j - 1 + Ny) % Ny;
    int k_12 = (k - 1 + Nz) % Nz;
    int index_12 = k_12 * Nx * Ny + j_12 * Nx + i_12;

    int i_13 = (i + 1 + Nx) % Nx;
    int j_13 = (j - 1 + Ny) % Ny;
    int k_13 = k;
    int index_13 = k_13 * Nx * Ny + j_13 * Nx + i_13;

    int i_14 = (i - 1 + Nx) % Nx;
    int j_14 = (j + 1 + Ny) % Ny;
    int k_14 = k;
    int index_14 = k_14 * Nx * Ny + j_14 * Nx + i_14;

    int i_15 = (i + 1 + Nx) % Nx;
    int j_15 = j;
    int k_15 = (k - 1 + Nz) % Nz;
    int index_15 = k_15 * Nx * Ny + j_15 * Nx + i_15;

    int i_16 = (i - 1 + Nx) % Nx;
    int j_16 = j;
    int k_16 = (k + 1 + Nz) % Nz;
    int index_16 = k_16 * Nx * Ny + j_16 * Nx + i_16;

    int i_17 = i;
    int j_17 = (j + 1 + Ny) % Ny;
    int k_17 = (k - 1 + Nz) % Nz;
    int index_17 = k_17 * Nx * Ny + j_17 * Nx + i_17;

    int i_18 = i;
    int j_18 = (j - 1 + Ny) % Ny;
    int k_18 = (k + 1 + Nz) % Nz;
    int index_18 = k_18 * Nx * Ny + j_18 * Nx + i_18;

    
    d_Fx_1[index] = (1. / 18) * d_psi_1[index] * (4.7 * d_psi_1[index_1] - 6.0 * d_psi_2[index_1]) + (-1. / 18) * d_psi_1[index] * (4.7 * d_psi_1[index_2] - 6.0 * d_psi_2[index_2])
        + (1. / 36) * d_psi_1[index] * (4.7 * d_psi_1[index_7] - 6.0 * d_psi_2[index_7]) + (-1. / 36) * d_psi_1[index] * (4.7 * d_psi_1[index_8] - 6.0 * d_psi_2[index_8])
        + (1. / 36) * d_psi_1[index] * (4.7 * d_psi_1[index_9] - 6.0 * d_psi_2[index_9]) + (-1. / 36) * d_psi_1[index] * (4.7 * d_psi_1[index_10] - 6.0 * d_psi_2[index_10])
        + (1. / 36) * d_psi_1[index] * (4.7 * d_psi_1[index_13] - 6.0 * d_psi_2[index_13]) + (-1. / 36) * d_psi_1[index] * (4.7 * d_psi_1[index_14] - 6.0 * d_psi_2[index_14])
        + (1. / 36) * d_psi_1[index] * (4.7 * d_psi_1[index_15] - 6.0 * d_psi_2[index_15]) + (-1. / 36) * d_psi_1[index] * (4.7 * d_psi_1[index_16] - 6.0 * d_psi_2[index_16]);
    
    d_Fy_1[index] = (1. / 18) * d_psi_1[index] * (4.7 * d_psi_1[index_3] - 6.0 * d_psi_2[index_3]) + (-1. / 18) * d_psi_1[index] * (4.7 * d_psi_1[index_4] - 6.0 * d_psi_2[index_4])
        + (1. / 36) * d_psi_1[index] * (4.7 * d_psi_1[index_7] - 6.0 * d_psi_2[index_7]) + (-1. / 36) * d_psi_1[index] * (4.7 * d_psi_1[index_8] - 6.0 * d_psi_2[index_8])
        + (1. / 36) * d_psi_1[index] * (4.7 * d_psi_1[index_11] - 6.0 * d_psi_2[index_11]) + (-1. / 36) * d_psi_1[index] * (4.7 * d_psi_1[index_12] - 6.0 * d_psi_2[index_12])
        - (1. / 36) * d_psi_1[index] * (4.7 * d_psi_1[index_13] - 6.0 * d_psi_2[index_13]) + (1. / 36) * d_psi_1[index] * (4.7 * d_psi_1[index_14] - 6.0 * d_psi_2[index_14])
        + (1. / 36) * d_psi_1[index] * (4.7 * d_psi_1[index_17] - 6.0 * d_psi_2[index_17]) + (-1. / 36) * d_psi_1[index] * (4.7 * d_psi_1[index_18] - 6.0 * d_psi_2[index_18]);
    
    d_Fz_1[index] = (1. / 18) * d_psi_1[index] * (4.7 * d_psi_1[index_5] - 6.0 * d_psi_2[index_5]) + (-1. / 18) * d_psi_1[index] * (4.7 * d_psi_1[index_6] - 6.0 * d_psi_2[index_6])
        + (1. / 36) * d_psi_1[index] * (4.7 * d_psi_1[index_9] - 6.0 * d_psi_2[index_9]) + (-1. / 36) * d_psi_1[index] * (4.7 * d_psi_1[index_10] - 6.0 * d_psi_2[index_10])
        + (1. / 36) * d_psi_1[index] * (4.7 * d_psi_1[index_11] - 6.0 * d_psi_2[index_11]) + (-1. / 36) * d_psi_1[index] * (4.7 * d_psi_1[index_12] - 6.0 * d_psi_2[index_12])
        - (1. / 36) * d_psi_1[index] * (4.7 * d_psi_1[index_15] - 6.0 * d_psi_2[index_15]) + (1. / 36) * d_psi_1[index] * (4.7 * d_psi_1[index_16] - 6.0 * d_psi_2[index_16])
        - (1. / 36) * d_psi_1[index] * (4.7 * d_psi_1[index_17] - 6.0 * d_psi_2[index_17]) + (-1. / 36) * d_psi_1[index] * (4.7 * d_psi_1[index_18] - 6.0 * d_psi_2[index_18]);
    
    d_Fx_2[index] = (1. / 18) * d_psi_2[index] * (0.0 * d_psi_2[index_1] - 6.0 * d_psi_1[index_1]) + (-1. / 18) * d_psi_2[index] * (0.0 * d_psi_2[index_2] - 6.0 * d_psi_1[index_2])
        + (1. / 36) * d_psi_2[index] * (0.0 * d_psi_2[index_7] - 6.0 * d_psi_1[index_7]) + (-1. / 36) * d_psi_2[index] * (0.0 * d_psi_2[index_8] - 6.0 * d_psi_1[index_8])
        + (1. / 36) * d_psi_2[index] * (0.0 * d_psi_2[index_9] - 6.0 * d_psi_1[index_9]) + (-1. / 36) * d_psi_2[index] * (0.0 * d_psi_2[index_10] - 6.0 * d_psi_1[index_10])
        + (1. / 36) * d_psi_2[index] * (0.0 * d_psi_2[index_13] - 6.0 * d_psi_1[index_13]) + (-1. / 36) * d_psi_2[index] * (0.0 * d_psi_2[index_14] - 6.0 * d_psi_1[index_14])
        + (1. / 36) * d_psi_2[index] * (0.0 * d_psi_2[index_15] - 6.0 * d_psi_1[index_15]) + (-1. / 36) * d_psi_2[index] * (0.0 * d_psi_2[index_16] - 6.0 * d_psi_1[index_16]);
    
    d_Fy_2[index] = (1. / 18) * d_psi_2[index] * (0.0 * d_psi_2[index_3] - 6.0 * d_psi_1[index_3]) + (-1. / 18) * d_psi_2[index] * (0.0 * d_psi_2[index_4] - 6.0 * d_psi_1[index_4])
        + (1. / 36) * d_psi_2[index] * (0.0 * d_psi_2[index_7] - 6.0 * d_psi_1[index_7]) + (-1. / 36) * d_psi_2[index] * (0.0 * d_psi_2[index_8] - 6.0 * d_psi_1[index_8])
        + (1. / 36) * d_psi_2[index] * (0.0 * d_psi_2[index_11] - 6.0 * d_psi_1[index_11]) + (-1. / 36) * d_psi_2[index] * (0.0 * d_psi_2[index_12] - 6.0 * d_psi_1[index_12])
        - (1. / 36) * d_psi_2[index] * (0.0 * d_psi_2[index_13] - 6.0 * d_psi_1[index_13]) + (1. / 36) * d_psi_2[index] * (0.0 * d_psi_2[index_14] - 6.0 * d_psi_1[index_14])
        + (1. / 36) * d_psi_2[index] * (0.0 * d_psi_2[index_17] - 6.0 * d_psi_1[index_17]) + (-1. / 36) * d_psi_2[index] * (0.0 * d_psi_2[index_18] - 6.0 * d_psi_1[index_18]);
    
    d_Fz_2[index] = (1. / 18) * d_psi_2[index] * (0.0 * d_psi_2[index_5] - 6.0 * d_psi_1[index_5]) + (-1. / 18) * d_psi_2[index] * (0.0 * d_psi_2[index_6] - 6.0 * d_psi_1[index_6])
        + (1. / 36) * d_psi_2[index] * (0.0 * d_psi_2[index_9] - 6.0 * d_psi_1[index_9]) + (-1. / 36) * d_psi_2[index] * (0.0 * d_psi_2[index_10] - 6.0 * d_psi_1[index_10])
        + (1. / 36) * d_psi_2[index] * (0.0 * d_psi_2[index_11] - 6.0 * d_psi_1[index_11]) + (-1. / 36) * d_psi_2[index] * (0.0 * d_psi_2[index_12] - 6.0 * d_psi_1[index_12])
        - (1. / 36) * d_psi_2[index] * (0.0 * d_psi_2[index_15] - 6.0 * d_psi_1[index_15]) + (1. / 36) * d_psi_2[index] * (0.0 * d_psi_2[index_16] - 6.0 * d_psi_1[index_16])
        - (1. / 36) * d_psi_2[index] * (0.0 * d_psi_2[index_17] - 6.0 * d_psi_1[index_17]) + (-1. / 36) * d_psi_2[index] * (0.0 * d_psi_2[index_18] - 6.0 * d_psi_1[index_18]);

}

// 8. velocity

__global__ void computeVelocity(double* __restrict__ d_f_1, double* __restrict__ d_f_2, double* __restrict__ d_rho_1, double* __restrict__ d_rho_2, double* __restrict__ d_Fx_1, double* __restrict__ d_Fy_1, double* __restrict__ d_Fz_1, double* __restrict__ d_Fx_2, double* __restrict__ d_Fy_2, double* __restrict__ d_Fz_2, double* __restrict__ d_ux, double* __restrict__ d_uy, double* __restrict__ d_uz)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    int k = threadIdx.z + blockIdx.z * blockDim.z;
    int index = k * Nx * Ny + j * Nx + i;
    int Nlattice = Nx * Ny * Nz;
    
    d_ux[index] = (d_f_1[index + Nlattice] - d_f_1[index + Nlattice * 2] - d_f_1[index + Nlattice * 7] - d_f_1[index + Nlattice * 8] + d_f_1[index + Nlattice * 9] - d_f_1[index + Nlattice * 10]
        + d_f_1[index + Nlattice * 11] + d_f_1[index + Nlattice * 12] + d_f_1[index + Nlattice * 13] - d_f_1[index + Nlattice * 14] + d_f_1[index + Nlattice * 15] - d_f_1[index + Nlattice * 16]
        + d_f_2[index + Nlattice] - d_f_2[index + Nlattice * 2] - d_f_2[index + Nlattice * 7] - d_f_2[index + Nlattice * 8] + d_f_2[index + Nlattice * 9] - d_f_2[index + Nlattice * 10]
        + d_f_2[index + Nlattice * 11] + d_f_2[index + Nlattice * 12] + d_f_2[index + Nlattice * 13] - d_f_2[index + Nlattice * 14] + d_f_2[index + Nlattice * 15] - d_f_2[index + Nlattice * 16]
        + 0.5 * (d_Fx_1[index] + d_Fx_2[index])) / (d_rho_1[index] + d_rho_2[index]);

    d_uy[index] = (d_f_1[index + Nlattice] - d_f_1[index + Nlattice * 2] - d_f_1[index + Nlattice * 7] - d_f_1[index + Nlattice * 8] + d_f_1[index + Nlattice * 9] - d_f_1[index + Nlattice * 10]
        + d_f_1[index + Nlattice * 11] + d_f_1[index + Nlattice * 12] + d_f_1[index + Nlattice * 13] - d_f_1[index + Nlattice * 14] + d_f_1[index + Nlattice * 15] - d_f_1[index + Nlattice * 16]
        + d_f_2[index + Nlattice] - d_f_2[index + Nlattice * 2] - d_f_2[index + Nlattice * 7] - d_f_2[index + Nlattice * 8] + d_f_2[index + Nlattice * 9] - d_f_2[index + Nlattice * 10]
        + d_f_2[index + Nlattice * 11] + d_f_2[index + Nlattice * 12] + d_f_2[index + Nlattice * 13] - d_f_2[index + Nlattice * 14] + d_f_2[index + Nlattice * 15] - d_f_2[index + Nlattice * 16]
        + 0.5 * (d_Fy_1[index] + d_Fy_2[index])) / (d_rho_1[index] + d_rho_2[index]);

    d_uz[index] = (d_f_1[index + Nlattice] - d_f_1[index + Nlattice * 2] - d_f_1[index + Nlattice * 7] - d_f_1[index + Nlattice * 8] + d_f_1[index + Nlattice * 9] - d_f_1[index + Nlattice * 10]
        + d_f_1[index + Nlattice * 11] + d_f_1[index + Nlattice * 12] + d_f_1[index + Nlattice * 13] - d_f_1[index + Nlattice * 14] + d_f_1[index + Nlattice * 15] - d_f_1[index + Nlattice * 16]
        + d_f_2[index + Nlattice] - d_f_2[index + Nlattice * 2] - d_f_2[index + Nlattice * 7] - d_f_2[index + Nlattice * 8] + d_f_2[index + Nlattice * 9] - d_f_2[index + Nlattice * 10]
        + d_f_2[index + Nlattice * 11] + d_f_2[index + Nlattice * 12] + d_f_2[index + Nlattice * 13] - d_f_2[index + Nlattice * 14] + d_f_2[index + Nlattice * 15] - d_f_2[index + Nlattice * 16]
        + 0.5 * (d_Fz_1[index] + d_Fz_2[index])) / (d_rho_1[index] + d_rho_2[index]);
}

// 9. stream

__global__ void stream(double* __restrict__ d_f_1, double* __restrict__ d_f_2, double* __restrict__ d_f_post_1, double* __restrict__ d_f_post_2, double* __restrict__ d_Fx_1, double* __restrict__ d_Fy_1, double* __restrict__ d_Fz_1, double* __restrict__ d_Fx_2, double* __restrict__ d_Fy_2, double* __restrict__ d_Fz_2, double* __restrict__ d_ux, double* __restrict__ d_uy, double* __restrict__ d_uz, double* __restrict__ d_rho_1, double* __restrict__ d_rho_2)
{
    double feq_1[19], feq_2[19]; 
    int indexf[19];
    double tmp_rho_1, tmp_rho_2, tmp_ux, tmp_uy, tmp_uz, ux2, uy2, uz2, uxyz2, uxy2, uxz2, uyz2, uxy, uxz, uyz;
    const double tauA = 1.0;
    const double tauB = 1.0;
    double omega_1 = 1.0 / tauA;
    double omega_2 = 1.0 / tauB;
    const int Q = 19;
    const int cx[19] = { 0,1,-1,0,0,0,0,1,-1,1,-1,0,0,1,-1,1,-1,0,0 };
    const int cy[19] = { 0,0,0,1,-1,0,0,1,-1,0,0,1,-1,-1,1,0,0,1,-1 };
    const int cz[19] = { 0,0,0,0,0,1,-1,0,0,1,-1,1,-1,0,0,-1,1,-1,1 };
    const double w[19] = { 1. / 3,1. / 18 ,1. / 18 ,1. / 18 ,1. / 18 ,1. / 18 ,1. / 18 ,1. / 36,1. / 36, 1. / 36, 1. / 36, 1. / 36, 1. / 36, 1. / 36, 1. / 36, 1. / 36, 1. / 36, 1. / 36, 1. / 36 };


    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    int k = threadIdx.z + blockIdx.z * blockDim.z;
    int index = k * Nx * Ny + j * Nx + i;
    int Nlattice = Nx * Ny * Nz;
    
    for (int q = 0; q < 19; q++)
    {
        int i_1 = (i + cx[q] + Nx) % Nx;
        int j_1 = (j + cy[q] + Ny) % Ny;
        int k_1 = (k + cz[q] + Nz) % Nz;
        indexf[q] = k_1 * Nx * Ny + j_1 * Nx + i_1;
    }

    tmp_rho_1 = d_rho_1[index];
    tmp_rho_2 = d_rho_2[index];
    tmp_ux = d_ux[index];
    tmp_uy = d_uy[index];
    tmp_uz = d_uz[index];
    ux2 = tmp_ux * tmp_ux;
    uy2 = tmp_uy * tmp_uy;
    uz2 = tmp_uz * tmp_uz;
    uxyz2 = ux2 + uy2 + uz2;
    uxy2 = ux2 + uy2;
    uxz2 = ux2 + uz2;
    uyz2 = uy2 + uz2;
    uxy = 2.0f * tmp_ux * tmp_uy;
    uxz = 2.0f * tmp_ux * tmp_uz;
    uyz = 2.0f * tmp_uy * tmp_uz;

    feq_1[0] = tmp_rho_1 * w[0] * (1.0f - 1.5f * uxyz2);
    feq_1[1] = tmp_rho_1 * w[1] * (1.0f + 3.0f * tmp_ux + 4.5f * ux2 - 1.5f * uxyz2);
    feq_1[2] = tmp_rho_1 * w[2] * (1.0f - 3.0f * tmp_ux + 4.5f * ux2 - 1.5f * uxyz2);
    feq_1[3] = tmp_rho_1 * w[3] * (1.0f + 3.0f * tmp_uy + 4.5f * uy2 - 1.5f * uxyz2);
    feq_1[4] = tmp_rho_1 * w[4] * (1.0f - 3.0f * tmp_uy + 4.5f * uy2 - 1.5f * uxyz2);
    feq_1[5] = tmp_rho_1 * w[5] * (1.0f + 3.0f * tmp_uz + 4.5f * uz2 - 1.5f * uxyz2);
    feq_1[6] = tmp_rho_1 * w[6] * (1.0f - 3.0f * tmp_uz + 4.5f * uz2 - 1.5f * uxyz2);
    feq_1[7] = tmp_rho_1 * w[7] * (1.0f + 3.0f * (tmp_ux + tmp_uy) + 4.5f * (uxy2 + uxy) - 1.5f * uxyz2);
    feq_1[8] = tmp_rho_1 * w[8] * (1.0f + 3.0f * (tmp_ux - tmp_uy) + 4.5f * (uxy2 - uxy) - 1.5f * uxyz2);
    feq_1[9] = tmp_rho_1 * w[9] * (1.0f + 3.0f * (tmp_uy - tmp_ux) + 4.5f * (uxy2 - uxy) - 1.5f * uxyz2);
    feq_1[10] = tmp_rho_1 * w[10] * (1.0f - 3.0f * (tmp_ux + tmp_uy) + 4.5f * (uxy2 + uxy) - 1.5f * uxyz2);
    feq_1[11] = tmp_rho_1 * w[11] * (1.0f + 3.0f * (tmp_ux + tmp_uz) + 4.5f * (uxz2 + uxz) - 1.5f * uxyz2);
    feq_1[12] = tmp_rho_1 * w[12] * (1.0f + 3.0f * (tmp_ux - tmp_uz) + 4.5f * (uxz2 - uxz) - 1.5f * uxyz2);
    feq_1[13] = tmp_rho_1 * w[13] * (1.0f + 3.0f * (tmp_uz - tmp_ux) + 4.5f * (uxz2 - uxz) - 1.5f * uxyz2);
    feq_1[14] = tmp_rho_1 * w[14] * (1.0f - 3.0f * (tmp_ux + tmp_uz) + 4.5f * (uxz2 + uxz) - 1.5f * uxyz2);
    feq_1[15] = tmp_rho_1 * w[15] * (1.0f + 3.0f * (tmp_uy + tmp_uz) + 4.5f * (uyz2 + uyz) - 1.5f * uxyz2);
    feq_1[16] = tmp_rho_1 * w[16] * (1.0f + 3.0f * (tmp_uz - tmp_uy) + 4.5f * (uyz2 - uyz) - 1.5f * uxyz2);
    feq_1[17] = tmp_rho_1 * w[17] * (1.0f + 3.0f * (tmp_uy - tmp_uz) + 4.5f * (uyz2 - uyz) - 1.5f * uxyz2);
    feq_1[18] = tmp_rho_1 * w[18] * (1.0f - 3.0f * (tmp_uy + tmp_uz) + 4.5f * (uyz2 + uyz) - 1.5f * uxyz2);

    feq_2[0] = tmp_rho_2 * w[0] * (1.0f - 1.5f * uxyz2);
    feq_2[1] = tmp_rho_2 * w[1] * (1.0f + 3.0f * tmp_ux + 4.5f * ux2 - 1.5f * uxyz2);
    feq_2[2] = tmp_rho_2 * w[2] * (1.0f - 3.0f * tmp_ux + 4.5f * ux2 - 1.5f * uxyz2);
    feq_2[3] = tmp_rho_2 * w[3] * (1.0f + 3.0f * tmp_uy + 4.5f * uy2 - 1.5f * uxyz2);
    feq_2[4] = tmp_rho_2 * w[4] * (1.0f - 3.0f * tmp_uy + 4.5f * uy2 - 1.5f * uxyz2);
    feq_2[5] = tmp_rho_2 * w[5] * (1.0f + 3.0f * tmp_uz + 4.5f * uz2 - 1.5f * uxyz2);
    feq_2[6] = tmp_rho_2 * w[6] * (1.0f - 3.0f * tmp_uz + 4.5f * uz2 - 1.5f * uxyz2);
    feq_2[7] = tmp_rho_2 * w[7] * (1.0f + 3.0f * (tmp_ux + tmp_uy) + 4.5f * (uxy2 + uxy) - 1.5f * uxyz2);
    feq_2[8] = tmp_rho_2 * w[8] * (1.0f + 3.0f * (tmp_ux - tmp_uy) + 4.5f * (uxy2 - uxy) - 1.5f * uxyz2);
    feq_2[9] = tmp_rho_2 * w[9] * (1.0f + 3.0f * (tmp_uy - tmp_ux) + 4.5f * (uxy2 - uxy) - 1.5f * uxyz2);
    feq_2[10] = tmp_rho_2 * w[10] * (1.0f - 3.0f * (tmp_ux + tmp_uy) + 4.5f * (uxy2 + uxy) - 1.5f * uxyz2);
    feq_2[11] = tmp_rho_2 * w[11] * (1.0f + 3.0f * (tmp_ux + tmp_uz) + 4.5f * (uxz2 + uxz) - 1.5f * uxyz2);
    feq_2[12] = tmp_rho_2 * w[12] * (1.0f + 3.0f * (tmp_ux - tmp_uz) + 4.5f * (uxz2 - uxz) - 1.5f * uxyz2);
    feq_2[13] = tmp_rho_2 * w[13] * (1.0f + 3.0f * (tmp_uz - tmp_ux) + 4.5f * (uxz2 - uxz) - 1.5f * uxyz2);
    feq_2[14] = tmp_rho_2 * w[14] * (1.0f - 3.0f * (tmp_ux + tmp_uz) + 4.5f * (uxz2 + uxz) - 1.5f * uxyz2);
    feq_2[15] = tmp_rho_2 * w[15] * (1.0f + 3.0f * (tmp_uy + tmp_uz) + 4.5f * (uyz2 + uyz) - 1.5f * uxyz2);
    feq_2[16] = tmp_rho_2 * w[16] * (1.0f + 3.0f * (tmp_uz - tmp_uy) + 4.5f * (uyz2 - uyz) - 1.5f * uxyz2);
    feq_2[17] = tmp_rho_2 * w[17] * (1.0f + 3.0f * (tmp_uy - tmp_uz) + 4.5f * (uyz2 - uyz) - 1.5f * uxyz2);
    feq_2[18] = tmp_rho_2 * w[18] * (1.0f - 3.0f * (tmp_uy + tmp_uz) + 4.5f * (uyz2 + uyz) - 1.5f * uxyz2);

    //d_f_post_1[index] = d_f_1[index] * (1.0 - omega_1) + feq_1[0] * omega_1 + w[0] * (1 - 0.5 * omega_1) *
        //((3.0 * (cx[0] - d_ux[index]) + 9.0 * cx[0] * (cx[0] * d_ux[index] + cy[0] * d_uy[index] + cz[0] * d_uz[index])) * d_Fx_1[index]
            //+ (3.0 * (cy[0] - d_uy[index]) + 9.0 * cy[0] * (cx[0] * d_ux[index] + cy[0] * d_uy[index] + cz[0] * d_uz[index])) * d_Fy_1[index]
            //+ (3.0 * (cz[0] - d_uz[index]) + 9.0 * cz[0] * (cx[0] * d_ux[index] + cy[0] * d_uy[index] + cz[0] * d_uz[index])) * d_Fz_1[index]);

    //d_f_post_1[index_1 + Nlattice] = d_f_1[index + Nlattice] * (1.0 - omega_1) + feq_1[1] * omega_1 + w[1] * (1 - 0.5 * omega_1) *
        //((3.0 * (cx[1] - d_ux[index]) + 9.0 * cx[1] * (cx[1] * d_ux[index] + cy[1] * d_uy[index] + cz[1] * d_uz[index])) * d_Fx_1[index]
            //+ (3.0 * (cy[1] - d_uy[index]) + 9.0 * cy[1] * (cx[1] * d_ux[index] + cy[1] * d_uy[index] + cz[1] * d_uz[index])) * d_Fy_1[index]
            //+ (3.0 * (cz[1] - d_uz[index]) + 9.0 * cz[1] * (cx[1] * d_ux[index] + cy[1] * d_uy[index] + cz[1] * d_uz[index])) * d_Fz_1[index]);

    //d_f_post_1[index_2 + Nlattice * 2] = d_f_1[index + Nlattice * 2] * (1.0 - omega_1) + feq_1[2] * omega_1 + w[2] * (1 - 0.5 * omega_1) *
        //((3.0 * (cx[2] - d_ux[index]) + 9.0 * cx[2] * (cx[2] * d_ux[index] + cy[2] * d_uy[index] + cz[2] * d_uz[index])) * d_Fx_1[index]
            //+ (3.0 * (cy[2] - d_uy[index]) + 9.0 * cy[2] * (cx[2] * d_ux[index] + cy[2] * d_uy[index] + cz[2] * d_uz[index])) * d_Fy_1[index]
            //+ (3.0 * (cz[2] - d_uz[index]) + 9.0 * cz[2] * (cx[2] * d_ux[index] + cy[2] * d_uy[index] + cz[2] * d_uz[index])) * d_Fz_1[index]);

    //d_f_post_1[index_3 + Nlattice * 3] = d_f_1[index + Nlattice * 3] * (1.0 - omega_1) + feq_1[3] * omega_1 + w[3] * (1 - 0.5 * omega_1) *
        //((3.0 * (cx[3] - d_ux[index]) + 9.0 * cx[3] * (cx[3] * d_ux[index] + cy[3] * d_uy[index] + cz[3] * d_uz[index])) * d_Fx_1[index]
            //+ (3.0 * (cy[3] - d_uy[index]) + 9.0 * cy[3] * (cx[3] * d_ux[index] + cy[3] * d_uy[index] + cz[3] * d_uz[index])) * d_Fy_1[index]
            //+ (3.0 * (cz[3] - d_uz[index]) + 9.0 * cz[3] * (cx[3] * d_ux[index] + cy[3] * d_uy[index] + cz[3] * d_uz[index])) * d_Fz_1[index]);

    //d_f_post_1[index_4 + Nlattice * 4] = d_f_1[index + Nlattice * 4] * (1.0 - omega_1) + feq_1[4] * omega_1 + w[4] * (1 - 0.5 * omega_1) *
        //((3.0 * (cx[4] - d_ux[index]) + 9.0 * cx[4] * (cx[4] * d_ux[index] + cy[4] * d_uy[index] + cz[4] * d_uz[index])) * d_Fx_1[index]
            //+ (3.0 * (cy[4] - d_uy[index]) + 9.0 * cy[4] * (cx[4] * d_ux[index] + cy[4] * d_uy[index] + cz[4] * d_uz[index])) * d_Fy_1[index]
            //+ (3.0 * (cz[4] - d_uz[index]) + 9.0 * cz[4] * (cx[4] * d_ux[index] + cy[4] * d_uy[index] + cz[4] * d_uz[index])) * d_Fz_1[index]);

    for (int q = 0; q < 19; q++)
    {
        d_f_post_1[indexf[q] + Nlattice * q] = d_f_1[index + Nlattice * q] * (1.0 - omega_1) + feq_1[q] * omega_1 + w[q] * (1 - 0.5 * omega_1) *
            ((3.0 * (cx[q] - d_ux[index]) + 9.0 * cx[q] * (cx[q] * d_ux[index] + cy[q] * d_uy[index] + cz[q] * d_uz[index])) * d_Fx_1[index]
                + (3.0 * (cy[q] - d_uy[index]) + 9.0 * cy[q] * (cx[q] * d_ux[index] + cy[q] * d_uy[index] + cz[q] * d_uz[index])) * d_Fy_1[index]
                + (3.0 * (cz[q] - d_uz[index]) + 9.0 * cz[q] * (cx[q] * d_ux[index] + cy[q] * d_uy[index] + cz[q] * d_uz[index])) * d_Fz_1[index]);

        d_f_post_2[indexf[q] + Nlattice * q] = d_f_2[index + Nlattice * q] * (1.0 - omega_1) + feq_2[q] * omega_2 + w[q] * (1 - 0.5 * omega_2) *
            ((3.0 * (cx[q] - d_ux[index]) + 9.0 * cx[q] * (cx[q] * d_ux[index] + cy[q] * d_uy[index] + cz[q] * d_uz[index])) * d_Fx_2[index]
                + (3.0 * (cy[q] - d_uy[index]) + 9.0 * cy[q] * (cx[q] * d_ux[index] + cy[q] * d_uy[index] + cz[q] * d_uz[index])) * d_Fy_2[index]
                + (3.0 * (cz[q] - d_uz[index]) + 9.0 * cz[q] * (cx[q] * d_ux[index] + cy[q] * d_uy[index] + cz[q] * d_uz[index])) * d_Fz_2[index]);
    }
    
    for (int q = 0; q < 19; q++)
    {
        d_f_1[index + Nlattice * q] = d_f_post_1[index + Nlattice * q];

        d_f_2[index + Nlattice * q] = d_f_post_2[index + Nlattice * q];
    }
}

// 0. Main loop

int main()
{

    const int Nstep = 5000;
    
    // allocate memory on CPU and GPU 
    h_ux = (double*)malloc(sizeof(double) * Nlattice);
    h_uy = (double*)malloc(sizeof(double) * Nlattice);
    h_uz = (double*)malloc(sizeof(double) * Nlattice);
    h_rho_1 = (double*)malloc(sizeof(double) * Nlattice);
    h_rho_2 = (double*)malloc(sizeof(double) * Nlattice);
    h_psi_1 = (double*)malloc(sizeof(double) * Nlattice);
    h_psi_2 = (double*)malloc(sizeof(double) * Nlattice);
    h_f_1 = (double*)malloc(sizeof(double) * Nlattice * Q);
    h_f_2 = (double*)malloc(sizeof(double) * Nlattice * Q);
    h_f_post_1 = (double*)malloc(sizeof(double) * Nlattice * Q);
    h_f_post_2 = (double*)malloc(sizeof(double) * Nlattice * Q);
    test = (double*)malloc(sizeof(double) * Nlattice);
    //testq = (double*)malloc(sizeof(double) * Nlattice * Q);////////////////////////////test

    cudaMalloc((void**)&d_f_1, Nlattice * Q * sizeof(double));
    cudaMalloc((void**)&d_f_2, Nlattice * Q * sizeof(double));
    cudaMalloc((void**)&d_f_post_1, Nlattice * Q * sizeof(double));
    cudaMalloc((void**)&d_f_post_2, Nlattice * Q * sizeof(double));
    cudaMalloc((void**)&d_Fx_1, Nlattice * sizeof(double));
    cudaMalloc((void**)&d_Fx_2, Nlattice * sizeof(double));
    cudaMalloc((void**)&d_Fy_1, Nlattice * sizeof(double));
    cudaMalloc((void**)&d_Fy_2, Nlattice * sizeof(double));
    cudaMalloc((void**)&d_Fz_1, Nlattice * sizeof(double));
    cudaMalloc((void**)&d_Fz_2, Nlattice * sizeof(double));
    cudaMalloc((void**)&d_rho_1, Nlattice * sizeof(double));
    cudaMalloc((void**)&d_rho_2, Nlattice * sizeof(double));
    cudaMalloc((void**)&d_psi_1, Nlattice * sizeof(double));
    cudaMalloc((void**)&d_psi_2, Nlattice * sizeof(double));
    cudaMalloc((void**)&d_ux, Nlattice * sizeof(double));
    cudaMalloc((void**)&d_uy, Nlattice * sizeof(double));
    cudaMalloc((void**)&d_uz, Nlattice * sizeof(double));

    Initialization();
    // initialization on GPU
    cudaMemcpy(d_f_1, h_f_1, Nlattice * Q * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_f_2, h_f_2, Nlattice * Q * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_f_post_1, h_f_post_1, Nlattice * Q * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_f_post_2, h_f_post_2, Nlattice * Q * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_psi_1, h_psi_1, Nlattice * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_psi_2, h_psi_2, Nlattice * sizeof(double), cudaMemcpyHostToDevice);

    for (int step = 0; step < Nstep; step++)
    {
        computeDensity << <grid, block >> > (Nlattice, d_f_1, d_f_2, d_rho_1, d_rho_2, d_psi_1, d_psi_2);
        cudaDeviceSynchronize();

        computeSCForces << <grid, block >> > (d_psi_1, d_psi_2, d_Fx_1, d_Fy_1, d_Fz_1, d_Fx_2, d_Fy_2, d_Fz_2);
        cudaDeviceSynchronize();

        computeVelocity << <grid, block >> > (d_f_1, d_f_2, d_rho_1, d_rho_2, d_Fx_1, d_Fy_1, d_Fz_1, d_Fx_2, d_Fy_2, d_Fz_2, d_ux, d_uy, d_uz);
        cudaDeviceSynchronize();

        stream << <grid, block >> > (d_f_1, d_f_2, d_f_post_1, d_f_post_2, d_Fx_1, d_Fy_1, d_Fz_1, d_Fx_2, d_Fy_2, d_Fz_2, d_ux, d_uy, d_uz, d_rho_1, d_rho_2);
        cudaDeviceSynchronize();
    }
    
    cudaMemcpy(test, d_rho_1, Nx * Ny * Nz * sizeof(double), cudaMemcpyDeviceToHost);///////////////////////////////////////test

    for (int z = 0; z < Nz; z++)//////////////////////////////////////////////////////test
    {
        for (int y = 0; y < Ny; y++)
        {
            
            for (int x = 0; x < Nx; x++)
            {
                int k = z * Nx * Ny + y * Nx + x;
                cout << x+1 << "\t" << y+1 << "\t " << z+1 << "\t " << test[k] << endl;
            }
        }
    }

    //for (int z = 0; z < Nz; z++)//////////////////////////////////////////////////////test
    //{
    //    cout << "z=" << z << endl;
    //    for (int y = 0; y < Ny; y++)
    //    {
    //        cout << "y=" << y << endl;
    //        for (int x = 0; x < Nx; x++)
    //        {
    //            const int k = z * Nx * Ny + y * Nx + x;
    //            cout << test[k] << " ";
    //        }cout << endl;
    //    }cout << endl;
    //}cout << endl;

    free(h_f_1);
    free(h_f_2);
    free(h_f_post_1);
    free(h_f_post_2);
    free(h_rho_1);
    free(h_rho_2);
    free(h_psi_1);
    free(h_psi_2);
    free(h_ux);
    free(h_uy);
    free(h_uz);
    free(test);

    cudaFree(d_f_1);
    cudaFree(d_f_2);
    cudaFree(d_f_post_1);
    cudaFree(d_f_post_2);
    cudaFree(d_rho_1);
    cudaFree(d_rho_2);
    cudaFree(d_psi_1);
    cudaFree(d_psi_2);
    cudaFree(d_ux);
    cudaFree(d_uy);
    cudaFree(d_uz);
    cudaFree(d_Fx_1);
    cudaFree(d_Fy_1);
    cudaFree(d_Fz_1);
    cudaFree(d_Fx_2);
    cudaFree(d_Fy_2);
    cudaFree(d_Fz_2);
    
    return 0;
}
