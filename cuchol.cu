#include "cuda_runtime.h"
#include "device_launch_paraMeters.h"

#include<iostream>
#include <fstream>
#include<iomanip>
#include<stdlib.h>
#include<stdio.h>
#include<assert.h>

#include <cusolverDn.h>
#include <cublas_v2.h>
#include <cuda_runtime_api.h>

#include "Utilities.cuh"

#define prec_save 10

/******************************************/
/* SET HERMITIAN POSITIVE DEFINITE MATRIX */
/******************************************/
// --- Credit to: https://math.stackexchange.com/questions/357980/how-to-generate-random-symmetric-positive-definite-matrices-using-matlab
void setPDMatrix(double * __restrict h_A, const int N) {

    // --- Initialize random seed
    srand(time(NULL));

    double *h_A_temp = (double *)malloc(N * N * sizeof(double));

    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            h_A_temp[i * N + j] = (float)rand() / (float)RAND_MAX;

    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++) 
            h_A[i * N + j] = 0.5 * (h_A_temp[i * N + j] + h_A_temp[j * N + i]);

    for (int i = 0; i < N; i++) h_A[i * N + i] = h_A[i * N + i] + N;

}

/************************************/
/* SAVE REAL ARRAY FROM CPU TO FILE */
/************************************/
template <class T>
void saveCPUrealtxt(const T * h_in, const char *filename, const int M) {

    std::ofstream outfile;
    outfile.open(filename);
    for (int i = 0; i < M; i++) outfile << std::setprecision(prec_save) << h_in[i] << "\n";
    outfile.close();

}

/************************************/
/* SAVE REAL ARRAY FROM GPU TO FILE */
/************************************/
template <class T>
void saveGPUrealtxt(const T * d_in, const char *filename, const int M) {

    T *h_in = (T *)malloc(M * sizeof(T));

    gpuErrchk(cudaMemcpy(h_in, d_in, M * sizeof(T), cudaMemcpyDeviceToHost));

    std::ofstream outfile;
    outfile.open(filename);
    for (int i = 0; i < M; i++) outfile << std::setprecision(prec_save) << h_in[i] << "\n";
    outfile.close();

}

/********/
/* MAIN */
/********/
int main(){

    const int N = 1000;

    // --- CUDA solver initialization
    cusolverDnHandle_t solver_handle;
    cusolveSafeCall(cusolverDnCreate(&solver_handle));

    // --- CUBLAS initialization
    cublasHandle_t cublas_handle;
    cublasSafeCall(cublasCreate(&cublas_handle));

    /***********************/
    /* SETTING THE PROBLEM */
    /***********************/
    // --- Setting the host, N x N matrix
    double *h_A = (double *)malloc(N * N * sizeof(double));
    setPDMatrix(h_A, N);

    // --- Allocate device space for the input matrix 
    double *d_A; gpuErrchk(cudaMalloc(&d_A, N * N * sizeof(double)));

    // --- Move the relevant matrix from host to device
    gpuErrchk(cudaMemcpy(d_A, h_A, N * N * sizeof(double), cudaMemcpyHostToDevice));

    /****************************************/
    /* COMPUTING THE CHOLESKY DECOMPOSITION */
    /****************************************/
    // --- cuSOLVE input/output parameters/arrays
    int work_size = 0;
    int *devInfo;           gpuErrchk(cudaMalloc(&devInfo, sizeof(int)));

    // --- CUDA CHOLESKY initialization
    cusolveSafeCall(cusolverDnDpotrf_bufferSize(solver_handle, CUBLAS_FILL_MODE_LOWER, N, d_A, N, &work_size));

    // --- CUDA POTRF execution
    double *work;   gpuErrchk(cudaMalloc(&work, work_size * sizeof(double)));
    cusolveSafeCall(cusolverDnDpotrf(solver_handle, CUBLAS_FILL_MODE_LOWER, N, d_A, N, work, work_size, devInfo));
    int devInfo_h = 0;  gpuErrchk(cudaMemcpy(&devInfo_h, devInfo, sizeof(int), cudaMemcpyDeviceToHost));
    if (devInfo_h != 0) std::cout << "Unsuccessful potrf execution\n\n" << "devInfo = " << devInfo_h << "\n\n";

    // --- At this point, the lower triangular part of A contains the elements of L. 
    /***************************************/
    /* CHECKING THE CHOLESKY DECOMPOSITION */
    /***************************************/
    saveCPUrealtxt(h_A, "D:\\Project\\solveSquareLinearSystemCholeskyCUDA\\solveSquareLinearSystemCholeskyCUDA\\h_A.txt", N * N);
    saveGPUrealtxt(d_A, "D:\\Project\\solveSquareLinearSystemCholeskyCUDA\\solveSquareLinearSystemCholeskyCUDA\\d_A.txt", N * N);

    cusolveSafeCall(cusolverDnDestroy(solver_handle));

    return 0;

}