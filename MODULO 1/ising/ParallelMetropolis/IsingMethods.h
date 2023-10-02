#ifndef ISING_METHODS_H
#define ISING_METHODS_H

#include<cuda_runtime.h>
#include<curand_kernel.h>

#include<fstream>
#include<math.h>
#include<thread>

#include<TTree.h>
#include<TFile.h>

///METHOD TO CHECK ERRORS FROM THE GP
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}


/*********************************************
/********** LATTICE FUNCTIONS ****************
/********************************************/

//function to convert from 2D coordinates to 1D coordinates, enforcing periodic boundary conditions (FOR LATTICE ARRAY)
__host__ __device__ constexpr int flatten(int i, int j, int L);


//function to map true->1 false->-1 (FOR LATTICE ARRAY)
__device__ constexpr signed char convert(bool value);


//function to get the value of spin from a lattice
__device__ signed char get(bool* lattice, int i, int j, int L);


/*********************************************
/******* RANDOM GENERATOR FUNCTIONS **********
/********************************************/

//initialize random generator, different for each simulation
__global__ void initRandom(curandState *states, int seed);


//get a random site from a lattice, used for Metropolis algorithm
__device__ void randomLatticeSite(curandState *states, int *i0, int *j0);


/*********************************************
/********* COPY TO DISK FUNCTION *************
/********************************************/

//function to copy to the disk
__host__ void saveResults(int dataPoints, int nResamplings, int nBeta, int L, long long* dMagCumulant, long long* dEnergyCumulant);


/*********************************************
/********* SIMULATION KERNEL *****************
/********************************************/

//simulation kernel, implementing single step Metropolis algorithm
__global__ void simulation(int batchNum, int batchSize, double *dBeta, signed short* dMagMarkov, signed short* dEnergyMarkov, signed short* dMagCurrent, signed short* dEnergyCurrent, bool* dLattices, curandState* dStates);

//bootstrapping the data by blocking technique
__global__ void bootstrap(int nResamplings, int batchSize, int blockingSize, signed short* dMagMarkov, signed short* dEnergyMarkov, long long* dMagCumulant, long long* dEnergyCumulant, curandState* dStatesBS);

#endif
