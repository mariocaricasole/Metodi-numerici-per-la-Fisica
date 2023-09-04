#include<cuda_runtime.h>
#include<curand_kernel.h>

#include<fstream>
#include<math.h>

#include<TTree.h>
#include<TFile.h>

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}



__host__ __device__ int flatten(int i, int j, int L)
{
    //periodic boundary conditions
    int m=i;
    int n=j;

    if(i<0)
        m = L-1;
    else if(i>L-1)
        m = 0;

    if(j<0)
        n = L-1;
    else if(j>L-1)
        n = 0;

    return m + n*L;
}



__device__ double get(bool* lattice, int i, int j, int L)
{
    bool value = lattice[flatten(i,j,L)];
    double res;
    if(value)
        res=1.;
    else
        res=-1.;

    return res;
}



__global__ void initRandom(curandState *states, int seed)
{
    unsigned short simIdx = threadIdx.x + blockIdx.x*blockDim.x;

    //initialize random generator on each thread
    curand_init(seed, simIdx, 0, &states[simIdx]);
}



__device__ void randomLatticeSite(curandState *states, int *i0, int *j0){

    unsigned short simIdx = threadIdx.x + blockDim.x*blockIdx.x;
    unsigned char L = 20 + blockIdx.x*10;
    curandState localState = states[simIdx];

    *i0 = (int)truncf(L*curand_uniform(&localState));
    *j0 = (int)truncf(L*curand_uniform(&localState));

    states[simIdx] = localState;
}



__global__ void simulation(int dataPoints, double *dBeta, double* dMag, double* dChi, double* dHeat, double* dMagMarkov, double* dEnergyMarkov, curandState* dStates)
{
    //get thread index
    unsigned short idx = threadIdx.x + blockIdx.x*blockDim.x;

    //get simulation parameters and random generator from the thread index
    double beta = 0.44 + 0.1*pow(2*(double)threadIdx.x/blockDim.x - 1,3);
    unsigned char L = 20 + blockIdx.x*10;
    curandState localState = dStates[idx];

    //declare and initialize lattice in private memory, use only booleans to save space (true = 1, false = -1)
    bool* lattice = (bool*)malloc(L*L*sizeof(bool));
    for(unsigned short i=0; i<L*L; i++)
        lattice[i]=true;    //cold state

    //declare useful variables
    int i0, j0;
    double acceptance, force;
    double actualMag = 1.;
    double actualEnergy = -2.;
    double avgEnergy=0., avgMag=0., avgEnergySq=0., avgMagSq=0.;

    //run metropolis algorithm
    for(unsigned long long i=0; i<100*dataPoints; i++)
    {
        //take a random site
        randomLatticeSite(&localState, &i0, &j0);
        //evaluate "force"
        force = get(lattice,i0-1,j0,L)+ get(lattice,i0+1,j0,L) + get(lattice,i0,j0-1,L) + get(lattice,i0,j0+1,L);

        //calculate acceptance
        acceptance = min(double(1.), exp(-2*get(lattice,i0,j0,L)*force*beta));

        //accept-reject step
        if(acceptance > curand_uniform(&localState))
        {
            actualEnergy += 2*get(lattice,i0,j0,L)*force/(L*L);
            actualMag -= 2*get(lattice,i0,j0,L)/(L*L);
            lattice[flatten(i0,j0,L)] = !lattice[flatten(i0,j0,L)];
        }

        //reduce data, extract only the first 10^5
        if(i%100==0)
        {
            if(i/100 < 1e5)
            {
                dMagMarkov[idx*100000 + i/100] = actualMag;
                dEnergyMarkov[idx*100000 + i/100] = actualEnergy;
            }

            //magnetization quantities
            avgMag += abs(actualMag);
            avgMagSq += pow(actualMag,2);

            //energy quantities
            avgEnergy += actualEnergy;
            avgEnergySq += pow(actualEnergy,2);
        }
    }

    dStates[idx] = localState;
    dMag[idx] = avgMag/dataPoints;
    dChi[idx] = L*L*(avgMagSq/dataPoints - pow(avgMag/dataPoints,2));
    dHeat[idx] = L*L*(avgEnergySq/dataPoints - pow(avgEnergy/dataPoints,2));

    if(blockIdx.x==0)
        dBeta[threadIdx.x] = beta;
}

int main()
{
    //simulation parameters
    const unsigned char L = 4;
    const unsigned char nBeta = 64;
    const int dataPoints = 1e6;

    //allocate random generators
    curandState *dStates;
    cudaMallocManaged(&dStates, L*nBeta*sizeof(curandState));

    //allocate host memory container for simulation results
    double *hMag = (double*)malloc(L*nBeta*sizeof(double));
    double *hChi = (double*)malloc(L*nBeta*sizeof(double));
    double *hHeat = (double*)malloc(L*nBeta*sizeof(double));
    double *hMagMarkov = (double*)malloc(L*nBeta*100000*sizeof(double));
    double *hEnergyMarkov = (double*)malloc(L*nBeta*100000*sizeof(double));
    double *hBeta = (double*)malloc(nBeta*sizeof(double));

    //allocate device memory too
    double *dMag, *dChi, *dHeat, *dBeta, *dMagMarkov, *dEnergyMarkov;
    cudaMallocManaged(&dMag, L*nBeta*sizeof(double));
    cudaMallocManaged(&dChi, L*nBeta*sizeof(double));
    cudaMallocManaged(&dHeat, L*nBeta*sizeof(double));
    cudaMallocManaged(&dMagMarkov, L*nBeta*100000*sizeof(double));
    cudaMallocManaged(&dEnergyMarkov, L*nBeta*100000*sizeof(double));
    cudaMallocManaged(&dBeta, nBeta*sizeof(double));

    //run simulation kernel
    initRandom<<<L,nBeta>>>(dStates,1221);

    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    simulation<<<L,nBeta>>>(dataPoints, dBeta, dMag, dChi, dHeat, dMagMarkov, dEnergyMarkov, dStates);

    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    //copy back array to host
    cudaMemcpy(hMag, dMag, L*nBeta*sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(hChi, dChi, L*nBeta*sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(hHeat, dHeat, L*nBeta*sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(hMagMarkov, dMagMarkov, L*nBeta*100000*sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(hEnergyMarkov, dEnergyMarkov, L*nBeta*100000*sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(hBeta, dBeta, nBeta*sizeof(double), cudaMemcpyDeviceToHost);

    //output to file what we got in a TTree
    TFile file("thermodynamics.root", "recreate");
    TTree t1("t1","tree holding interesting quantities");
    TTree t2("t2", "tree holding first 1e5 data points");
    double currentChi, currentMag, currentHeat, currentMarkovMag, currentMarkovEnergy;

    t1.Branch("mag", &currentMag, "mag/D");
    t1.Branch("chi", &currentChi, "chi/D");
    t1.Branch("heat", &currentHeat, "heat/D");
    t2.Branch("magMarkov", &currentMarkovMag, "magMarkov/D");
    t2.Branch("energyMarkov", &currentMarkovEnergy, "energyMarkov/D");

    for(int i=0; i < L*nBeta*100000; i++)
    {
        if(i < L*nBeta)
        {
            currentMag = hMag[i];
            currentChi = hChi[i];
            currentHeat = hHeat[i];
            t1.Fill();
        }

        currentMarkovEnergy = hEnergyMarkov[i];
        currentMarkovMag = hMagMarkov[i];
        t2.Fill();
    }
    t1.Write();
    t2.Write();

    //free device memory
    cudaFree(dStates);
    cudaFree(dMag);
    cudaFree(dChi);
    cudaFree(dHeat);
    cudaFree(dBeta);
    cudaFree(dMagMarkov);
    cudaFree(dEnergyMarkov);

    //free host memory
    free(hMag);
    free(hChi);
    free(hHeat);
    free(hBeta);
    free(hMagMarkov);
    free(hEnergyMarkov);

    return 0;
}
