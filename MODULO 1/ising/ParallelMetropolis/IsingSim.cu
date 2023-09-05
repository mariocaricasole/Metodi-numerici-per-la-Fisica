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



__device__ signed char get(bool* lattice, int i, int j, int L)
{
    bool value = lattice[flatten(i,j,L)];
    signed char res;
    if(value)
        res=1;
    else
        res=-1;

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



__global__ void simulation(int dataPoints, double *dBeta, signed char* dMagMarkov, signed char* dEnergyMarkov, curandState* dStates)
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
    float acceptance;
    signed char force;
    signed char dMag = 0;
    signed char dEnergy = 0;

    //run metropolis algorithm
    for(unsigned long long i=0; i<100*dataPoints; i++)
    {
        //take a random site
        randomLatticeSite(&localState, &i0, &j0);
        //evaluate "force"
        force = get(lattice,i0-1,j0,L)+ get(lattice,i0+1,j0,L) + get(lattice,i0,j0-1,L) + get(lattice,i0,j0+1,L);

        //calculate acceptance
        acceptance = min(float(1.), expf(-2*get(lattice,i0,j0,L)*force*beta));

        //accept-reject step
        if(acceptance > curand_uniform(&localState))
        {
            dEnergy += get(lattice,i0,j0,L)*force;
            dMag -= get(lattice,i0,j0,L);
            lattice[flatten(i0,j0,L)] = !lattice[flatten(i0,j0,L)];
        }

        //reduce data, extract only the first 10^5
        if(i%100==0)
        {
            dMagMarkov[idx*dataPoints + i/100] = dMag;
            dEnergyMarkov[idx*dataPoints + i/100] = dEnergy;
            dMag = 0;
            dEnergy = 0;
        }
    }

    dStates[idx] = localState;

    if(blockIdx.x==0)
        dBeta[threadIdx.x] = beta;
}

int main()
{
    //simulation parameters
    const unsigned char L = 4;
    const unsigned char nBeta = 64;
    const int dataPoints = 1e5;
    const unsigned long long size = L*nBeta*dataPoints;

    //allocate random generators
    curandState *dStates;
    cudaMallocManaged(&dStates, L*nBeta*sizeof(curandState));

    //allocate unified memory
    double *dBeta;
    signed char *dMagMarkov, *dEnergyMarkov;
    cudaMallocManaged(&dMagMarkov, size*sizeof(signed char));
    cudaMallocManaged(&dEnergyMarkov, size*sizeof(signed char));
    cudaMallocManaged(&dBeta, nBeta*sizeof(double));

    //run simulation kernel
    initRandom<<<L,nBeta>>>(dStates,1221);

    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    simulation<<<L,nBeta>>>(dataPoints, dBeta, dMagMarkov, dEnergyMarkov, dStates);

    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    //output to file
    TFile file("data.root", "recreate");
    TTree t1("t1", "data points");
    signed char currentMarkovMag, currentMarkovEnergy;

    t1.Branch("magMarkov", &currentMarkovMag, "magMarkov/B");
    t1.Branch("energyMarkov", &currentMarkovEnergy, "energyMarkov/B");

    for(unsigned long long i=0; i < size; i++)
    {
        currentMarkovEnergy = dEnergyMarkov[i];
        currentMarkovMag = dMagMarkov[i];
        t1.Fill();
    }
    t1.Write();

    //free device memory
    cudaFree(dStates);
    cudaFree(dBeta);
    cudaFree(dMagMarkov);
    cudaFree(dEnergyMarkov);

    return 0;
}
