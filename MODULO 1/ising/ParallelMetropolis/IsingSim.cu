#include<cuda_runtime.h>
#include<curand_kernel.h>

#include<fstream>
#include<math.h>

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

__device__ float get(bool* lattice, int i, int j, int L)
{
    bool value = lattice[flatten(i,j,L)];
    float res;
    if(value)
        res=1.;
    else
        res=-1.;

    return res;
}

__global__ void initRandom(curandState *states, int seed)
{
    int simIdx = threadIdx.x + blockIdx.x*blockDim.x;

    //initialize random generator on each thread
    curand_init(seed, simIdx, 0, &states[simIdx]);
}



__device__ void randomLatticeSite(curandState *states, int *i0, int *j0){

    int simIdx = threadIdx.x + blockDim.x*blockIdx.x;
    int L = 20 + blockIdx.x*10;
    curandState localState = states[simIdx];

    float randFloati0 = curand_uniform(&localState);
    float randFloatj0 = curand_uniform(&localState);

    randFloati0 *= L;
    randFloatj0 *= L;

    int randi0 = (int)truncf(randFloati0);
    int randj0  =(int)truncf(randFloatj0);

    *i0 = randi0;
    *j0 = randj0;

    states[simIdx] = localState;
}

__global__ void simulation(int dataPoints, float* dMag, float* dChi, curandState* dStates)
{
    //get thread index
    int idx = threadIdx.x + blockIdx.x*blockDim.x;

    //get simulation parameters and random generator from the thread index
    float beta = 0.34 + 0.2*threadIdx.x/64;
    int L = 20 + blockIdx.x*10;
    curandState localState = dStates[idx];

    //declare and initialize lattice in private memory, use only booleans to save space (true = 1, false = -1)
    bool* lattice = (bool*)malloc(L*L*sizeof(bool));
    for(int i=0; i<L*L; i++)
        lattice[i]=true;

    //declare useful variables
    int force, i0, j0;
    float acceptance;
    double actualMag = 1.;
    double avgMag = 0., avgMagSq = 0.;

    //run metropolis algorithm
    for(int i=0; i<100*dataPoints; i++)
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
            actualMag -= 2*get(lattice,i0,j0,L)/(L*L);
            lattice[flatten(i0,j0,L)] = !lattice[flatten(i0,j0,L)];
        }

        //extract data
        if(i%100==0)
        {
            avgMag += abs(actualMag);
            avgMagSq += pow(actualMag,2);
        }
    }

    dStates[idx] = localState;
    dMag[idx] = avgMag/dataPoints;
    dChi[idx] = L*L*(avgMagSq/dataPoints - pow(avgMag/dataPoints,2));

    if(dChi[idx]<0)
        printf("%f - %f \n", avgMagSq/dataPoints, pow(avgMag/dataPoints,2));
}

int main()
{
    //simulation parameters
    const int L = 4;
    const int nBeta = 64;
    const int dataPoints = 1e7;

    //allocate random generators
    curandState *dStates;
    cudaMallocManaged(&dStates, L*nBeta*sizeof(curandState));

    //allocate host memory container for simulation results
    float *hMag = (float*)malloc(L*nBeta*sizeof(float));
    float *hChi = (float*)malloc(L*nBeta*sizeof(float));

    //allocate device memory too
    float *dMag, *dChi;
    cudaMallocManaged(&dMag, L*nBeta*sizeof(float));
    cudaMallocManaged(&dChi, L*nBeta*sizeof(float));

    //run simulation kernel
    initRandom<<<L,nBeta>>>(dStates,1221);

    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    simulation<<<L,nBeta>>>(dataPoints,dMag, dChi, dStates);

    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    //copy back array to host
    cudaMemcpy(hMag, dMag, L*nBeta*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(hChi, dChi, L*nBeta*sizeof(float), cudaMemcpyDeviceToHost);

    //output to file what we got
    std::ofstream mag, chi;
    mag.open("mag.txt");
    chi.open("chi.txt");
    for(int i=0; i<nBeta*L; i++)
    {
        mag << hMag[i] << std::endl;
        chi << hChi[i] << std::endl;
    }

    mag.close();
    chi.close();

    //free device memory
    cudaFree(dStates);
    cudaFree(dMag);

    return 0;
}
