#include "IsingMethods.h"

/*********************************************
/********* COPY TO DISK FUNCTION *************
/********************************************/

__host__ void saveResults(int dataPoints, int nResamplings, int nBeta, int L, long long* dMagCumulant, long long* dEnergyCumulant)
{
    std::string fileName = "results.root";
    TFile file = TFile(fileName.c_str(), "recreate");
    TTree tree = TTree("data", "results of the bootstrap method");

    double currentMag, currentEnergy;
    unsigned short simIdx, resampleIdx;

    tree.Branch("resampleIdx", &resampleIdx, "resampleIdx/s");
    tree.Branch("simIdx", &simIdx, "simIdx/s");
    tree.Branch("magMarkov", &currentMag, "magMarkov/D");
    tree.Branch("energyMarkov", &currentEnergy, "energyMarkov/D");

    for(int n=0; n<nResamplings*nBeta*L;n++)
    {
        resampleIdx = n/(nBeta*L);
        simIdx = n%(nBeta*L);
        currentMag = double(dMagCumulant[n])/dataPoints;
        currentEnergy = double(dEnergyCumulant[n])/dataPoints;
        tree.Fill();
    }
    tree.Write();
    file.Close();
}


/*********************************************
/********* SIMULATION KERNEL *****************
/********************************************/

__global__ void simulation(int batchNum, int batchSize, double *dBeta, signed short* dMagMarkov, signed short* dEnergyMarkov, signed short* dMagCurrent, signed short* dEnergyCurrent, bool* dLattices, curandState* dStates)
{
    //get thread index
    unsigned short idx = threadIdx.x + blockIdx.x*blockDim.x;

    //get simulation parameters and random generator from the thread and block index
    double beta = 0.4 + 0.1*pow(2*(double)threadIdx.x/blockDim.x - 1,3);
    unsigned char L = 10 + blockIdx.x*10;

    //initialize inter-batch arrays if this is the first run
    if(batchNum == 0)
    {
            dMagCurrent[idx] = L*L;
            dEnergyCurrent[idx] = -2*L*L;
    }

    //get values from inter-batch arrays
    curandState localState = dStates[idx];
    signed short dMag = dMagCurrent[idx];
    signed short dEnergy = dEnergyCurrent[idx];

    //make a copy from global memory of the local lattice
    int latticeIdx=0;
    for(unsigned char i=0; i<blockIdx.x; i++)
        latticeIdx += pow((10+i*10),2)*blockDim.x;

    latticeIdx += L*L*threadIdx.x;
    bool* localLattice = &dLattices[latticeIdx];

    //declare useful variables
    int i0, j0;
    float acceptance;
    signed char force;

    //run metropolis algorithm
    for(unsigned long long i=0; i<100*batchSize; i++)
    {
        //take a random site
        randomLatticeSite(&localState, &i0, &j0);
        //evaluate "force"
        force = get(localLattice,i0-1,j0,L)+ get(localLattice,i0+1,j0,L) + get(localLattice,i0,j0-1,L) + get(localLattice,i0,j0+1,L);

        //calculate acceptance
        acceptance = min(float(1.), expf(-2*get(localLattice,i0,j0,L)*force*beta));

        //accept-reject step
        if(acceptance > curand_uniform(&localState))
        {
            dEnergy += 2*get(localLattice,i0,j0,L)*force;
            dMag -= 2*get(localLattice,i0,j0,L);
            localLattice[flatten(i0,j0,L)] = !localLattice[flatten(i0,j0,L)];
        }

        //reduce data
        if(i%100==0)
        {
            dMagMarkov[idx*batchSize + i/100] = abs(dMag);
            dEnergyMarkov[idx*batchSize + i/100] = dEnergy;
        }
    }

    //copy back current state of the random generator and inter-batch arrays
    dStates[idx] = localState;
    dMagCurrent[idx] = dMag;
    dEnergyCurrent[idx] = dEnergy;

    if(blockIdx.x==0)
        dBeta[threadIdx.x] = beta;
}


//bootstrapping the data by blocking technique
__global__ void bootstrap(int nResamplings, int batchSize, int blockingSize, signed short* dMagMarkov, signed short* dEnergyMarkov, long long* dMagCumulant, long long* dEnergyCumulant, curandState* dStatesBS)
{
    //evaluate simulation index and simulation variables
    unsigned short idx = threadIdx.x + blockIdx.x*blockDim.x;
    unsigned short L = gridDim.x;
    unsigned short nBeta = blockDim.x;

    //get random generator state
    curandState localState = dStatesBS[idx];

    //define ancillary variables
    long long magCumulant=0, energyCumulant=0;
    int poissonRandom=1;

    for(int i=0; i<nResamplings*batchSize;i++)
    {
        //get resample number
        int resample = i/batchSize;

        //on the first pass, don't resample, on the others draw a Poisson weight for each data block
        if(resample!=0 and (i+1)%blockingSize==0)
            poissonRandom = curand_poisson(&localState,1);

        magCumulant += dMagMarkov[idx*batchSize + i%batchSize]*poissonRandom;
        energyCumulant += dEnergyMarkov[idx*batchSize + i%batchSize]*poissonRandom;

        if((i+1)%batchSize)
        {
            dMagCumulant[nBeta*L*resample + idx] += magCumulant;
            dEnergyCumulant[nBeta*L*resample + idx] += energyCumulant;
            magCumulant=0;
            energyCumulant=0;
        }
    }

    dStatesBS[idx] = localState;
}


/*********************************************
/********** LATTICE FUNCTIONS ****************
/********************************************/

__host__ __device__ constexpr int flatten(int i, int j, int L)
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


__device__ constexpr signed char convert(bool value)
{
    signed char res=-1;

    if(value)
        res=1;

    return res;
}


__device__ signed char get(bool* lattice, int i, int j, int L)
{
    bool value = lattice[flatten(i,j,L)];
    return convert(value);
}



/*********************************************
/******* RANDOM GENERATOR FUNCTIONS **********
/********************************************/

__global__ void initRandom(curandState *states, int seed)
{
    unsigned short simIdx = threadIdx.x + blockIdx.x*blockDim.x;

    //initialize random generator on each thread
    curand_init(seed, simIdx, 0, &states[simIdx]);
}


__device__ void randomLatticeSite(curandState *states, int *i0, int *j0){

    unsigned short simIdx = threadIdx.x + blockDim.x*blockIdx.x;
    unsigned char L = 10 + blockIdx.x*10;
    curandState localState = states[simIdx];

    *i0 = (int)truncf(L*curand_uniform(&localState));
    *j0 = (int)truncf(L*curand_uniform(&localState));

    states[simIdx] = localState;
}
