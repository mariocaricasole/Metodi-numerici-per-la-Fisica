#include "IsingMethods.h"

/*********************************************
/********* COPY TO DISK FUNCTION *************
/********************************************/

__host__ void copyToDisk(int i, int batchSize, int memSize, signed char* dMagMarkov, signed short* dEnergyMarkov)
{
    std::string fileName = "batch" + std::to_string(i) + ".root";
    TFile file = TFile(fileName.c_str(), "recreate");
    TTree tree = TTree("data", "markov step data");

    signed char currentMag;
    signed short currentEnergy;
    unsigned short simIdx;

    tree.Branch("simIdx", &simIdx, "simIdx/s");
    tree.Branch("magMarkov", &currentMag, "magMarkov/B");
    tree.Branch("energyMarkov", &currentEnergy, "energyMarkov/S");

    for(int n=0; n<memSize;n++)
    {
        simIdx = n/batchSize;
        currentMag = dMagMarkov[n];
        currentEnergy = dEnergyMarkov[n];
        tree.Fill();
    }
    tree.Write();
    file.Close();
}


/*********************************************
/********* SIMULATION KERNEL *****************
/********************************************/

__global__ void simulation(int batchSize, double *dBeta, signed char* dMagMarkov, signed short* dEnergyMarkov, bool* dLattices, curandState* dStates)
{
    //get thread index
    unsigned short idx = threadIdx.x + blockIdx.x*blockDim.x;

    //get simulation parameters and random generator from the thread index
    double beta = 0.44 + 0.1*pow(2*(double)threadIdx.x/blockDim.x - 1,3);
    unsigned char L = 20 + blockIdx.x*10;
    curandState localState = dStates[idx];

    //make a copy from global memory of the local lattice
    int latticeIdx=0;
    for(unsigned char i=0; i<blockIdx.x; i++)
        latticeIdx += pow((20+i*blockDim.x),2)*64;

    latticeIdx += L*L*threadIdx.x;
    bool* localLattice = &dLattices[latticeIdx];

    //declare useful variables
    int i0, j0;
    float acceptance;
    signed char force;
    signed char dMag = 0;
    signed short dEnergy = 0;

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
            dEnergy += get(localLattice,i0,j0,L)*force;
            dMag -= get(localLattice,i0,j0,L);
            localLattice[flatten(i0,j0,L)] = !localLattice[flatten(i0,j0,L)];
        }

        //reduce data
        if(i%100==0)
        {
            dMagMarkov[idx*batchSize + i/100] = dMag;
            dEnergyMarkov[idx*batchSize + i/100] = dEnergy;
            dMag = 0;
            dEnergy = 0;
        }
    }

    //copy back current state of the random generator
    dStates[idx] = localState;

    if(blockIdx.x==0)
        dBeta[threadIdx.x] = beta;
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
    unsigned char L = 20 + blockIdx.x*10;
    curandState localState = states[simIdx];

    *i0 = (int)truncf(L*curand_uniform(&localState));
    *j0 = (int)truncf(L*curand_uniform(&localState));

    states[simIdx] = localState;
}
