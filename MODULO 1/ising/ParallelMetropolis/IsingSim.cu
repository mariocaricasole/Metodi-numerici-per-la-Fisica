#include "IsingMethods.h"
#include <thread>

int main()
{
    /************************ PARAMETERS ************************/
    //simulation parameters
    const unsigned char L = 4;
    const unsigned char nBeta = 64;
    const unsigned char nResamplings = 64;
    const int dataPoints = 1e5;
    const int batchSize = 1e5;
    const int blockingSize = 1e4;
    const unsigned long long memSize = L*nBeta*batchSize;

    /******************* MEMORY ALLOCATIONS *********************/
    //allocate random generators
    curandState *dStates, *dStatesBS;
    cudaMallocManaged(&dStates, L*nBeta*sizeof(curandState));
    cudaMallocManaged(&dStatesBS, L*nBeta*sizeof(curandState));

    //allocate unified memory
    double *dBeta;
    signed short **dMagMarkov, **dEnergyMarkov;
    signed short *dMagCurrent, *dEnergyCurrent;
    long long *dMagCumulant, *dEnergyCumulant;
    bool* dLattices;

    //allocate twin memory registers for simultaneous execution and copy of batches, make a host register for later use (planned interleaved GPU bootstrapping and CPU copy to disk)
    cudaMallocHost(&dMagMarkov, 2*sizeof(signed short*));
    cudaMallocHost(&dEnergyMarkov, 2*sizeof(signed short*));

    for(int i=0;i<2;i++)
    {
        cudaMallocManaged(&dMagMarkov[i], memSize*sizeof(signed short));
        cudaMallocManaged(&dEnergyMarkov[i], memSize*sizeof(signed short));
    }

    //allocating array of registers for magnetization and energy for inter-batch use
    cudaMallocManaged(&dMagCurrent, nBeta*L*sizeof(signed short));
    cudaMallocManaged(&dEnergyCurrent, nBeta*L*sizeof(signed short));

    //allocating array of cumulants for holding the sums during bootstrapping
    cudaMallocManaged(&dMagCumulant, nBeta*L*nResamplings*sizeof(long long));
    cudaMallocManaged(&dEnergyCumulant, nBeta*L*nResamplings*sizeof(long long));

    //allocating array of betas and lattices memory for inter-batch use
    cudaMallocManaged(&dBeta, nBeta*sizeof(double));
    cudaMallocManaged(&dLattices, 5400*nBeta*sizeof(bool));

    /*********************** INITIALIZATIONS ********************/
    //initialize all lattices to cold state
    for(unsigned int i=0; i<5400*nBeta; i++)
        dLattices[i]=true;

    //run random generator initialization kernel
    initRandom<<<L,nBeta>>>(dStates,1221);
    initRandom<<<L,nBeta>>>(dStatesBS, 1234);

    //create event to signal end of kernel execution
    cudaEvent_t simulationDone, bootstrapDone;
    cudaEventCreate(&simulationDone);
    cudaEventCreate(&bootstrapDone);

    //create two streams, one for copy the other for execution of kernel
    cudaStream_t Stream1, Stream2;
    cudaStreamCreate(&Stream1);
    cudaStreamCreate(&Stream2);
//
    /***************************** SIMULATION RUN *************************************/
    //run the kernel using one memory, while copying data to disk on the other memory
    int nBatches = dataPoints/batchSize;
    for(int i=0; i<nBatches;i++){
        if(i!=0)
        {
            //wait for kernel execution to be completed
            cudaStreamWaitEvent(Stream2, simulationDone);
            bootstrap<<<L,nBeta, 0, Stream2>>>(nResamplings, batchSize, blockingSize, dMagMarkov[(i-1)%2], dEnergyMarkov[(i-1)%2], dMagCumulant, dEnergyCumulant, dStatesBS);
            cudaEventRecord(bootstrapDone, Stream2);
        }
        simulation<<<L,nBeta, 0, Stream1>>>(i,batchSize, dBeta, dMagMarkov[i%2], dEnergyMarkov[i%2], dMagCurrent, dEnergyCurrent, dLattices, dStates);
        cudaEventRecord(simulationDone, Stream1);
        if(i!=0)
            cudaStreamWaitEvent(Stream1,bootstrapDone);
    }
    //copy the last batch produced
    cudaStreamWaitEvent(Stream2, simulationDone);
    bootstrap<<<L,nBeta, 0, Stream2>>>(nResamplings, batchSize, blockingSize, dMagMarkov[(nBatches-1)%2], dEnergyMarkov[(nBatches-1)%2], dMagCumulant, dEnergyCumulant, dStatesBS);
    cudaDeviceSynchronize();

    //save the results to the disk
    saveResults(dataPoints, nResamplings, nBeta, L, dMagCumulant, dEnergyCumulant);

    /********************************* CLEAN UP ****************************************/
    //destroy streams
    cudaStreamDestroy(Stream1);
    cudaStreamDestroy(Stream2);

    //free device memory
    cudaFree(dStates);
    cudaFree(dStatesBS);
    cudaFree(dBeta);
    for(int i=0;i<2;i++)
    {
        cudaFree(dMagMarkov[i]);
        cudaFree(dEnergyMarkov[i]);
    }
    cudaFree(dMagMarkov);
    cudaFree(dEnergyMarkov);
    cudaFree(dMagCurrent);
    cudaFree(dEnergyCurrent);
    cudaFree(dMagCumulant);
    cudaFree(dEnergyCumulant);
    cudaFree(dLattices);


    return 0;
}
