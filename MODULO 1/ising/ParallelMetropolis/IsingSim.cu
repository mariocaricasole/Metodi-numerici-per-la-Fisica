#include "IsingMethods.h"
#include <thread>

int main()
{
    /************************ PARAMETERS ************************/
    //simulation parameters
    const unsigned char L = 4;
    const unsigned char nBeta = 64;
    const int dataPoints = 2e5;
    const int batchSize = 1e5;
    const unsigned long long memSize = L*nBeta*batchSize;

    /******************* MEMORY ALLOCATIONS *********************/
    //allocate random generators
    curandState *dStates;
    cudaMallocManaged(&dStates, L*nBeta*sizeof(curandState));

    //allocate unified memory
    double *dBeta;
    signed char **dMagMarkov;
    signed short **dEnergyMarkov;
    bool* dLattices;

    cudaMallocHost(&dMagMarkov, 3*sizeof(signed char*));
    cudaMallocHost(&dEnergyMarkov, 3*sizeof(signed short*));

    for(int i=0;i<2;i++)
    {
        cudaMallocManaged(&dMagMarkov[i], memSize*sizeof(signed char));
        cudaMallocManaged(&dEnergyMarkov[i], memSize*sizeof(signed short));
    }
    cudaMallocHost(&dMagMarkov[2], memSize*sizeof(signed char));
    cudaMallocHost(&dEnergyMarkov[2], memSize*sizeof(signed short));
    cudaMallocManaged(&dBeta, nBeta*sizeof(double));
    cudaMallocManaged(&dLattices, 5400*nBeta*sizeof(bool));

    /*********************** INITIALIZATIONS ********************/
    //initialize all lattices to cold state
    for(unsigned int i=0; i<5400*nBeta; i++)
        dLattices[i]=true;

    //run random generator initialization kernel
    initRandom<<<L,nBeta>>>(dStates,1221);

    //create event to signal end of kernel execution
    cudaEvent_t kernelDone;
    cudaEventCreate(&kernelDone);

    //create two streams, one for copy the other for execution of kernel
    cudaStream_t copyStream, executeStream;
    cudaStreamCreate(&copyStream);
    cudaStreamCreate(&executeStream);
//
    /***************************** SIMULATION RUN *************************************/
    //run the kernel using one memory, while copying data to disk on the other memory
    for(int i=0; i<dataPoints/batchSize;i++){
        if(i!=0)
        {
            cudaStreamWaitEvent(copyStream, kernelDone);
            cudaMemcpyAsync(dMagMarkov[2], dMagMarkov[(i-1)%2], memSize*sizeof(signed char), cudaMemcpyDeviceToHost, copyStream);
            cudaMemcpyAsync(dEnergyMarkov[2], dEnergyMarkov[(i-1)%2], memSize*sizeof(signed short), cudaMemcpyDeviceToHost, copyStream);
        }
        simulation<<<L,nBeta, 0, executeStream>>>(batchSize, dBeta, dMagMarkov[i%2], dEnergyMarkov[i%2], dLattices, dStates);
        cudaEventRecord(kernelDone, executeStream);
        if(i!=0)
        {
            cudaStreamSynchronize(copyStream);
            copyToDisk(i-1, batchSize, memSize, dMagMarkov[2], dEnergyMarkov[2]);
        }
    }
    //last copy
    cudaStreamWaitEvent(copyStream, kernelDone);
    cudaMemcpyAsync(dMagMarkov[2], dMagMarkov[(dataPoints/batchSize -1)%2], memSize*sizeof(signed char), cudaMemcpyDeviceToHost, copyStream);
    cudaMemcpyAsync(dEnergyMarkov[2], dEnergyMarkov[(dataPoints/batchSize -1)%2], memSize*sizeof(signed short), cudaMemcpyDeviceToHost, copyStream);
    cudaStreamSynchronize(copyStream);
    copyToDisk(dataPoints/batchSize-1, batchSize, memSize, dMagMarkov[2], dEnergyMarkov[2]);
    cudaDeviceSynchronize();

    /********************************* CLEAN UP ****************************************/
    //destroy streams
    cudaStreamDestroy(copyStream);
    cudaStreamDestroy(executeStream);

    //free device memory
    cudaFree(dStates);
    cudaFree(dBeta);
    for(int i=0;i<3;i++)
    {
        cudaFree(dMagMarkov[i]);
        cudaFree(dEnergyMarkov[i]);
    }
    cudaFree(dMagMarkov);
    cudaFree(dEnergyMarkov);
    cudaFree(dLattices);


    return 0;
}
