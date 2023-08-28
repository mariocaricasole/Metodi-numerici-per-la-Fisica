#include <iostream>
#include <fstream>
#include "IsingLattice.h"

int main(){
    //define and initialize simulation parameters
    int Larr[4] = {20,30,40,50};
    int nBeta = 50;
    float dBeta = (0.54 - 0.34)/nBeta;
    float beta;

    //open file to save data of the Markov chain run
    std::ofstream fData;
    std::ofstream fBeta;
    std::ofstream fLarr;
    fBeta.open("beta.txt");
    fLarr.open("L.txt");

    IsingLattice* lattice;
    for(int L : Larr)
    {
        fLarr << L << std::endl;
        std::string fileName = "data" + std::to_string(L) + ".txt";
        std::cout<< fileName << std::endl;
        fData.open(fileName);
        lattice = new IsingLattice(L);
        for(int betaIdx = 0; betaIdx < nBeta; betaIdx++)
        {
            beta = 0.34 + betaIdx*dBeta;
            if(L==Larr[0])
            {
                fBeta << beta << std::endl;
            }

            int counter=0;
            //Markov chain evolution
                for(int i=0;i<10000000;i++){
                    lattice->metropolisUpdate(beta);
                    counter+=1;
                    if(counter==100)
                    {
                        fData << lattice->E() << " " << lattice->M() << std::endl;
                        counter=0;
                    }
                }
        }
        fData.close();
    }

    fBeta.close();
    fLarr.close();
    delete lattice;

    return 0;
}

