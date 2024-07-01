#include<iostream>
#include<fstream>
#include"Path.h"

int main(){
    // define parameters of the simulation
    int N = 100;
    int Nmc = 1e7;
    float beta = 5;
    float eta = beta/N;
    float delta = eta*eta;

    // initialize path
    Path* path;
    path = new Path(N);

    // open fData
    std::ofstream fData;
    fData.open("pathEvolution.txt");

    //Markov chain evolution
    float acceptance = 0;
    for(int i=0;i<Nmc; i++){
        for(int j=0; j<N; j++){
            acceptance += path->metropolisUpdate(delta,eta);
        }
        if(i%(int)1e2==0){
            for(int j=0; j<N; j++)
                fData << path->x(j) << "\t";
            fData << std::endl;
        }
    }
    fData.close();

    std::cout << "Acceptance ratio: " << acceptance/(N*Nmc) << std::endl;
    delete path;
    return 0;
}

