#include "IsingLattice.h"
#include <iostream>

IsingLattice::IsingLattice(int L) : m_L(L)
{
    //initialize distributions
    m_generator = std::default_random_engine(std::time(0));
    m_randProb = std::uniform_real_distribution<float>(0., 1.);
    m_randIdx = std::uniform_int_distribution<int>(0,L-1);

    //initialize array
    m_array = new float*[L];
    for(int i=0;i<L;i++)
    {
        m_array[i] = new float[L];
        for(int j=0;j<L;j++)
        {
            m_array[i][j] = 1.;  //cold start
        }
    }
    
    //cold start
    m_E = -2;
    m_M = 1;
}

IsingLattice::~IsingLattice()
{
    for(int i=0; i<m_L; i++)
    {
        delete[] m_array[i];
    }

    delete[] m_array;
}


int* IsingLattice::periodicNeighboursIndexes(int i0, int j0)
{
    int il, ih, jl, jh;

    if(i0 == 0)
        il = m_L-1;
    else
        il = i0-1;

    if(i0 == m_L-1)
        ih = 0;
    else
        ih = i0+1;

    if(j0 == 0)
        jl = m_L - 1;
    else
        jl = j0-1;

    if(j0 == m_L-1)
        jh = 0;
    else
        jh = j0+1;

    int* res = new int[4];
    res[0] = il;
    res[1] = ih;
    res[2] = jl;
    res[3] = jh;

    return res;
}


void IsingLattice::metropolisUpdate(float beta)
{
    //choose a random site from the IsingLattice
    int i0 = m_randIdx(m_generator);
    int j0 = m_randIdx(m_generator);

//     std::cout << i0 << " " << j0 << std::endl;

    //take all the nearby lattice points
    int* neighIdx = periodicNeighboursIndexes(i0,j0);
    float neighbours[4] = {m_array[neighIdx[0]][j0], m_array[neighIdx[1]][j0], m_array[i0][neighIdx[2]], m_array[i0][neighIdx[3]]};
    //evaluate acceptance ratio
    float force = 0.;
    for(auto item : neighbours)
    {
        force+=item;
    }

    float acceptance = std::min(1., exp(-2*m_array[i0][j0]*force*beta));

    //accept-reject: if accept flip the spin and adjust parameters
    if(acceptance > m_randProb(m_generator))
    {
        m_M = m_M - 2*m_array[i0][j0]/(m_L*m_L);
        m_E = m_E + 2*force*m_array[i0][j0]/(m_L*m_L);
        m_array[i0][j0]*=-1;
    }

    delete neighIdx;
}
