#include "Path.h"
#include <iostream>

Path::Path(int N) : m_N(N)
{
    //initialize distributions
    m_generator = std::default_random_engine(std::time(0));
    m_randProb = std::uniform_real_distribution<float>(0., 1.);
    m_randIdx = std::uniform_int_distribution<int>(0,N);

    //initialize array
    m_x = new float[N];
    for(int i=0;i<N;i++)
        m_x[i] = 0;
}

Path::~Path()
{
    delete[] m_x;
}


int* Path::periodicNeighboursIndexes(int i0)
{
    int i_before, i_after;

    if(i0 == 0)
        i_before = m_N-1;
    else
        i_before = i0-1;

    if(i0 == m_N-1)
        i_after = 0;
    else
        i_after = i0+1;

    int* res = new int[2];
    res[0] = i_before;
    res[1] = i_after;

    return res;
}


int Path::metropolisUpdate(float delta, float eta)
{
    //choose a random site from the path and perform a random movement
    int i0 = m_randIdx(m_generator);
    float xp = m_x[i0] - delta + 2*delta*m_randProb(m_generator);

    //take the next element and the element before, using periodic boundary conditions
    int* neighIdx = periodicNeighboursIndexes(i0);
    float neighbours[2] = {m_x[neighIdx[0]], m_x[neighIdx[1]]};

    //evaluate acceptance ratio
    float force = neighbours[1] + neighbours[0];
    float dS = (xp*xp - m_x[i0]*m_x[i0])*(eta/2 + 1/eta) - 1/eta * force * (xp - m_x[i0]);

    int res;
    //accept-reject: if accept flip the spin and adjust parameters
    if(dS<=0)
    {
        m_x[i0] = xp;
        res=1;
    }
    else if(exp(-dS) > m_randProb(m_generator)){
        m_x[i0] = xp;
        res=1;
    }
    else{
        res=0;
    }

    delete neighIdx;

    return res;
}
