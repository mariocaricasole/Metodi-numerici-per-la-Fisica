#ifndef ISINGLATTICE_H
#define ISINGLATTICE_H

#include <random>
#include <ctime>

class IsingLattice
{
public:
    IsingLattice(int L);
    ~IsingLattice();
    
    //getters
    inline float E() const { return m_E;}
    inline float M() const { return m_M;}

    //update lattice using metropolis
    void metropolisUpdate(float beta);

private:
    //attributes
    int m_L;
    float **m_array;
    float m_E, m_M;

    //private member functions
    int* periodicNeighboursIndexes(int i0, int j0);

    //random generators
    std::default_random_engine m_generator;
    std::uniform_real_distribution<float> m_randProb;
    std::uniform_int_distribution<int> m_randIdx;

};

#endif
