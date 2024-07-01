#ifndef PATH_H
#define PATH_H

#include <random>
#include <ctime>

class Path
{
public:
    Path(int N);
    ~Path();
    
    //getters
    inline float x(int idx) const { return m_x[idx];}

    //update lattice using metropolis
    int metropolisUpdate(float delta, float eta);

private:
    //attributes
    int m_N;
    float * m_x;

    //private member functions
    int* periodicNeighboursIndexes(int i0);

    //random generators
    std::default_random_engine m_generator;
    std::uniform_real_distribution<float> m_randProb;
    std::uniform_int_distribution<int> m_randIdx;

};

#endif
