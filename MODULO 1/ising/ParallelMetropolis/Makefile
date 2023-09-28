objects = IsingMethods.o IsingSim.o

all : $(objects)
	nvcc -ccbin g++ --compiler-options -L/home/mario/Software/root/lib -lGui -lCore -lImt -lRIO -lNet -lHist -lGraf -lGraf3d -lGpad -lROOTVecOps -lTree -lTreePlayer -lRint -lPostscript -lMatrix -lPhysics -lMathCore -lThread -lMultiProc -lROOTDataFrame -lm -ldl -I/home/mario/Software/root/include -gencode arch=compute_50,code=sm_50 $(objects) -o IsingSim

%.o: %.cu
	nvcc -ccbin g++ --compiler-options -L/home/mario/Software/root/lib -lGui -lCore -lImt -lRIO -lNet -lHist -lGraf -lGraf3d -lGpad -lROOTVecOps -lTree -lTreePlayer -lRint -lPostscript -lMatrix -lPhysics -lMathCore -lThread -lMultiProc -lROOTDataFrame -lm -ldl -I/home/mario/Software/root/include -I. -gencode arch=compute_50,code=sm_50 -dc $< -o $@

clean:
	rm -f *o IsingSim
