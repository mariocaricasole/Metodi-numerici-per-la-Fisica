IsingSim: IsingSim.o
	nvcc -ccbin g++ --compiler-options -L/home/mario/Software/root/lib -lGui -lCore -lImt -lRIO -lNet -lHist -lGraf -lGraf3d -lGpad -lROOTVecOps -lTree -lTreePlayer -lRint -lPostscript -lMatrix -lPhysics -lMathCore -lThread -lMultiProc -lROOTDataFrame -lm -ldl -I/home/mario/Software/root/include -gencode arch=compute_50,code=sm_50 -o $@ $^

IsingSim.o: IsingSim.cu
	nvcc -ccbin g++ --compiler-options -L/home/mario/Software/root/lib -lGui -lCore -lImt -lRIO -lNet -lHist -lGraf -lGraf3d -lGpad -lROOTVecOps -lTree -lTreePlayer -lRint -lPostscript -lMatrix -lPhysics -lMathCore -lThread -lMultiProc -lROOTDataFrame -lm -ldl -I/home/mario/Software/root/include -gencode arch=compute_50,code=sm_50 -o $@ -dc $^

clean:
	rm -f IsingSim IsingSim.o

clobber: clean
