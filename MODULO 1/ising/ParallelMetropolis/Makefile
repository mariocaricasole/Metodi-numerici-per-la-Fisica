IsingSim: IsingSim.o
	nvcc -ccbin g++ -gencode arch=compute_50,code=sm_50 -o $@ $^

IsingSim.o: IsingSim.cu
	nvcc -ccbin g++ --std=c++20 -gencode arch=compute_50,code=sm_50 -o $@ -dc $^

clean:
	rm -f IsingSim IsingSim.o

clobber: clean
