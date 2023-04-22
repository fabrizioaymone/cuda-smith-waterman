# Usage:
# make        # compile all binary
# make clean  # remove ALL binaries and objects

.PHONY = all clean

CC = nvcc -x cu

OPTIMIZEFLAG = -O3

all: main.out

main.out: main.c cpu.c gpu_pure.cu gpu_conventional.cu
	$(CC) -O3 main.c cpu.c gpu_pure.cu gpu_conventional.cu -o main.out


clean:
	rm *.out
