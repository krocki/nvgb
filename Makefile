GCC ?= gcc
CUDA ?= /usr/local/cuda
NVCC ?= ${CUDA}/bin/nvcc

TARGETS := nv_gb
NVCC_FLAGS := -O2 --disable-warnings -ccbin gcc
GCC_FLAGS := -O2

all: ${TARGETS}

glgb: glmain.o nv_gb_gl.o
	${NVCC} $? ${GCC_FLAGS} -lGL -lGLU -lglut -o glgb
glmain: glmain.o glcuda.o
	${NVCC} $? ${GCC_FLAGS} -lGL -lGLU -lglut -o glmain

%.o: %.c
	${NVCC} ${NVCC_FLAGS} -o $@ -c $<
%.o: %.cu
	${NVCC} ${NVCC_FLAGS} -o $@ -c $<
%: %.o
	${NVCC} ${NVCC_FLAGS} $< -o $@
clean:
	rm -rf *.o ${TARGETS}
