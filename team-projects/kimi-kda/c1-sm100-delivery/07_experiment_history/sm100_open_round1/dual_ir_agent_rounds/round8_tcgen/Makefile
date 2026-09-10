CUDA_HOME ?= /usr/local/cuda-13.0
NVCC ?= $(if $(wildcard $(CUDA_HOME)/bin/nvcc),$(CUDA_HOME)/bin/nvcc,nvcc)
CUOBJDUMP ?= $(if $(wildcard $(CUDA_HOME)/bin/cuobjdump),$(CUDA_HOME)/bin/cuobjdump,cuobjdump)
ARCH ?= 103a

TARGET := round8_v16_preferred
SOURCE := round8_v16_preferred.cu
BASE_SOURCE := phase6_probe_preferred.cu

NVCCFLAGS := -O3 -std=c++17 --expt-relaxed-constexpr -lineinfo \
	-gencode arch=compute_$(ARCH),code=sm_$(ARCH) \
	-Xptxas=-v,--warn-on-spills

ifeq ($(GUARDRAILS),1)
NVCCFLAGS += -Xptxas=-g-tmem-access-check
endif

.PHONY: all clean guardrails sass

all: $(TARGET)

$(TARGET): $(SOURCE) $(BASE_SOURCE)
	$(NVCC) $(NVCCFLAGS) -o $@ $(SOURCE)

guardrails:
	$(MAKE) clean
	$(MAKE) GUARDRAILS=1 $(TARGET)

sass: $(TARGET)
	$(CUOBJDUMP) --dump-sass $(TARGET) > $(TARGET).sass

clean:
	rm -f $(TARGET) $(TARGET).sass
