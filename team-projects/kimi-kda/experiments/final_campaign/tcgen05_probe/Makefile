CUDA_HOME ?= /usr/local/cuda-13.0
NVCC ?= $(if $(wildcard $(CUDA_HOME)/bin/nvcc),$(CUDA_HOME)/bin/nvcc,nvcc)
CUOBJDUMP ?= $(if $(wildcard $(CUDA_HOME)/bin/cuobjdump),$(CUDA_HOME)/bin/cuobjdump,cuobjdump)
ARCH ?= 103a
TARGET := phase6_probe
SOURCE := phase6_probe.cu

NVCCFLAGS := -O3 -std=c++17 --expt-relaxed-constexpr -lineinfo \
	-gencode arch=compute_$(ARCH),code=sm_$(ARCH) \
	-Xptxas=-v,--warn-on-spills

# CUDA 13 can insert runtime TMEM bounds checks.  Use for correctness bring-up,
# then rebuild without them before collecting performance numbers.
ifeq ($(GUARDRAILS),1)
NVCCFLAGS += -Xptxas=-g-tmem-access-check
endif

.PHONY: all clean sass guardrails

all: $(TARGET)

$(TARGET): $(SOURCE)
	$(NVCC) $(NVCCFLAGS) -o $@ $<

guardrails:
	$(MAKE) clean
	$(MAKE) GUARDRAILS=1 $(TARGET)

sass: $(TARGET)
	$(CUOBJDUMP) --dump-sass $(TARGET) > $(TARGET).sass

clean:
	rm -f $(TARGET) $(TARGET).sass
