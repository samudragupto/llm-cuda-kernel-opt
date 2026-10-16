# ==============================================================================
# Makefile: Fused Residual-Add + RMSNorm Benchmark (Ada Lovelace, sm_89)
# ==============================================================================

NVCC       ?= nvcc
ARCH       ?= sm_89
CXXFLAGS   ?= -O3 -std=c++17 -arch=$(ARCH) -lineinfo --use_fast_math
SRC        := benchmark_rmsnorm.cu
TARGET     := benchmark_rmsnorm

# OS Detection & Environment setup
ifeq ($(OS),Windows_NT)
    EXE         := $(TARGET).exe
    RM          := del /F /Q
    VCVARS      := "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
    COMPILE_CMD := cmd /c "call $(VCVARS) >nul && $(NVCC) $(CXXFLAGS) $(SRC) -o $(EXE)"
else
    EXE         := $(TARGET)
    RM          := rm -f
    COMPILE_CMD := $(NVCC) $(CXXFLAGS) $(SRC) -o $(EXE)
endif

.PHONY: all clean run profile graph help

all: $(EXE)

$(EXE): $(SRC)
	@echo [NVCC] Compiling $(SRC) for $(ARCH)...
	@$(COMPILE_CMD)
	@echo [OK] Successfully built $(EXE)

run: $(EXE)
	@echo [RUN] Executing $(EXE) (eager mode, DRAM-streaming)...
	@./$(EXE)

graph: $(EXE)
	@echo [RUN] Executing $(EXE) (CUDA Graph mode)...
	@./$(EXE) --graph

profile: $(EXE)
	@bash ./profile.sh

clean:
	@echo [CLEAN] Removing build and profiling artifacts...
	-$(RM) $(EXE) *.exp *.lib *.o *.ncu-rep *.csv results_*.txt

help:
	@echo Available targets:
	@echo   make all      - Compile $(SRC) targeting $(ARCH) (default)
	@echo   make run      - Run the standard benchmark suite
	@echo   make graph    - Run the benchmark with CUDA Graph replay
	@echo   make profile  - Run Nsight Compute profiling via profile.sh
	@echo   make clean    - Remove binary, object, and profiling output files
