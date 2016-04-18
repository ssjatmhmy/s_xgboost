export CC  = gcc
export CXX = g++
export CFLAGS = -Wall -msse2  -Wno-unknown-pragmas -fopenmp -g

# specify tensor path
BIN = xgboost
OBJ =
.PHONY: clean all

all: $(BIN) $(OBJ)
export LDFLAGS= -pthread -lm 

xgboost: src/xgboost_main.cpp src/gbm/*.h src/learner/*.h src/*.h src/tree/*.h src/tree/*.hpp

$(BIN) : 
	$(CXX) $(CFLAGS) $(LDFLAGS) -o $@ $(filter %.cpp %.o %.c, $^)

$(OBJ) : 
	$(CXX) -c $(CFLAGS) -o $@ $(firstword $(filter %.cpp %.c, $^))

install:
	cp -f -r $(BIN)  $(INSTALL_PATH)

clean:
	$(RM) $(OBJ) $(BIN) *~
