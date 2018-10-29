CXX = g++
WARN = -Wall -Wextra -pedantic
OPTIM = -O3 -march=native
STD = -std=c++17
INCLUDE = -I$(DNEST4_PATH)
FLAGS = $(STD) $(OPTIM) $(WARN) $(INCLUDE)

default:
	$(CXX) $(FLAGS) -c *.cpp

