CXX = g++
WARN = -Wall -Wextra -pedantic
OPTIM = -O3 -march=native
STD = -std=c++17
FLAGS = $(STD) $(OPTIM) $(WARN)

default:
	$(CXX) $(FLAGS) -c *.cpp

