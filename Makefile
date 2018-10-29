CXX = g++
WARN = -Wall -Wextra -pedantic
OPTIM = -O3 -march=native -DEIGEN_NO_DEBUG
STD = -std=c++17
INCLUDE = -I $(DNEST4_PATH)
FLAGS = $(STD) $(OPTIM) $(WARN) $(INCLUDE)
LINK = -L $(DNEST4_PATH)/DNest4/code

default:
	$(CXX) $(FLAGS) -c *.cpp
	$(CXX) $(LINK) -o main *.o -ldnest4
	rm -f *.o
