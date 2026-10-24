CXX=g++
CXX_FLAGS=-std=c++20 -g -fsanitize=address -Iincludes -Wall -Wextra -Werror -pedantic -O0 -Wno-error=unused-parameter

exec: bin/exec
tests: bin/tests

bin/exec: ./src/m1_box_checkpoint.cpp
	$(CXX) $(CXX_FLAGS) $^ -o $@