CXX ?= c++
CXXFLAGS ?= -std=c++20 -g -fsanitize=address -Wall -Wextra -Werror -pedantic -O0

.PHONY: all exec tests tests-slow plot-m3

all: exec

exec: bin/exec

tests: exec
	./bin/exec checkpoints/m1_checkpoint.txt

tests-slow: exec
	./bin/exec test-slow checkpoints/m3_energy_scaling.csv

plot-m3: tests-slow
	pdflatex -halt-on-error -interaction=nonstopmode -output-directory=checkpoints \
		-jobname=m3_energy_scaling plots/m3_energy_scaling.tex

bin/exec: src/pasta_cpu.cpp | bin checkpoints
	$(CXX) $(CXXFLAGS) $< -o $@

bin checkpoints:
	mkdir -p $@
