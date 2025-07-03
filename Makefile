CXX = g++-14
CXXFLAGS = -std=c++11 -Wall -g -fopenmp -Iinclude
PY = python3
HEADER = header
PLOT = plotter
PROGRAM = tsp
PARAMS = 2 3 4 5 6 7 8
file := $(file)

target:
	@echo file: $(file)
	@echo numP: $(numP)

all: $(PROGRAM)

$(PROGRAM): $(PROGRAM).cpp
	$(CXX) $(CXXFLAGS) -o $@ $<
	

.PHONY: clean

clean:
	rm -f $(PROGRAM)
	rm -f *.csv

run: $(PROGRAM)
	$(PY) $(HEADER).py
	@for param in $(PARAMS); do \
		echo "Running $(PROGRAM) with parameter: $(file) and $$param"; \
		./$(PROGRAM) $(file) $$param; \
		echo "------------------------------"; \
	done

plot:
	$(PY) $(PLOT).py
