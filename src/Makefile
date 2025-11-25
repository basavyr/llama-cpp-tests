CXX = g++
CXX_FLAGS = -std=c++17 -O3 -Wall -Wextra

TARGET = simple-test

SOURCES = simple-test.cpp

$(TARGET): $(SOURCES)
	$(CXX) $(CXX_FLAGS) -o $(TARGET).o $(SOURCES)

clean:
	rm -f *.o *.out $(TARGET)