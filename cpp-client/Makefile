CXX      = g++
CXXFLAGS = -std=c++17 -Wall -Wextra -O2
TARGET   = policy-cli
SRCS     = main.cpp http.cpp json.cpp
OBJS     = $(SRCS:.cpp=.o)

ITEST    = test_json

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(ITEST): test_json.cpp json.cpp json.h
	$(CXX) $(CXXFLAGS) -o $@ test_json.cpp json.cpp

test: $(ITEST)
	./$(ITEST)

clean:
	rm -f $(OBJS) $(TARGET) $(ITEST)

.PHONY: all clean test