all: clean softmax logistic

CFLAGS = -Wall --std=c++14 -O3 -Iinclude/

softmax: fast_parser murmurhash util
	g++ $(CFLAGS) -fopenmp -pthread -mavx softmax.cpp fast_parser.o MurmurHash.o util.o -o softmax
	g++ $(CFLAGS) -fopenmp -pthread -mavx coarse_mission_softmax.cpp fast_parser.o MurmurHash.o util.o -o coarse_mission_softmax
	g++ $(CFLAGS) -fopenmp -pthread -mavx fine_mission_softmax.cpp fast_parser.o MurmurHash.o util.o -o fine_mission_softmax

logistic: fast_parser murmurhash util
	g++ $(CFLAGS) -fopenmp -pthread -mavx mission_logistic.cpp fast_parser.o MurmurHash.o util.o -o mission_logistic

parser: fast_parser murmurhash
	g++ $(CFLAGS) -fopenmp -pthread parser_main.cpp fast_parser.o MurmurHash.o -o parser

fast_parser:
	g++ $(CFLAGS) -o fast_parser.o -c fast_parser.cpp

murmurhash:
	g++ $(CFLAGS) -o MurmurHash.o -c MurmurHash.cpp

util:
	g++ $(CFLAGS) -mavx -o util.o -c util.cpp

clean:
	rm -rf MurmurHash.o
	rm -rf fast_parser.o
	rm -rf util.o
	rm -rf mission_logistic
	rm -rf fine_mission_softmax
	rm -rf coarse_mission_softmax
	rm -rf softmax
