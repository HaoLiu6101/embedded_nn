
CC = gcc
CFLAGS = -Wall -Wextra -std=c11

all: main

main: main.o math_nn.o
	$(CC) $(CFLAGS) -o main main.o math_nn.o -lm

main.o: main.c
	$(CC) $(CFLAGS) -c main.c

math_nn.o: math_nn.c
	$(CC) $(CFLAGS) -c math_nn.c

clean:
	rm -f *.o main

run: all
	./main