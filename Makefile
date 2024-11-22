CC = gcc
CFLAGS = -Wall -Wextra -std=c11
SRC = main.c # Only include main.c if math_nn is not needed
OBJ_DIR = build
OBJ = $(patsubst %.c,$(OBJ_DIR)/%.o,$(SRC))
DEPS = $(wildcard *.h)

all: main

main: $(OBJ)
	$(CC) $(CFLAGS) -o $@ $^ -lm

$(OBJ_DIR)/%.o: %.c $(DEPS)
	@mkdir -p $(OBJ_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

build: $(OBJ)
	$(CC) $(CFLAGS) -o main $(OBJ) -lm

clean:
	rm -rf $(OBJ_DIR) main

run: all
	./main