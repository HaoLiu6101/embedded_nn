CC = gcc
CFLAGS = -Wall -Wextra -std=c11
OBJ_DIR = build
DEPS = $(wildcard *.h)

# Define source files for main executable
MAIN_SRC = main.c math_nn.c
MAIN_OBJ = $(MAIN_SRC:%.c=$(OBJ_DIR)/%.o)

# Define source files for test executable
TEST_SRC = test_math_nn.c math_nn.c
TEST_OBJ = $(TEST_SRC:%.c=$(OBJ_DIR)/%.o)

# Pattern rule to build object files
$(OBJ_DIR)/%.o: %.c $(DEPS)
	@mkdir -p $(OBJ_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

all: build

build: $(MAIN_OBJ)
	$(CC) $(CFLAGS) -o main $(MAIN_OBJ) -lm

run: build
	./main

test: $(TEST_OBJ)
	$(CC) $(CFLAGS) -o test_math_nn $(TEST_OBJ) -lm
	./test_math_nn

clean:
	rm -rf $(OBJ_DIR) main test_math_nn