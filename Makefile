# Compiler and flags
CC = gcc
CFLAGS = -Wall -Wextra -std=c11 -Iinclude  # Include directory for headers

# Directories
OBJ_DIR = build
INCLUDE_DIR = include
SRC_DIR = lib

# Source and object files
MAIN_SRC = main.c
LIB_SRC = $(wildcard $(SRC_DIR)/*.c)  # Automatically include all .c files in lib/
MAIN_OBJ = $(OBJ_DIR)/main.o
LIB_OBJ = $(LIB_SRC:$(SRC_DIR)/%.c=$(OBJ_DIR)/%.o)

# Create the object directory if it doesn't exist
$(OBJ_DIR):
	@mkdir -p $(OBJ_DIR)

# Rule to build object files for library sources
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.c | $(OBJ_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

# Rule to build the main object file from main.c located at root level
$(MAIN_OBJ): $(MAIN_SRC) | $(OBJ_DIR)
	$(CC) $(CFLAGS) -c $< -o $@


# Rule to link the executable
all: $(MAIN_OBJ) $(LIB_OBJ)
	$(CC) -o main $(MAIN_OBJ) $(LIB_OBJ) -lm  # Link all object files into the final executable

run: all
	./main

clean:
	rm -rf $(OBJ_DIR) main




# # Define source files for test executable
# TEST_SRC = $(wildcard $(SRC_DIR)/*.c)
# TEST_OBJ = $(TEST_SRC:$(SRC_DIR)/%.c=$(OBJ_DIR)/%.o)



# test: $(TEST_OBJ)
# 	$(CC) $(CFLAGS) -o test_math_nn $(TEST_OBJ) -lm
# 	./test_math_nn