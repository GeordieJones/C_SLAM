CC = gcc
# Added -finstrument-functions so GCC injects entry/exit hooks for your function tree
CFLAGS = -O3 -Wall -Wextra -finstrument-functions \
         -I. \
         -I../../metrics/libraries \
         -I../../neural_network/libraries

CFLAGS += -g -O2

CFLAGS += -no-pie

LDFLAGS = -lm

# Consolidated list of all required source dependencies
SRC_COMMON = distance_detection_train.c \
      pipelines.c \
      ../../neural_network/libraries/trainer.c \
      ../../neural_network/libraries/layer.c \
      ../../neural_network/libraries/network.c \
      ../../neural_network/libraries/dataset.c \
      ../../neural_network/libraries/tensor.c \
	  ../../neural_network/libraries/activation.c

      #../../metrics/libraries/profiler.c \


TARGET = train_distance

all: $(TARGET)

#$(TARGET): $(SRC)
	#$(CC) $(CFLAGS) $(SRC) $(LDFLAGS) -o $(TARGET)

$(TARGET): $(SRC_COMMON) ../../metrics/libraries/timer.c
	# 1. Compile the profiler explicitly WITHOUT self-instrumentation
	#$(CC) $(CFLAGS) -fno-instrument-functions -c ../../metrics/libraries/profiler.c -o profiler.o
	$(CC) $(CFLAGS) -fno-instrument-functions -c ../../metrics/libraries/timer.c -o timer.o
	# 2. Compile everything else WITH instrumentation and link together
	#$(CC) $(CFLAGS) $(SRC_COMMON) profiler.o $(LDFLAGS) -o $(TARGET)
	$(CC) $(CFLAGS) $(SRC_COMMON) timer.o $(LDFLAGS) -o $(TARGET)



clean:
	rm -f $(TARGET)
