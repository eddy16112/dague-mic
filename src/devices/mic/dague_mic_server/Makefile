CXXFLAGS += -g -I /usr/include
LDFLAGS +=
#EXEC_DIR = 
MIC_BUILD_FLAGS = -Wall $< -lscif -lpthread -o
HOST_BUILD_FLAGS = -Wall -DHOST $< -lscif -lpthread -o
HOST_CC = gcc


MIC_CC_PATH =/usr/linux-k1om-4.7/bin
MIC_CC = x86_64-k1om-linux-gcc

.PHONY: all

all: build


build:	dague_mic_server

dague_mic_server: dague_mic_server.c
	$(MIC_CC_PATH)/$(MIC_CC) $(CXXFLAGS) $(LDFLAGS) $(MIC_BUILD_FLAGS) $@

#mic_server: mic_server.c 
#	icc -mmic $(CXXFLAGS) $(LDFLAGS) $(MIC_BUILD_FLAGS) $@
clean:
	rm dague_mic_server
