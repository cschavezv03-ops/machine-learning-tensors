CC = gcc
CFLAGS = -Wall -Wextra -O2
LDLIBS = -lTensors_cpu -lm -ltcl -ltk
LDFLAGS = -Wl,-rpath,/usr/local/lib

.PHONY: all clean

all: quiz01

quiz01: quiz01.c
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS) $(LDLIBS)

clean:
	rm -f quiz01
