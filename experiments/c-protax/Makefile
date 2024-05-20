C = gcc
CFLAGS = -g # debug #-O2 
LDFLAGS =
LIBS = -lm

PROGS = classify_v0 classify_v1 classify_v2
PROGS_SRCS = $(PROGS:=.c)

ROUTINES_SRCS = routines_taxonomy.c routines_sequence.c routines_model.c
ROUTINES_OBJS = $(ROUTINES_SRCS:.c=.o)

DEFS = defs.h

all: $(PROGS)

objs: $(PROGS_SRCS:.c=.o) $(ROUTINES_OBJS)

%.o: %.c $(DEFS)
	$(C) $(CFLAGS) -c $< -o $@

$(PROGS) : %: %.o $(ROUTINES_OBJS) $(DEFS)
	$(C) $(CFLAGS) $(LDFLAGS) $< -o $@ $(ROUTINES_OBJS) $(LIBS)

clean:
	rm -f $(PROGS) *.o