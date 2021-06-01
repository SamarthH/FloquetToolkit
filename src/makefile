IDIR =../include
CC=gcc
CFLAGS=-I$(IDIR) -Wall 
EXTRAFLAGS=

ODIR=obj
LDIR =../lib

LIBS= -lm -lgsl -lgslcblas

_DEPS = diffEqSolvers.h
DEPS = $(patsubst %,$(IDIR)/%,$(_DEPS))

_OBJ = diffEqSolvers.o tests.o
OBJ = $(patsubst %,$(ODIR)/%,$(_OBJ))


$(ODIR)/%.o: %.c $(DEPS)
	$(CC) -L$(LDIR) -c -o $@ $< $(CFLAGS) $(EXTRAFLAGS) $(LIBS)

tests: $(OBJ)
	$(CC) -o $@ $^ $(CFLAGS) $(EXTRAFLAGS) $(LIBS)


.PHONY: clean

clean:
	rm -f $(ODIR)/*.o *~ core $(IDIR)/*~ 

nest:
	set OMP_DYNAMIC=true
	set OMP_NESTED=true