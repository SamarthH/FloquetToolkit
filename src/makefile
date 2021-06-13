IDIR =../include
CC=gcc
CFLAGS=-I$(IDIR) -Wall 
EXTRAFLAGS= -Ofast

ODIR=obj
LDIR =../lib

LIBS= -lm -lgsl -lgslcblas -fopenmp

_DEPS = diffEqSolvers.h floquet.h
DEPS = $(patsubst %,$(IDIR)/%,$(_DEPS))

_OBJ = diffEqSolvers.o floquet.o
OBJ = $(patsubst %,$(ODIR)/%,$(_OBJ))


$(ODIR)/%.o: %.c $(DEPS)
	$(CC) -L$(LDIR) -c -o $@ $< $(CFLAGS) $(EXTRAFLAGS) $(LIBS)

mathieu_undamped: $(OBJ) mathieu_undamped.o
	$(CC) -o $@ $^ $(CFLAGS) $(EXTRAFLAGS) $(LIBS)
	mkdir -p ../$@
	mv $@ ../$@/$@
	cp plotter_3d.py ../$@/plotter_3d.py

mathieu_damped_k_0_1: $(OBJ) mathieu_damped_k_0_1.o
	$(CC) -o $@ $^ $(CFLAGS) $(EXTRAFLAGS) $(LIBS)
	mkdir -p ../$@
	mv $@ ../$@/$@
	cp plotter_3d.py ../$@/plotter_3d.py

mathieu_damped_k_0_5: $(OBJ) mathieu_damped_k_0_5.o
	$(CC) -o $@ $^ $(CFLAGS) $(EXTRAFLAGS) $(LIBS)
	mkdir -p ../$@
	mv $@ ../$@/$@
	cp plotter_3d.py ../$@/plotter_3d.py

mathieu_damped_k_1: $(OBJ) mathieu_damped_k_1.o
	$(CC) -o $@ $^ $(CFLAGS) $(EXTRAFLAGS) $(LIBS)
	mkdir -p ../$@
	mv $@ ../$@/$@
	cp plotter_3d.py ../$@/plotter_3d.py

mathieu_damped_k_10: $(OBJ) mathieu_damped_k_10.o
	$(CC) -o $@ $^ $(CFLAGS) $(EXTRAFLAGS) $(LIBS)
	mkdir -p ../$@
	mv $@ ../$@/$@
	cp plotter_3d.py ../$@/plotter_3d.py

hill_meissner: $(OBJ) hill_meissner.o
	$(CC) -o $@ $^ $(CFLAGS) $(EXTRAFLAGS) $(LIBS)
	mkdir -p ../$@
	mv $@ ../$@/$@
	cp plotter_3d.py ../$@/plotter_3d.py

hill_meissner_damped1: $(OBJ) hill_meissner_damped1.o
	$(CC) -o $@ $^ $(CFLAGS) $(EXTRAFLAGS) $(LIBS)
	mkdir -p ../$@
	mv $@ ../$@/$@
	cp plotter_3d.py ../$@/plotter_3d.py

hill_meissner_damped2: $(OBJ) hill_meissner_damped2.o
	$(CC) -o $@ $^ $(CFLAGS) $(EXTRAFLAGS) $(LIBS)
	mkdir -p ../$@
	mv $@ ../$@/$@
	cp plotter_3d.py ../$@/plotter_3d.py

population_dynamics: $(OBJ) population_dynamics.o
	$(CC) -o $@ $^ $(CFLAGS) $(EXTRAFLAGS) $(LIBS)
	mkdir -p ../$@
	mv $@ ../$@/$@
	cp plotter_2d.py ../$@/plotter_2d.py

mathieu: mathieu_undamped mathieu_damped_k_1 mathieu_damped_k_10 mathieu_damped_k_0_1 mathieu_damped_k_0_5

meissner: hill_meissner hill_meissner_damped1 hill_meissner_damped2

all: mathieu meissner population_dynamics_1

.PHONY: clean

clean:
	rm *.o
	rm -f $(ODIR)/*.o *~ core $(IDIR)/*~ 

nest:
	set OMP_DYNAMIC=true
	set OMP_NESTED=true