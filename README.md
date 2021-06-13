# FloquetToolkit
This is an attempt to create a C library to perform Floquet Analysis on a general system.

## How to Install
To compile this, you would require GSL. In Debian-based distributions, it can be obtained by using 
```bash
sudo apt install libgsl-dev
```

Just go to src and then run 
```bash
make all
```
to compile all the programs.
To compile just for specific cases, use
```bash
make mathieu
```
to get the programs corresponding to mathieu equation.

```bash
make meissner
```
to get the programs corresponding to hill-meissner equation.

```bash
make population_dynamics
```
to get the programs corresponding to the results on population dynamics.

## How to Run
To run this, just go to the respective directories and run the compiled binaries.
They will generate some data files. To plot them and get the plots, run 
```bash
python3 plotter3d.py
```
```bash
python3 plotter2d.py
```
Depending on the type of file that is available. plotter3d.py generated plots when 2 parameters are involved and plotter2d generates plots for when only 1 parameter is involved.