MODULE_TOPDIR = ../..

PGM = r.sun

LIBES = $(GPROJLIB) $(RASTERLIB) $(GISLIB) $(MATHLIB) $(PROJLIB) -lOpenCL -Wall -W -g -p -O2
DEPENDENCIES = $(GPROJDEP) $(RASTERDEP) $(GISDEP)
EXTRA_INC = $(PROJINC) -Wall -O2 -W -g -p

include $(MODULE_TOPDIR)/include/Make/Module.make

default: cmd
