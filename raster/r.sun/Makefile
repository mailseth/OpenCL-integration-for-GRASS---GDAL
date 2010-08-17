MODULE_TOPDIR = ../..

PGM = r.sun

LIBES = $(GPROJLIB) $(RASTERLIB) $(GISLIB) $(MATHLIB) $(PROJLIB) -framework OpenCL -Wall -Werror -g -p -O2
DEPENDENCIES = $(GPROJDEP) $(RASTERDEP) $(GISDEP)
EXTRA_INC = $(PROJINC) -framework OpenCL -Wall -O2 -Werror -g -p

include $(MODULE_TOPDIR)/include/Make/Module.make

default: cmd
