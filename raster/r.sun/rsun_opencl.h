#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include <grass/gis.h>
#include <grass/gprojects.h>
#include <grass/glocale.h>

#ifdef __APPLE__
#include <OpenCL/OpenCL.h>
#else
#include <CL/opencl.h>
#endif

struct OCLCalc {
	cl_command_queue queue;
	cl_context context;
    cl_device_id dev;
    
    cl_kernel calcKern;
	size_t calcGroupSize;
    
    cl_kernel consKern;
	size_t consGroupSize;
};

struct OCLConstants {
	double invstepx;
	double invstepy;
	double xmin;
	double ymin;
	double xmax;
	double ymax;
	double civilTime;
	double tim;
	double step;
	double horizonStep;
	double singleLinke;
	double singleAlbedo;
	double singleSlope;
	double singleAspect;
	double cbh;
	double cdh;
	double dist;
	double TOLER;
	double offsetx;
	double offsety;
	double declination;
	int n;
	int m;
	int saveMemory;
	int day;
	int ttime;
	int numPartitions;
	int proj_eq_ll;
	int someRadiation;
	int numRows;
	int ll_correction;
	int aspin;
	int slopein;
	int linkein;
	int albedo;
	int latin;
	int longin;
	int coefbh;
	int coefdh;
	int incidout;
	int beam_rad;
	int insol_time;
	int diff_rad;
	int refl_rad;
	int glob_rad;
    double degreeInMeters;
    double zmax;
    
    float linke_max, linke_min;
    float albedo_max, albedo_min;
    float lat_max, lat_min;
    float lon_max, lon_min;
    float sunrise_min, sunrise_max;
    float sunset_min, sunset_max;
    float beam_max, beam_min;
    float insol_max, insol_min;
    float diff_max, diff_min;
    float refl_max, refl_min;
    float globrad_max, globrad_min;
};

struct OCLCalc *make_environ_cl(struct OCLConstants *oclConst,
                                struct SolarRadVar *sunRadVar,
                                struct SunGeometryConstDay *sunGeom,
                                struct GridGeometry *gridGeom,
                                int sugDev,
                                cl_int *clErr );

cl_int free_environ_cl(struct OCLCalc *oclCalc);

cl_int calculate_core_cl(unsigned int partOff,
                         struct OCLCalc *oclCalc,
                         struct OCLConstants *oclConst,
                         struct GridGeometry *gridGeom,
                         unsigned char *horizonarray,
                         
                         float **z, float **o, float **s, float **li, float **a,
                         float **latitudeArray, float **longitudeArray,
                         float **cbhr, float **cdhr,
                         float **lumcl, float **beam, float **insol,
                         float **diff, float **refl, float **globrad );
