#include <OpenCL/OpenCL.h>

#ifdef __cplusplus /* If this is a C++ compiler, use C linkage */
extern "C" {
#endif

typedef enum {
    OCL_Bilinear=10,
    OCL_Cubic=11,
    OCL_CubicSpline=12,
    OCL_Lanczos=13
} OCLResampAlg;
    
struct oclWarper {
    cl_command_queue queue;
    cl_context context;
    cl_device_id dev;
    cl_kernel kern1;
    cl_kernel kern4;
    
    int srcWidth;
    int srcHeight;
    int dstWidth;
    int dstHeight;
    
    int useUnifiedSrcDensity;
    int useUnifiedSrcValid;
    int useDstDensity;
    int useDstValid;
    
    int numBands;
    int numImages;
    OCLResampAlg resampAlg;
    
    cl_channel_type imageFormat;
    cl_mem *realWorkCL;
    union {
        void **v;
        char **c;
        unsigned char **uc;
        short **s;
        unsigned short **us;
        float **f;
    } realWork;
    
    cl_mem *imagWorkCL;
    union {
        void **v;
        char **c;
        unsigned char **uc;
        short **s;
        unsigned short **us;
        float **f;
    } imagWork;
    
    cl_mem *dstRealWorkCL;
    union {
        void **v;
        char **c;
        unsigned char **uc;
        short **s;
        unsigned short **us;
        float **f;
    } dstRealWork;
    
    cl_mem *dstImagWorkCL;
    union {
        void **v;
        char **c;
        unsigned char **uc;
        short **s;
        unsigned short **us;
        float **f;
    } dstImagWork;
    
    unsigned int imgChSize1;
    cl_channel_order imgChOrder1;
    unsigned int imgChSize4;
    cl_channel_order imgChOrder4;
	char    useVec;
    
    cl_mem useBandSrcValidCL;
    char *useBandSrcValid;
    
    cl_mem nBandSrcValidCL;
    float *nBandSrcValid;
    
    cl_mem xyWorkCL;
    float *xyWork;
    
    int xyWidth;
    int xyHeight;
    int coordMult;
    
    unsigned int xyChSize;
    cl_channel_order xyChOrder;
    
    cl_mem fDstNoDataRealCL;
    float *fDstNoDataReal;
};

struct oclWarper* GDALWarpKernelOpenCL_createEnv(int srcWidth, int srcHeight,
                                                 int dstWidth, int dstHeight,
                                                 cl_channel_type imageFormat,
                                                 int numBands, int coordMult,
                                                 int useImag, int useBandSrcValid,
                                                 float *fDstDensity,
                                                 double *dfDstNoDataReal,
                                                 OCLResampAlg resampAlg, cl_int *envErr);

cl_int GDALWarpKernelOpenCL_setSrcValid(struct oclWarper *warper,
                                        int *bandSrcValid, int bandNum);

cl_int GDALWarpKernelOpenCL_setSrcImg(struct oclWarper *warper, void *imgData,
                                      int bandNum);

cl_int GDALWarpKernelOpenCL_setDstImg(struct oclWarper *warper, void *imgData,
                                      int bandNum);

cl_int GDALWarpKernelOpenCL_setCoordRow(struct oclWarper *warper,
                                        double *rowSrcX, double *rowSrcY,
                                        double srcXOff, double srcYOff,
                                        int *success, int rowNum);

cl_int GDALWarpKernelOpenCL_runResamp(struct oclWarper *warper,
                                      float *unifiedSrcDensity,
                                      unsigned int *unifiedSrcValid,
                                      float *dstDensity,
                                      unsigned int *dstValid,
                                      double dfXScale, double dfYScale,
                                      double dfXFilter, double dfYFilter,
                                      int nXRadius, int nYRadius,
                                      int nFiltInitX, int nFiltInitY);

cl_int GDALWarpKernelOpenCL_getRow(struct oclWarper *warper,
                                   void **rowReal, void **rowImag,
                                   int rowNum, int bandNum);

cl_int GDALWarpKernelOpenCL_deleteEnv(struct oclWarper *warper);

#ifdef __cplusplus /* If this is a C++ compiler, end C linkage */
}
#endif
