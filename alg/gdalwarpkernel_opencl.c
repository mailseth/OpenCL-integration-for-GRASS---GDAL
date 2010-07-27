#include <assert.h>
#include <stdio.h>
#include "cpl_string.h"
#include "gdalwarpkernel_opencl.h"

#define handleErr(err) if((err) != CL_SUCCESS) { \
    printf("Error at file %s line %d; Err val: %d\n", __FILE__, __LINE__, err); \
    printCLErr(err); \
    while(1){}\
    return err; \
}

#define handleErrRetNULL(err) if((err) != CL_SUCCESS) { \
    (*clErr) = err; \
    printf("Error at file %s line %d; Err val: %d\n", __FILE__, __LINE__, err); \
    printCLErr(err); \
    return NULL; \
}

#define freeCLMem(clMem, fallBackMem) { \
    if ((clMem) != NULL) { \
        handleErr(err = clReleaseMemObject(clMem)); \
        clMem = NULL; \
        fallBackMem = NULL; \
    } else if ((fallBackMem) != NULL) { \
        CPLFree(fallBackMem); \
        fallBackMem = NULL; \
    } \
}



#ifndef NDEBUG
int device_stats(cl_device_id device_id)
{
	
	int err;
	size_t returned_size;
    int i;
	
	// Report the device vendor and device name
    // 
    cl_char vendor_name[1024] = {0};
    cl_char device_name[1024] = {0};
	cl_char device_profile[1024] = {0};
	cl_char device_extensions[1024] = {0};
    
	cl_device_local_mem_type local_mem_type;
    cl_ulong global_mem_size, global_mem_cache_size;
	cl_ulong max_mem_alloc_size;
	cl_uint clock_frequency, vector_width, max_compute_units;
	size_t max_work_item_dims,max_work_group_size, max_work_item_sizes[3];
	
	cl_uint vector_types[] = {CL_DEVICE_PREFERRED_VECTOR_WIDTH_CHAR, CL_DEVICE_PREFERRED_VECTOR_WIDTH_SHORT, CL_DEVICE_PREFERRED_VECTOR_WIDTH_INT,CL_DEVICE_PREFERRED_VECTOR_WIDTH_LONG,CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT,CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE}; 
	char *vector_type_names[] = {"char","short","int","long","float","double"};
	
	err = clGetDeviceInfo(device_id, CL_DEVICE_VENDOR, sizeof(vendor_name), vendor_name, &returned_size);
    err|= clGetDeviceInfo(device_id, CL_DEVICE_NAME, sizeof(device_name), device_name, &returned_size);
	err|= clGetDeviceInfo(device_id, CL_DEVICE_PROFILE, sizeof(device_profile), device_profile, &returned_size);
	err|= clGetDeviceInfo(device_id, CL_DEVICE_EXTENSIONS, sizeof(device_extensions), device_extensions, &returned_size);
	err|= clGetDeviceInfo(device_id, CL_DEVICE_LOCAL_MEM_TYPE, sizeof(local_mem_type), &local_mem_type, &returned_size);
	err|= clGetDeviceInfo(device_id, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(global_mem_size), &global_mem_size, &returned_size);
	err|= clGetDeviceInfo(device_id, CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE, sizeof(global_mem_cache_size), &global_mem_cache_size, &returned_size);
	err|= clGetDeviceInfo(device_id, CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof(max_mem_alloc_size), &max_mem_alloc_size, &returned_size);
	err|= clGetDeviceInfo(device_id, CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof(clock_frequency), &clock_frequency, &returned_size);
	err|= clGetDeviceInfo(device_id, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(max_work_group_size), &max_work_group_size, &returned_size);
	err|= clGetDeviceInfo(device_id, CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, sizeof(max_work_item_dims), &max_work_item_dims, &returned_size);
	err|= clGetDeviceInfo(device_id, CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof(max_work_item_sizes), max_work_item_sizes, &returned_size);
	err|= clGetDeviceInfo(device_id, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(max_compute_units), &max_compute_units, &returned_size);
	
	printf("Vendor: %s\n", vendor_name);
	printf("Device Name: %s\n", device_name);
	printf("Profile: %s\n", device_profile);
	printf("Supported Extensions: %s\n\n", device_extensions);
	
	printf("Local Mem Type (Local=1, Global=2): %i\n",(int)local_mem_type);
	printf("Global Mem Size (MB): %i\n",(int)global_mem_size/(1024*1024));
	printf("Global Mem Cache Size (Bytes): %i\n",(int)global_mem_cache_size);
	printf("Max Mem Alloc Size (MB): %ld\n",(long int)max_mem_alloc_size/(1024*1024));
	
	printf("Clock Frequency (MHz): %i\n\n",clock_frequency);
	
	for(i=0;i<6;i++){
		err|= clGetDeviceInfo(device_id, vector_types[i], sizeof(clock_frequency), &vector_width, &returned_size);
		printf("Vector type width for: %s = %i\n",vector_type_names[i],vector_width);
	}
	
	printf("\nMax Work Group Size: %lu\n",max_work_group_size);
	printf("Max Work Item Dims: %lu\n",max_work_item_dims);
	for(i=0;i<max_work_item_dims;i++) 
		printf("Max Work Items in Dim %lu: %lu\n",(long unsigned)(i+1),(long unsigned)max_work_item_sizes[i]);
	
	printf("Max Compute Units: %i\n",max_compute_units);
	printf("\n");
	
	return CL_SUCCESS;
}

void printImgFmt(cl_channel_order order, cl_channel_type type)
{
    switch (order)
    {
        case CL_R:
            printf("CL_R ");
            break;
        case CL_A:
            printf("CL_A ");
            break;
        case CL_RG:
            printf("CL_RG ");
            break;
        case CL_RA:
            printf("CL_RA ");
            break;
        case CL_RGB:
            printf("CL_RGB ");
            break;
        case CL_RGBA:
            printf("CL_RGBA ");
            break;
        case CL_BGRA:
            printf("CL_BGRA ");
            break;
        case CL_ARGB:
            printf("CL_ARGB ");
            break;
        case CL_INTENSITY:
            printf("CL_INTENSITY ");
            break;
        case CL_LUMINANCE:
            printf("CL_LUMINANCE ");
            break;
    }
    
    switch (type)
    {
        case CL_SNORM_INT8:
            printf("CL_SNORM_INT8");
            break;
        case CL_SNORM_INT16:
            printf("CL_SNORM_INT16");
            break;
        case CL_UNORM_INT8:
            printf("CL_UNORM_INT8");
            break;
        case CL_UNORM_INT16:
            printf("CL_UNORM_INT16");
            break;
        case CL_UNORM_SHORT_565:
            printf("CL_UNORM_SHORT_565");
            break;
        case CL_UNORM_SHORT_555:
            printf("CL_UNORM_SHORT_555");
            break;
        case CL_UNORM_INT_101010:
            printf("CL_UNORM_INT_101010");
            break;
        case CL_SIGNED_INT8:
            printf("CL_SIGNED_INT8");
            break;
        case CL_SIGNED_INT16:
            printf("CL_SIGNED_INT16");
            break;
        case CL_SIGNED_INT32:
            printf("CL_SIGNED_INT32");
            break;
        case CL_UNSIGNED_INT8:
            printf("CL_UNSIGNED_INT8");
            break;
        case CL_UNSIGNED_INT16:
            printf("CL_UNSIGNED_INT16");
            break;
        case CL_UNSIGNED_INT32:
            printf("CL_UNSIGNED_INT32");
            break;
        case CL_HALF_FLOAT:
            printf("CL_HALF_FLOAT");
            break;
        case CL_FLOAT:
            printf("CL_FLOAT");
            break;
    }
    
    printf("\n");
}
#endif

void printCLErr(cl_int err)
{
    switch (err)
    {
        case CL_SUCCESS:
            printf("CL_SUCCESS\n");
            break;
        case CL_DEVICE_NOT_FOUND:
            printf("CL_DEVICE_NOT_FOUND\n");
            break;
        case CL_DEVICE_NOT_AVAILABLE:
            printf("CL_DEVICE_NOT_AVAILABLE\n");
            break;
        case CL_COMPILER_NOT_AVAILABLE:
            printf("CL_COMPILER_NOT_AVAILABLE\n");
            break;
        case CL_MEM_OBJECT_ALLOCATION_FAILURE:
            printf("CL_MEM_OBJECT_ALLOCATION_FAILURE\n");
            break;
        case CL_OUT_OF_RESOURCES:
            printf("CL_OUT_OF_RESOURCES\n");
            break;
        case CL_OUT_OF_HOST_MEMORY:
            printf("CL_OUT_OF_HOST_MEMORY\n");
            break;
        case CL_PROFILING_INFO_NOT_AVAILABLE:
            printf("CL_PROFILING_INFO_NOT_AVAILABLE\n");
            break;
        case CL_MEM_COPY_OVERLAP:
            printf("CL_MEM_COPY_OVERLAP\n");
            break;
        case CL_IMAGE_FORMAT_MISMATCH:
            printf("CL_IMAGE_FORMAT_MISMATCH\n");
            break;
        case CL_IMAGE_FORMAT_NOT_SUPPORTED:
            printf("CL_IMAGE_FORMAT_NOT_SUPPORTED\n");
            break;
        case CL_BUILD_PROGRAM_FAILURE:
            printf("CL_BUILD_PROGRAM_FAILURE\n");
            break;
        case CL_MAP_FAILURE:
            printf("CL_MAP_FAILURE\n");
            break;
        case CL_INVALID_VALUE:
            printf("CL_INVALID_VALUE\n");
            break;
        case CL_INVALID_DEVICE_TYPE:
            printf("CL_INVALID_DEVICE_TYPE\n");
            break;
        case CL_INVALID_PLATFORM:
            printf("CL_INVALID_PLATFORM\n");
            break;
        case CL_INVALID_DEVICE:
            printf("CL_INVALID_DEVICE\n");
            break;
        case CL_INVALID_CONTEXT:
            printf("CL_INVALID_CONTEXT\n");
            break;
        case CL_INVALID_QUEUE_PROPERTIES:
            printf("CL_INVALID_QUEUE_PROPERTIES\n");
            break;
        case CL_INVALID_COMMAND_QUEUE:
            printf("CL_INVALID_COMMAND_QUEUE\n");
            break;
        case CL_INVALID_HOST_PTR:
            printf("CL_INVALID_HOST_PTR\n");
            break;
        case CL_INVALID_MEM_OBJECT:
            printf("CL_INVALID_MEM_OBJECT\n");
            break;
        case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR:
            printf("CL_INVALID_IMAGE_FORMAT_DESCRIPTOR\n");
            break;
        case CL_INVALID_IMAGE_SIZE:
            printf("CL_INVALID_IMAGE_SIZE\n");
            break;
        case CL_INVALID_SAMPLER:
            printf("CL_INVALID_SAMPLER\n");
            break;
        case CL_INVALID_BINARY:
            printf("CL_INVALID_BINARY\n");
            break;
        case CL_INVALID_BUILD_OPTIONS:
            printf("CL_INVALID_BUILD_OPTIONS\n");
            break;
        case CL_INVALID_PROGRAM:
            printf("CL_INVALID_PROGRAM\n");
            break;
        case CL_INVALID_PROGRAM_EXECUTABLE:
            printf("CL_INVALID_PROGRAM_EXECUTABLE\n");
            break;
        case CL_INVALID_KERNEL_NAME:
            printf("CL_INVALID_KERNEL_NAME\n");
            break;
        case CL_INVALID_KERNEL_DEFINITION:
            printf("CL_INVALID_KERNEL_DEFINITION\n");
            break;
        case CL_INVALID_KERNEL:
            printf("CL_INVALID_KERNEL\n");
            break;
        case CL_INVALID_ARG_INDEX:
            printf("CL_INVALID_ARG_INDEX\n");
            break;
        case CL_INVALID_ARG_VALUE:
            printf("CL_INVALID_ARG_VALUE\n");
            break;
        case CL_INVALID_ARG_SIZE:
            printf("CL_INVALID_ARG_SIZE\n");
            break;
        case CL_INVALID_KERNEL_ARGS:
            printf("CL_INVALID_KERNEL_ARGS\n");
            break;
        case CL_INVALID_WORK_DIMENSION:
            printf("CL_INVALID_WORK_DIMENSION\n");
            break;
        case CL_INVALID_WORK_GROUP_SIZE:
            printf("CL_INVALID_WORK_GROUP_SIZE\n");
            break;
        case CL_INVALID_WORK_ITEM_SIZE:
            printf("CL_INVALID_WORK_ITEM_SIZE\n");
            break;
        case CL_INVALID_GLOBAL_OFFSET:
            printf("CL_INVALID_GLOBAL_OFFSET\n");
            break;
        case CL_INVALID_EVENT_WAIT_LIST:
            printf("CL_INVALID_EVENT_WAIT_LIST\n");
            break;
        case CL_INVALID_EVENT:
            printf("CL_INVALID_EVENT\n");
            break;
        case CL_INVALID_OPERATION:
            printf("CL_INVALID_OPERATION\n");
            break;
        case CL_INVALID_GL_OBJECT:
            printf("CL_INVALID_GL_OBJECT\n");
            break;
        case CL_INVALID_BUFFER_SIZE:
            printf("CL_INVALID_BUFFER_SIZE\n");
            break;
        case CL_INVALID_MIP_LEVEL:
            printf("CL_INVALID_MIP_LEVEL\n");
            break;
        case CL_INVALID_GLOBAL_WORK_SIZE:
            printf("CL_INVALID_GLOBAL_WORK_SIZE\n");
            break;
    }
}

/*
 Finds an appropirate OpenCL device. If the user specifies a preference, the
 code for it should be here (but not currently supported). For debugging, it's
 always easier to use CL_DEVICE_TYPE_CPU because then printf() can be called
 from the kernel. If debugging is on, we can print the name and stats about the
 device we're using.
 */
cl_device_id get_device()
{
	cl_int err = 0;
	cl_device_id device = NULL;
#ifndef NDEBUG
    size_t returned_size = 0;
    cl_char vendor_name[1024] = {0};
    cl_char device_name[1024] = {0};
#endif
    
    // Find the GPU CL device, this is what we really want
    // If there is no GPU device is CL capable, fall back to CPU
    err = clGetDeviceIDs(NULL, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
//    err = clGetDeviceIDs(NULL, CL_DEVICE_TYPE_CPU, 1, &device, NULL);
    if (err != CL_SUCCESS)
    {
        // Find the CPU CL device, as a fallback
        err = clGetDeviceIDs(NULL, CL_DEVICE_TYPE_CPU, 1, &device, NULL);
        assert(err == CL_SUCCESS);
    }
    assert(device);
    
#ifndef NDEBUG
    // Get some information about the returned device
    err = clGetDeviceInfo(device, CL_DEVICE_VENDOR, sizeof(vendor_name), 
                          vendor_name, &returned_size);
    err |= clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(device_name), 
                           device_name, &returned_size);
    assert(err == CL_SUCCESS);
    printf("Connecting to %s %s...\n", vendor_name, device_name);
	
//    device_stats(device);
#endif
    
    return device;
}

/*
 Given that not all OpenCL devices support the same image formats, we need to
 make do with what we have. This leads to wasted space, but as OpenCL matures
 I hope it'll get better.
 */
cl_int set_supported_formats(struct oclWarper *warper,
                             cl_channel_order minOrderSize,
                             cl_channel_order *chosenOrder,
                             unsigned int *chosenSize,
                             cl_channel_type dataType )
{
    cl_image_format *fmtBuf = (cl_image_format *)calloc(256, sizeof(cl_image_format));
    cl_uint numRet;
    int i;
    int extraSpace = 9999;
    cl_int err = CL_SUCCESS;
    
    //Find what we *can* handle
    handleErr(err = clGetSupportedImageFormats(warper->context,
                                               CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                               CL_MEM_OBJECT_IMAGE2D,
                                               256, fmtBuf, &numRet));
    for (i = 0; i < numRet; ++i) {
        int thisOrderSize = 0;
        switch (fmtBuf[i].image_channel_order)
        {
			//Only support formats which use the channels in order (x,y,z,w)
            case CL_R:
            case CL_INTENSITY:
            case CL_LUMINANCE:
                thisOrderSize = 1;
                break;
            case CL_RG:
                thisOrderSize = 2;
                break;
            case CL_RGB:
                thisOrderSize = 3;
                break;
            case CL_RGBA:
                thisOrderSize = 4;
                break;
        }
        
        //Choose an order with the least wasted space
        if (fmtBuf[i].image_channel_data_type == dataType &&
            minOrderSize <= thisOrderSize &&
            extraSpace > thisOrderSize - minOrderSize ) {
			
			//Set the vector size, order, & remember wasted space
            (*chosenSize) = thisOrderSize;
            (*chosenOrder) = fmtBuf[i].image_channel_order;
            extraSpace = thisOrderSize - minOrderSize;
        }
    }
    
    free(fmtBuf);
    return CL_SUCCESS;
}

/*
 Allocate some pinned memory that we can use as an intermediate buffer. We're
 using the pinned memory to assemble the data before transferring it to the
 device. The reason we're using pinned RAM is because the transfer speed from
 host RAM to device RAM is faster than non-pinned. The disadvantage is that
 pinned RAM is a scarce OS resource. I'm making the assumption that the user
 has as much pinned host RAM available as total device RAM because device RAM
 tends to be similarly scarce. However, if the pinned memory fails we fall back
 to using a regular memory allocation.
 
 Returns CL_SUCCESS on success and other CL_* errors when something goes wrong.
 */
cl_int alloc_pinned_mem(struct oclWarper *warper, int imgNum, size_t dataSz,
                        void **wrkPtr, cl_mem *wrkCL)
{
	cl_int err = CL_SUCCESS;
    wrkCL[imgNum] = clCreateBuffer(warper->context,
                                   CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR,
                                   dataSz, NULL, &err);

    if (err == CL_SUCCESS) {
        wrkPtr[imgNum] = (void *)clEnqueueMapBuffer(warper->queue, wrkCL[imgNum],
                                                    CL_FALSE, CL_MAP_WRITE,
                                                    0, dataSz, 0, NULL, NULL, &err);
        handleErr(err);
    } else {
        wrkCL[imgNum] = NULL;
#ifndef NDEBUG
        printf("Using fallback memory!\n");
#endif
        //Fallback to regular allocation
        wrkPtr[imgNum] = (void *)CPLMalloc(dataSz);
        
        if (wrkPtr[imgNum] == NULL)
            handleErr(err = CL_OUT_OF_HOST_MEMORY);
    }
    
    return CL_SUCCESS;
}

/*
 Allocates the working host memory for all bands of the image in the warper
 structure. This includes both the source image buffers and the destination
 buffers. This memory is located on the host, so we can assemble the image.
 Reasons for buffering it like this include reading each row from disk and
 de-interleaving bands and parts of bands. Then they can be copied to the device
 as a single operation fit for use as an OpenCL memory object.
 
 Returns CL_SUCCESS on success and other CL_* errors when something goes wrong.
 */
cl_int alloc_working_arr(struct oclWarper *warper,
                         size_t ptrSz, size_t dataSz, size_t *fmtSz)
{
	cl_int err = CL_SUCCESS;
    int i;
    size_t srcDataSz, dstDataSz;
    
    //Find the best channel order for this format
    err = set_supported_formats(warper, 1,
                                &(warper->imgChOrder), &(warper->imgChSize),
                                warper->imageFormat);
    handleErr(err);
    
    //Calc the sizes we need
    srcDataSz = dataSz * warper->srcWidth * warper->srcHeight * warper->imgChSize;
    dstDataSz = dataSz * warper->dstWidth * warper->dstHeight * warper->imgChSize;
    
    //Alloc space for pointers to the main image data
    warper->realWork.v = (void **)CPLMalloc(ptrSz*warper->numImages);
    warper->dstRealWork.v = (void **)CPLMalloc(ptrSz*warper->numImages);
    if (warper->realWork.v == NULL || warper->dstRealWork.v == NULL)
        handleErr(err = CL_OUT_OF_HOST_MEMORY);
    
    if (warper->imagWorkCL != NULL) {
        //Alloc space for pointers to the extra channel, if it exists
        warper->imagWork.v = (void **)CPLMalloc(ptrSz*warper->numImages);
        warper->dstImagWork.v = (void **)CPLMalloc(ptrSz*warper->numImages);
        if (warper->imagWork.v == NULL || warper->dstImagWork.v == NULL)
            handleErr(err = CL_OUT_OF_HOST_MEMORY);
    } else {
        warper->imagWork.v = NULL;
        warper->dstImagWork.v = NULL;
    }
    
    //Allocate pinned memory for each band's image
    for (i = 0; i < warper->numImages; ++i) {
        handleErr(err = alloc_pinned_mem(warper, i, srcDataSz,
                                         warper->realWork.v,
                                         warper->realWorkCL));
        
        handleErr(err = alloc_pinned_mem(warper, i, dstDataSz,
                                         warper->dstRealWork.v,
                                         warper->dstRealWorkCL));
    }
    
    if (warper->imagWorkCL != NULL) {
        //Allocate pinned memory for each band's extra channel, if exists
        for (i = 0; i < warper->numImages; ++i) {
            handleErr(err = alloc_pinned_mem(warper, i, srcDataSz,
                                             warper->imagWork.v,
                                             warper->imagWorkCL));
            
            handleErr(err = alloc_pinned_mem(warper, i, dstDataSz,
                                             warper->dstImagWork.v,
                                             warper->dstImagWorkCL));
        }
    }
    
    return CL_SUCCESS;
}

/*
 Assemble and create the kernel. For optimization, portabilaty, and
 implimentation limitation reasons, the program is actually assembled from
 several strings, then compiled with as many invariants as possible defined by
 the preprocessor. There is also quite a bit of error-catching code in here
 because the kernel is where many bugs show up.
 
 Returns CL_SUCCESS on success and other CL_* errors in the error buffer when
 something goes wrong.
 */
cl_kernel get_kernel(struct oclWarper *warper,
                     double dfXScale, double dfYScale, double dfXFilter, double dfYFilter,
                     int nXRadius, int nYRadius, int nFiltInitX, int nFiltInitY,
                     cl_int *clErr )
{
	cl_program program;
    cl_kernel kernel;
	cl_int err = CL_SUCCESS;
    char *buffer = (char *)calloc(128000, sizeof(char));
    char *progBuf = (char *)calloc(128000, sizeof(char));
    float dstMinVal, dstMaxVal;
    
    const char *outType;
    const char *useVec = "";
    const char *kernGenFuncs =
// ********************* General Funcs ********************
"void clampToDst(float fReal,\n"
                "__global outType *dstPtr,\n"
                "unsigned int iDstOffset,\n"
                "__constant float *fDstNoDataReal,\n"
                "int bandNum)\n"
"{\n"
	"fReal *= dstMaxVal;\n"
    
    "if (fReal < dstMinVal)\n"
        "dstPtr[iDstOffset] = (outType)dstMinVal;\n"
    "else if (fReal > dstMaxVal)\n"
        "dstPtr[iDstOffset] = (outType)dstMaxVal;\n"
    "else\n"
        "dstPtr[iDstOffset] = (dstMinVal < 0) ? (outType)floor(fReal + 0.5f) : (outType)(fReal + 0.5f);\n"
    
    "if (useDstNoDataReal && bandNum >= 0 &&\n"
        "fDstNoDataReal[bandNum] == dstPtr[iDstOffset])\n"
    "{\n"
        "if (dstPtr[iDstOffset] == dstMinVal)\n"
            "dstPtr[iDstOffset] = dstMinVal + 1;\n"
        "else\n"
            "dstPtr[iDstOffset] --;\n"
    "}\n"
"}\n"

"void setPixel(__global outType *dstReal,\n"
              "__global outType *dstImag,\n"
              "__global float *dstDensity,\n"
              "__global int *nDstValid,\n"
              "__constant float *fDstNoDataReal,\n"
              "const int bandNum,\n"
              "vecf fDensity, vecf fReal, vecf fImag)\n"
"{\n"
    "unsigned int iDstOffset = get_global_id(1)*iDstWidth + get_global_id(0);\n"
    
"#ifdef USE_VEC\n"
    "if (fDensity.x < 0.9999f || fDensity.y < 0.9999f ||\n"
        "fDensity.z < 0.9999f || fDensity.w < 0.9999f )\n"
"#else\n"
    "if (fDensity < 0.9999f)\n"
"#endif\n"
    "{\n"
        "vecf fDstReal, fDstImag, fDstDensity;\n"
        "fDstImag = 0.0f;\n"
        
"#ifdef USE_VEC\n"
        "fDstReal = vload4(iDstOffset*4, dstReal);\n"
        "if (useImag)\n"
            "fDstImag = vload4(iDstOffset*4, dstImag);\n"
"#else\n"
        "fDstReal = dstReal[iDstOffset];\n"
        "if (useImag)\n"
            "fDstImag = dstImag[iDstOffset];\n"
"#endif\n"
        
        "if (useDstDensity)\n"
            "fDstDensity = dstDensity[iDstOffset];\n"
        "else if (useDstValid &&\n"
                 "!((nDstValid[iDstOffset>>5] & (0x01 << (iDstOffset & 0x1f))) ))\n"
            "fDstDensity = 0.0f;\n"
        "else\n"
            "fDstDensity = 1.0f;\n"
        
        "float fDstInfluence = (1.0f - fDensity) * fDstDensity;\n"
        
        // Density should be checked for <= 0.0 & handled by the calling function
        "fReal = (fReal * fDensity + fDstReal * fDstInfluence) / (fDensity + fDstInfluence);\n"
        "if (useImag)\n"
            "fImag = (fImag * fDensity + fDstImag * fDstInfluence) / (fDensity + fDstInfluence);\n"
    "}\n"
    
"#ifdef USE_VEC\n"
    "clampToDst(fReal.x, dstReal, iDstOffset, fDstNoDataReal, bandNum);\n"
    "if (useImag)\n"
        "clampToDst(fImag.x, dstImag, iDstOffset, fDstNoDataReal, -1);\n"
    "clampToDst(fReal.y, dstReal, iDstOffset+iDstHeight*iDstWidth, fDstNoDataReal, bandNum);\n"
    "if (useImag)\n"
        "clampToDst(fImag.y, dstImag, iDstOffset+iDstHeight*iDstWidth, fDstNoDataReal, -1);\n"
    "clampToDst(fReal.z, dstReal, iDstOffset+iDstHeight*iDstWidth*2, fDstNoDataReal, bandNum);\n"
    "if (useImag)\n"
        "clampToDst(fImag.z, dstImag, iDstOffset+iDstHeight*iDstWidth*2, fDstNoDataReal, -1);\n"
    "clampToDst(fReal.w, dstReal, iDstOffset+iDstHeight*iDstWidth*3, fDstNoDataReal, bandNum);\n"
    "if (useImag)\n"
        "clampToDst(fImag.w, dstImag, iDstOffset+iDstHeight*iDstWidth*3, fDstNoDataReal, -1);\n"
"#else\n"
    "clampToDst(fReal, dstReal, iDstOffset, fDstNoDataReal, bandNum);\n"
    "if (useImag)\n"
        "clampToDst(fImag, dstImag, iDstOffset, fDstNoDataReal, -1);\n"
"#endif\n"
"}\n"

"int getPixel(__read_only image2d_t srcReal,\n"
             "__read_only image2d_t srcImag,\n"
             "__global float *fUnifiedSrcDensity,\n"
             "__global int *nUnifiedSrcValid,\n"
             "__constant char *useBandSrcValid,\n"
             "__global int *nBandSrcValid,\n"
             "const int2 iSrc,\n"
             "int bandNum,\n"
             "vecf *fDensity, vecf *fReal, vecf *fImag)\n"
"{\n"
    "int iSrcOffset = 0, iBandValidLen = 0, iSrcOffsetMask = 0;\n"
    "int bHasValid = FALSE;\n"
    
    // Clamp the src offset values if needed
    "if(useUnifiedSrcDensity || useUnifiedSrcValid || useUseBandSrcValid){\n"
        "int iSrcX = iSrc.x;\n"
        "int iSrcY = iSrc.y;\n"
        
        // Needed because the offset isn't clamped in OpenCL hardware
        "if(iSrcX < 0)\n"
            "iSrcX = 0;\n"
        "else if(iSrcX >= iSrcWidth)\n"
            "iSrcX = iSrcWidth - 1;\n"
            
        "if(iSrcY < 0)\n"
            "iSrcY = 0;\n"
        "else if(iSrcY >= iSrcHeight)\n"
            "iSrcY = iSrcHeight - 1;\n"
            
        "iSrcOffset = iSrcY*iSrcWidth + iSrcX;\n"
        "iBandValidLen = 1 + ((iSrcWidth*iSrcHeight)>>5);\n"
        "iSrcOffsetMask = (0x01 << (iSrcOffset & 0x1f));\n"
    "}\n"
    
    "if (useUnifiedSrcValid &&\n"
        "!((nUnifiedSrcValid[iSrcOffset>>5] & iSrcOffsetMask) ) )\n"
        "return FALSE;\n"
    
"#ifdef USE_VEC\n"
    "if (!useUseBandSrcValid || !useBandSrcValid[bandNum] ||\n"
        "((nBandSrcValid[(iSrcOffset>>5)+iBandValidLen*bandNum    ] & iSrcOffsetMask)) )\n"
        "bHasValid = TRUE;\n"
    
    "if (!useUseBandSrcValid || !useBandSrcValid[bandNum+1] ||\n"
        "((nBandSrcValid[(iSrcOffset>>5)+iBandValidLen*(1+bandNum)] & iSrcOffsetMask)) )\n"
        "bHasValid = TRUE;\n"
    
    "if (!useUseBandSrcValid || !useBandSrcValid[bandNum+2] ||\n"
        "((nBandSrcValid[(iSrcOffset>>5)+iBandValidLen*(2+bandNum)] & iSrcOffsetMask)) )\n"
        "bHasValid = TRUE;\n"
    
    "if (!useUseBandSrcValid || !useBandSrcValid[bandNum+3] ||\n"
        "((nBandSrcValid[(iSrcOffset>>5)+iBandValidLen*(3+bandNum)] & iSrcOffsetMask)) )\n"
        "bHasValid = TRUE;\n"
"#else\n"
    "if (!useUseBandSrcValid || !useBandSrcValid[bandNum] ||\n"
        "((nBandSrcValid[(iSrcOffset>>5)+iBandValidLen*bandNum    ] & iSrcOffsetMask)) )\n"
        "bHasValid = TRUE;\n"
"#endif\n"
    
    "if (!bHasValid)\n"
        "return FALSE;\n"
    
    "const sampler_t samp =  CLK_NORMALIZED_COORDS_FALSE |\n"
                            "CLK_ADDRESS_CLAMP_TO_EDGE |\n"
                            "CLK_FILTER_NEAREST;\n"
    
"#ifdef USE_VEC\n"
    "(*fReal) = read_imagef(srcReal, samp, iSrc);\n"
    "if (useImag)\n"
        "(*fImag) = read_imagef(srcImag, samp, iSrc);\n"
"#else\n"
    "(*fReal) = read_imagef(srcReal, samp, iSrc).x;\n"
    "if (useImag)\n"
        "(*fImag) = read_imagef(srcImag, samp, iSrc).x;\n"
"#endif\n"
    
    "if (useUnifiedSrcDensity) {\n"
        "(*fDensity) = fUnifiedSrcDensity[iSrcOffset];\n"
    "} else {\n"
        "(*fDensity) = 1.0f;\n"
        "return TRUE;\n"
    "}\n"
    
"#ifdef USE_VEC\n"
    "return  (*fDensity).x > 0.0f || (*fDensity).y > 0.0f ||\n"
            "(*fDensity).z > 0.0f || (*fDensity).w > 0.0f;\n"
"#else\n"
    "return (*fDensity) > 0.0f;\n"
"#endif\n"
"}\n"

"int isValid(__global float *fUnifiedSrcDensity,\n"
            "__global int *nUnifiedSrcValid,\n"
            "float2 fSrcCoords )\n"
"{\n"
    "if (fSrcCoords.x < 0.0f || fSrcCoords.y < 0.0f)\n"
        "return FALSE;\n"
    
    "int iSrcX = (int) (fSrcCoords.x - 0.5f);\n"
    "int iSrcY = (int) (fSrcCoords.y - 0.5f);\n"
    
    "if( iSrcX < 0 || iSrcX >= iSrcWidth || iSrcY < 0 || iSrcY >= iSrcHeight )\n"
        "return FALSE;\n"
    
    "int iSrcOffset = iSrcX + iSrcY * iSrcWidth;\n"
    
    "if (useUnifiedSrcDensity && fUnifiedSrcDensity[iSrcOffset] < 0.00001f)\n"
        "return FALSE;\n"
    
    "if (useUnifiedSrcValid &&\n"
        "!(nUnifiedSrcValid[iSrcOffset>>5] & (0x01 << (iSrcOffset & 0x1f))) )\n"
        "return FALSE;\n"
    
    "return TRUE;\n"
"}\n"

"float2 getSrcCoords(__read_only image2d_t srcCoords,\n"
                    "float2 fDst)\n"
"{\n"
    "float4  fSrcCoords = read_imagef(srcCoords,\n"
                                     "CLK_NORMALIZED_COORDS_TRUE |\n"
                                     "CLK_ADDRESS_CLAMP_TO_EDGE |\n"
                                     "CLK_FILTER_LINEAR,\n"
                                     "fDst);\n"
    
    "return (float2)(fSrcCoords.x, fSrcCoords.y);\n"
"}\n";

    const char *kernCubic =
// "************************ Cubic ************************\n"
"vecf cubicConvolution(float dist1, float dist2, float dist3,\n"
                       "vecf f0, vecf f1, vecf f2, vecf f3)\n"
"{\n"
    "return   (  -f0 +    f1  - f2 + f3) * dist3\n"
           "+ (2.0f*(f0 - f1) + f2 - f3) * dist2\n"
           "+ (  -f0          + f2     ) * dist1\n"
           "+             f1;\n"
"}\n"

// ************************ Cubic ************************
"__kernel void resamp(__read_only image2d_t srcCoords,\n"
                     "__read_only image2d_t srcReal,\n"
                     "__read_only image2d_t srcImag,\n"
                     "__global float *fUnifiedSrcDensity,\n"
                     "__global int *nUnifiedSrcValid,\n"
                     "__constant char *useBandSrcValid,\n"
                     "__global int *nBandSrcValid,\n"
                     "__global outType *dstReal,\n"
                     "__global outType *dstImag,\n"
                     "__constant float *fDstNoDataReal,\n"
                     "__global float *dstDensity,\n"
                     "__global int *nDstValid,\n"
                     "const int bandNum)\n"
"{\n"
    "int i;\n"
    "float2  fDst = (float2)((0.5f+get_global_id(0))/((float)iDstWidth),\n"
                            "(0.5f+get_global_id(1))/((float)iDstHeight));\n"
    
    // Check & return when the thread group overruns the image size
    "if (fDst.x > 1.0f || fDst.y > 1.0f)\n"
        "return;\n"
    
    "float2  fSrc = getSrcCoords(srcCoords, fDst);\n"
    
    "if (!isValid(fUnifiedSrcDensity, nUnifiedSrcValid, fSrc))\n"
        "return;\n"
    
    "int     iSrcX = (int) floor( fSrc.x - 0.5f );\n"
    "int     iSrcY = (int) floor( fSrc.y - 0.5f );\n"
    "float   fDeltaX = fSrc.x - 0.5f - (float)iSrcX;\n"
    "float   fDeltaY = fSrc.y - 0.5f - (float)iSrcY;\n"
    "float   fDeltaX2 = fDeltaX * fDeltaX;\n"
    "float   fDeltaY2 = fDeltaY * fDeltaY;\n"
    "float   fDeltaX3 = fDeltaX2 * fDeltaX;\n"
    "float   fDeltaY3 = fDeltaY2 * fDeltaY;\n"
    "vecf    afReal[4], afImag[4], afDens[4];\n"
    
    // Loop over rows
    "for (i = -1; i < 3; ++i)\n"
    "{\n"
        "vecf    fReal1, fReal2, fReal3, fReal4;\n"
        "vecf    fImag1, fImag2, fImag3, fImag4;\n"
        "vecf    fDens1, fDens2, fDens3, fDens4;\n"
        
        //Get all the pixels for this row
        "getPixel(srcReal, srcImag, fUnifiedSrcDensity, nUnifiedSrcValid,\n"
                 "useBandSrcValid, nBandSrcValid, (int2)(iSrcX-1, iSrcY+i),\n"
                 "bandNum, &fDens1, &fReal1, &fImag1);\n"
        
        "getPixel(srcReal, srcImag, fUnifiedSrcDensity, nUnifiedSrcValid,\n"
                 "useBandSrcValid, nBandSrcValid, (int2)(iSrcX  , iSrcY+i),\n"
                 "bandNum, &fDens2, &fReal2, &fImag2);\n"
        
        "getPixel(srcReal, srcImag, fUnifiedSrcDensity, nUnifiedSrcValid,\n"
                 "useBandSrcValid, nBandSrcValid, (int2)(iSrcX+1, iSrcY+i),\n"
                 "bandNum, &fDens3, &fReal3, &fImag3);\n"
        
        "getPixel(srcReal, srcImag, fUnifiedSrcDensity, nUnifiedSrcValid,\n"
                 "useBandSrcValid, nBandSrcValid, (int2)(iSrcX+2, iSrcY+i),\n"
                 "bandNum, &fDens4, &fReal4, &fImag4);\n"
   
        // Process this row
        "afReal[i+1] = cubicConvolution(fDeltaX, fDeltaX2, fDeltaX3, fReal1, fReal2, fReal3, fReal4);\n"
        "if (useImag)\n"
            "afImag[i+1] = cubicConvolution(fDeltaX, fDeltaX2, fDeltaX3, fImag1, fImag2, fImag3, fImag4);\n"
        "afDens[i+1] = cubicConvolution(fDeltaX, fDeltaX2, fDeltaX3, fDens1, fDens2, fDens3, fDens4);\n"
    "}\n"
    
    "vecf fFinImag;\n"
    "if (useImag)\n"
        "fFinImag = cubicConvolution(fDeltaY, fDeltaY2, fDeltaY3, afImag[0], afImag[1], afImag[2], afImag[3]);\n"
    "else\n"
        "fFinImag = 0.0f;\n"
    
    // Compute and save final pixel
    "setPixel(dstReal, dstImag, dstDensity, nDstValid, fDstNoDataReal, bandNum,\n"
             "cubicConvolution(fDeltaY, fDeltaY2, fDeltaY3, afDens[0], afDens[1], afDens[2], afDens[3]),\n"
             "cubicConvolution(fDeltaY, fDeltaY2, fDeltaY3, afReal[0], afReal[1], afReal[2], afReal[3]),\n"
             "fFinImag );\n"
"}\n";

    const char *kernResampler =
// "************************ LanczosSinc ************************\n"

"float lanczosSinc( float dfX, float dfR )\n"
"{\n"
    "if ( dfX > dfR || dfX < -dfR)\n"
        "return 0.0f;\n"
    "if ( dfX == 0.0f )\n"
        "return 1.0f;\n"
    
    "float dfPIX = PI * dfX;\n"
    "return ( sin(dfPIX) / dfPIX ) * ( sin(dfPIX / dfR) * dfR / dfPIX );\n"
"}\n"

// "************************ Bicubic Spline ************************\n"

"float bSpline( float x )\n"
"{\n"
    "float xp2 = x + 2.0f;\n"
    "float xp1 = x + 1.0f;\n"
    "float xm1 = x - 1.0f;\n"
    "float xp2c = xp2 * xp2 * xp2;\n"
    
    "return (((xp2 > 0.0f)?((xp1 > 0.0f)?((x > 0.0f)?((xm1 > 0.0f)?\n"
                                                     "-4.0f * xm1*xm1*xm1:0.0f) +\n"
                                         "6.0f * x*x*x:0.0f) +\n"
                           "-4.0f * xp1*xp1*xp1:0.0f) +\n"
             "xp2c:0.0f) ) * 0.166666666666666666666f;\n"
"}\n"

// "************************ General Resampler ************************\n"

"__kernel void resamp(__read_only image2d_t srcCoords,\n"
                     "__read_only image2d_t srcReal,\n"
                     "__read_only image2d_t srcImag,\n"
                     "__global float *fUnifiedSrcDensity,\n"
                     "__global int *nUnifiedSrcValid,\n"
                     "__constant char *useBandSrcValid,\n"
                     "__global int *nBandSrcValid,\n"
                     "__global outType *dstReal,\n"
                     "__global outType *dstImag,\n"
                     "__constant float *fDstNoDataReal,\n"
                     "__global float *dstDensity,\n"
                     "__global int *nDstValid,\n"
                     "const int bandNum)\n"
"{\n"
    "float2  fDst = (float2)((0.5f+get_global_id(0))/((float)iDstWidth),\n"
                            "(0.5f+get_global_id(1))/((float)iDstHeight));\n"
    
    //"Check & return when the thread group overruns the image size\n"
    "if (fDst.x >= 1.0f || fDst.y >= 1.0f)\n"
        "return;\n"
    
    "float2  fSrc = getSrcCoords(srcCoords, fDst);\n"
    
    "if (!isValid(fUnifiedSrcDensity, nUnifiedSrcValid, fSrc))\n"
        "return;\n"
    
    "int     iSrcX = (int) floor( fSrc.x - 0.5f );\n"
    "int     iSrcY = (int) floor( fSrc.y - 0.5f );\n"
    "float   fDeltaX = fSrc.x - 0.5f - (float)iSrcX;\n"
    "float   fDeltaY = fSrc.y - 0.5f - (float)iSrcY;\n"
    
    "vecf  fAccumulatorReal = 0.0f, fAccumulatorImag = 0.0f;\n"
    "vecf  fAccumulatorDensity = 0.0f;\n"
    "float fAccumulatorWeight = 0.0f;\n"
    "int   i, j;\n"

     // "Loop over pixel rows in the kernel\n"
    "for ( j = nFiltInitY; j <= nYRadius; ++j )\n"
    "{\n"
        "float   fWeight1;\n"
        "int2 iSrc = (int2)(0, iSrcY + j);\n"
        
        // "Skip sampling over edge of image\n"
        "if ( iSrc.y < 0 || iSrc.y >= iSrcHeight )\n"
            "continue;\n"
    
        // "Select the resampling algorithm\n"
        "if ( doCubicSpline )\n"
            // "Calculate the Y weight\n"
            "fWeight1 = ( fYScale < 1.0f ) ?\n"
                "bSpline(((float)j) * fYScale) * fYScale :\n"
                "bSpline(((float)j) - fDeltaY);\n"
        "else\n"
            "fWeight1 = ( fYScale < 1.0f ) ?\n"
                "lanczosSinc(j * fYScale, fYFilter) * fYScale :\n"
                "lanczosSinc(j - fDeltaY, fYFilter);\n"
        
        // "Iterate over pixels in row\n"
        "for ( i = nFiltInitX; i <= nXRadius; ++i )\n"
        "{\n"
            "float fWeight2;\n"
            "vecf fDensity = 0.0f, fReal = 0.0f, fImag = 0.0f;\n"
            "iSrc.x = iSrcX + i;\n"
            
            // Skip sampling at edge of image
            // Skip sampling when invalid pixel
            "if ( iSrc.x < 0 || iSrc.x >= iSrcWidth || \n"
                  "!getPixel(srcReal, srcImag, fUnifiedSrcDensity,\n"
                            "nUnifiedSrcValid, useBandSrcValid, nBandSrcValid,\n"
                            "iSrc, bandNum, &fDensity, &fReal, &fImag) )\n"
                "continue;\n"
    
            // Choose among possible algorithms
            "if ( doCubicSpline )\n"
                // Calculate & save the X weight
                "fWeight2 = fWeight1 * ((fXScale < 1.0f ) ?\n"
                    "bSpline((float)i * fXScale) * fXScale :\n"
                    "bSpline(fDeltaX - (float)i));\n"
            "else\n"
                // Calculate & save the X weight
                "fWeight2 = fWeight1 * ((fXScale < 1.0f ) ?\n"
                    "lanczosSinc(i * fXScale, fXFilter) * fXScale :\n"
                    "lanczosSinc(i - fDeltaX, fXFilter));\n"
            
            // Accumulate!
            "fAccumulatorReal += fReal * fWeight2;\n"
            "fAccumulatorImag += fImag * fWeight2;\n"
            "fAccumulatorDensity += fDensity * fWeight2;\n"
            "fAccumulatorWeight += fWeight2;\n"
        "}\n"
    "}\n"

    /* FIXME: make this work with vector data. It'll look something like this:
    "#ifdef USE_VEC\n"
    "if (fDensity.x < 0.0001f && fDensity.y < 0.0001f &&\n"
    "fDensity.z < 0.0001f && fDensity.w < 0.0001f )\n"
    "return;\n"
    */
    "if ( fAccumulatorWeight < 0.000001f || fAccumulatorDensity < 0.000001f )\n"
    "{\n"
        "setPixel(dstReal, dstImag, dstDensity, nDstValid, fDstNoDataReal, bandNum,\n"
                 "0.0f, 0.0f, 0.0f);\n"
    "}\n"
    // "Calculate the output taking into account weighting\n"
    "else if ( fAccumulatorWeight < 0.99999f || fAccumulatorWeight > 1.00001f )\n"
    "{\n"
        "vecf fFinImag;\n"
        "if (useImag)\n"
            "fFinImag = fAccumulatorImag / fAccumulatorWeight;\n"
        "else\n"
            "fFinImag = 0.0f;\n"
        
        "setPixel(dstReal, dstImag, dstDensity, nDstValid, fDstNoDataReal, bandNum,\n"
                 "fAccumulatorDensity / fAccumulatorWeight,\n"
                 "fAccumulatorReal / fAccumulatorWeight,\n"
                 "fFinImag);\n"
    "} else {\n"
        "setPixel(dstReal, dstImag, dstDensity, nDstValid, fDstNoDataReal, bandNum,\n"
                 "fAccumulatorDensity,\n"
                 "fAccumulatorReal,\n"
                 "fAccumulatorImag);\n"
    "}\n"
"}\n";
    
    //Defines based on image format
    switch (warper->imageFormat) {
        case CL_FLOAT:
            dstMinVal = -MAXFLOAT;
            dstMaxVal = MAXFLOAT;
            outType = "float";
            break;
        case CL_SNORM_INT8:
            dstMinVal = -128.0;
            dstMaxVal = 127.0;
            outType = "char";
            break;
        case CL_UNORM_INT8:
            dstMinVal = 0.0;
            dstMaxVal = 255.0;
            outType = "uchar";
            break;
        case CL_SNORM_INT16:
            dstMinVal = -32768.0;
            dstMaxVal = 32767.0;
            outType = "short";
            break;
        case CL_UNORM_INT16:
            dstMinVal = 0.0;
            dstMaxVal = 65535.0;
            outType = "ushort";
            break;
    }
    
    //Use vector format? (currently unsupported)
    if(0)
        useVec = "-D USE_VEC";
    
    //Assemble the kernel from parts. The compiler is unable to handle multiple
    //kernels in one string with more than a few __constant modifiers each.
    if (warper->resampAlg == OCL_Cubic)
        sprintf(progBuf, "%s\n%s", kernGenFuncs, kernCubic);
    else
        sprintf(progBuf, "%s\n%s", kernGenFuncs, kernResampler);
    
    //Actually make the program from assembled source
    program = clCreateProgramWithSource(warper->context, 1, (const char**)&progBuf,
                                        NULL, &err);
    handleErrRetNULL(err);
    
    //Assemble the compiler arg string for speed. All invariants should be defined here.
    sprintf(buffer, "-cl-fast-relaxed-math -Werror -D vecf=float -D FALSE=0 -D TRUE=1 "
            "-D iSrcWidth=%d -D iSrcHeight=%d -D iDstWidth=%d -D iDstHeight=%d "
            "-D useUnifiedSrcDensity=%d -D useUnifiedSrcValid=%d "
            "-D useDstDensity=%d -D useDstValid=%d -D useImag=%d "
            "-D fXScale=%010ff -D fYScale=%010ff -D fXFilter=%010ff -D fYFilter=%010ff "
            "-D nXRadius=%d -D nYRadius=%d -D nFiltInitX=%d -D nFiltInitY=%d "
            "-D PI=%010ff -D outType=%s -D dstMinVal=%010ff -D dstMaxVal=%010ff "
            "-D useDstNoDataReal=%d %s -D doCubicSpline=%d "
            "-D useUseBandSrcValid=%d",
            warper->srcWidth, warper->srcHeight, warper->dstWidth, warper->dstHeight,
            warper->useUnifiedSrcDensity, warper->useUnifiedSrcValid,
            warper->useDstDensity, warper->useDstValid, warper->imagWorkCL != NULL,
            dfXScale, dfYScale, dfXFilter, dfYFilter,
            nXRadius, nYRadius, nFiltInitX, nFiltInitY,
            M_PI, outType, dstMinVal, dstMaxVal,
            warper->fDstNoDataRealCL != NULL, useVec, warper->resampAlg == OCL_CubicSpline,
            warper->nBandSrcValidCL != NULL);

    (*clErr) = err = clBuildProgram(program, 1, &(warper->dev), buffer, NULL, NULL);
    
    //Detailed debugging info
    if (err != CL_SUCCESS)
    {
        err = clGetProgramBuildInfo(program, warper->dev, CL_PROGRAM_BUILD_LOG,
                                    128000*sizeof(char), buffer, NULL);
        handleErrRetNULL(err);
        
        printf("Build Log:\n%s\n", buffer);
        printf("Error: Failed to build program executable!\n");
        printCLErr(err);
        
        printf("%s\n", buffer);
        
        err = clGetProgramBuildInfo(program, warper->dev, CL_PROGRAM_BUILD_STATUS,
                                    128000*sizeof(char), buffer, NULL);
        handleErrRetNULL(err);
        
        printf("Build Status:\n");
        if(buffer[0] == CL_BUILD_NONE)
            printf("CL_BUILD_NONE\n");
        else if(buffer[0] == CL_BUILD_ERROR)
            printf("CL_BUILD_ERROR\n");
        else if(buffer[0] == CL_BUILD_SUCCESS)
            printf("CL_BUILD_SUCCESS\n");
        else if(buffer[0] == CL_BUILD_IN_PROGRESS)
            printf("CL_BUILD_IN_PROGRESS\n");
        
        printf("Program Source:\n%s\n", progBuf);
        return NULL;
    }
    
    kernel = clCreateKernel(program, "resamp", &err);
    handleErrRetNULL(err);
    
    err = clReleaseProgram(program);
    handleErrRetNULL(err);
    
    free(buffer);
    free(progBuf);
    return kernel;
}

/*
 Alloc & copy the coordinate data from host working memory to the device. The
 working memory should be a pinned, linear, array of floats. This allows us to
 allocate and copy all data in one step. The pointer to the device memory is
 saved and set as the appropriate argument number.
 
 Returns CL_SUCCESS on success and other CL_* errors when something goes wrong.
 */
cl_int set_coord_data (struct oclWarper *warper, cl_mem *xy)
{
    cl_int err = CL_SUCCESS;
    cl_image_format imgFmt;
    
    //Copy coord data to the device
    imgFmt.image_channel_order = warper->xyChOrder;
    imgFmt.image_channel_data_type = CL_FLOAT;
    (*xy) = clCreateImage2D(warper->context,
                            CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, &imgFmt,
                            (size_t) warper->dstWidth,
                            (size_t) warper->dstHeight,
                            (size_t) sizeof(float) * warper->xyChSize * warper->dstWidth,
                            warper->xyWork, &err);
    handleErr(err);
    
    //Free the source memory, now that it's copied we don't need it
    freeCLMem(warper->xyWorkCL, warper->xyWork);
    
    //Set up argument
    handleErr(err = clSetKernelArg(warper->kern, 0, sizeof(cl_mem), xy));
    
    return CL_SUCCESS;
}

/*
 Sets the unified density & valid data structures. These are optional structures
 from GDAL, and as such if they are NULL a small placeholder memory segment is
 defined. This is because the spec is unclear on if a NULL value can be passed
 as a kernel argument in place of memory. If it's not NULL, the data is copied
 from the working memory to the device memory. After that, we check if we are
 using the per-band validity mask, and set that as appropriate. At the end, the
 CL mem is passed as the kernel arguments.
 
 Returns CL_SUCCESS on success and other CL_* errors when something goes wrong.
 */
cl_int set_unified_data(struct oclWarper *warper,
                        cl_mem *unifiedSrcDensityCL, cl_mem *unifiedSrcValidCL,
                        float *unifiedSrcDensity, unsigned int *unifiedSrcValid,
                        cl_mem *useBandSrcValidCL, cl_mem *nBandSrcValidCL)
{
    cl_int err = CL_SUCCESS;
    size_t sz = warper->srcWidth * warper->srcHeight;
    int useValid = warper->nBandSrcValidCL != NULL;
    //32 bits in the mask
    int validSz = sizeof(int) * (1 + (sz >> 5));
    
    //Copy unifiedSrcDensity if it exists
    if (unifiedSrcDensity == NULL) {
        //Alloc dummy device RAM
        (*unifiedSrcDensityCL) = clCreateBuffer(warper->context, CL_MEM_READ_ONLY, 1, NULL, &err);
        handleErr(err);
    } else {
        //Alloc & copy all density data
        (*unifiedSrcDensityCL) = clCreateBuffer(warper->context,
                                                CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                                sizeof(float) * sz, unifiedSrcDensity, &err);
        handleErr(err);
    }
    
    //Copy unifiedSrcValid if it exists
    if (unifiedSrcValid == NULL) {
        //Alloc dummy device RAM
        (*unifiedSrcValidCL) = clCreateBuffer(warper->context, CL_MEM_READ_ONLY, 1, NULL, &err);
        handleErr(err);
    } else {
        //Alloc & copy all validity data
        (*unifiedSrcValidCL) = clCreateBuffer(warper->context,
                                              CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                              validSz, unifiedSrcValid, &err);
        handleErr(err);
    }
    
    // Set the band validity usage
    if(useValid) {
        (*useBandSrcValidCL) = clCreateBuffer(warper->context,
                                              CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                              sizeof(char) * warper->numBands,
                                              warper->useBandSrcValid, &err);
        handleErr(err);
    } else {
        //Make a fake image so we don't have a NULL pointer
        (*useBandSrcValidCL) = clCreateBuffer(warper->context, CL_MEM_READ_ONLY, 1, NULL, &err);
        handleErr(err);
    }
    
    //Do a more thorough check for validity
    if (useValid) {
        int i;
        useValid = FALSE;
        for (i = 0; i < warper->numBands; ++i)
            if (warper->useBandSrcValid[i])
                useValid = TRUE;
    }
    
    //And the validity mask if needed
    if (useValid) {
        (*nBandSrcValidCL) = clCreateBuffer(warper->context,
                                            CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                            warper->numBands * validSz,
                                            warper->nBandSrcValid, &err);
        handleErr(err);
    } else {
        //Make a fake image so we don't have a NULL pointer
        (*nBandSrcValidCL) = clCreateBuffer(warper->context, CL_MEM_READ_ONLY, 1, NULL, &err);
        handleErr(err);
    }

    //Set up arguments
    handleErr(err = clSetKernelArg(warper->kern, 3, sizeof(cl_mem), unifiedSrcDensityCL));
    handleErr(err = clSetKernelArg(warper->kern, 4, sizeof(cl_mem), unifiedSrcValidCL));
    handleErr(err = clSetKernelArg(warper->kern, 5, sizeof(cl_mem), useBandSrcValidCL));
    handleErr(err = clSetKernelArg(warper->kern, 6, sizeof(cl_mem), nBandSrcValidCL));
    
    return CL_SUCCESS;
}

/*
 Here we set the per-band raster data. First priority is the real raster data,
 of course. Then, if applicable, we set the additional image channel. Once this
 data is copied to the device, it can be freed on the host, so that is done
 here. Finally the appropriate kernel arguments are set.
 
 Returns CL_SUCCESS on success and other CL_* errors when something goes wrong.
 */
cl_int set_src_rast_data (struct oclWarper *warper, int iNum, size_t sz,
                          cl_mem *srcReal, cl_mem *srcImag)
{
    cl_image_format imgFmt;
    cl_int err = CL_SUCCESS;
    int useImagWork = warper->imagWork.v != NULL && warper->imagWork.v[iNum] != NULL;
    
    //Set up image vars
    imgFmt.image_channel_order = warper->imgChOrder;
    imgFmt.image_channel_data_type = warper->imageFormat;
    
    //Create & copy the source image
    (*srcReal) = clCreateImage2D(warper->context,
                                 CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, &imgFmt,
                                 (size_t) warper->srcWidth,
                                 (size_t) warper->srcHeight,
                                 sz * warper->srcWidth * warper->imgChSize,
                                 warper->realWork.v[iNum], &err);
    handleErr(err);
    
    //And the source image parts if needed
    if (useImagWork) {
        (*srcImag) = clCreateImage2D(warper->context,
                                     CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, &imgFmt,
                                     (size_t) warper->srcWidth,
                                     (size_t) warper->srcHeight,
                                     sz * warper->srcWidth * warper->imgChSize,
                                     warper->imagWork.v[iNum], &err);
        handleErr(err);
    } else {
        //Make a fake image so we don't have a NULL pointer
        (*srcImag) = clCreateImage2D(warper->context,
                                     CL_MEM_READ_ONLY, &imgFmt,
                                     1, 1, sz * warper->imgChSize, NULL, &err);
        handleErr(err);
    }

    //Free the source memory, now that it's copied we don't need it
    freeCLMem(warper->realWorkCL[iNum], warper->realWork.v[iNum]);
    if (warper->imagWork.v != NULL) {
        freeCLMem(warper->imagWorkCL[iNum], warper->imagWork.v[iNum]);
    }
    
    //Set up per-band arguments
    handleErr(err = clSetKernelArg(warper->kern, 1, sizeof(cl_mem), srcReal));
    handleErr(err = clSetKernelArg(warper->kern, 2, sizeof(cl_mem), srcImag));
    
    return CL_SUCCESS;
}

/*
 Set the destination data for the raster. Although it's the output, it still
 is copied to the device because some blending is done there. First the real
 data is allocated and copied, then the imag data is allocated and copied if
 needed. They are then set as the appropriate arguments to the kernel.
 
 Returns CL_SUCCESS on success and other CL_* errors when something goes wrong.
 */
cl_int set_dst_rast_data(struct oclWarper *warper, int iImg, size_t sz,
                         cl_mem *dstReal, cl_mem *dstImag)
{
    cl_int err = CL_SUCCESS;
    sz *= warper->dstWidth * warper->dstHeight * warper->imgChSize;
    
    //Copy the dst real data
    (*dstReal) = clCreateBuffer(warper->context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                sz, warper->dstRealWork.v[iImg], &err);
    handleErr(err);
    
    //Copy the dst imag data if exists
    if (warper->dstImagWork.v != NULL && warper->dstImagWork.v[iImg] != NULL) {
        (*dstImag) = clCreateBuffer(warper->context,
                                    CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                    sz, warper->dstImagWork.v[iImg], &err);
        handleErr(err);
    } else {
        (*dstImag) = clCreateBuffer(warper->context, CL_MEM_READ_WRITE, 1, NULL, &err);
        handleErr(err);
    }
    
    //Set up per-band arguments
    handleErr(err = clSetKernelArg(warper->kern, 7, sizeof(cl_mem), dstReal));
    handleErr(err = clSetKernelArg(warper->kern, 8, sizeof(cl_mem), dstImag));
    
    return CL_SUCCESS;
}

/*
 Read the final raster data back from the graphics card to working memory. This
 copies both the real memory and the imag memory if appropriate.
 
 Returns CL_SUCCESS on success and other CL_* errors when something goes wrong.
 */
cl_int get_dst_rast_data(struct oclWarper *warper, int iImg,
                         cl_mem dstReal, cl_mem dstImag)
{
    cl_int err = CL_SUCCESS;
    size_t sz = warper->dstWidth * warper->dstHeight;
    
    //Check the word size
    switch (warper->imageFormat) {
        case CL_FLOAT:
            sz *= sizeof(float);
            break;
        case CL_SNORM_INT8:
            sz *= sizeof(char);
            break;
        case CL_UNORM_INT8:
            sz *= sizeof(unsigned char);
            break;
        case CL_SNORM_INT16:
            sz *= sizeof(short);
            break;
        case CL_UNORM_INT16:
            sz *= sizeof(unsigned short);
            break;
    }
    
    //Copy from dev into working memory
    handleErr(err = clEnqueueReadBuffer(warper->queue, dstReal,
                                        CL_FALSE, 0, sz, warper->dstRealWork.v[iImg],
                                        0, NULL, NULL));
    
    //If we are expecting the imag channel, then copy it back also
    if (warper->dstImagWork.v != NULL && warper->dstImagWork.v[iImg] != NULL) {
        handleErr(err = clEnqueueReadBuffer(warper->queue, dstImag,
                                            CL_FALSE, 0, sz, warper->dstImagWork.v[iImg],
                                            0, NULL, NULL));
    }
    
    //The copy requests were non-blocking, so we'll need to make sure they finish.
    handleErr(err = clFinish(warper->queue));
    
    return CL_SUCCESS;
}

/*
 Set the destination image density & validity mask on the device. This is used
 to blend the final output image with the existing buffer. This handles the
 unified structures that apply to all bands. After the buffers are created and
 copied, they are set as kernel arguments.
 
 Returns CL_SUCCESS on success and other CL_* errors when something goes wrong.
 */
cl_int set_dst_data(struct oclWarper *warper,
                    cl_mem *dstDensityCL, cl_mem *dstValidCL, cl_mem *dstNoDataRealCL,
                    float *dstDensity, unsigned int *dstValid, float *dstNoDataReal)
{
    cl_int err = CL_SUCCESS;
    size_t sz = warper->dstWidth * warper->dstHeight;
    
    //Copy the no-data value(s)
    if (dstNoDataReal == NULL) {
        (*dstNoDataRealCL) = clCreateBuffer(warper->context, CL_MEM_READ_ONLY, 1, NULL, &err);
        handleErr(err);
    } else {
        (*dstNoDataRealCL) = clCreateBuffer(warper->context,
                                         CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                         sizeof(float) * warper->numBands, dstNoDataReal, &err);
        handleErr(err);
    }
    
    //Copy unifiedSrcDensity if it exists
    if (dstDensity == NULL) {
        (*dstDensityCL) = clCreateBuffer(warper->context, CL_MEM_READ_ONLY, 1, NULL, &err);
        handleErr(err);
    } else {
        (*dstDensityCL) = clCreateBuffer(warper->context,
                                         CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                         sizeof(float) * sz, dstDensity, &err);
        handleErr(err);
    }
    
    //Copy unifiedSrcValid if it exists
    if (dstValid == NULL) {
        (*dstValidCL) = clCreateBuffer(warper->context, CL_MEM_READ_ONLY, 1, NULL, &err);
        handleErr(err);
    } else {
        (*dstValidCL) = clCreateBuffer(warper->context,
                                       CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                       sizeof(int) * ((1 + sz) >> 5), dstValid, &err);
        handleErr(err);
    }
    
    //Set up arguments
    handleErr(err = clSetKernelArg(warper->kern,  9, sizeof(cl_mem), dstNoDataRealCL));
    handleErr(err = clSetKernelArg(warper->kern, 10, sizeof(cl_mem), dstDensityCL));
    handleErr(err = clSetKernelArg(warper->kern, 11, sizeof(cl_mem), dstValidCL));
    
    return CL_SUCCESS;
}

/*
 Go ahead and execute the kernel. This handles some housekeeping stuff like the
 run dimensions. When running in debug mode, it times the kernel call and prints
 the execution time.
 
 Returns CL_SUCCESS on success and other CL_* errors when something goes wrong.
 */
cl_int execute_kern(struct oclWarper *warper, size_t loc_size)
{
    cl_int err = CL_SUCCESS;
    cl_event ev;
    size_t ceil_runs[2];
    size_t group_size[2];
#ifndef NDEBUG
    size_t start_time = 0;
    size_t end_time;
#endif
    
    // Use a likely X-dimension which is a power of 2
    if (loc_size >= 512)
        group_size[0] = 32;
    else if (loc_size >= 64)
        group_size[0] = 16;
    else if (loc_size > 8)
        group_size[0] = 8;
    else
        group_size[0] = 1;
    
    if (group_size[0] > loc_size)
        group_size[1] = group_size[0]/loc_size;
    else
        group_size[1] = 1;
    
    //Round up num_runs to find the dim of the block of pixels we'll be processing
    if(warper->dstWidth % group_size[0])
        ceil_runs[0] = warper->dstWidth + group_size[0] - warper->dstWidth % group_size[0];
    else
        ceil_runs[0] = warper->dstWidth;
    
    if(warper->dstHeight % group_size[1])
        ceil_runs[1] = warper->dstHeight + group_size[1] - warper->dstHeight % group_size[1];
    else
        ceil_runs[1] = warper->dstHeight;
    
#ifndef NDEBUG
    handleErr(err = clSetCommandQueueProperty(warper->queue, CL_QUEUE_PROFILING_ENABLE, CL_TRUE, NULL));
#endif
    
    // Run the calculation by enqueuing it and forcing the 
    // command queue to complete the task
    handleErr(err = clEnqueueNDRangeKernel(warper->queue, warper->kern, 2, NULL, 
                                           ceil_runs, group_size, 0, NULL, &ev));
    handleErr(err = clFinish(warper->queue));
    
#ifndef NDEBUG
    handleErr(err = clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_START,
                                            sizeof(size_t), &start_time, NULL));
    handleErr(err = clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_END,
                                            sizeof(size_t), &end_time, NULL));
    assert(end_time != 0);
    assert(start_time != 0);
    handleErr(err = clReleaseEvent(ev));
    
    printf("Kernel Time: %15lu\n", (long int)((end_time-start_time)/100000));
#endif
    return CL_SUCCESS;
}

/*
 Copy data from a raw source to the warper's working memory. If the imag
 channel is expected, then the data will be de-interlaced into component blocks
 of memory.
 
 Returns CL_SUCCESS on success and other CL_* errors when something goes wrong.
 */
cl_int set_img_data(struct oclWarper *warper, void *srcImgData,
                    unsigned int width, unsigned int height,
                    void *dstReal, void *dstImag)
{
    unsigned int imgChSize = warper->imgChSize;
    unsigned int iSrcY, i;
    
    if (warper->imagWorkCL == NULL && imgChSize == 1) {
        //Set memory size & location depending on the data type
        //This is the ideal code path for speed
        switch (warper->imageFormat) {
            case CL_UNORM_INT8:
                memcpy(dstReal, srcImgData, width*height*sizeof(unsigned char));
                break;
            case CL_SNORM_INT8:
                memcpy(dstReal, srcImgData, width*height*sizeof(char));
                break;
            case CL_UNORM_INT16:
                memcpy(dstReal, srcImgData, width*height*sizeof(unsigned short));
                break;
            case CL_SNORM_INT16:
                memcpy(dstReal, srcImgData, width*height*sizeof(short));
                break;
            case CL_FLOAT:
                memcpy(dstReal, srcImgData, width*height*sizeof(float));
                break;
        }
    } else if (warper->imagWorkCL == NULL) {
        //We need to space the values due to OpenCL implementation reasons
        for( iSrcY = 0; iSrcY < height; iSrcY++ )
        {
            int pxOff = width*iSrcY;
            //Copy & deinterleave interleaved data
            switch (warper->imageFormat) {
                case CL_UNORM_INT8:
                {
                    unsigned char *realDst = &(((unsigned char *)dstReal)[pxOff]);
                    unsigned char *dataSrc = &(((unsigned char *)srcImgData)[pxOff]);
                    for (i = 0; i < width; ++i)
                        realDst[imgChSize*i] = dataSrc[i];
                }
                    break;
                case CL_SNORM_INT8:
                {
                    char *realDst = &(((char *)dstReal)[pxOff]);
                    char *dataSrc = &(((char *)srcImgData)[pxOff]);
                    for (i = 0; i < width; ++i)
                        realDst[imgChSize*i] = dataSrc[i];
                }
                    break;
                case CL_UNORM_INT16:
                {
                    unsigned short *realDst = &(((unsigned short *)dstReal)[pxOff]);
                    unsigned short *dataSrc = &(((unsigned short *)srcImgData)[pxOff]);
                    for (i = 0; i < width; ++i)
                        realDst[imgChSize*i] = dataSrc[i];
                }
                    break;
                case CL_SNORM_INT16:
                {
                    short *realDst = &(((short *)dstReal)[pxOff]);
                    short *dataSrc = &(((short *)srcImgData)[pxOff]);
                    for (i = 0; i < width; ++i)
                        realDst[imgChSize*i] = dataSrc[i];
                }
                    break;
                case CL_FLOAT:
                {
                    float *realDst = &(((float *)dstReal)[pxOff]);
                    float *dataSrc = &(((float *)srcImgData)[pxOff]);
                    for (i = 0; i < width; ++i)
                        realDst[imgChSize*i] = dataSrc[i];
                }
                    break;
            }
        }
    } else {
        //Copy, deinterleave, & space interleaved data
        for( iSrcY = 0; iSrcY < height; iSrcY++ )
        {
            int pxOff = width*iSrcY;
            switch (warper->imageFormat) {
                case CL_FLOAT:
                {
                    float *realDst = &(((float *)dstReal)[pxOff]);
                    float *imagDst = &(((float *)dstImag)[pxOff]);
                    float *dataSrc = &(((float *)srcImgData)[pxOff]);
                    for (i = 0; i < width; ++i) {
                        realDst[imgChSize*i] = dataSrc[i*2  ];
                        imagDst[imgChSize*i] = dataSrc[i*2+1];
                    }
                }
                    break;
                case CL_SNORM_INT8:
                {
                    char *realDst = &(((char *)dstReal)[pxOff]);
                    char *imagDst = &(((char *)dstImag)[pxOff]);
                    char *dataSrc = &(((char *)srcImgData)[pxOff]);
                    for (i = 0; i < width; ++i) {
                        realDst[imgChSize*i] = dataSrc[i*2  ];
                        imagDst[imgChSize*i] = dataSrc[i*2+1];
                    }
                }
                    break;
                case CL_UNORM_INT8:
                {
                    unsigned char *realDst = &(((unsigned char *)dstReal)[pxOff]);
                    unsigned char *imagDst = &(((unsigned char *)dstImag)[pxOff]);
                    unsigned char *dataSrc = &(((unsigned char *)srcImgData)[pxOff]);
                    for (i = 0; i < width; ++i) {
                        realDst[imgChSize*i] = dataSrc[i*2  ];
                        imagDst[imgChSize*i] = dataSrc[i*2+1];
                    }
                }
                    break;
                case CL_SNORM_INT16:
                {
                    short *realDst = &(((short *)dstReal)[pxOff]);
                    short *imagDst = &(((short *)dstImag)[pxOff]);
                    short *dataSrc = &(((short *)srcImgData)[pxOff]);
                    for (i = 0; i < width; ++i) {
                        realDst[imgChSize*i] = dataSrc[i*2  ];
                        imagDst[imgChSize*i] = dataSrc[i*2+1];
                    }
                }
                    break;
                case CL_UNORM_INT16:
                {
                    unsigned short *realDst = &(((unsigned short *)dstReal)[pxOff]);
                    unsigned short *imagDst = &(((unsigned short *)dstImag)[pxOff]);
                    unsigned short *dataSrc = &(((unsigned short *)srcImgData)[pxOff]);
                    for (i = 0; i < width; ++i) {
                        realDst[imgChSize*i] = dataSrc[i*2  ];
                        imagDst[imgChSize*i] = dataSrc[i*2+1];
                    }
                }
                    break;
            }
        }
    }
    
    return CL_SUCCESS;
}

/*
 Creates the struct which inits & contains the OpenCL context & environment.
 Inits wired(?) space to buffer the image in host RAM. Chooses the OpenCL
 device, perhaps the user can choose it later? This would also choose the
 appropriate OpenCL image format (R, RG, RGBA, or multiples thereof). Space
 for metadata can be allocated as required, though.
 
 Supported image formats are:
 CL_FLOAT, CL_SNORM_INT8, CL_UNORM_INT8, CL_SNORM_INT16, CL_UNORM_INT16
 32-bit int formats won't keep precision when converted to floats internally
 and doubles are generally not supported on the GPU image formats.
 */
struct oclWarper* GDALWarpKernelOpenCL_createEnv(int srcWidth, int srcHeight,
                                                 int dstWidth, int dstHeight,
                                                 cl_channel_type imageFormat, int numBands,
                                                 int useImag, int useBandSrcValid,
                                                 float *fDstDensity,
                                                 double *dfDstNoDataReal,
                                                 OCLResampAlg resampAlg, cl_int *clErr)
{
    struct oclWarper *warper;
    int i;
	cl_int err = CL_SUCCESS;
    size_t fmtSize, sz;
    
    warper = (struct oclWarper *)CPLMalloc(sizeof(struct oclWarper));
    if (warper == NULL)
        handleErrRetNULL(err = CL_OUT_OF_HOST_MEMORY);
    
    //Init passed vars
    warper->srcWidth = srcWidth;
    warper->srcHeight = srcHeight;
    warper->dstWidth = dstWidth;
    warper->dstHeight = dstHeight;
    
    warper->numBands = numBands;
    warper->imageFormat = imageFormat;
    warper->resampAlg = resampAlg;
    
    warper->imagWorkCL = NULL;
    warper->dstImagWorkCL = NULL;
    warper->useBandSrcValidCL = NULL;
    warper->useBandSrcValid = NULL;
    warper->nBandSrcValidCL = NULL;
    warper->nBandSrcValid = NULL;
    warper->fDstNoDataRealCL = NULL;
    warper->fDstNoDataReal = NULL;
    
    // Note in the future we may have the option for float4 vectors, thus this may be different than numBands
    warper->numImages = numBands;
    
    //Make the pointer space for the real images
    warper->realWorkCL = (cl_mem *)CPLMalloc(sizeof(cl_mem)*warper->numImages);
    warper->dstRealWorkCL = (cl_mem *)CPLMalloc(sizeof(cl_mem)*warper->numImages);
    if (warper->realWorkCL == NULL || warper->dstRealWorkCL == NULL)
        handleErrRetNULL(err = CL_OUT_OF_HOST_MEMORY);
    
    //Make space for the per-channel Imag data (if exists)
    if (useImag) {
        warper->imagWorkCL = (cl_mem *)CPLMalloc(sizeof(cl_mem)*warper->numImages);
        warper->dstImagWorkCL = (cl_mem *)CPLMalloc(sizeof(cl_mem)*warper->numImages);
        if (warper->imagWorkCL == NULL || warper->imagWorkCL == NULL)
            handleErrRetNULL(err = CL_OUT_OF_HOST_MEMORY);
    }
    
    //Make space for the per-band BandSrcValid data (if exists)
    if (useBandSrcValid) {
        //32 bits in the mask
        size_t sz = warper->numBands * (1 + (warper->srcWidth * warper->srcHeight >> 5));
        
        //Allocate some space for the validity of the validity mask
        err = alloc_pinned_mem(warper, 0, warper->numBands*sizeof(char),
                               (void **)&(warper->useBandSrcValid),
                               &(warper->useBandSrcValidCL));
        handleErrRetNULL(err);
        
        for (i = 0; i < warper->numBands; ++i)
            warper->useBandSrcValid[i] = FALSE;
        
        //Allocate one array for all the band validity masks
        //Remember that the masks don't use much memeory (they're bitwise)
        err = alloc_pinned_mem(warper, 0, sz * sizeof(int),
                               (void **)&(warper->nBandSrcValid),
                               &(warper->nBandSrcValidCL));
        handleErrRetNULL(err);
    }
    
    //Make space for the per-band 
    if (dfDstNoDataReal != NULL) {
        alloc_pinned_mem(warper, 0, warper->numBands,
                         (void **)&(warper->fDstNoDataReal), &(warper->fDstNoDataRealCL));
        
        //Copy over values
        for (i = 0; i < warper->numBands; ++i)
            warper->fDstNoDataReal[i] = dfDstNoDataReal[i];
    }
    
    //Setup the OpenCL environment
    warper->dev = get_device();
    warper->context = clCreateContext(0, 1, &(warper->dev), NULL, NULL, &err);
    handleErrRetNULL(err);
    warper->queue = clCreateCommandQueue(warper->context, warper->dev, 0, &err);
    handleErrRetNULL(err);
    
    //Alloc working host image memory
    //We'll be copying into these buffers soon
    switch (imageFormat) {
        case CL_FLOAT:
            err = alloc_working_arr(warper, sizeof(float *), sizeof(float), &fmtSize);
            break;
        case CL_SNORM_INT8:
            err = alloc_working_arr(warper, sizeof(char *), sizeof(char), &fmtSize);
            break;
        case CL_UNORM_INT8:
            err = alloc_working_arr(warper, sizeof(unsigned char *), sizeof(unsigned char), &fmtSize);
            break;
        case CL_SNORM_INT16:
            err = alloc_working_arr(warper, sizeof(short *), sizeof(short), &fmtSize);
            break;
        case CL_UNORM_INT16:
            err = alloc_working_arr(warper, sizeof(unsigned short *), sizeof(unsigned short), &fmtSize);
            break;
    }
    handleErrRetNULL(err);
    
    //Find a good & compable image channel order for the Lat/Long arr
    err = set_supported_formats(warper, 2,
                                &(warper->xyChOrder), &(warper->xyChSize),
                                CL_FLOAT);
    handleErrRetNULL(err);
    
    //Alloc coord memory
    sz = sizeof(float) * warper->dstWidth * warper->dstHeight * warper->xyChSize;
    err = alloc_pinned_mem(warper, 0, sz, (void **)&(warper->xyWork),
                           &(warper->xyWorkCL));
    handleErrRetNULL(err);
    
    //Ensure everything is finished allocating, copying, & mapping
    err = clFinish(warper->queue);
    handleErrRetNULL(err);
    
    (*clErr) = CL_SUCCESS;
    return warper;
}

/*
 Copy the validity mask for an image band to the warper.
 
 Returns CL_SUCCESS on success and other CL_* errors when something goes wrong.
 */
cl_int GDALWarpKernelOpenCL_setSrcValid(struct oclWarper *warper,
                                        int *bandSrcValid, int bandNum)
{
    //32 bits in the mask
    int stride = 1 + (warper->srcWidth * warper->srcHeight >> 5);
    
    //Copy bandSrcValid
    assert(warper->nBandSrcValid != NULL);
    memcpy(&(warper->nBandSrcValid[bandNum*stride]), bandSrcValid, sizeof(int) * stride);
    warper->useBandSrcValid[bandNum] = TRUE;
    
    return CL_SUCCESS;
}

/*
 Sets the source image real & imag into the host memory so that it is
 permuted (ex. RGBA) for better graphics card access.
 
 Returns CL_SUCCESS on success and other CL_* errors when something goes wrong.
 */
cl_int GDALWarpKernelOpenCL_setSrcImg(struct oclWarper *warper, void *imgData,
                                      int bandNum)
{
    void *imagWorkPtr = NULL;
    
    if (warper->imagWorkCL != NULL)
        imagWorkPtr = warper->imagWork.v[bandNum];
    
    return set_img_data(warper, imgData, warper->srcWidth, warper->srcHeight,
                        warper->realWork.v[bandNum], imagWorkPtr);
}

/*
 Sets the destination image real & imag into the host memory so that it is
 permuted (ex. RGBA) for better graphics card access.
 
 Returns CL_SUCCESS on success and other CL_* errors when something goes wrong.
 */
cl_int GDALWarpKernelOpenCL_setDstImg(struct oclWarper *warper, void *imgData,
                                      int bandNum)
{
    void *dstImagWorkPtr = NULL;
    
    if (warper->dstImagWorkCL != NULL)
        dstImagWorkPtr = warper->dstImagWork.v[bandNum];
    
    return set_img_data(warper, imgData, warper->dstWidth, warper->dstHeight,
                        warper->dstRealWork.v[bandNum], dstImagWorkPtr);
}

/*
 Inputs the source coordinates for a row of the destination pixels. Invalid
 coordinates are set as -99.0, which should be out of the image bounds. Sets
 the coordinates as ready to be used in OpenCL image memory: interleaved and
 minus the offset. By using image memory, we should be able to eventually use
 a smaller texture for coordinates and use OpenCL's built-in interpolation
 to save memory.
 
 Returns CL_SUCCESS on success and other CL_* errors when something goes wrong.
 */
cl_int GDALWarpKernelOpenCL_setCoordRow(struct oclWarper *warper,
                                        double *rowSrcX, double *rowSrcY,
                                        double srcXOff, double srcYOff,
                                        int *success, int rowNum)
{
    int i;
    int dstWidth = warper->dstWidth;
    int xyChSize = warper->xyChSize;
    float *xyPtr = &(warper->xyWork[rowNum * dstWidth * xyChSize]);
    
    for (i = 0; i < dstWidth; ++i) {
        if (success[i]) {
            xyPtr[0] = rowSrcX[i] - srcXOff;
            xyPtr[1] = rowSrcY[i] - srcYOff;
        } else {
            xyPtr[0] = -99.0f;
            xyPtr[1] = -99.0f;
        }
        xyPtr += xyChSize;
    }
    return CL_SUCCESS;
}

/*
 Copies all data to the device RAM, frees the host RAM, runs the
 appropriate resampling kernel, mallocs output space, & copies the data
 back from the device RAM for each band. Also check to make sure that
 setRow*() was called the appropriate number of times to init all image
 data.
 
 Returns CL_SUCCESS on success and other CL_* errors when something goes wrong.
 */
cl_int GDALWarpKernelOpenCL_runResamp(struct oclWarper *warper,
                                      float *unifiedSrcDensity,
                                      unsigned int *unifiedSrcValid,
                                      float *dstDensity,
                                      unsigned int *dstValid,
                                      double dfXScale, double dfYScale,
                                      double dfXFilter, double dfYFilter,
                                      int nXRadius, int nYRadius,
                                      int nFiltInitX, int nFiltInitY)
{
    int i;
	cl_int err = CL_SUCCESS;
    cl_mem xy, unifiedSrcDensityCL, unifiedSrcValidCL;
    cl_mem dstDensityCL, dstValidCL, dstNoDataRealCL;
    cl_mem useBandSrcValidCL, nBandSrcValidCL;
	size_t groupSize, wordSize;
    
    warper->useUnifiedSrcDensity = unifiedSrcDensity != NULL;
    warper->useUnifiedSrcValid = unifiedSrcValid != NULL;

    //Check the word size
    switch (warper->imageFormat) {
        case CL_FLOAT:
            wordSize = sizeof(float);
            break;
        case CL_SNORM_INT8:
            wordSize = sizeof(char);
            break;
        case CL_UNORM_INT8:
            wordSize = sizeof(unsigned char);
            break;
        case CL_SNORM_INT16:
            wordSize = sizeof(short);
            break;
        case CL_UNORM_INT16:
            wordSize = sizeof(unsigned short);
            break;
    }
    
    //Compile the kernel; the invariants are being compiled into the code
    warper->kern = get_kernel(warper,
                              dfXScale, dfYScale, dfXFilter, dfYFilter,
                              nXRadius, nYRadius, nFiltInitX, nFiltInitY, &err);
    handleErr(err);
    
    //Copy coord data to the device
    handleErr(err = set_coord_data(warper, &xy));
    
    //Copy unified density & valid data
    handleErr(err = set_unified_data(warper, &unifiedSrcDensityCL, &unifiedSrcValidCL,
                                     unifiedSrcDensity, unifiedSrcValid,
                                     &useBandSrcValidCL, &nBandSrcValidCL));
    
    //Copy output density & valid data
    handleErr(set_dst_data(warper, &dstDensityCL, &dstValidCL, &dstNoDataRealCL,
                           dstDensity, dstValid, warper->fDstNoDataReal));
    
    //What's the recommended group size?
	handleErr(clGetKernelWorkGroupInfo(warper->kern, warper->dev, CL_KERNEL_WORK_GROUP_SIZE,
                                       sizeof(size_t), &groupSize, NULL));
    
    //Loop over each image
    for (i = 0; i < warper->numImages; ++i)
    {
        cl_mem srcImag, srcReal;
        cl_mem dstReal, dstImag;
        
        //Create & copy the source image
        handleErr(err = set_src_rast_data(warper, i, wordSize, &srcReal, &srcImag));
        
        //Create & copy the output image
        handleErr(err = set_dst_rast_data(warper, i, wordSize, &dstReal, &dstImag));
        
        //Set the bandNum
        handleErr(err = clSetKernelArg(warper->kern, 12, sizeof(int), &i));
        
        //Run the kernel
        handleErr(err = execute_kern(warper, groupSize));
        
        //Free loop CL mem
        handleErr(err = clReleaseMemObject(srcReal));
        handleErr(err = clReleaseMemObject(srcImag));
        
        //Copy the back output results
        handleErr(err = get_dst_rast_data(warper, i, dstReal, dstImag));
       
        //Free remaining CL mem
        handleErr(err = clReleaseMemObject(dstReal));
        handleErr(err = clReleaseMemObject(dstImag));
    }
    
    //Free remaining CL mem
    handleErr(err = clReleaseMemObject(xy));
    handleErr(err = clReleaseMemObject(unifiedSrcDensityCL));
    handleErr(err = clReleaseMemObject(unifiedSrcValidCL));
    handleErr(err = clReleaseMemObject(useBandSrcValidCL));
    handleErr(err = clReleaseMemObject(nBandSrcValidCL));
    handleErr(err = clReleaseMemObject(dstDensityCL));
    handleErr(err = clReleaseMemObject(dstValidCL));
    handleErr(err = clReleaseMemObject(dstNoDataRealCL));

    return CL_SUCCESS;
}

/*
 Sets pointers to the floating point data in the warper. The pointers
 are internal to the warper structure, so don't free() them. If the imag
 channel is in use, it will receive a pointer. Otherwise it'll be set to NULL.
 These are pointers to floating point data, so the caller will need to
 manipulate the output as appropriate before saving the data.
 
 Returns CL_SUCCESS on success and other CL_* errors when something goes wrong.
 */
cl_int GDALWarpKernelOpenCL_getRow(struct oclWarper *warper,
                                   void **rowReal, void **rowImag,
                                   int rowNum, int bandNum)
{
    int memOff = rowNum * warper->dstWidth;
    
    //Return pointers into the warper's data
    switch (warper->imageFormat) {
        case CL_FLOAT:
            (*rowReal) = &(warper->dstRealWork.f[bandNum][memOff]);
            break;
        case CL_SNORM_INT8:
            (*rowReal) = &(warper->dstRealWork.c[bandNum][memOff]);
            break;
        case CL_UNORM_INT8:
            (*rowReal) = &(warper->dstRealWork.uc[bandNum][memOff]);
            break;
        case CL_SNORM_INT16:
            (*rowReal) = &(warper->dstRealWork.s[bandNum][memOff]);
            break;
        case CL_UNORM_INT16:
            (*rowReal) = &(warper->dstRealWork.us[bandNum][memOff]);
            break;
    }
    
    if (warper->dstImagWorkCL == NULL) {
        (*rowImag) = NULL;
    } else {
        switch (warper->imageFormat) {
            case CL_FLOAT:
                (*rowImag) = &(warper->dstImagWork.f[bandNum][memOff]);
                break;
            case CL_SNORM_INT8:
                (*rowImag) = &(warper->dstImagWork.c[bandNum][memOff]);
                break;
            case CL_UNORM_INT8:
                (*rowImag) = &(warper->dstImagWork.uc[bandNum][memOff]);
                break;
            case CL_SNORM_INT16:
                (*rowImag) = &(warper->dstImagWork.s[bandNum][memOff]);
                break;
            case CL_UNORM_INT16:
                (*rowImag) = &(warper->dstImagWork.us[bandNum][memOff]);
                break;
        }
    }
    
    return CL_SUCCESS;
}

/*
 Free the OpenCL warper environment. It should check everything for NULL, so
 be sure to mark free()ed pointers as NULL or it'll be double free()ed.
 
 Returns CL_SUCCESS on success and other CL_* errors when something goes wrong.
 */
cl_int GDALWarpKernelOpenCL_deleteEnv(struct oclWarper *warper)
{
    int i;
	cl_int err = CL_SUCCESS;
    
    for (i = 0; i < warper->numImages; ++i) {
        // Run free!!
        freeCLMem(warper->realWorkCL[i], warper->realWork.v[i]);
        freeCLMem(warper->dstRealWorkCL[i], warper->dstRealWork.v[i]);
        
        //(As applicable)
        if(warper->imagWork.v != NULL) {
            freeCLMem(warper->imagWorkCL[i], warper->imagWork.v[i]);
        }
        if(warper->dstImagWork.v != NULL) {
            freeCLMem(warper->dstImagWorkCL[i], warper->dstImagWork.v[i]);
        }
    }
    
    //Free cl_mem
    freeCLMem(warper->useBandSrcValidCL, warper->useBandSrcValid);
    freeCLMem(warper->nBandSrcValidCL, warper->nBandSrcValid);
    freeCLMem(warper->xyWorkCL, warper->xyWork);
    freeCLMem(warper->fDstNoDataRealCL, warper->fDstNoDataReal);
    
    //Free pointers to cl_mem*
    if (warper->realWorkCL != NULL)
        CPLFree(warper->realWorkCL);
    if (warper->dstRealWorkCL != NULL)
        CPLFree(warper->dstRealWorkCL);
    
    if (warper->imagWorkCL != NULL)
        CPLFree(warper->imagWorkCL);
    if (warper->dstImagWorkCL != NULL)
        CPLFree(warper->dstImagWorkCL);

    if (warper->realWork.v != NULL)
        CPLFree(warper->realWork.v);
    if (warper->dstRealWork.v != NULL)
        CPLFree(warper->dstRealWork.v);
    
    if (warper->imagWork.v != NULL)
        CPLFree(warper->imagWork.v);
    if (warper->dstImagWork.v != NULL)
        CPLFree(warper->dstImagWork.v);
    
    //Free OpenCL structures
    if (warper->kern != NULL)
        clReleaseKernel(warper->kern);
    if (warper->queue != NULL)
        clReleaseCommandQueue(warper->queue);
    if (warper->context != NULL)
        clReleaseContext(warper->context);
    
    CPLFree(warper);
    
    return CL_SUCCESS;
}