/*******************************************************************************
 r.sun: rsun_opencl.c. This is the OpenCL implimentation of r.sun. It was
 written by Seth Price in 2010 during the Google Summer of Code.
 (C) 2010 Copyright Seth Price
 email: seth@pricepages.org
 *******************************************************************************/
/*
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License
 * as published by the Free Software Foundation; either version 2
 * of the License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the
 *   Free Software Foundation, Inc.,
 *   59 Temple Place - Suite 330,
 *   Boston, MA  02111-1307, USA.
 */

#include "sunradstruct.h"
#include "local_proto.h"
#include "rsunglobals.h"
#include "rsun_opencl.h"

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
cl_device_id get_device(cl_int *clErr)
{
    cl_int err = CL_SUCCESS;
	cl_device_id device = NULL;
#ifndef NDEBUG
    size_t returned_size = 0;
    cl_char vendor_name[1024] = {0};
    cl_char device_name[1024] = {0};
#endif
    
    // Find the GPU CL device, this is what we really want
    // If there is no GPU device is CL capable, fall back to CPU
    //err = clGetDeviceIDs(NULL, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    err = clGetDeviceIDs(NULL, CL_DEVICE_TYPE_CPU, 1, &device, NULL);
    if (err != CL_SUCCESS) {
        // Find the CPU CL device, as a fallback
        err = clGetDeviceIDs(NULL, CL_DEVICE_TYPE_CPU, 1, &device, NULL);
        handleErrRetNULL(err);
    }
    
#ifndef NDEBUG
    // Get some information about the returned device
    err = clGetDeviceInfo(device, CL_DEVICE_VENDOR, sizeof(vendor_name), 
                          vendor_name, &returned_size);
    handleErrRetNULL(err);
    err = clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(device_name), 
                          device_name, &returned_size);
    handleErrRetNULL(err);
    printf("Connecting to %s %s...\n", vendor_name, device_name);
#endif
    
    return device;
}

/*
 Go ahead and execute the kernel. This handles some housekeeping stuff like the
 run dimensions. When running in debug mode, it times the kernel call and prints
 the execution time.
 
 run_kern(cmd_queue, kern, numThreads, groupSize);
 
 Returns CL_SUCCESS on success and other CL_* errors when something goes wrong.
 */
cl_int run_kern(cl_command_queue queue, cl_kernel kern, size_t glob_size, size_t group_size)
{
    cl_int err = CL_SUCCESS;
    cl_event ev;
#ifndef NDEBUG
    size_t start_time = 0;
    size_t end_time;

    handleErr(err = clSetCommandQueueProperty(queue, CL_QUEUE_PROFILING_ENABLE, CL_TRUE, NULL));
#endif
    
    
    
    // Run the calculation by enqueuing it and forcing the 
    // command queue to complete the task
    handleErr(err = clEnqueueNDRangeKernel(queue, kern, 1, NULL, 
                                           &glob_size, &group_size, 0, NULL, &ev));
    handleErr(err = clFinish(queue));
    
#ifndef NDEBUG
    handleErr(err = clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_START,
                                            sizeof(size_t), &start_time, NULL));
    handleErr(err = clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_END,
                                            sizeof(size_t), &end_time, NULL));
    assert(end_time != 0);
    assert(start_time != 0);
    handleErr(err = clReleaseEvent(ev));
    printf("Kernel Time: %10lu\n", (long int)((end_time-start_time)/100000));
#endif
    return CL_SUCCESS;
}

cl_int make_thread_mem_cl(cl_command_queue cmd_queue, cl_context context, cl_kernel kern,
                          unsigned int numThreads, unsigned int locSize,
                          cl_mem *sunGeom_cl, cl_mem *sunVarGeom_cl, cl_mem *sunSlopeGeom_cl)
{
    cl_int err = CL_SUCCESS;
    unsigned int sz;
    
    //Allocate space for structures
    sz = sizeof(float) * numThreads * 8;
    (*sunGeom_cl) = clCreateBuffer(context, CL_MEM_READ_WRITE, sz, NULL, &err);
    printf("%d %d\n", sz, numThreads);
    handleErr(err);
    sz = sizeof(float) * numThreads * 12;
    (*sunVarGeom_cl) = clCreateBuffer(context, CL_MEM_READ_WRITE, sz, NULL, &err);
    handleErr(err);
    sz = sizeof(float) * numThreads * 4;
    (*sunSlopeGeom_cl) = clCreateBuffer(context, CL_MEM_READ_WRITE, sz, NULL, &err);
    handleErr(err);
    
    //Set up these as arguments
    //FIXME: Best case scenario is everything is local mem instead of global, but that probably won't happen. Test, though.
    err = clSetKernelArg(kern, 0, sizeof(cl_mem), sunGeom_cl);
    handleErr(err);
    err = clSetKernelArg(kern, 1, sizeof(cl_mem), sunVarGeom_cl);
    handleErr(err);
    err = clSetKernelArg(kern, 2, sizeof(cl_mem), sunSlopeGeom_cl);
    handleErr(err);
    err = clSetKernelArg(kern, 3, sizeof(float)*locSize*8, NULL);
    handleErr(err);
    
    return CL_SUCCESS;
}

cl_int make_hoz_mem_cl(cl_command_queue cmd_queue, cl_context context, cl_kernel kern,
                       unsigned int numThreads, int useHoz, unsigned char *hozArr,
                       cl_mem *horizon_cl)
{
    cl_int err = CL_SUCCESS;
    unsigned int sz;
    
    //Set up memory for the horizon if needed
    if (useHoz){
        sz = sizeof(unsigned char) * numThreads;
        assert(sz >= 0);
        (*horizon_cl) = clCreateBuffer(context, CL_MEM_READ_ONLY, sz, NULL, &err);
        handleErr(err);
        
        err = clEnqueueWriteBuffer(cmd_queue, (*horizon_cl), CL_TRUE, 0, sz,
                                   (void*)hozArr, 0, NULL, NULL);
        handleErr(err);
    } else {
        //Make a token cl device malloc
        (*horizon_cl) = clCreateBuffer(context, CL_MEM_READ_ONLY, 1, NULL, &err);
        handleErr(err);
    }
    
    //Set up these as arguments
    err = clSetKernelArg(kern, 4, sizeof(cl_mem), horizon_cl);
    handleErr(err);
    
    return CL_SUCCESS;
}

cl_int make_input_raster_cl(cl_command_queue cmd_queue, cl_context context, cl_kernel kern,
                            unsigned int x, unsigned int y, int useData, unsigned int argNum,
                            float **src_gs, cl_mem *dst_cl)
{
    int numThreads = x*y;
    cl_int err;
    
    //Set up work space
    if (useData) {
        int i;
        
        //Allocate full buffers
        unsigned int sz = sizeof(float) * numThreads;
        assert(sz >= 0);
        (*dst_cl) = clCreateBuffer(context, CL_MEM_READ_ONLY, sz, NULL, &err);
        handleErr(err);
        cl_mem src_work_cl = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR, sz, NULL, &err);
        handleErr(err);
        float *src_work = (float *)clEnqueueMapBuffer(cmd_queue, src_work_cl, CL_TRUE, CL_MAP_WRITE,
                                                      0, sz, 0, NULL, NULL, &err);
        handleErr(err);
        
        //Copy data to buffer
        for(i = 0; i < y; ++i)
            memcpy(&(src_work[i*x]), src_gs[i], sizeof(float)*x);
        
        //Copy data to divice memory
        err = clEnqueueWriteBuffer(cmd_queue, (*dst_cl), CL_TRUE, 0, sz,
                                   (void*)src_work, 0, NULL, NULL);
        handleErr(err);
        
        //Clean up mem space
        handleErr(err = clReleaseMemObject(src_work_cl));
    } else {
        //Token memory space
        (*dst_cl) = clCreateBuffer(context, CL_MEM_READ_ONLY, 1, NULL, &err);
        handleErr(err);
    }
    
    //Set it up as an argument
    handleErr(err = clSetKernelArg(kern, argNum, sizeof(cl_mem), dst_cl));
    return CL_SUCCESS;
}

cl_int make_output_raster_cl(cl_command_queue cmd_queue, cl_context context, cl_kernel kern,
                           unsigned int x, unsigned int y, unsigned int useData,
                           unsigned int argNum, cl_mem *out_cl)
{
    int numThreads = x*y;
    cl_int err;
    
    if (useData) {
        //Allocate mem for writing
        unsigned int sz = sizeof(float) * numThreads;
        assert(sz >= 0);
        (*out_cl) = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sz, NULL, &err);
        handleErr(err);
    } else {
        //Token memory space
        (*out_cl) = clCreateBuffer(context, CL_MEM_WRITE_ONLY, 1, NULL, &err);
        handleErr(err);
    }
    
    //Set it up as an argument
    handleErr(err = clSetKernelArg(kern, argNum, sizeof(cl_mem), out_cl));
    
    return CL_SUCCESS;
}

cl_int copy_output_cl(cl_command_queue queue, cl_context context, cl_kernel kern,
                      unsigned int x, unsigned int y, unsigned int hasData,
                      float **dstArr, cl_mem clSrc)
{
    cl_int err;
    int i;
    unsigned int sz = sizeof(float) * x * y;
    
    if (!hasData)
        return CL_SUCCESS;
    
    //Make some pinned working memory
    cl_mem src_work_cl = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR, sz, NULL, &err);
    handleErr(err);
    float *src_work = (float *)clEnqueueMapBuffer(queue, src_work_cl, CL_TRUE, CL_MAP_WRITE,
                                                  0, sz, 0, NULL, NULL, &err);
    handleErr(err);
    
    //Copy data to host
    err = clEnqueueWriteBuffer(queue, clSrc, CL_TRUE, 0, sz, (void*)src_work, 0, NULL, NULL);
    handleErr(err);
    
    //Copy data back to GRASS
    for(i = 0; i < y; ++i)
        memcpy(src_work, &(src_work[i*x]), sizeof(float)*x);
    
    //Clean up mem space
    handleErr(err = clReleaseMemObject(src_work_cl));
    
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
cl_kernel get_kernel(cl_context context, cl_device_id dev,
                     int numThreads,
                     struct OCLConstants *oclConst,
                     struct SolarRadVar *sunRadVar,
                     struct SunGeometryConstDay *sungeom,
                     struct GridGeometry *gridGeom, cl_int *clErr )
{
	cl_program program;
    cl_kernel kernel;
	cl_int err = CL_SUCCESS;
    char *buffer = (char *)calloc(128000, sizeof(char));
    char *progBuf = (char *)calloc(128000, sizeof(char));
    
    const char *kernFunc =
"void com_par_const(float *sunGeom_lum_C11,\n"
                   "float *sunGeom_lum_C13,\n"
                   "float *sunGeom_lum_C22,\n"
                   "float *sunGeom_lum_C31,\n"
                   "float *sunGeom_lum_C33,\n"
                   "float *sunGeom_timeAngle,\n"
                   "float *sunGeom_sunrise_time,\n"
                   "float *sunGeom_sunset_time,\n"
                   "const float gridGeom_sinlat,\n"
                   "const float gridGeom_coslat,\n"
                   "const float longitTime)\n"
"{\n"
    "unsigned int gid = get_global_id(0);\n"
    "unsigned int gsz = n*m;\n"
    
    "*sunGeom_lum_C11 =  gridGeom_sinlat*cosdecl;\n"
    "*sunGeom_lum_C13 = -gridGeom_coslat*sindecl;\n"
    "*sunGeom_lum_C22 = cosdecl;\n"
    "*sunGeom_lum_C31 = gridGeom_coslat*cosdecl;\n"
    "*sunGeom_lum_C33 = gridGeom_sinlat*sindecl;\n"
    
    "if (fabs(*sunGeom_lum_C31) < EPS)\n"
        "return;\n"
    
    "if (civilTimeFlag)\n"
        "*sunGeom_timeAngle -= (timeOffset + longitTime) * HOURANGLE;\n"
    
    "float pom = -(*sunGeom_lum_C33) / (*sunGeom_lum_C31);\n"
    
    "if (fabs(pom) <= 1.0f) {\n"
        "pom = acos(pom) * rad2deg;\n"
        "*sunGeom_sunrise_time = (90.0f - pom) / 15.0f + 6.0f;\n"
        "*sunGeom_sunset_time = (pom - 90.0f) / 15.0f + 18.0f;\n"
    "} else if (pom < 0.0f) {\n"
        //Sun is ABOVE the surface during the whole day
        "*sunGeom_sunrise_time = 0.0f;\n"
        "*sunGeom_sunset_time = 24.0f;\n"
    "} else if (fabs(pom) - 1.0f <= EPS) {\n"
        //The sun is BELOW the surface during the whole day
        "*sunGeom_sunrise_time = 12.0f;\n"
        "*sunGeom_sunset_time = 12.0f;\n"
    "}\n"
"}\n"

"void com_par(float *sunGeom_sunrise_time,\n"
             "float *sunGeom_sunset_time,\n"
             "float *sunVarGeom_solarAltitude,\n"
             "float *sunVarGeom_sinSolarAltitude,\n"
             "float *sunVarGeom_tanSolarAltitude,\n"
             "float *sunVarGeom_solarAzimuth,\n"
             "float *sunVarGeom_sunAzimuthAngle,\n"
             "float *sunVarGeom_stepsinangle,\n"
             "float *sunVarGeom_stepcosangle,\n"
             "const float sunGeom_lum_C11,\n"
             "const float sunGeom_lum_C13,\n"
             "const float sunGeom_lum_C22,\n"
             "const float sunGeom_lum_C31,\n"
             "const float sunGeom_lum_C33,\n"
             "const float sunGeom_timeAngle,\n"
             "const float latitude,\n"
             "const float longitude)\n"
"{\n"
    "float costimeAngle = cos(sunGeom_timeAngle);\n"
    
    "*sunVarGeom_sinSolarAltitude = sunGeom_lum_C31 * costimeAngle + sunGeom_lum_C33;\n"
    
    "if (fabs(sunGeom_lum_C31) < EPS) {\n"
        "if (fabs(*sunVarGeom_sinSolarAltitude) >= EPS) {\n"
            "if (*sunVarGeom_sinSolarAltitude > 0.0f) {\n"
                //Sun is ABOVE area during the whole day
                "*sunGeom_sunrise_time = 0.0f;\n"
                "*sunGeom_sunset_time = 24.0f;\n"
            "} else {\n"
                "*sunVarGeom_solarAltitude = 0.0f;\n"
                "*sunVarGeom_solarAzimuth = UNDEF;\n"
                "return;\n"
            "}\n"
        "} else {\n"
            //The Sun is ON HORIZON during the whole day
            "*sunGeom_sunrise_time = 0.0f;\n"
            "*sunGeom_sunset_time = 24.0f;\n"
        "}\n"
    "}\n"
    
    	/* vertical angle of the sun */
    "*sunVarGeom_solarAltitude = asin(*sunVarGeom_sinSolarAltitude);\n"
    "*sunVarGeom_tanSolarAltitude = tan(*sunVarGeom_solarAltitude);\n"
    
    "float lum_Lx = -sunGeom_lum_C22 * sin(*sunVarGeom_sinSolarAltitude);\n"
    "float lum_Ly = sunGeom_lum_C11 * costimeAngle + sunGeom_lum_C13;\n"
    "float pom = sqrt(lum_Lx*lum_Lx + lum_Ly*lum_Ly);\n"
    
    "if (fabs(pom) > EPS) {\n"
        "*sunVarGeom_solarAzimuth = acos(lum_Ly / pom);\n"	/* horiz. angle of the Sun */
        "if (lum_Lx < 0)\n"
            "*sunVarGeom_solarAzimuth = pi2 - *sunVarGeom_solarAzimuth;\n"
    "} else {\n"
        "*sunVarGeom_solarAzimuth = UNDEF;\n"
    "}\n"
    
    "if (*sunVarGeom_solarAzimuth < 0.5 * PI)\n"
        "*sunVarGeom_sunAzimuthAngle = 0.5f * PI - *sunVarGeom_solarAzimuth;\n"
    "else\n"
        "*sunVarGeom_sunAzimuthAngle = 2.5f * PI - *sunVarGeom_solarAzimuth;\n"
    
    "*sunVarGeom_stepsinangle = stepxy * sin(*sunVarGeom_sunAzimuthAngle);\n"
    "*sunVarGeom_stepcosangle = stepxy * cos(*sunVarGeom_sunAzimuthAngle);\n"
    
    "return;\n"
"}\n"

"float where_is_point(__global float *z,\n"
                     "float *sunVarGeom_zp,\n"
                     "const float gridGeom_xx0,\n"
                     "const float gridGeom_yy0,\n"
                     "const float gridGeom_xg0,\n"
                     "const float gridGeom_yg0,\n"
                     "const float coslatsq)\n"
"{\n"
    //Offset 0.5 cell size to get the right cell i, j
    "int i = gridGeom_xx0 * invstepx + offsetx;\n"
    "int j = gridGeom_yy0 * invstepy + offsety;\n"
    
    //Check bounds
    "if (i > n - 1 || j > m - 1)\n"
        //Return *something*
        "return 9999999.9f;\n"
    
    "float dx = ((float)(i * stepx)) - gridGeom_xg0;\n"
    "float dy = ((float)(j * stepy)) - gridGeom_yg0;\n"
    
    "unsigned int gid = get_global_id(0);\n"
    "unsigned int gsz = n*m;\n"
    
    "*sunVarGeom_zp = z[j*n+i];\n"
    
    //Used to be distance()
    "if (ll_correction)\n"
        "return DEGREEINMETERS * sqrt(coslatsq * dx*dx + dy*dy);\n"
    "else\n"
        "return sqrt(dx*dx + dy*dy);\n"
"}\n"

"int searching(__global float *z,\n"
              "float *sunVarGeom_zp,\n"
              "float *gridGeom_xx0,\n"
              "float *gridGeom_yy0,\n"
              "const float sunVarGeom_z_orig,\n"
              "const float sunVarGeom_tanSolarAltitude,\n"
              "const float sunVarGeom_stepsinangle,\n"
              "const float sunVarGeom_stepcosangle,\n"
              "const float gridGeom_xg0,\n"
              "const float gridGeom_yg0,\n"
              "const float coslatsq)\n"
"{\n"
    "unsigned int gid = get_global_id(0);\n"
    "unsigned int gsz = n*m;\n"
    "int success = 0;\n"
    
    "if (*sunVarGeom_zp == UNDEFZ)\n"
        "return 0;\n"
    
    "*gridGeom_xx0 += sunVarGeom_stepcosangle;\n"
    "*gridGeom_yy0 += sunVarGeom_stepsinangle;\n"
    
    "if (   ((*gridGeom_xx0 + (0.5f * stepx)) < 0.0f)\n"
        "|| ((*gridGeom_xx0 + (0.5f * stepx)) > deltx)\n"
        "|| ((*gridGeom_yy0 + (0.5f * stepy)) < 0.0f)\n"
        "|| ((*gridGeom_yy0 + (0.5f * stepy)) > delty)) {\n"
        
        "success = 3;\n"
    "} else {\n"
        "success = 1;\n"
        
        "float length = where_is_point(z, sunVarGeom_zp,\n"
                        "*gridGeom_xx0, *gridGeom_yy0, gridGeom_xg0, gridGeom_xg0, coslatsq);\n"
        "float z2 = sunVarGeom_z_orig +\n"
                        "EARTHRADIUS * (1.0f - cos(length / EARTHRADIUS)) +\n"
                        "length * sunVarGeom_tanSolarAltitude;\n"
        
        "if (z2 < *sunVarGeom_zp)\n"
            "success = 2;\n"		/* shadow */
        "if (z2 > zmax)\n"
            "success = 3;\n"		/* no test needed all visible */
    "}\n"
    
    "if (success != 1) {\n"
        "*gridGeom_xx0 = gridGeom_xg0;\n"
        "*gridGeom_yy0 = gridGeom_yg0;\n"
    "}\n"
    
    "return success;\n"
"}\n"

"float lumcline2(__global float *horizonArr,\n"
                "__global float *z,\n"
                "int *sunVarGeom_isShadow,\n"
                "float *sunVarGeom_zp,\n"
                "float *gridGeom_xx0,\n"
                "float *gridGeom_yy0,\n"
                "const int horizonOff,\n"
                "const float sunGeom_timeAngle,\n"
                "const float sunVarGeom_z_orig,\n"
                "const float sunVarGeom_solarAltitude,\n"
                "const float sunVarGeom_tanSolarAltitude,\n"
                "const float sunVarGeom_sunAzimuthAngle,\n"
                "const float sunVarGeom_stepsinangle,\n"
                "const float sunVarGeom_stepcosangle,\n"
                "const float sunSlopeGeom_longit_l,\n"
                "const float sunSlopeGeom_lum_C31_l,\n"
                "const float sunSlopeGeom_lum_C33_l,\n"
                "const float gridGeom_xg0,\n"
                "const float gridGeom_yg0,\n"
                "const float coslatsq)\n"
"{\n"
    "float s = 0.0f;\n"
    
    "*sunVarGeom_isShadow = 0;\n"	/* no shadow */
    
    "if (useShadowFlag) {\n"
        "if (useHorizonDataFlag) {\n"
            /* Start is due east, sungeom->timeangle = -pi/2 */
            "float horizPos = sunVarGeom_sunAzimuthAngle / horizonInterval;\n"
            "int lowPos = (int) horizPos;\n"
            "int highPos = lowPos + 1;\n"
            
            "if (highPos == arrayNumInt)\n"
                "highPos = 0;\n"
            
            "float horizonHeight = invScale * ((1.0f - (horizPos - lowPos)) * horizonArr[lowPos+horizonOff]\n"
                                                "+ (horizPos - lowPos) * horizonArr[highPos+horizonOff]);\n"
            
            "*sunVarGeom_isShadow = horizonHeight > sunVarGeom_solarAltitude;\n"
            
            "if (!(*sunVarGeom_isShadow))\n"
                "s = sunSlopeGeom_lum_C31_l * cos(-sunGeom_timeAngle - sunSlopeGeom_longit_l)\n"
                    "+ sunSlopeGeom_lum_C33_l;\n"	/* Jenco */
        "} else {\n"
            "int r;\n"
            "do {\n"
                "r = searching(z, sunVarGeom_zp, gridGeom_xx0, gridGeom_yy0, sunVarGeom_z_orig,\n"
                        "sunVarGeom_tanSolarAltitude, sunVarGeom_stepsinangle,\n"
                        "sunVarGeom_stepcosangle, gridGeom_yg0, gridGeom_yg0, coslatsq);\n"
            "} while (r == 1);\n"
            
            "if (r == 2)\n"
                "*sunVarGeom_isShadow = 1;\n"	/* shadow */
            "else\n"
                "s = sunSlopeGeom_lum_C31_l * cos(-sunGeom_timeAngle - sunSlopeGeom_longit_l)\n"
                    "+ sunSlopeGeom_lum_C33_l;\n"	/* Jenco */
        "}\n"
    "} else {\n"
        "s = sunSlopeGeom_lum_C31_l * cos(-sunGeom_timeAngle - sunSlopeGeom_longit_l)\n"
            "+ sunSlopeGeom_lum_C33_l;\n"	/* Jenco */
    "}\n"
    
    "if (s < 0.0f)\n"
        "return 0.0f;\n"
    "else\n"
        "return s;\n"
"}\n"

"float brad(__global float *s,\n"
           "__global float *li,\n"
           "__global float *cbhr,\n"
           "float *bh,\n"
           "const float sunVarGeom_z_orig\n,"
           "const float sunVarGeom_solarAltitude\n,"
           "const float sunVarGeom_sinSolarAltitude\n,"
           "const float sunSlopeGeom_aspect\n,"
           "const float sh)\n"
"{\n"
    "unsigned int gid = get_global_id(0);\n"
    "float h0refract = sunVarGeom_solarAltitude + 0.061359f *\n"
        "(0.1594f + sunVarGeom_solarAltitude * (1.123f + 0.065656f * sunVarGeom_solarAltitude)) /\n"
        "(1.0f + sunVarGeom_solarAltitude * (28.9344f + 277.3971f * sunVarGeom_solarAltitude));\n"
    
    "float opticalAirMass = exp(-sunVarGeom_z_orig / 8434.5f) / (sin(h0refract) +\n"
                        "0.50572f * pow(h0refract * rad2deg + 6.07995f, -1.6364f));\n"
    "float rayl, bhc, slope, linke;\n"
    
    "if(slopein)\n"
        "slope = singleSlope;\n"
    "else\n"
        "slope = s[gid] * deg2rad;\n"
    
    "if(linkein)\n"
        "linke = singleLinke;\n"
    "else\n"
        "linke = li[gid];\n"
    
    "if(coefbh)\n"
        "bhc = 1.0f;\n"
    "else\n"
        "bhc = cbhr[gid];\n"
    
    "if (opticalAirMass <= 20.0f)\n"
        "rayl = 1.0f / (6.6296f + opticalAirMass * (1.7513f + opticalAirMass *\n"
                "(-0.1202f + opticalAirMass * (0.0065f - opticalAirMass * 0.00013f))));\n"
    "else\n"
        "rayl = 1.0f / (10.4f + 0.718f * opticalAirMass);\n"
    
    "*bh = bhc * G_norm_extra * sunVarGeom_sinSolarAltitude *\n"
                    "exp(-rayl * opticalAirMass * 0.8662f * linke);\n"
    
    "if (sunSlopeGeom_aspect != UNDEF && slope != 0.0f)\n"
        "return *bh * sh / sunVarGeom_sinSolarAltitude;\n"
    "else\n"
        "return *bh;\n"
"}\n"

"float drad(__global float *s,\n"
           "__global float *li,\n"
           "__global float *a,\n"
           "__global float *cdhr,\n"
           
           "float *rr,\n"
           "const int sunVarGeom_isShadow,\n"
           "const float sunVarGeom_solarAltitude,\n"
           "const float sunVarGeom_sinSolarAltitude,\n"
           "const float sunVarGeom_solarAzimuth,\n"
           "const float sunSlopeGeom_aspect,\n"
           "const float sh,\n"
           "const float bh)\n"
"{\n"
    "unsigned int gid = get_global_id(0);\n"
    "float A1, gh, fg, slope, linke;\n"
    
    "if(slopein)\n"
        "slope = singleSlope;\n"
    "else\n"
        "slope = s[gid] * deg2rad;\n"
    
    "if(linkein)\n"
        "linke = singleLinke;\n"
    "else\n"
        "linke = li[gid];\n"
    
    "float dhc;\n"
    
    "float tn = -0.015843f + linke * (0.030543f + 0.0003797f * linke);\n"
    "float A1b = 0.26463f + linke * (-0.061581f + 0.0031408f * linke);\n"
    
    "if(coefdh)\n"
        "dhc = 1.0f;\n"
    "else\n"
        "dhc = cdhr[gid];\n"
    
    "if (A1b * tn < 0.0022f)\n"
        "A1 = 0.0022f / tn;\n"
    "else\n"
        "A1 = A1b;\n"

    "float dh = (A1 + (2.04020f + linke * (0.018945f - 0.011161f * linke)) *\n"
                        "sunVarGeom_sinSolarAltitude +\n"
                "(-1.3025f + linke * (0.039231f + 0.0085079f * linke)) *\n"
                        "sunVarGeom_sinSolarAltitude * sunVarGeom_sinSolarAltitude) *\n"
                "dhc * G_norm_extra * tn;\n"
    
    "if (sunSlopeGeom_aspect != UNDEF && slope != 0.0f) {\n"
        "float cosslope = cos(slope);\n"
        "float sinslope = sin(slope);\n"
        "float sinHalfSlope = sin(0.5f * slope);\n"
        "float fg = sinslope - slope * cosslope - PI * sinHalfSlope * sinHalfSlope;\n"
        "float r_sky = (1.0f + cosslope) * 0.5f;\n"
        "float kb = bh / (G_norm_extra * sunVarGeom_sinSolarAltitude);\n"
        "float fx = 0.0f;\n"
        "float alb;\n"
        
        "if(albedo)\n"
            "alb = singleAlbedo;\n"
        "else\n"
            "alb = a[gid];\n"
        
        "if (sunVarGeom_isShadow == 1 || sh <= 0.0f)\n"
            "fx = r_sky + fg * 0.252271f;\n"
        "else if (sunVarGeom_solarAltitude >= 0.1f)\n"
            "fx = ((0.00263f - kb * (0.712f + 0.6883f * kb)) * fg + r_sky) *\n"
                "(1.0f - kb) + kb * sh / sunVarGeom_sinSolarAltitude;\n"
        "else if (sunVarGeom_solarAltitude < 0.1f) {\n"
            "float a_ln = sunVarGeom_solarAzimuth - sunSlopeGeom_aspect;\n"
            
            "if (a_ln > M_PI)\n"
                "a_ln -= deg2rad;\n"
            "else if (a_ln < -M_PI)\n"
                "a_ln += deg2rad;\n"
            
            "fx = ((0.00263f - 0.712f * kb - 0.6883f * kb * kb) * fg + r_sky) *\n"
                "(1.0f - kb) + kb * sinslope * cos(a_ln) /\n"
                "(0.1f - 0.008f * sunVarGeom_solarAltitude);\n"
        "}\n"
        
       /* refl. rad */
        "(*rr) = alb * (bh + dh) * (1.0f - cosslope) * 0.5f;\n"
        "return dh * fx;\n"
    "} else {\n"	/* plane */
        "(*rr) = 0.0f;\n"
        "return dh;\n"
    "}\n"
"}\n"
 
"__kernel void calculate(__global float *sunGeom,\n"
                       "__global float *sunVarGeom,\n"
                       
                       "__global float *horizonArr,\n"
                       "__global float *z,\n"
                       "__global float *o,\n"
                       "__global float *s,\n"
                       "__global float *li,\n"
                       "__global float *a,\n"
                       "__global float *latitudeArray,\n"
                       "__global float *longitudeArray,\n"
                       "__global float *cbhr,\n"
                       "__global float *cdhr,\n"

                       "__global float *lumcl,\n"
                       "__global float *beam,\n"
                       "__global float *globrad,\n"
                       "__global float *insol,\n"
                       "__global float *diff,\n"
                       "__global float *refl )\n"
"{\n"
    "unsigned int gid = get_global_id(0);\n"
    "unsigned int gsz = n*m;\n"
    "unsigned int lid = get_local_id(0);\n"
    "unsigned int lsz = get_local_size(0);\n"
    "float longitTime = 0.0f;\n"
    "float o_orig, coslatsq;\n"
    "float gridGeom_xx0, gridGeom_yy0, gridGeom_xg0, gridGeom_yg0;\n"
    
    //Don't overrun arrays
    "if (gid >= gsz)\n"
        "return;\n"
    
    "if (civilTimeFlag)\n"
        "longitTime = -longitudeArray[gid] / 15.0f;\n"
    
    "gridGeom_xg0 = gridGeom_xx0 = stepx * gid / (float)m;\n"
    "gridGeom_yg0 = gridGeom_yy0 = (gid % m) * stepy;\n"
    
    "float gridGeom_xp = xmin + gridGeom_xx0;\n"
    "float gridGeom_yp = ymin + gridGeom_yy0;\n"
    
    "if (ll_correction) {\n"
        "float coslat = cos(deg2rad * gridGeom_yp);\n"
        "coslatsq = coslat * coslat;\n"
    "}\n"
    
    "float sunVarGeom_z_orig, z1, sunVarGeom_zp;\n"
    "sunVarGeom_z_orig = z1 = sunVarGeom_zp = z[gid];\n"
    
    "if (z1 == UNDEFZ)\n"
        "return;\n"
    
    "float latitude, longitude, aspect, slope;\n"
    
    "if (latin)\n"
        "latitude = latitudeArray[gid] * deg2rad;\n"
    
    "if (longin)\n"
        "longitude = longitudeArray[gid] * deg2rad;\n"
    
    "if (proj_eq_ll) {\n"		/* ll projection */
        "longitude = gridGeom_xp * deg2rad;\n"
        "latitude  = gridGeom_yp * deg2rad;\n"
    "}\n"
    
    "if (slopein)\n"
        "slope = singleSlope;\n"
    "else\n"
        "slope = s[gid];\n"
        
    "float cos_u = cos(pihalf - slope);\n"
    "float sin_u = sin(pihalf - slope);\n"
    "float cos_v = cos(pihalf + aspect);\n"
    "float sin_v = sin(pihalf + aspect);\n"
    "float sunGeom_timeAngle;\n"
    
    "if (ttime)\n"
        "sunGeom_timeAngle = tim;\n"
    
    "float gridGeom_sinlat = sin(-latitude);\n"
    "float gridGeom_coslat = cos(-latitude);\n"
    "float sin_phi_l = -gridGeom_coslat * cos_u * sin_v + gridGeom_sinlat * sin_u;\n"
    
    "float sunGeom_lum_C11, sunGeom_lum_C13, sunGeom_lum_C22;\n"
    "float sunGeom_lum_C31, sunGeom_lum_C33;\n"
    "float sunGeom_sunrise_time, sunGeom_sunset_time;\n"
    
    "if (incidout || someRadiation)\n"
        "com_par_const(&sunGeom_lum_C11, &sunGeom_lum_C13, &sunGeom_lum_C22,\n"
                      "&sunGeom_lum_C31, &sunGeom_lum_C33, &sunGeom_timeAngle,\n"
                      "&sunGeom_sunrise_time, &sunGeom_sunset_time,\n"
                      "gridGeom_sinlat, gridGeom_coslat, longitTime);\n"
    
    "float sunSlopeGeom_longit_l = atan(-cos_u * cos_v / (gridGeom_sinlat * cos_u * sin_v + gridGeom_coslat * sin_u));\n"
    "float sunSlopeGeom_lum_C31_l = cos(asin(sin_phi_l)) * cosdecl;\n"
    "float sunSlopeGeom_lum_C33_l = sin_phi_l * sindecl;\n"
    
    "float sunVarGeom_solarAltitude, sunVarGeom_sinSolarAltitude;\n"
    "float sunVarGeom_tanSolarAltitude, sunVarGeom_solarAzimuth;\n"
    "float sunVarGeom_stepsinangle, sunVarGeom_stepcosangle;\n"
    
    "int sunVarGeom_isShadow;\n"
    "float sunVarGeom_sunAzimuthAngle;\n"
    
    "if (incidout) {\n"
        "com_par(&sunGeom_sunrise_time, &sunGeom_sunset_time,\n"
                "&sunVarGeom_solarAltitude, &sunVarGeom_sinSolarAltitude,\n"
                "&sunVarGeom_tanSolarAltitude, &sunVarGeom_solarAzimuth,\n"
                "&sunVarGeom_sunAzimuthAngle,\n"
                "&sunVarGeom_stepsinangle, &sunVarGeom_stepcosangle,\n"
                "sunGeom_lum_C11, sunGeom_lum_C13, sunGeom_lum_C22,\n"
                "sunGeom_lum_C31, sunGeom_lum_C33, sunGeom_timeAngle,\n"
                "latitude, longitude);\n"

        "float lum = lumcline2(horizonArr, z,\n"
                              "&sunVarGeom_isShadow, &sunVarGeom_zp,\n"
                              "&gridGeom_xx0, &gridGeom_yy0, gid*arrayNumInt,\n"
                              "sunGeom_timeAngle, sunVarGeom_z_orig, sunVarGeom_solarAltitude,\n"
                              "sunVarGeom_tanSolarAltitude, sunVarGeom_sunAzimuthAngle,\n"
                              "sunVarGeom_stepsinangle, sunVarGeom_stepcosangle, sunSlopeGeom_longit_l,\n"
                              "sunSlopeGeom_lum_C31_l, sunSlopeGeom_lum_C33_l,\n"
                              "gridGeom_xg0, gridGeom_yg0, coslatsq);\n"
        
        "if (lum > 0.0f) {\n"
            "lumcl[gid] = rad2deg * asin(lum);\n"
        "} else {\n"
            "lumcl[gid] = UNDEFZ;\n"
        "}\n"
    "}\n"

    "if (someRadiation) {\n"
        //joules2() is inlined so I don't need to pass in basically *everything*
		"float sunSlopeGeom_aspect;\n"
    
		"if (aspin) {\n"
			"if (o[gid] != 0.0f)\n"
				"sunSlopeGeom_aspect = o[gid] * deg2rad;\n"
			"else\n"
				"sunSlopeGeom_aspect = UNDEF;\n"
		"} else {\n"
			"sunSlopeGeom_aspect = singleAspect;\n"
		"}\n"
	
		//Double precision so summation works better (shouldn't slow much)
		"double beam_e = 0.0;\n"
		"double diff_e = 0.0;\n"
		"double refl_e = 0.0;\n"
		"double insol_t = 0.0;\n"
		"int insol_count = 0;\n"
		
        "com_par(&sunGeom_sunrise_time, &sunGeom_sunset_time,\n"
                "&sunVarGeom_solarAltitude, &sunVarGeom_sinSolarAltitude,\n"
                "&sunVarGeom_tanSolarAltitude, &sunVarGeom_solarAzimuth,\n"
                "&sunVarGeom_sunAzimuthAngle,\n"
                "&sunVarGeom_stepsinangle, &sunVarGeom_stepcosangle,\n"
                "sunGeom_lum_C11, sunGeom_lum_C13, sunGeom_lum_C22,\n"
                "sunGeom_lum_C31, sunGeom_lum_C33, sunGeom_timeAngle,\n"
                "latitude, longitude);\n"
    
		"if (ttime) {\n"		/*irradiance */
            "float s0 = lumcline2(horizonArr, z,\n"
                    "&sunVarGeom_isShadow, &sunVarGeom_zp,\n"
                    "&gridGeom_xx0, &gridGeom_yy0, gid*arrayNumInt,\n"
                    "sunGeom_timeAngle, sunVarGeom_z_orig, sunVarGeom_solarAltitude,\n"
                    "sunVarGeom_tanSolarAltitude, sunVarGeom_sunAzimuthAngle,\n"
                    "sunVarGeom_stepsinangle, sunVarGeom_stepcosangle, sunSlopeGeom_longit_l,\n"
                    "sunSlopeGeom_lum_C31_l, sunSlopeGeom_lum_C33_l,\n"
                    "gridGeom_xg0, gridGeom_yg0, coslatsq);\n"
    
			"if (sunVarGeom[4*gsz+gid] > 0.0f) {\n"
				"float bh;\n"
				"if (sunVarGeom[gid] < 0.5f && s0 > 0.0f) {\n"
					"beam_e = brad(s, li, cbhr, &bh, sunVarGeom_z_orig,\n"
                            "sunVarGeom_solarAltitude, sunVarGeom_sinSolarAltitude,\n"
                            "sunSlopeGeom_aspect, s0);\n"	/* beam radiation */
				"} else {\n"
					"beam_e = 0.0f;\n"
					"bh = 0.0f;\n"
				"}\n"
				
				"float rr = 0.0f;\n"
				"if (diff_rad || glob_rad)\n"
					"diff_e = drad(s, li, a, cdhr, &rr, sunVarGeom_isShadow,\n"
                            "sunVarGeom_solarAltitude, sunVarGeom_sinSolarAltitude,\n"
                            "sunVarGeom_solarAzimuth, sunSlopeGeom_aspect, s0, bh);\n"	/* diffuse rad. */
		
				"if (refl_rad || glob_rad) {\n"
					"if (diff_rad && glob_rad)\n"
                        "drad(s, li, a, cdhr, &rr, sunVarGeom_isShadow,\n"
                            "sunVarGeom_solarAltitude, sunVarGeom_sinSolarAltitude,\n"
                            "sunVarGeom_solarAzimuth, sunSlopeGeom_aspect, s0, bh);\n"
					"refl_e = rr;\n"	/* reflected rad. */
				"}\n"
			"}\n"			/* solarAltitude */
		"} else {\n"
			/* all-day radiation */
			"float sunrise_time = sunGeom[5*gsz+gid];\n"
			"int srStepNo = sunrise_time / timeStep;\n"
			"float lastAngle = (sunGeom[6*gsz+gid] - 12.0f) * HOURANGLE;\n"
			"float firstTime;\n"
			
			"if ((sunrise_time - srStepNo * timeStep) > 0.5f * timeStep)\n"
				"firstTime = (srStepNo + 1.5f) * timeStep;\n"
			"else\n"
				"firstTime = (srStepNo + 0.5f) * timeStep;\n"
			
			"sunGeom_timeAngle = (firstTime - 12.0f) * HOURANGLE;\n"
			
			"do {\n"
                "com_par(&sunGeom_sunrise_time, &sunGeom_sunset_time,\n"
                        "&sunVarGeom_solarAltitude, &sunVarGeom_sinSolarAltitude,\n"
                        "&sunVarGeom_tanSolarAltitude, &sunVarGeom_solarAzimuth,\n"
                        "&sunVarGeom_sunAzimuthAngle,\n"
                        "&sunVarGeom_stepsinangle, &sunVarGeom_stepcosangle,\n"
                        "sunGeom_lum_C11, sunGeom_lum_C13, sunGeom_lum_C22,\n"
                        "sunGeom_lum_C31, sunGeom_lum_C33, sunGeom_timeAngle,\n"
                        "latitude, longitude);\n"
    
                "float s0 = lumcline2(horizonArr, z,\n"
                        "&sunVarGeom_isShadow, &sunVarGeom_zp,\n"
                        "&gridGeom_xx0, &gridGeom_yy0, gid*arrayNumInt,\n"
                        "sunGeom_timeAngle, sunVarGeom_z_orig, sunVarGeom_solarAltitude,\n"
                        "sunVarGeom_tanSolarAltitude, sunVarGeom_sunAzimuthAngle,\n"
                        "sunVarGeom_stepsinangle, sunVarGeom_stepcosangle, sunSlopeGeom_longit_l,\n"
                        "sunSlopeGeom_lum_C31_l, sunSlopeGeom_lum_C33_l,\n"
                        "gridGeom_xg0, gridGeom_yg0, coslatsq);\n"
    
				"if (sunVarGeom_solarAltitude > 0.0f) {\n"
					"float bh;\n"
					"if (sunVarGeom[gid] < 0.5f && s0 > 0.0f) {\n"
						"++insol_count;\n"
                        "beam_e += timeStep * brad(s, li, cbhr, &bh, sunVarGeom_z_orig,\n"
                                "sunVarGeom_solarAltitude, sunVarGeom_sinSolarAltitude,\n"
                                "sunSlopeGeom_aspect, s0);\n"
                    "} else {\n"
						"bh = 0.0f;\n"
					"}\n"
					
					"float rr = 0.0f;\n"
					"if (diff_rad || glob_rad)\n"
                        "diff_e += timeStep * drad(s, li, a, cdhr, &rr, sunVarGeom_isShadow,\n"
                                "sunVarGeom_solarAltitude, sunVarGeom_sinSolarAltitude,\n"
                                "sunVarGeom_solarAzimuth, sunSlopeGeom_aspect, s0, bh);\n"
					"if (refl_rad || glob_rad) {\n"
						"if (diff_rad && glob_rad)\n"
                            "drad(s, li, a, cdhr, &rr, sunVarGeom_isShadow,\n"
                                    "sunVarGeom_solarAltitude, sunVarGeom_sinSolarAltitude,\n"
                                    "sunVarGeom_solarAzimuth, sunSlopeGeom_aspect, s0, bh);\n"
						"refl_e += timeStep * rr;\n"
					"}\n"
				"}\n"			/* illuminated */
				
				"sunGeom_timeAngle += timeStep * HOURANGLE;\n"
			"} while (sunGeom_timeAngle > lastAngle);\n" /* we've got the sunset */
		"}\n"				/* all-day radiation */
		
		//Only apply values to where they're wanted
		"if(beam_rad)\n"
			"beam[gid] = beam_e;\n"
		"if(insol_time)\n"
		   "insol[gid] = timeStep*insol_count;\n"
		"if(diff_rad)\n"
			"diff[gid] = diff_e;\n"
		"if(refl_rad)\n"
			"refl[gid] = refl_e;\n"
		"if(glob_rad)\n"
			"globrad[gid] = beam_e + diff_e + refl_e;\n"
    "}\n"
"}\n";
    
    //Actually make the program from assembled source
    program = clCreateProgramWithSource(context, 1, (const char**)&kernFunc,
                                        NULL, &err);
    handleErrRetNULL(err);
    
    //Assemble the compiler arg string for speed. All invariants should be defined here.
    sprintf(buffer, "-cl-fast-relaxed-math -Werror -D FALSE=0 -D TRUE=1 "
            "-D invScale=%ff -D pihalf=%ff -D pi2=%ff -D deg2rad=%ff -D rad2deg=%ff "
            "-D invstepx=%ff -D invstepy=%ff -D xmin=%ff -D ymin=%ff -D xmax=%ff "
            "-D ymax=%ff -D civilTime=%ff -D timeStep=%ff -D horizonStep=%ff "
            "-D stepx=%ff -D stepy=%ff -D deltx=%ff -D delty=%ff "
            "-D stepxy=%ff -D horizonInterval=%ff -D singleLinke=%ff "
            "-D singleAlbedo=%ff -D singleSlope=%ff -D singleAspect=%ff -D cbh=%ff "
            "-D cdh=%ff -D dist=%ff -D TOLER=%ff -D offsetx=%ff -D offsety=%ff "
            "-D declination=%ff -D G_norm_extra=%ff -D timeOffset=%ff -D sindecl=%ff "
            "-D cosdecl=%ff -D zmax=%ff -D n=%d -D m=%d -D saveMemory=%d -D civilTimeFlag=%d "
            "-D day=%d -D ttime=%d -D numPartitions=%d -D arrayNumInt=%d -D tim=%d "
            "-D proj_eq_ll=%d -D someRadiation=%d -D numRows=%d -D numThreads=%d "
            "-D ll_correction=%d -D aspin=%d -D slopein=%d -D linkein=%d -D albedo=%d -D latin=%d "
            "-D longin=%d -D coefbh=%d -D coefdh=%d -D incidout=%d -D beam_rad=%d "
            "-D insol_time=%d -D diff_rad=%d -D refl_rad=%d -D glob_rad=%d "
            "-D useShadowFlag=%d -D useHorizonDataFlag=%d -D EPS=%ff -D HOURANGLE=%ff "
            "-D PI=%ff -D DEGREEINMETERS=%ff -D UNDEFZ=%ff -D EARTHRADIUS=%ff -D UNDEF=%ff",
            invScale, pihalf, pi2, deg2rad, rad2deg,
            oclConst->invstepx, oclConst->invstepy, oclConst->xmin, oclConst->ymin, oclConst->xmax,
            oclConst->ymax, oclConst->civilTime, oclConst->step, oclConst->horizonStep,
            gridGeom->stepx, gridGeom->stepy, gridGeom->deltx, gridGeom->delty,
            gridGeom->stepxy, getHorizonInterval(), oclConst->singleLinke,
            oclConst->singleAlbedo, oclConst->singleSlope, oclConst->singleAspect, oclConst->cbh,
            oclConst->cdh, oclConst->dist, oclConst->TOLER, oclConst->offsetx, oclConst->offsety,
            oclConst->declination, sunRadVar->G_norm_extra, getTimeOffset(), sungeom->sindecl,
            sungeom->cosdecl, oclConst->zmax, oclConst->n, oclConst->m, oclConst->saveMemory, useCivilTime(),
            oclConst->day, oclConst->ttime, oclConst->numPartitions, arrayNumInt, oclConst->tim,
            oclConst->proj_eq_ll, oclConst->someRadiation, oclConst->numRows, numThreads,
            oclConst->ll_correction, oclConst->aspin, oclConst->slopein, oclConst->linkein, oclConst->albedo, oclConst->latin,
            oclConst->longin, oclConst->coefbh, oclConst->coefdh, oclConst->incidout, oclConst->beam_rad,
            oclConst->insol_time, oclConst->diff_rad, oclConst->refl_rad, oclConst->glob_rad,
            useShadow(), useHorizonData(), EPS, HOURANGLE,
            M_PI, oclConst->degreeInMeters, UNDEFZ, EARTHRADIUS, UNDEF);
    
    (*clErr) = err = clBuildProgram(program, 1, &(dev), buffer, NULL, NULL);
    
    //Detailed debugging info
    if (err != CL_SUCCESS)
    {
        err = clGetProgramBuildInfo(program, dev, CL_PROGRAM_BUILD_LOG,
                                    128000*sizeof(char), buffer, NULL);
        handleErrRetNULL(err);
        
        printf("Build Log:\n%s\n", buffer);
        printf("Error: Failed to build program executable!\n");
        printCLErr(err);
        
        printf("%s\n", buffer);
        
        err = clGetProgramBuildInfo(program, dev, CL_PROGRAM_BUILD_STATUS,
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
    
    kernel = clCreateKernel(program, "calculate", &err);
    handleErrRetNULL(err);
    
    err = clReleaseProgram(program);
    handleErrRetNULL(err);
    
    free(buffer);
    free(progBuf);
    return kernel;
}

cl_int calculate_core_cl(int x, int y,
                         struct OCLConstants *oclConst,
                         struct SolarRadVar *sunRadVar,
                         struct SunGeometryConstDay *sunGeom,
                         struct GridGeometry *gridGeom,
                         unsigned char *horizonarray,
                         
                         float **z, float **o, float **s, float **li, float **a,
                         float **latitudeArray, float **longitudeArray,
                         float **cbhr, float **cdhr,
                         
                         float **lumcl, float **beam, float **insol,
                         float **diff, float **refl, float **globrad )
{
	cl_command_queue cmd_queue;
	cl_context context;
    cl_kernel kern;
	size_t groupSize;
    int numThreads = x*y;
    cl_int err;
    
    cl_device_id dev = get_device(&err);

    cl_mem  sunGeom_cl, sunVarGeom_cl, sunSlopeGeom_cl,
            horizon_cl, z_cl, o_cl, s_cl, li_cl, a_cl, lat_cl, long_cl, cbhr_cl, cdhr_cl,
            lumcl_cl, beam_cl, globrad_cl, insol_cl, diff_cl, refl_cl;
    
    // Now create a context to perform our calculation with the specified device 
    context = clCreateContext(0, 1, &dev, NULL, NULL, &err);
    handleErr(err);
    
    // And also a command queue for the context
    cmd_queue = clCreateCommandQueue(context, dev, 0, &err);
    handleErr(err);
    
	kern = get_kernel(context, dev, numThreads,
                      oclConst, sunRadVar, sunGeom, gridGeom, &err);
    handleErr(err);

    //What's the recommended group size?
	err = clGetKernelWorkGroupInfo(kern, dev, CL_KERNEL_WORK_GROUP_SIZE,
                                   sizeof(size_t), &groupSize, NULL);
    handleErr(err);
#ifndef NDEBUG
	printf("Recommended Size: %lu\n", groupSize);
#endif
    
    //Allocate and copy all the inputs
    make_thread_mem_cl(cmd_queue, context, kern, numThreads, groupSize,
                       &sunGeom_cl, &sunVarGeom_cl, &sunSlopeGeom_cl);
    
    make_hoz_mem_cl(cmd_queue, context, kern, numThreads, useHorizonData(), horizonarray, &horizon_cl);
    make_input_raster_cl(cmd_queue, context, kern, x, y, 1, 5, z, &z_cl);
    make_input_raster_cl(cmd_queue, context, kern, x, y, oclConst->aspin, 6, o, &o_cl);
    make_input_raster_cl(cmd_queue, context, kern, x, y, oclConst->slopein, 7, s, &s_cl);
    make_input_raster_cl(cmd_queue, context, kern, x, y, oclConst->linkein, 8, li, &li_cl);
    make_input_raster_cl(cmd_queue, context, kern, x, y, oclConst->albedo, 9, a, &a_cl);
    make_input_raster_cl(cmd_queue, context, kern, x, y, oclConst->latin, 10, latitudeArray, &lat_cl);
    make_input_raster_cl(cmd_queue, context, kern, x, y, oclConst->longin, 11, longitudeArray, &long_cl);
    make_input_raster_cl(cmd_queue, context, kern, x, y, oclConst->coefbh, 12, cbhr, &cbhr_cl);
    make_input_raster_cl(cmd_queue, context, kern, x, y, oclConst->coefdh, 13, cdhr, &cdhr_cl);
    
    //Make space for the outputs
    make_output_raster_cl(cmd_queue, context, kern, x, y, oclConst->incidout, 14, &lumcl_cl);
    make_output_raster_cl(cmd_queue, context, kern, x, y, oclConst->beam_rad, 15, &beam_cl);
    make_output_raster_cl(cmd_queue, context, kern, x, y, oclConst->glob_rad, 16, &globrad_cl);
    make_output_raster_cl(cmd_queue, context, kern, x, y, oclConst->insol_time, 17, &insol_cl);
    make_output_raster_cl(cmd_queue, context, kern, x, y, oclConst->diff_rad, 18, &diff_cl);
    make_output_raster_cl(cmd_queue, context, kern, x, y, oclConst->refl_rad, 19, &refl_cl);
    
    //Do the dirty work
    run_kern(cmd_queue, kern, numThreads, groupSize);
    
    //Release unneeded inputs
    handleErr(err = clReleaseMemObject(sunGeom_cl));
    handleErr(err = clReleaseMemObject(sunVarGeom_cl));
    handleErr(err = clReleaseMemObject(sunSlopeGeom_cl));
    
    handleErr(err = clReleaseMemObject(horizon_cl));
    handleErr(err = clReleaseMemObject(z_cl));
    handleErr(err = clReleaseMemObject(o_cl));
    handleErr(err = clReleaseMemObject(s_cl));
    handleErr(err = clReleaseMemObject(li_cl));
    handleErr(err = clReleaseMemObject(a_cl));
    handleErr(err = clReleaseMemObject(lat_cl));
    handleErr(err = clReleaseMemObject(long_cl));
    handleErr(err = clReleaseMemObject(cbhr_cl));
    handleErr(err = clReleaseMemObject(cdhr_cl));
    
    //Copy requested outputs 
    copy_output_cl(cmd_queue, context, kern, x, y, oclConst->incidout, lumcl, lumcl_cl);
    copy_output_cl(cmd_queue, context, kern, x, y, oclConst->beam_rad, beam, beam_cl);
    copy_output_cl(cmd_queue, context, kern, x, y, oclConst->insol_time, insol, insol_cl);
    copy_output_cl(cmd_queue, context, kern, x, y, oclConst->diff_rad, diff, diff_cl);
    copy_output_cl(cmd_queue, context, kern, x, y, oclConst->refl_rad, refl, refl_cl);
    copy_output_cl(cmd_queue, context, kern, x, y, oclConst->glob_rad, globrad, globrad_cl);
    
    //Release remaining resources
    handleErr(err = clReleaseMemObject(lumcl_cl));
    handleErr(err = clReleaseMemObject(beam_cl));
    handleErr(err = clReleaseMemObject(globrad_cl));
    handleErr(err = clReleaseMemObject(insol_cl));
    handleErr(err = clReleaseMemObject(diff_cl));
    handleErr(err = clReleaseMemObject(refl_cl));
    
    handleErr(err = clReleaseKernel(kern));
    handleErr(err = clReleaseCommandQueue(cmd_queue));
    handleErr(err = clReleaseContext(context));
    
    return CL_SUCCESS;
}
