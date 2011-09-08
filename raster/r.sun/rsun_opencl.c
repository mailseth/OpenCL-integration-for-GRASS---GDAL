/*******************************************************************************
 r.sun: rsun_opencl.c. This is the OpenCL implimentation of r.sun. It was
 written by Seth Price in 2010 for the Google Summer of Code.
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

#define  NUM_OPENCL_PARTITIONS 32

//These macros kill GRASS with G_fatal_error()
#define handleErr(err) if((err) != CL_SUCCESS) { \
    printCLErr(__FILE__, __LINE__, err); \
    return err; \
}

#define handleErrRetNULL(err) if((err) != CL_SUCCESS) { \
    (*clErr) = err; \
    printCLErr(__FILE__, __LINE__, err); \
    return NULL; \
}

/*
 Handle a OpenCL error by printing the file, line, then error. The trigger a
 fatal error.
 */
void printCLErr(char *fName, unsigned int lineNum, cl_int err)
{
    char *errStr = "";
    switch (err)
    {
        case CL_SUCCESS:
            errStr = "CL_SUCCESS";
            break;
        case CL_DEVICE_NOT_FOUND:
            errStr = "CL_DEVICE_NOT_FOUND";
            break;
        case CL_DEVICE_NOT_AVAILABLE:
            errStr = "CL_DEVICE_NOT_AVAILABLE";
            break;
        case CL_COMPILER_NOT_AVAILABLE:
            errStr = "CL_COMPILER_NOT_AVAILABLE";
            break;
        case CL_MEM_OBJECT_ALLOCATION_FAILURE:
            errStr = "CL_MEM_OBJECT_ALLOCATION_FAILURE";
            break;
        case CL_OUT_OF_RESOURCES:
            errStr = "CL_OUT_OF_RESOURCES";
            break;
        case CL_OUT_OF_HOST_MEMORY:
            errStr = "CL_OUT_OF_HOST_MEMORY";
            break;
        case CL_PROFILING_INFO_NOT_AVAILABLE:
            errStr = "CL_PROFILING_INFO_NOT_AVAILABLE";
            break;
        case CL_MEM_COPY_OVERLAP:
            errStr = "CL_MEM_COPY_OVERLAP";
            break;
        case CL_IMAGE_FORMAT_MISMATCH:
            errStr = "CL_IMAGE_FORMAT_MISMATCH";
            break;
        case CL_IMAGE_FORMAT_NOT_SUPPORTED:
            errStr = "CL_IMAGE_FORMAT_NOT_SUPPORTED";
            break;
        case CL_BUILD_PROGRAM_FAILURE:
            errStr = "CL_BUILD_PROGRAM_FAILURE";
            break;
        case CL_MAP_FAILURE:
            errStr = "CL_MAP_FAILURE";
            break;
        case CL_INVALID_VALUE:
            errStr = "CL_INVALID_VALUE";
            break;
        case CL_INVALID_DEVICE_TYPE:
            errStr = "CL_INVALID_DEVICE_TYPE";
            break;
        case CL_INVALID_PLATFORM:
            errStr = "CL_INVALID_PLATFORM";
            break;
        case CL_INVALID_DEVICE:
            errStr = "CL_INVALID_DEVICE";
            break;
        case CL_INVALID_CONTEXT:
            errStr = "CL_INVALID_CONTEXT";
            break;
        case CL_INVALID_QUEUE_PROPERTIES:
            errStr = "CL_INVALID_QUEUE_PROPERTIES";
            break;
        case CL_INVALID_COMMAND_QUEUE:
            errStr = "CL_INVALID_COMMAND_QUEUE";
            break;
        case CL_INVALID_HOST_PTR:
            errStr = "CL_INVALID_HOST_PTR";
            break;
        case CL_INVALID_MEM_OBJECT:
            errStr = "CL_INVALID_MEM_OBJECT";
            break;
        case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR:
            errStr = "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
            break;
        case CL_INVALID_IMAGE_SIZE:
            errStr = "CL_INVALID_IMAGE_SIZE";
            break;
        case CL_INVALID_SAMPLER:
            errStr = "CL_INVALID_SAMPLER";
            break;
        case CL_INVALID_BINARY:
            errStr = "CL_INVALID_BINARY";
            break;
        case CL_INVALID_BUILD_OPTIONS:
            errStr = "CL_INVALID_BUILD_OPTIONS";
            break;
        case CL_INVALID_PROGRAM:
            errStr = "CL_INVALID_PROGRAM";
            break;
        case CL_INVALID_PROGRAM_EXECUTABLE:
            errStr = "CL_INVALID_PROGRAM_EXECUTABLE";
            break;
        case CL_INVALID_KERNEL_NAME:
            errStr = "CL_INVALID_KERNEL_NAME";
            break;
        case CL_INVALID_KERNEL_DEFINITION:
            errStr = "CL_INVALID_KERNEL_DEFINITION";
            break;
        case CL_INVALID_KERNEL:
            errStr = "CL_INVALID_KERNEL";
            break;
        case CL_INVALID_ARG_INDEX:
            errStr = "CL_INVALID_ARG_INDEX";
            break;
        case CL_INVALID_ARG_VALUE:
            errStr = "CL_INVALID_ARG_VALUE";
            break;
        case CL_INVALID_ARG_SIZE:
            errStr = "CL_INVALID_ARG_SIZE";
            break;
        case CL_INVALID_KERNEL_ARGS:
            errStr = "CL_INVALID_KERNEL_ARGS";
            break;
        case CL_INVALID_WORK_DIMENSION:
            errStr = "CL_INVALID_WORK_DIMENSION";
            break;
        case CL_INVALID_WORK_GROUP_SIZE:
            errStr = "CL_INVALID_WORK_GROUP_SIZE";
            break;
        case CL_INVALID_WORK_ITEM_SIZE:
            errStr = "CL_INVALID_WORK_ITEM_SIZE";
            break;
        case CL_INVALID_GLOBAL_OFFSET:
            errStr = "CL_INVALID_GLOBAL_OFFSET";
            break;
        case CL_INVALID_EVENT_WAIT_LIST:
            errStr = "CL_INVALID_EVENT_WAIT_LIST";
            break;
        case CL_INVALID_EVENT:
            errStr = "CL_INVALID_EVENT";
            break;
        case CL_INVALID_OPERATION:
            errStr = "CL_INVALID_OPERATION";
            break;
        case CL_INVALID_GL_OBJECT:
            errStr = "CL_INVALID_GL_OBJECT";
            break;
        case CL_INVALID_BUFFER_SIZE:
            errStr = "CL_INVALID_BUFFER_SIZE";
            break;
        case CL_INVALID_MIP_LEVEL:
            errStr = "CL_INVALID_MIP_LEVEL";
            break;
        case CL_INVALID_GLOBAL_WORK_SIZE:
            errStr = "CL_INVALID_GLOBAL_WORK_SIZE";
            break;
    }
    G_fatal_error("Error at file %s line %d; %s\n", fName, lineNum, errStr);
}

/*
 Check if an extension is supported by comparing the string to the list of
 supported extensions.
 */
int ext_supported(cl_device_id dev, char *extName)
{
	size_t returned_size;
	cl_char dev_ext[1024] = {0};
	int i, len = strlen(extName);
    
    clGetDeviceInfo(dev, CL_DEVICE_EXTENSIONS, sizeof(dev_ext), dev_ext, &returned_size);

    for (i = 0; i < 1023; ++i)
        if (strncmp(extName, (char *)&(dev_ext[i]), len) == 0)
            return TRUE;
    
    return FALSE;
}

/*
 Print a list of OpenCL devices connected to this host. This list could be
 presented to the user so the user can select the appropriate device. This
 allows different devices to be used by different threads or selection of
 a fp64 device when needed.
 */
cl_int printDevList()
{
    cl_int err = CL_SUCCESS;
    
    cl_platform_id platform_id = NULL;
    cl_uint ret_num_platforms;
    
    cl_device_id dev[32];
    cl_uint num_dev;
    
    unsigned int i;
    
    // opencl clGetDeviceIDs needs a platform_id! NULL is no longer allowed!
    err = clGetPlatformIDs(1, &platform_id, &ret_num_platforms);
    handleErr(err);
    G_message(_("Found %i OpenCL platforms!"),ret_num_platforms);
    
    err = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_ALL, 32, dev, &num_dev);
    handleErr(err);
    
    G_message(_("Supported OpenCL devices:"));
    for (i = 0; i < num_dev; ++i){
        cl_char vendor_name[1024] = {0};
        cl_char device_name[1024] = {0};
        size_t returned_size = 0;
        char *dpSupport = "";
        
        err = clGetDeviceInfo(dev[i], CL_DEVICE_VENDOR, sizeof(vendor_name), 
                              vendor_name, &returned_size);
        handleErr(err);
        err = clGetDeviceInfo(dev[i], CL_DEVICE_NAME, sizeof(device_name), 
                              device_name, &returned_size);
        handleErr(err);
        if(ext_supported(dev[i], "cl_khr_fp64"))
            dpSupport = "(fp64 support)";
        
        G_message(_("%8d: %16s %64s %s"), i+1, vendor_name, device_name, dpSupport);
    }
    
    return CL_SUCCESS;
}

/*
 Finds an appropirate OpenCL device. If the user specifies a preference, we pick
 that device index from the list. If we are left to a default device, we'll print
 the table of available devices. For debugging, it's always easier to use
 CL_DEVICE_TYPE_CPU because then printf() can be called from the kernel. If
 debugging is on, we can print the name and stats about the device we're using.
 
 sug_dev should be an index starting at zero.
 */
cl_device_id get_device(int sug_dev, cl_int *clErr)
{
    cl_int err = CL_SUCCESS;
    
    cl_platform_id platform_id = NULL;
    cl_uint ret_num_platforms;
    
    cl_device_id device = NULL;
    size_t returned_size = 0;
    cl_char vendor_name[1024] = {0};
    cl_char device_name[1024] = {0};
    cl_device_id dev[32];
    cl_uint num_dev;
    
    // opencl clGetDeviceIDs needs a platform_id! NULL is no longer allowed!
    err = clGetPlatformIDs(1, &platform_id, &ret_num_platforms);
    handleErr(err);
    
    if (sug_dev >= 0 && sug_dev < 32) {
        err = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_ALL, 32, dev, &num_dev);
        handleErrRetNULL(err);
        if (num_dev > sug_dev) {
            device = dev[sug_dev];
        } else {
            G_warning(_("Device index %d too large for number of devices (%d)."),
                      sug_dev, num_dev);
            return get_device(-1, clErr);
        }
    } else {
        printDevList();
        
        // Find the GPU CL device, this is what we really want
        // If there is no GPU device is CL capable, fall back to CPU
        err = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
        if (err != CL_SUCCESS) {
            // Find the CPU CL device, as a fallback
            err = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_CPU, 1, &device, NULL);
            handleErrRetNULL(err);
        }
    }
    
    // Get some information about the returned device
    err = clGetDeviceInfo(device, CL_DEVICE_VENDOR, sizeof(vendor_name), 
                          vendor_name, &returned_size);
    // Fall back to default if invalid
    if (err == CL_INVALID_DEVICE && sug_dev != -1) {
        G_warning(_("Selected device %d is invalid. Defaulting..."), sug_dev);
        return get_device(-1, clErr);
    }
    handleErrRetNULL(err);
    
    err = clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(device_name), 
                          device_name, &returned_size);
    handleErrRetNULL(err);
    
    G_message(_("OpenCL enabled using %s %s..."), vendor_name, device_name);
    
    return device;
}

/*
 Figure out how many threads to make per group. This takes into account the
 number of partitions we've divided our global group into. It returns the
 number of threads per group.
 */
size_t get_thread_group(size_t num_threads, size_t group_size,
                        unsigned int num_partitions)
{
    size_t par_threads;
    
    //Overestimate number of threads per partition if needed
    if (num_threads % num_partitions)
        par_threads = num_threads/num_partitions + 1;
    else
        par_threads = num_threads/num_partitions;
    
    //Round to the higher multiple of the group size
    if (par_threads % group_size)
        return par_threads + group_size - par_threads % group_size;
    else
        return par_threads;
}

/*
 Go ahead and execute the kernel. This handles some housekeeping stuff like the
 run dimensions. When running in debug mode, it times the kernel call and prints
 the execution time.
 
 We'll partition the OpenCL run into many sets of kernel runs. Otherwise we'll
 monopolize the GPU for too long and the watchdog timer will kill the job.
 32 partitions should be appropriate in the vast majority of instances. If you
 notice the display freezing for a few seconds, then the job fails with
 CL_INVALID_COMMAND_QUEUE, it's probably the fault of the watchdog timer kicking
 in. You'll need to increase the number of partitions and recompile the module.
 
 In my brief test, a value of 32 increased runtime by 2%. Seeing as it's already
 been decreased by ~2000% over the original code, I think this is acceptable.
 If you have no display connected to your graphics card, then the watchdog timer
 should be disabled and you can change this to 1 for maximum performance.
 
 Returns CL_SUCCESS on success and other CL_* errors when something goes wrong.
 */
cl_int run_kern(struct OCLCalc *calc, cl_kernel kern, size_t num_threads,
                size_t group_size, unsigned int num_partitions )
{
    cl_int err = CL_SUCCESS;
    int i;
    size_t start_time = 0, end_time;
    double tot_time = 0.0;
    size_t glob_size = get_thread_group(num_threads, group_size, num_partitions);
    handleErr(err = clSetCommandQueueProperty(calc->queue, CL_QUEUE_PROFILING_ENABLE,
                                              CL_TRUE, NULL));
    
    //Run the kernel on each partition
    for (i = 0; i < num_partitions; ++i) {
        cl_event ev;
        
        if (kern == calc->calcKern)
            handleErr(err = clSetKernelArg(kern, 17, sizeof(unsigned int), &i));
        
        // Run the calculation by enqueuing it and forcing the 
        // command queue to complete the task
        handleErr(err = clEnqueueNDRangeKernel(calc->queue, kern, 1, NULL, 
                                               &glob_size, &group_size, 0, NULL, &ev));
        if (err == CL_MEM_OBJECT_ALLOCATION_FAILURE)
            G_fatal_error(_("Unable to allocate enough memory (%d). Try increasing the number of partitions."), __LINE__);
        if (err == CL_INVALID_COMMAND_QUEUE)
            G_fatal_error(_("Kernel crashed. If this happened after the screen froze for a few seconds, "
                            "the GPU's watchdog timer may have been triggered. Try increasing "
                            "the number of partitions on the command line or "
                            "NUM_OPENCL_PARTITIONS in %s and recompile."), __FILE__);
        handleErr(err = clFinish(calc->queue));
        
        handleErr(err = clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_START,
                                                sizeof(size_t), &start_time, NULL));
        handleErr(err = clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_END,
                                                sizeof(size_t), &end_time, NULL));
        assert(end_time != 0);
        assert(start_time != 0);
        handleErr(err = clReleaseEvent(ev));
        tot_time += (end_time-start_time)/1000000000.0;
    }
    
    handleErr(err = clSetCommandQueueProperty(calc->queue, CL_QUEUE_PROFILING_ENABLE,
                                              CL_FALSE, NULL));
    G_verbose_message(_("OpenCL Partitions: % 3d; Total Kernel Time:%12.4f\n"), num_partitions, tot_time);
    return CL_SUCCESS;
}

/*
 Make and copy the memory for the horizon information. It's optional, so if it
 isn't being used we make a very small buffer instead. That way we can pass
 something to the kernel.
 
 Returns CL_SUCCESS on success and other CL_* errors when something goes wrong.
 */
cl_int make_hoz_mem_cl(struct OCLCalc *calc, unsigned int numThreads,
                       int useHoz, unsigned char *hozArr, cl_mem *horizon_cl)
{
    cl_int err = CL_SUCCESS;
    size_t sz;
    
    //Set up memory for the horizon if needed
    if (useHoz){
        sz = sizeof(unsigned char) * numThreads;
        assert(sz >= 0);
        (*horizon_cl) = clCreateBuffer(calc->context, CL_MEM_READ_ONLY, sz, NULL, &err);
        handleErr(err);
        
        err = clEnqueueWriteBuffer(calc->queue, (*horizon_cl), CL_TRUE, 0, sz,
                                   (void*)hozArr, 0, NULL, NULL);
        handleErr(err);
    } else {
        //Make a token cl device malloc
        (*horizon_cl) = clCreateBuffer(calc->context, CL_MEM_READ_ONLY, 1, NULL, &err);
        handleErr(err);
    }
    
    //Set up these as arguments
    err = clSetKernelArg(calc->calcKern, 0, sizeof(cl_mem), horizon_cl);
    handleErr(err);
    
    return CL_SUCCESS;
}

/*
 Make and copy the memory for an input buffer. It's often optional, so if it
 isn't being used we make a very small buffer instead. That way we can pass
 something to the kernel. The data is assembled in temp pinned memory, then
 copied to the device in one large block. (Transfers are faster from pinned
 memory.)
 
 Returns CL_SUCCESS on success and other CL_* errors when something goes wrong.
 */
cl_int make_input_raster_cl(struct OCLCalc *calc, unsigned int x, unsigned int y,
                            int useData, unsigned int argNum, float **src_gs,
                            cl_mem *dst_cl)
{
    int numThreads = x*y;
    cl_int err;
    
    //Set up work space
    if (useData) {
        int i;
        
        //Allocate full buffers
        unsigned int sz = sizeof(float) * numThreads;
        assert(sz >= 0);
        (*dst_cl) = clCreateBuffer(calc->context, CL_MEM_READ_ONLY, sz, NULL, &err);
        if (err == CL_INVALID_BUFFER_SIZE)
            G_fatal_error(_("Unable to allocate enough memory (%d). Try increasing the number of partitions."), __LINE__);
        handleErr(err);
        cl_mem src_work_cl = clCreateBuffer(calc->context, CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR, sz, NULL, &err);
        handleErr(err);
        float *src_work = (float *)clEnqueueMapBuffer(calc->queue, src_work_cl, CL_TRUE, CL_MAP_WRITE,
                                                      0, sz, 0, NULL, NULL, &err);
        handleErr(err);
        
        //Copy data to buffer
        for(i = 0; i < y; ++i)
            memcpy(&(src_work[i*x]), src_gs[i], sizeof(float)*x);
        
        //Copy data to divice memory
        err = clEnqueueWriteBuffer(calc->queue, (*dst_cl), CL_TRUE, 0, sz,
                                   (void*)src_work, 0, NULL, NULL);
        handleErr(err);
        
        //Clean up mem space
        handleErr(err = clReleaseMemObject(src_work_cl));
    } else {
        //Token memory space
        (*dst_cl) = clCreateBuffer(calc->context, CL_MEM_READ_ONLY, 1, NULL, &err);
        handleErr(err);
    }
    
    //Set it up as an argument
    handleErr(err = clSetKernelArg(calc->calcKern, argNum, sizeof(cl_mem), dst_cl));
    return CL_SUCCESS;
}

/*
 Make some empty device buffers to write output to. All output is optional, so
 if it's unneeded, only a small token buffer is created so we still have
 something to pass around.
 
 Returns CL_SUCCESS on success and other CL_* errors when something goes wrong.
 */
cl_int make_output_raster_cl(struct OCLCalc *calc, unsigned int x, unsigned int y,
                             unsigned int useData, unsigned int argNum, cl_mem *out_cl)
{
    int numThreads = x*y;
    cl_int err;
    
    if (useData) {
        //Allocate mem for writing
        size_t sz = sizeof(float) * numThreads;
        assert(sz >= 0);
        (*out_cl) = clCreateBuffer(calc->context, CL_MEM_WRITE_ONLY, sz, NULL, &err);
        handleErr(err);
    } else {
        //Token memory space
        (*out_cl) = clCreateBuffer(calc->context, CL_MEM_WRITE_ONLY, 1, NULL, &err);
        handleErr(err);
    }
    
    //Set it up as an argument
    handleErr(err = clSetKernelArg(calc->calcKern, argNum, sizeof(cl_mem), out_cl));
    
    return CL_SUCCESS;
}

/*
 Copy the OpenCL results back from the device to the host and free the device
 memory. Pinned memory is used as an intermediate buffer, then freed at the end
 of the function.
 
 Returns CL_SUCCESS on success and other CL_* errors when something goes wrong.
 */
cl_int copy_output_cl(struct OCLCalc *calc, unsigned int x, unsigned int y,
                      unsigned int hasData, float **dstArr, cl_mem clSrc)
{
    cl_int err;
    int i;
    size_t sz = sizeof(float) * x * y;
    
    if (!hasData) {
        handleErr(err = clReleaseMemObject(clSrc));
        return CL_SUCCESS;
    }
    
    //Make some pinned working memory
    cl_mem src_work_cl = clCreateBuffer(calc->context,
                                        CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR,
                                        sz, NULL, &err);
    handleErr(err);
    float *src_work = (float *)clEnqueueMapBuffer(calc->queue, src_work_cl,
                                                  CL_TRUE, CL_MAP_WRITE,
                                                  0, sz, 0, NULL, NULL, &err);
    handleErr(err);
    
    //Copy data to host
    err = clEnqueueReadBuffer(calc->queue, clSrc, CL_TRUE, 0, sz, (void*)src_work, 0, NULL, NULL);
    handleErr(err);
    
    //Copy data back to GRASS
    for (i = 0; i < y; ++i)
        memcpy(dstArr[i], &(src_work[i*x]), sizeof(float)*x);
    
    //Clean up mem space
    handleErr(err = clReleaseMemObject(src_work_cl));
    handleErr(err = clReleaseMemObject(clSrc));
    
    return CL_SUCCESS;
}

/*
 Support outputting the min/max values from the kernel. Min/Max is initially
 calculated & saved per work group, so if there are large work groups we don't
 need an obscene amount of memory to calculate it. This also reduces global
 memory accesses.
 
 Returns CL_SUCCESS on success and other CL_* errors when something goes wrong.
 */
cl_int make_min_max_cl(struct OCLCalc *calc, struct OCLConstants *oclConst,
                       cl_mem *min_max_cl)
{
    cl_int err;
    int numThreads = oclConst->numRows * oclConst->n;
    size_t glob_size = get_thread_group(numThreads, calc->calcGroupSize, NUM_OPENCL_PARTITIONS);
    size_t grp_stride = NUM_OPENCL_PARTITIONS * glob_size / calc->calcGroupSize;
    size_t num_grps;
    
    if (numThreads % calc->calcGroupSize)
        num_grps = numThreads / calc->calcGroupSize + 1;
    else
        num_grps = numThreads / calc->calcGroupSize;
        
    // Allocate full buffers
    size_t sz = sizeof(float) * 22 * grp_stride;
    assert(sz >= 0);
    (*min_max_cl) = clCreateBuffer(calc->context, CL_MEM_READ_WRITE, sz, NULL, &err);
    if (err == CL_INVALID_BUFFER_SIZE)
        G_fatal_error(_("Unable to allocate enough memory (%d). Try increasing the number of partitions."), __LINE__);
    handleErr(err);
    
    // Set it up as an argument
    handleErr(err = clSetKernelArg(calc->calcKern, 16, sizeof(cl_mem), min_max_cl));
    
    // Make local space for the reduce function, also
	handleErr(err = clSetKernelArg(calc->calcKern, 19, sizeof(float) * calc->calcGroupSize, NULL));
    
    // We can do all consalidate args here, too
    handleErr(err = clSetKernelArg(calc->consKern, 0, sizeof(cl_mem), min_max_cl));
	handleErr(err = clSetKernelArg(calc->consKern, 1, sizeof(unsigned int), &grp_stride));
	handleErr(err = clSetKernelArg(calc->consKern, 2, sizeof(unsigned int), &num_grps));
	handleErr(err = clSetKernelArg(calc->consKern, 3, sizeof(float) * calc->consGroupSize, NULL));
    
    return CL_SUCCESS;
}

/*
 Support outputting the min/max values from the kernel. Read each of the min/max
 values that have been computed by the kernel.
 
 Returns CL_SUCCESS on success and other CL_* errors when something goes wrong.
 */
cl_int copy_min_max_cl(struct OCLCalc *calc, struct OCLConstants *oclConst,
                       cl_mem min_max_cl)
{
    cl_int err;
    size_t glob_size = get_thread_group(oclConst->numRows * oclConst->n, calc->calcGroupSize, NUM_OPENCL_PARTITIONS);
    size_t stride_sz = sizeof(float) * NUM_OPENCL_PARTITIONS * glob_size / calc->calcGroupSize;
    
    // Copy data back to GRASS, one value at a time
    
    // Linke atmospheric turbidity coefficient
    err = clEnqueueReadBuffer(calc->queue, min_max_cl, CL_FALSE, 0,
                              sizeof(float), &(oclConst->linke_min), 0, NULL, NULL);
    handleErr(err);
    err = clEnqueueReadBuffer(calc->queue, min_max_cl, CL_FALSE, stride_sz,
                              sizeof(float), &(oclConst->linke_max), 0, NULL, NULL);
    handleErr(err);
    
    // Albedo
    err = clEnqueueReadBuffer(calc->queue, min_max_cl, CL_FALSE, stride_sz*2,
                              sizeof(float), &(oclConst->albedo_min), 0, NULL, NULL);
    handleErr(err);
    err = clEnqueueReadBuffer(calc->queue, min_max_cl, CL_FALSE, stride_sz*3,
                              sizeof(float), &(oclConst->albedo_max), 0, NULL, NULL);
    handleErr(err);
    
    // Lat
    err = clEnqueueReadBuffer(calc->queue, min_max_cl, CL_FALSE, stride_sz*4,
                              sizeof(float), &(oclConst->lat_min), 0, NULL, NULL);
    handleErr(err);
    err = clEnqueueReadBuffer(calc->queue, min_max_cl, CL_FALSE, stride_sz*5,
                              sizeof(float), &(oclConst->lat_max), 0, NULL, NULL);
    handleErr(err);
    
    // Lon
    err = clEnqueueReadBuffer(calc->queue, min_max_cl, CL_FALSE, stride_sz*6,
                              sizeof(float), &(oclConst->lon_min), 0, NULL, NULL);
    handleErr(err);
    err = clEnqueueReadBuffer(calc->queue, min_max_cl, CL_FALSE, stride_sz*7,
                              sizeof(float), &(oclConst->lon_max), 0, NULL, NULL);
    handleErr(err);
    
    // Sunrise
    err = clEnqueueReadBuffer(calc->queue, min_max_cl, CL_FALSE, stride_sz*8,
                              sizeof(float), &(oclConst->sunrise_min), 0, NULL, NULL);
    handleErr(err);
    err = clEnqueueReadBuffer(calc->queue, min_max_cl, CL_FALSE, stride_sz*9,
                              sizeof(float), &(oclConst->sunrise_max), 0, NULL, NULL);
    handleErr(err);
    
    // Sunset
    err = clEnqueueReadBuffer(calc->queue, min_max_cl, CL_FALSE, stride_sz*10,
                              sizeof(float), &(oclConst->sunset_min), 0, NULL, NULL);
    handleErr(err);
    err = clEnqueueReadBuffer(calc->queue, min_max_cl, CL_FALSE, stride_sz*11,
                              sizeof(float), &(oclConst->sunset_max), 0, NULL, NULL);
    handleErr(err);
    
    // Beam
    err = clEnqueueReadBuffer(calc->queue, min_max_cl, CL_FALSE, stride_sz*12,
                              sizeof(float), &(oclConst->beam_min), 0, NULL, NULL);
    handleErr(err);
    err = clEnqueueReadBuffer(calc->queue, min_max_cl, CL_FALSE, stride_sz*13,
                              sizeof(float), &(oclConst->beam_max), 0, NULL, NULL);
    handleErr(err);
    
    // Diffuse
    err = clEnqueueReadBuffer(calc->queue, min_max_cl, CL_FALSE, stride_sz*14,
                              sizeof(float), &(oclConst->diff_min), 0, NULL, NULL);
    handleErr(err);
    err = clEnqueueReadBuffer(calc->queue, min_max_cl, CL_FALSE, stride_sz*15,
                              sizeof(float), &(oclConst->diff_max), 0, NULL, NULL);
    handleErr(err);

    // Reflected
    err = clEnqueueReadBuffer(calc->queue, min_max_cl, CL_FALSE, stride_sz*16,
                              sizeof(float), &(oclConst->refl_min), 0, NULL, NULL);
    handleErr(err);
    err = clEnqueueReadBuffer(calc->queue, min_max_cl, CL_FALSE, stride_sz*17,
                              sizeof(float), &(oclConst->refl_max), 0, NULL, NULL);
    handleErr(err);
    
    // Insolation Time
    err = clEnqueueReadBuffer(calc->queue, min_max_cl, CL_FALSE, stride_sz*18,
                              sizeof(float), &(oclConst->insol_min), 0, NULL, NULL);
    handleErr(err);
    err = clEnqueueReadBuffer(calc->queue, min_max_cl, CL_FALSE, stride_sz*19,
                              sizeof(float), &(oclConst->insol_max), 0, NULL, NULL);
    handleErr(err);
    
    // Global
    err = clEnqueueReadBuffer(calc->queue, min_max_cl, CL_FALSE, stride_sz*20,
                              sizeof(float), &(oclConst->globrad_min), 0, NULL, NULL);
    handleErr(err);
    err = clEnqueueReadBuffer(calc->queue, min_max_cl, CL_FALSE, stride_sz*21,
                              sizeof(float), &(oclConst->globrad_max), 0, NULL, NULL);
    handleErr(err);
    
    //Make sure we're done before releasing the memory
    handleErr(err = clFinish(calc->queue));

    //Clean up mem space
    handleErr(err = clReleaseMemObject(min_max_cl));
    
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
                     struct OCLConstants *oclConst,
                     char *kernName,
                     struct SolarRadVar *sunRadVar,
                     struct SunGeometryConstDay *sungeom,
                     struct GridGeometry *gridGeom, cl_int *clErr )
{
	cl_program program;
    cl_kernel kernel;
	cl_int err = CL_SUCCESS;
    char *buffer = (char *)calloc(128000, sizeof(char));
    int latin = oclConst->latin;
    int longin = oclConst->longin;
    char *useDouble = "";
    
    const char *kernFunc =
/*
 There are a handful of places that floating point precision becomes a bit
 insufficent. It's best to use double precision there if possible.
 */
"#ifdef useDouble\n"
"#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n"
"#define BEST_FP double\n"
"#else\n"
"#define BEST_FP float\n"
"#endif\n"

/*
 min_reduce_and_store() & max_reduce_and_store() take a value, store it in very
 fast local memory, and then all of the threads of the group reduce it down to
 a single value. That value is then stored in global memory. Note that this
 function has been optimized so that the group size must be a power of two.
 This function is much faster than the atomic functions, but has overhead in the
 form of local memory storage & register usage.
 */
"void max_reduce_and_store(__local float *sdata,\n"
                          "__global float *store_arr,\n"
                          "float value,\n"
                          "int store_off)\n"
"{\n"
    //Note that this draws from NVIDIA's reduction example:
    //- Doesn't use % operator.
    //- Uses contiguous threads.
    //- Uses sequential addressing -- no divergence or bank conflicts.
    //- Is completely unrolled.
    // local size must be a power of 2 and (>= 64 or == 1)
    "unsigned int lsz = get_local_size(0);\n"
    "unsigned int lid = get_local_id(0);\n"
    "sdata[lid] = value;\n"
    "barrier(CLK_LOCAL_MEM_FENCE);\n"

    // do reduction in shared mem
    "if (lsz != 1) {\n"
        // We assume that the maximum group size is < 1024.
        "if (lsz >= 512) {if (lid < 256) {sdata[lid] = max(sdata[lid], sdata[lid + 256]);} barrier(CLK_LOCAL_MEM_FENCE);}\n"
        "if (lsz >= 256) {if (lid < 128) {sdata[lid] = max(sdata[lid], sdata[lid + 128]);} barrier(CLK_LOCAL_MEM_FENCE);}\n"
        "if (lsz >= 128) {if (lid <  64) {sdata[lid] = max(sdata[lid], sdata[lid +  64]);} barrier(CLK_LOCAL_MEM_FENCE);}\n"

        //Avoid extra 'if' statements by only using local size >= 64 || == 1
        "if (lid < 32) {sdata[lid] = max(sdata[lid], sdata[lid + 32]);} barrier(CLK_LOCAL_MEM_FENCE);\n"
        "if (lid < 16) {sdata[lid] = max(sdata[lid], sdata[lid + 16]);} barrier(CLK_LOCAL_MEM_FENCE);\n"
        "if (lid <  8) {sdata[lid] = max(sdata[lid], sdata[lid +  8]);} barrier(CLK_LOCAL_MEM_FENCE);\n"
        "if (lid <  4) {sdata[lid] = max(sdata[lid], sdata[lid +  4]);} barrier(CLK_LOCAL_MEM_FENCE);\n"
        "if (lid <  2) {sdata[lid] = max(sdata[lid], sdata[lid +  2]);} barrier(CLK_LOCAL_MEM_FENCE);\n"
        "if (lid <  1) {sdata[lid] = max(sdata[lid], sdata[lid +  1]);} barrier(CLK_LOCAL_MEM_FENCE);\n"
    "}\n"

    // write result for this block to global mem 
    "if (lid == 0)\n"
        "store_arr[store_off] = sdata[0];\n"
    "barrier(CLK_LOCAL_MEM_FENCE);\n"
"}\n"
    
"void min_reduce_and_store(__local float *sdata,\n"
                            "__global float *store_arr,\n"
                            "float value,\n"
                            "int store_off)\n"
"{\n"
    //Note that this draws from NVIDIA's reduction example:
    //- Doesn't use % operator.
    //- Uses contiguous threads.
    //- Uses sequential addressing -- no divergence or bank conflicts.
    //- Is completely unrolled.
    // local size must be a power of 2 and (>= 64 or == 1)
    "unsigned int lsz = get_local_size(0);\n"
    "unsigned int lid = get_local_id(0);\n"
    "sdata[lid] = value;\n"
    "barrier(CLK_LOCAL_MEM_FENCE);\n"
    
    // do reduction in shared mem
    "if (lsz != 1) {\n"
        // We assume that the maximum group size is < 1024.
        "if (lsz >= 512) {if (lid < 256) {sdata[lid] = min(sdata[lid], sdata[lid + 256]);} barrier(CLK_LOCAL_MEM_FENCE);}\n"
        "if (lsz >= 256) {if (lid < 128) {sdata[lid] = min(sdata[lid], sdata[lid + 128]);} barrier(CLK_LOCAL_MEM_FENCE);}\n"
        "if (lsz >= 128) {if (lid <  64) {sdata[lid] = min(sdata[lid], sdata[lid +  64]);} barrier(CLK_LOCAL_MEM_FENCE);}\n"
        
        // Avoid extra 'if' statements by only using local size >= 64 || == 1
        "if (lid < 32) {sdata[lid] = min(sdata[lid], sdata[lid + 32]);} barrier(CLK_LOCAL_MEM_FENCE);\n"
        "if (lid < 16) {sdata[lid] = min(sdata[lid], sdata[lid + 16]);} barrier(CLK_LOCAL_MEM_FENCE);\n"
        "if (lid <  8) {sdata[lid] = min(sdata[lid], sdata[lid +  8]);} barrier(CLK_LOCAL_MEM_FENCE);\n"
        "if (lid <  4) {sdata[lid] = min(sdata[lid], sdata[lid +  4]);} barrier(CLK_LOCAL_MEM_FENCE);\n"
        "if (lid <  2) {sdata[lid] = min(sdata[lid], sdata[lid +  2]);} barrier(CLK_LOCAL_MEM_FENCE);\n"
        "if (lid <  1) {sdata[lid] = min(sdata[lid], sdata[lid +  1]);} barrier(CLK_LOCAL_MEM_FENCE);\n"
    "}\n"
    
    // write result for this block to global mem 
    "if (lid == 0)\n"
        "store_arr[store_off] = sdata[0];\n"
    "barrier(CLK_LOCAL_MEM_FENCE);\n"
"}\n"

/*
 global_min_and_reduce() & global_max_and_reduce() are used to reduce an entire
 set of values in global memory. Only the first work group is used in order to
 locally syncronize at speed. Due to the power of exponential reduction, this
 should still be acceptable.
 */
"void global_max_and_reduce(__local float *reduce_s,\n"
                            "__global float *reduce_arr,\n"
                            "unsigned int beg_off,\n"
                            "unsigned int arr_len)\n"
"{\n"
    "unsigned int lsz = get_local_size(0);\n"
    "unsigned int lid = get_local_id(0);\n"
    "unsigned int i;"
    "float value = reduce_arr[beg_off + lid];\n"

    //Reduce the entire array using one work group
    "for(i = lid + lsz; i < arr_len; i += lsz)\n"
        "if (i < arr_len)\n"
            "value = max(value, reduce_arr[beg_off+i]);\n"

    "max_reduce_and_store(reduce_s, reduce_arr, value, beg_off);\n"
"}\n"

"void global_min_and_reduce(__local float *reduce_s,\n"
                            "__global float *reduce_arr,\n"
                            "unsigned int beg_off,\n"
                            "unsigned int arr_len)\n"
"{\n"
    "unsigned int lsz = get_local_size(0);\n"
    "unsigned int lid = get_local_id(0);\n"
    "unsigned int i;"
    "float value = reduce_arr[beg_off + lid];\n"
    
    //Reduce the entire array using one work group
    "for(i = lid + lsz; i < arr_len; i += lsz)\n"
        "if (i < arr_len)\n"
            "value = min(value, reduce_arr[beg_off+i]);\n"
    
    "min_reduce_and_store(reduce_s, reduce_arr, value, beg_off);\n"
"}\n"

/*
 This kernel is run after the main calculate kernel. It calculates the min/max
 values for such things as latitude, sunrise, & sunset. It must be called
 seperately from the original kernel in order to globally synchronize memory.
 */
"__kernel void consolidate_min_max(__global float *min_max,\n"
                                  "unsigned int grp_stride,\n"
                                  "unsigned int orig_gnum,\n"
                                  "__local float *reduce_s)\n"
"{\n"
    "if (get_group_id(0) != 0)\n"
        "return;\n"
    "int i;\n"
    "for (i = 0; i < 22; i += 2) {\n"
        "global_min_and_reduce(reduce_s, min_max, i   *grp_stride, orig_gnum);\n"
        "global_max_and_reduce(reduce_s, min_max,(i+1)*grp_stride, orig_gnum);\n"
    "}\n"
"}\n"

/*
 Compute some constant parameters before the run starts
 */
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
        "pom = degrees(acos(pom));\n"
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

/*
 Compute more parameters before the main calculations
 */
"void com_par(float *sunGeom_sunrise_time,\n"
             "float *sunGeom_sunset_time,\n"
             "float *sunVarGeom_solarAltitude,\n"
             "float *sunVarGeom_sinSolarAltitude,\n"
             "float *sunVarGeom_tanSolarAltitude,\n"
             "BEST_FP *sunVarGeom_solarAzimuth,\n"
             "BEST_FP *sunVarGeom_sunAzimuthAngle,\n"
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
    
    "float lum_Lx = -sunGeom_lum_C22 * sin(sunGeom_timeAngle);\n"
    "float lum_Ly = sunGeom_lum_C11 * costimeAngle + sunGeom_lum_C13;\n"
    
    // vertical angle of the sun
    "*sunVarGeom_solarAltitude = asin(*sunVarGeom_sinSolarAltitude);\n"
    "*sunVarGeom_tanSolarAltitude = tan(*sunVarGeom_solarAltitude);\n"
    
    // horiz. angle of the Sun
    "*sunVarGeom_solarAzimuth = acos(lum_Ly * rsqrt(lum_Lx*lum_Lx + lum_Ly*lum_Ly));\n"
    "if (lum_Lx < 0)\n"
        "*sunVarGeom_solarAzimuth = pi2 - *sunVarGeom_solarAzimuth;\n"
    
    "if (*sunVarGeom_solarAzimuth < pihalf)\n"
        "*sunVarGeom_sunAzimuthAngle = pihalf - *sunVarGeom_solarAzimuth;\n"
    "else\n"
        "*sunVarGeom_sunAzimuthAngle = 2.5 * PI - *sunVarGeom_solarAzimuth;\n"
    
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
    
    "*sunVarGeom_zp = z[i + j*n];\n"
    
    //Used to be distance()
    "if (ll_correction) {\n"
        "float dx = ((float)(i * stepx)) - gridGeom_xg0;\n"
        "float dy = ((float)(j * stepy)) - gridGeom_yg0;\n"
        "return DEGREEINMETERS * sqrt(coslatsq * dx*dx + dy*dy);\n"
    "} else\n"
        "return distance((float2)(i * stepx, j * stepy),\n"
                        "(float2)(gridGeom_xg0, gridGeom_yg0));\n"
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
              "const float coslatsq,\n"
              "const float zmax)\n"
"{\n"
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
            "success = 2;\n"		// shadow

        "if (z2 > zmax)\n"
            "success = 3;\n"		// no test needed all visible
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
                "const BEST_FP sunVarGeom_sunAzimuthAngle,\n"
                "const float sunVarGeom_stepsinangle,\n"
                "const float sunVarGeom_stepcosangle,\n"
                "const float sunSlopeGeom_longit_l,\n"
                "const float sunSlopeGeom_lum_C31_l,\n"
                "const float sunSlopeGeom_lum_C33_l,\n"
                "const float gridGeom_xg0,\n"
                "const float gridGeom_yg0,\n"
                "const float coslatsq,\n"
                "const float zmax)\n"
"{\n"
    "float s = 0.0f;\n"
    
    "*sunVarGeom_isShadow = 0;\n"	// no shadow
    
    "if (useShadowFlag) {\n"
        "if (useHorizonDataFlag) {\n"
            // Start is due east, sungeom->timeangle = -pi/2
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
                    "+ sunSlopeGeom_lum_C33_l;\n"	// Jenco
        "} else {\n"
            "int r;\n"
            "do {\n"
                "r = searching(z, sunVarGeom_zp, gridGeom_xx0, gridGeom_yy0, sunVarGeom_z_orig,\n"
                        "sunVarGeom_tanSolarAltitude, sunVarGeom_stepsinangle,\n"
                        "sunVarGeom_stepcosangle, gridGeom_yg0, gridGeom_yg0, coslatsq, zmax);\n"
            "} while (r == 1);\n"
            
            "if (r == 2)\n"
                "*sunVarGeom_isShadow = 1;\n"	// shadow
            "else\n"
                "s = sunSlopeGeom_lum_C31_l * cos(-sunGeom_timeAngle - sunSlopeGeom_longit_l)\n"
                    "+ sunSlopeGeom_lum_C33_l;\n"	// Jenco
        "}\n"
    "} else {\n"
        "s = sunSlopeGeom_lum_C31_l * cos(-sunGeom_timeAngle - sunSlopeGeom_longit_l)\n"
            "+ sunSlopeGeom_lum_C33_l;\n"	// Jenco
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
           "const float sh,\n"
           "const float linke,\n"
           "const unsigned int gid)\n"
"{\n"
    "float h0refract = sunVarGeom_solarAltitude + 0.061359f *\n"
        "(0.1594f + sunVarGeom_solarAltitude * (1.123f + 0.065656f * sunVarGeom_solarAltitude)) /\n"
        "(1.0f + sunVarGeom_solarAltitude * (28.9344f + 277.3971f * sunVarGeom_solarAltitude));\n"
    
    "float opticalAirMass = exp(-sunVarGeom_z_orig / 8434.5f) / (sin(h0refract) +\n"
                        "0.50572f * pow(degrees(h0refract) + 6.07995f, -1.6364f));\n"
    "float rayl, bhc, slope;\n"
    
    "if(slopein)\n"
        "slope = radians(s[gid]);\n"
    "else\n"
        "slope = singleSlope;\n"
    
    "if(coefbh)\n"
        "bhc = cbhr[gid];\n"
    "else\n"
        "bhc = 1.0f;\n"
    
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

/*
 Compute the direct radiation
 */
"float drad(__global float *s,\n"
           "__global float *li,\n"
           "__global float *a,\n"
           "__global float *cdhr,\n"
           "__global float *min_max,\n"
           
           "float *rr,\n"
           "const int sunVarGeom_isShadow,\n"
           "const float sunVarGeom_solarAltitude,\n"
           "const float sunVarGeom_sinSolarAltitude,\n"
           "const BEST_FP sunVarGeom_solarAzimuth,\n"
           "const float sunSlopeGeom_aspect,\n"
           "const float sh,\n"
           "const float bh,\n"
           "const float linke,\n"
           "const float alb,\n"
           "const unsigned int gid,\n"
           "__local float *reduce_s)\n"
"{\n"
    "float A1, gh, fg, slope, dhc;\n"
    
    "if(slopein)\n"
        "slope = radians(s[gid]);\n"
    "else\n"
        "slope = singleSlope;\n"
    
    "float tn = -0.015843f + linke * (0.030543f + 0.0003797f * linke);\n"
    "float A1b = 0.26463f + linke * (-0.061581f + 0.0031408f * linke);\n"
    
    "if(coefdh)\n"
        "dhc = cdhr[gid];\n"
    "else\n"
        "dhc = 1.0f;\n"
    
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
        
        "if (sunVarGeom_isShadow == 1 || sh <= 0.0f)\n"
            "fx = r_sky + fg * 0.252271f;\n"
        "else if (sunVarGeom_solarAltitude >= 0.1f)\n"
            "fx = ((0.00263f - kb * (0.712f + 0.6883f * kb)) * fg + r_sky) *\n"
                "(1.0f - kb) + kb * sh / sunVarGeom_sinSolarAltitude;\n"
        "else if (sunVarGeom_solarAltitude < 0.1f) {\n"
            "float a_ln = sunVarGeom_solarAzimuth - sunSlopeGeom_aspect;\n"
            
            "if (a_ln > PI)\n"
                "a_ln -= deg2rad;\n"
            "else if (a_ln < -PI)\n"
                "a_ln += deg2rad;\n"
            
            "fx = ((0.00263f - 0.712f * kb - 0.6883f * kb * kb) * fg + r_sky) *\n"
                "(1.0f - kb) + kb * sinslope * cos(a_ln) /\n"
                "(0.1f - 0.008f * sunVarGeom_solarAltitude);\n"
        "}\n"
        
       // refl. rad
        "(*rr) = alb * (bh + dh) * (1.0f - cosslope) * 0.5f;\n"
        "return dh * fx;\n"
    "} else {\n"	// plane
        "(*rr) = 0.0f;\n"
        "return dh;\n"
    "}\n"
"}\n"

/*
 Main compute code. It includes the joules2() function because almost all
 variables would have been passed into it, and it was only called once anyway.
 */
"__kernel void calculate(__global float *horizonArr,\n"
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
                        "__global float *refl,\n"

                        "__global float *min_max,\n"
                        "unsigned int partNum,\n"
                        "const float zmax,\n"
                        "__local float *reduce_s)\n"
"{\n"
    "unsigned int gid = get_global_id(0)+partNum*get_global_size(0);\n"
    "unsigned int gsz = n*numRows;\n"
    "float longitTime = 0.0f;\n"
    "float coslatsq;\n"
    "float gridGeom_xx0, gridGeom_yy0, gridGeom_xg0, gridGeom_yg0;\n"
    
    "gridGeom_xg0 = gridGeom_xx0 = stepx * (gid % m);\n"
    "gridGeom_yg0 = gridGeom_yy0 = stepy * (gid / m);\n"
    "float gridGeom_xp = xmin + gridGeom_xx0;\n"
    "float gridGeom_yp = ymin + gridGeom_yy0;\n"
    
    "if (ll_correction) {\n"
        "float coslat = cos(radians(gridGeom_yp));\n"
        "coslatsq = coslat * coslat;\n"
    "}\n"
    
    "float sunVarGeom_z_orig, sunVarGeom_zp;\n"
    "if (gid < gsz)\n"
        "sunVarGeom_z_orig = sunVarGeom_zp = z[gid];\n"
    
    "float linke, alb;\n"
    "float latitude, longitude;\n"
    "float sunGeom_sunrise_time, sunGeom_sunset_time;\n"
    "BEST_FP beam_e = 0.0;\n"
    "BEST_FP diff_e = 0.0;\n"
    "BEST_FP refl_e = 0.0;\n"
    "BEST_FP insol_t = 0.0;\n"
    
    //Don't overrun arrays
	//Skip if no elevation info
    "if (gid < gsz && sunVarGeom_z_orig != UNDEFZ) {\n"
        "if (civilTimeFlag)\n"
            "longitTime = -longitudeArray[gid] / 15.0f;\n"
        "if (proj_eq_ll) {\n"   //	ll projection
            "longitude = radians(gridGeom_xp);\n"
            "latitude  = radians(gridGeom_yp);\n"
        "} else {\n"
            "latitude = radians(latitudeArray[gid]);\n"
            "longitude = radians(longitudeArray[gid]);\n"
        "}\n"
        
        "float sunSlopeGeom_aspect, sunSlopeGeom_slope;\n"
        
        "if (aspin) {\n"
            "if (o[gid] != 0.0f)\n"
                "sunSlopeGeom_aspect = radians(o[gid]);\n"
            "else\n"
                "sunSlopeGeom_aspect = UNDEF;\n"
        "} else\n"
            "sunSlopeGeom_aspect = singleAspect;\n"
        
        "if (slopein)\n"
            "sunSlopeGeom_slope = radians(s[gid]);\n"
        "else\n"
            "sunSlopeGeom_slope = singleSlope;\n"
        
        "float cos_u = cos(pihalf - sunSlopeGeom_slope);\n"
        "float sin_u = sin(pihalf - sunSlopeGeom_slope);\n"
        "float cos_v = cos(pihalf + sunSlopeGeom_aspect);\n"
        "float sin_v = sin(pihalf + sunSlopeGeom_aspect);\n"
        "float gridGeom_sinlat = sin(-latitude);\n"
        "float gridGeom_coslat = cos(-latitude);\n"
        "float sunGeom_timeAngle = 0.0f;\n"
        
        "if (ttime)\n"
            "sunGeom_timeAngle = tim;\n"
        
        "float sin_phi_l = -gridGeom_coslat * cos_u * sin_v + gridGeom_sinlat * sin_u;\n"
        "float sunSlopeGeom_longit_l = atan2(-cos_u * cos_v, "
                                "gridGeom_sinlat * cos_u * sin_v + gridGeom_coslat * sin_u);\n"
        "float sunSlopeGeom_lum_C31_l = cos(asin(sin_phi_l)) * cosdecl;\n"
        "float sunSlopeGeom_lum_C33_l = sin_phi_l * sindecl;\n"
        
        "float sunGeom_lum_C11, sunGeom_lum_C13, sunGeom_lum_C22;\n"
        "float sunGeom_lum_C31, sunGeom_lum_C33;\n"
        
        "if (incidout || someRadiation)\n"
            "com_par_const(&sunGeom_lum_C11, &sunGeom_lum_C13, &sunGeom_lum_C22,\n"
                          "&sunGeom_lum_C31, &sunGeom_lum_C33, &sunGeom_timeAngle,\n"
                          "&sunGeom_sunrise_time, &sunGeom_sunset_time,\n"
                          "gridGeom_sinlat, gridGeom_coslat, longitTime);\n"
        
        "float sunVarGeom_solarAltitude, sunVarGeom_sinSolarAltitude;\n"
        "float sunVarGeom_tanSolarAltitude;\n"
        "float sunVarGeom_stepsinangle, sunVarGeom_stepcosangle;\n"
        "int sunVarGeom_isShadow;\n"
        "BEST_FP sunVarGeom_sunAzimuthAngle, sunVarGeom_solarAzimuth;\n"

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
                                  "gridGeom_xg0, gridGeom_yg0, coslatsq, zmax);\n"
        
            "if (lum > 0.0f)\n"
                "lumcl[gid] = degrees(asin(lum));\n"
            "else\n"
                "lumcl[gid] = UNDEFZ;\n"
        "}\n"

        "if (linkein)\n"
            "linke = li[gid];\n"
        "else\n"
            "linke = singleLinke;\n"
    
        "if(albedo)\n"
            "alb = a[gid];\n"
        "else\n"
            "alb = singleAlbedo;\n"
    
         "if (someRadiation) {\n"
            //joules2() is inlined so I don't need to pass in basically *everything*
            "int insol_count = 0;\n"
            
            "com_par(&sunGeom_sunrise_time, &sunGeom_sunset_time,\n"
                    "&sunVarGeom_solarAltitude, &sunVarGeom_sinSolarAltitude,\n"
                    "&sunVarGeom_tanSolarAltitude, &sunVarGeom_solarAzimuth,\n"
                    "&sunVarGeom_sunAzimuthAngle,\n"
                    "&sunVarGeom_stepsinangle, &sunVarGeom_stepcosangle,\n"
                    "sunGeom_lum_C11, sunGeom_lum_C13, sunGeom_lum_C22,\n"
                    "sunGeom_lum_C31, sunGeom_lum_C33, sunGeom_timeAngle,\n"
                    "latitude, longitude);\n"

            "if (ttime) {\n"		//irradiance
                "float s0 = lumcline2(horizonArr, z,\n"
                        "&sunVarGeom_isShadow, &sunVarGeom_zp,\n"
                        "&gridGeom_xx0, &gridGeom_yy0, gid*arrayNumInt,\n"
                        "sunGeom_timeAngle, sunVarGeom_z_orig, sunVarGeom_solarAltitude,\n"
                        "sunVarGeom_tanSolarAltitude, sunVarGeom_sunAzimuthAngle,\n"
                        "sunVarGeom_stepsinangle, sunVarGeom_stepcosangle, sunSlopeGeom_longit_l,\n"
                        "sunSlopeGeom_lum_C31_l, sunSlopeGeom_lum_C33_l,\n"
                        "gridGeom_xg0, gridGeom_yg0, coslatsq, zmax);\n"
        
                "if (sunVarGeom_solarAltitude > 0.0f) {\n"
                    "float bh;\n"
                    "if (!sunVarGeom_isShadow && s0 > 0.0f) {\n"
                        "beam_e = brad(s, li, cbhr, &bh, sunVarGeom_z_orig,\n"
                                "sunVarGeom_solarAltitude, sunVarGeom_sinSolarAltitude,\n"
                                "sunSlopeGeom_aspect, s0, linke, gid);\n"	// beam radiation
                    "} else {\n"
                        "beam_e = 0.0f;\n"
                        "bh = 0.0f;\n"
                    "}\n"
                    
                    "float rr = 0.0f;\n"
                    "if (diff_rad || glob_rad)\n"
                        // diffuse rad.
                        "diff_e = drad(s, li, a, cdhr, min_max, &rr, sunVarGeom_isShadow,\n"
                                "sunVarGeom_solarAltitude, sunVarGeom_sinSolarAltitude,\n"
                                "sunVarGeom_solarAzimuth, sunSlopeGeom_aspect, s0, bh, linke, alb, gid, reduce_s);\n"
            
                    "if (refl_rad || glob_rad) {\n"
                        "if (diff_rad && glob_rad)\n"
                            "drad(s, li, a, cdhr, min_max, &rr, sunVarGeom_isShadow,\n"
                                "sunVarGeom_solarAltitude, sunVarGeom_sinSolarAltitude,\n"
                                "sunVarGeom_solarAzimuth, sunSlopeGeom_aspect, s0, bh, linke, alb, gid, reduce_s);\n"
                        "refl_e = rr;\n"	// reflected rad.
                    "}\n"
                "}\n"			// solarAltitude
            "} else {\n"
                // all-day radiation
                "int srStepNo = sunGeom_sunrise_time / timeStep;\n"
                "float lastAngle = (sunGeom_sunset_time - 12.0f) * HOURANGLE;\n"
                "float firstTime;\n"
                "int passNum = 1;\n"
        
                "if ((sunGeom_sunrise_time - srStepNo * timeStep) > 0.5f * timeStep)\n"
                    "firstTime = ((srStepNo + 1.5f) * timeStep - 12.0f) * HOURANGLE;\n"
                "else\n"
                    "firstTime = ((srStepNo + 0.5f) * timeStep - 12.0f) * HOURANGLE;\n"
        
                "sunGeom_timeAngle = firstTime;\n"
        
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
                            "gridGeom_xg0, gridGeom_yg0, coslatsq, zmax);\n"
        
                    "if (sunVarGeom_solarAltitude > 0.0f) {\n"
                        "float bh;\n"
                        "if (!sunVarGeom_isShadow && s0 > 0.0f) {\n"
                            "++insol_count;\n"
                            "beam_e += timeStep * brad(s, li, cbhr, &bh, sunVarGeom_z_orig,\n"
                                    "sunVarGeom_solarAltitude, sunVarGeom_sinSolarAltitude,\n"
                                    "sunSlopeGeom_aspect, s0, linke, gid);\n"
                        "} else {\n"
                            "bh = 0.0f;\n"
                        "}\n"
                        
                        "float rr = 0.0f;\n"
                        "if (diff_rad || glob_rad)\n"
                            "diff_e += timeStep * drad(s, li, a, cdhr, min_max, &rr, sunVarGeom_isShadow,\n"
                                    "sunVarGeom_solarAltitude, sunVarGeom_sinSolarAltitude,\n"
                                    "sunVarGeom_solarAzimuth, sunSlopeGeom_aspect, s0, bh, linke, alb, gid, reduce_s);\n"
                        "if (refl_rad || glob_rad) {\n"
                            "if (diff_rad && glob_rad)\n"
                                "drad(s, li, a, cdhr, min_max, &rr, sunVarGeom_isShadow,\n"
                                        "sunVarGeom_solarAltitude, sunVarGeom_sinSolarAltitude,\n"
                                        "sunVarGeom_solarAzimuth, sunSlopeGeom_aspect, s0, bh, linke, alb, gid, reduce_s);\n"
                            "refl_e += timeStep * rr;\n"
                        "}\n"
                    "}\n"			// illuminated
        
                    "sunGeom_timeAngle = firstTime + passNum * timeStep * HOURANGLE;\n"
                    "++passNum;\n"
                "} while (sunGeom_timeAngle <= lastAngle);\n" // we've got the sunset
            "}\n"				// all-day radiation
    
            "insol_t = timeStep*insol_count;\n"
            
            //Only apply values to where they're wanted
            "if(beam_rad)\n"
                "beam[gid] = beam_e;\n"
            "if(insol_time)\n"
               "insol[gid] = insol_t;\n"
            "if(diff_rad)\n"
                "diff[gid] = diff_e;\n"
            "if(refl_rad)\n"
                "refl[gid] = refl_e;\n"
            "if(glob_rad)\n"
                "globrad[gid] = beam_e + diff_e + refl_e;\n"
        "}\n"
    "}\n"
    
    "int gnum = get_num_groups(0)*NUM_OPENCL_PARTITIONS;\n"
    "int gpid = get_group_id(0)+partNum*get_num_groups(0);\n"
    "int isValid = gid < gsz && sunVarGeom_z_orig != UNDEFZ;\n"
    "float global_e = beam_e + diff_e + refl_e;\n"
    
    //This gets ugly because all threads must enter *_reduce_and_store() for it
    //to work properly, and we must give invalid values for each if invalid source
    "if (linkein) {\n"
        "if (!isValid)\n"
            "linke = 100.0f;\n"
        "min_reduce_and_store(reduce_s, min_max, linke, gpid);\n"
        "if (!isValid)\n"
            "linke = 0.0f;\n"
        "max_reduce_and_store(reduce_s, min_max, linke, gnum+gpid);\n"
    "}\n"
    "if (albedo) {\n"
        "if (!isValid)\n"
            "alb = 1.0f;\n"
        "min_reduce_and_store(reduce_s, min_max, alb, 2*gnum+gpid);\n"
        "if (!isValid)\n"
            "alb = 0.0f;\n"
        "max_reduce_and_store(reduce_s, min_max, alb, 3*gnum+gpid);\n"
    "}\n"
    
    "if (!isValid) {\n"
        "latitude = 90.0f;\n"
        "longitude = 180.0f;\n"
        "sunGeom_sunrise_time = 24.0f;\n"
        "sunGeom_sunset_time = 24.0f;\n"
        "beam_e = MAXFLOAT;\n"
        "diff_e = MAXFLOAT;\n"
        "refl_e = MAXFLOAT;\n"
        "insol_t = MAXFLOAT;\n"
        "global_e = MAXFLOAT;\n"
    "}\n"

    "min_reduce_and_store(reduce_s, min_max, latitude, 4*gnum+gpid);\n"
    "min_reduce_and_store(reduce_s, min_max, longitude, 6*gnum+gpid);\n"
    "min_reduce_and_store(reduce_s, min_max, sunGeom_sunrise_time, 8*gnum+gpid);\n"
    "min_reduce_and_store(reduce_s, min_max, sunGeom_sunset_time, 10*gnum+gpid);\n"
    "min_reduce_and_store(reduce_s, min_max, beam_e, 12*gnum+gpid);\n"
    "min_reduce_and_store(reduce_s, min_max, diff_e, 14*gnum+gpid);\n"
    "min_reduce_and_store(reduce_s, min_max, refl_e, 16*gnum+gpid);\n"
    "min_reduce_and_store(reduce_s, min_max, insol_t, 18*gnum+gpid);\n"
    "min_reduce_and_store(reduce_s, min_max, global_e, 20*gnum+gpid);\n"
    
    "if (!isValid) {\n"
        "latitude = -90.0f;\n"
        "longitude = -180.0f;\n"
        "sunGeom_sunrise_time = 0.0f;\n"
        "sunGeom_sunset_time = 0.0f;\n"
        "beam_e = 0.0f;\n"
        "diff_e = 0.0f;\n"
        "refl_e = 0.0f;\n"
        "insol_t = 0.0f;\n"
        "global_e = 0.0f;\n"
    "}\n"
    
    "max_reduce_and_store(reduce_s, min_max, latitude, 5*gnum+gpid);\n"
    "max_reduce_and_store(reduce_s, min_max, longitude, 7*gnum+gpid);\n"
    "max_reduce_and_store(reduce_s, min_max, sunGeom_sunrise_time, 9*gnum+gpid);\n"
    "max_reduce_and_store(reduce_s, min_max, sunGeom_sunset_time, 11*gnum+gpid);\n"
    "max_reduce_and_store(reduce_s, min_max, beam_e, 13*gnum+gpid);\n"
    "max_reduce_and_store(reduce_s, min_max, diff_e, 15*gnum+gpid);\n"
    "max_reduce_and_store(reduce_s, min_max, refl_e, 17*gnum+gpid);\n"
    "max_reduce_and_store(reduce_s, min_max, insol_t, 19*gnum+gpid);\n"
    "max_reduce_and_store(reduce_s, min_max, global_e, 21*gnum+gpid);\n"
"}\n";

    //Actually make the program from assembled source
    program = clCreateProgramWithSource(context, 1, (const char**)&kernFunc,
                                        NULL, &err);
    handleErrRetNULL(err);
    
    // Set lat/long input constants
	if (!oclConst->proj_eq_ll && (!oclConst->latin || !oclConst->longin)) {
        // We'll be creating these arrays on the fly
        latin = TRUE;
        longin = TRUE;
    }
    
    if (ext_supported(dev, "cl_khr_fp64"))
        useDouble = "-D useDouble";
    
    // Assemble the compiler arg string for speed. All invariants should be defined
    // here. I'm using "%015.15lff" to format the numbers to maintain full precision
    // it really makes a difference for some calculations.
    sprintf(buffer, "-cl-fast-relaxed-math -cl-mad-enable -Werror "
            "-D invScale=%015.15lff -D pihalf=%015.15lff -D pi2=%015.15lf -D deg2rad=%015.15lff "
            "-D invstepx=%015.15lff -D invstepy=%015.15lff -D xmin=%015.15lff -D ymin=%015.15lff -D xmax=%015.15lff "
            "-D ymax=%015.15lff -D civilTime=%015.15lff -D tim=%015.15lff -D timeStep=%015.15lff -D horizonStep=%015.15lff "
            "-D stepx=%015.15lff -D stepy=%015.15lff -D deltx=%015.15lff -D delty=%015.15lff "
            "-D stepxy=%015.15lf -D horizonInterval=%015.15lff -D singleLinke=%015.15lff "
            "-D singleAlbedo=%015.15lff -D singleSlope=%015.15lff -D singleAspect=%015.15lff -D cbh=%015.15lff "
            "-D cdh=%015.15lff -D dist=%015.15lff -D TOLER=%015.15lff -D offsetx=%015.15lff -D offsety=%015.15lff "
            "-D declination=%015.15lff -D G_norm_extra=%015.15lff -D timeOffset=%015.15lff -D sindecl=%015.15lff "
            "-D cosdecl=%015.15lff -D n=%d -D m=%d -D saveMemory=%d -D civilTimeFlag=%d "
            "-D day=%d -D ttime=%d -D numPartitions=%d -D arrayNumInt=%d "
            "-D proj_eq_ll=%d -D someRadiation=%d -D numRows=%d "
            "-D ll_correction=%d -D aspin=%d -D slopein=%d -D linkein=%d -D albedo=%d -D latin=%d "
            "-D longin=%d -D coefbh=%d -D coefdh=%d -D incidout=%d -D beam_rad=%d "
            "-D insol_time=%d -D diff_rad=%d -D refl_rad=%d -D glob_rad=%d "
            "-D useShadowFlag=%d -D useHorizonDataFlag=%d -D EPS=%015.15lff -D HOURANGLE=%015.15lff "
            "-D PI=%015.15lf -D DEGREEINMETERS=%015.15lff -D UNDEFZ=%015.15lff -D EARTHRADIUS=%015.15lff -D UNDEF=%015.15lff "
            "%s -D NUM_OPENCL_PARTITIONS=%d ",
            invScale, pihalf, pi2, deg2rad,
            oclConst->invstepx, oclConst->invstepy, oclConst->xmin, oclConst->ymin, oclConst->xmax,
            oclConst->ymax, oclConst->civilTime, oclConst->tim, oclConst->step, oclConst->horizonStep,
            gridGeom->stepx, gridGeom->stepy, gridGeom->deltx, gridGeom->delty,
            gridGeom->stepxy, getHorizonInterval(), oclConst->singleLinke,
            oclConst->singleAlbedo, oclConst->singleSlope, oclConst->singleAspect, oclConst->cbh,
            oclConst->cdh, oclConst->dist, oclConst->TOLER, oclConst->offsetx, oclConst->offsety,
            oclConst->declination, sunRadVar->G_norm_extra, getTimeOffset(), sungeom->sindecl,
            sungeom->cosdecl, oclConst->n, oclConst->m, oclConst->saveMemory, useCivilTime(),
            oclConst->day, oclConst->ttime, oclConst->numPartitions, arrayNumInt,
            oclConst->proj_eq_ll, oclConst->someRadiation, oclConst->numRows,
            oclConst->ll_correction, oclConst->aspin, oclConst->slopein, oclConst->linkein, oclConst->albedo, latin,
            longin, oclConst->coefbh, oclConst->coefdh, oclConst->incidout, oclConst->beam_rad,
            oclConst->insol_time, oclConst->diff_rad, oclConst->refl_rad, oclConst->glob_rad,
            useShadow(), useHorizonData(), EPS, HOURANGLE,
            M_PI, oclConst->degreeInMeters, UNDEFZ, EARTHRADIUS, UNDEF,
            useDouble, NUM_OPENCL_PARTITIONS);
    
    (*clErr) = err = clBuildProgram(program, 1, &(dev), buffer, NULL, NULL);
    
    //Detailed debugging info
    if (err != CL_SUCCESS)
    {
        err = clGetProgramBuildInfo(program, dev, CL_PROGRAM_BUILD_LOG,
                                    128000*sizeof(char), buffer, NULL);
        handleErrRetNULL(err);
        
        //Print the build error msg
        printf("Build Log:\n%s\n", buffer);
        printf("Error: Failed to build program executable!\n");
        printCLErr(__FILE__, __LINE__, err);
        
        //Print build status in case that's useful
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
        
        //Dump the source so we have a line number reference
        printf("Program Source:\n%s\n", kernFunc);
        return NULL;
    }
    
    //Compile the kernel from the program
    kernel = clCreateKernel(program, kernName, &err);
    handleErrRetNULL(err);
    
    //Release the program now that we have the kernel
    err = clReleaseProgram(program);
    handleErrRetNULL(err);
    
    free(buffer);
    return kernel;
}

/*
 Make a struct containing the constant values between partitions, the compiled
 kernel, group sizes, device id, queue, & context.
 */
struct OCLCalc *make_environ_cl(struct OCLConstants *oclConst,
                                struct SolarRadVar *sunRadVar,
                                struct SunGeometryConstDay *sunGeom,
                                struct GridGeometry *gridGeom,
                                int sugDev,
                                cl_int *clErr )
{
    cl_int err;
	size_t groupSize;
    struct OCLCalc *calc = (struct OCLCalc *)malloc(sizeof(struct OCLCalc));
    
    calc->dev = get_device(sugDev, &err);
    
    // Now create a context to perform our calculation with the specified device 
    calc->context = clCreateContext(0, 1, &(calc->dev), NULL, NULL, &err);
    handleErrRetNULL(err);
    
    // And also a command queue for the context
    calc->queue = clCreateCommandQueue(calc->context, calc->dev, 0, &err);
    handleErrRetNULL(err);
    
    //Compile the kernels
	calc->calcKern = get_kernel(calc->context, calc->dev, oclConst, "calculate",
                                sunRadVar, sunGeom, gridGeom, &err);
    handleErrRetNULL(err);
    
	calc->consKern = get_kernel(calc->context, calc->dev, oclConst, "consolidate_min_max",
                                sunRadVar, sunGeom, gridGeom, &err);
    handleErrRetNULL(err);
    
    //What's the recommended group size?
	err = clGetKernelWorkGroupInfo(calc->calcKern, calc->dev, CL_KERNEL_WORK_GROUP_SIZE,
                                   sizeof(size_t), &groupSize, NULL);
    handleErrRetNULL(err);
    
    // *_reduce_and_store() requires group size >= 64 && power of 2 || == 1
    if (groupSize >= 512)
        groupSize = 512;
    else if (groupSize >= 256)
        groupSize = 256;
    else if (groupSize >= 128)
        groupSize = 128;
    else if (groupSize >= 64)
        groupSize = 64;
    else
        groupSize = 1;
	G_verbose_message(_("Calculate Group Size:   %lu"), groupSize);
    
    calc->calcGroupSize = groupSize;
    
    //What's the recommended group size?
	err = clGetKernelWorkGroupInfo(calc->consKern, calc->dev, CL_KERNEL_WORK_GROUP_SIZE,
                                   sizeof(size_t), &groupSize, NULL);
    handleErrRetNULL(err);
    
    // *_reduce_and_store() requires group size >= 64 || == 1
    if (groupSize < 64)
        groupSize = 1;
	G_verbose_message(_("Consolidate Group Size: %lu"), groupSize);
    
    calc->consGroupSize = groupSize;
    
    (*clErr) = CL_SUCCESS;
    return calc;
}

/*
 Release all constant values in the calculator structure
 */
cl_int free_environ_cl(struct OCLCalc *oclCalc)
{
    cl_int err;
    
    handleErr(err = clReleaseKernel(oclCalc->calcKern));
    handleErr(err = clReleaseKernel(oclCalc->consKern));

    handleErr(err = clReleaseCommandQueue(oclCalc->queue));
    handleErr(err = clReleaseContext(oclCalc->context));
    
    G_free(oclCalc);
    
    return CL_SUCCESS;
}

/*
 The main entrance function to do the r.sun work. It's called from calculate()
 in main.c. We create the kernel, copy inputs to the device, run the kernel, and
 copy the output back.
 
 For a speed & efficiency bump, seperate this out so the constants are loaded
 once and it's compiled only once for multiple partition runs.
 
 Returns CL_SUCCESS on success and other CL_* errors when something goes wrong.
 */
cl_int calculate_core_cl(unsigned int partOff,
                         struct OCLCalc *calc,
                         struct OCLConstants *oclConst,
                         struct GridGeometry *gridGeom,
                         unsigned char *horizonarray,
                         
                         float **z, float **o, float **s, float **li, float **a,
                         float **latitudeArray, float **longitudeArray,
                         float **cbhr, float **cdhr,
                         
                         float **lumcl, float **beam, float **insol,
                         float **diff, float **refl, float **globrad )
{
    int xDim = oclConst->n;
    int yDim = oclConst->numRows;
    int numThreads = xDim*yDim;
    int k;
    cl_int err;
    int latMalloc = FALSE, lonMalloc = FALSE;
    
    cl_mem  horizon_cl, z_cl, o_cl, s_cl, li_cl, a_cl, lat_cl, long_cl, cbhr_cl, cdhr_cl,
            lumcl_cl, beam_cl, globrad_cl, insol_cl, diff_cl, refl_cl, min_max_cl;
    
    unsigned int numCopyRows;
    if (oclConst->m < partOff + yDim)
        numCopyRows = oclConst->m - partOff;
    else
        numCopyRows = yDim;
	
	//Construct the lat/long array if needed
	if (!oclConst->proj_eq_ll && (!oclConst->latin || !oclConst->longin)) {
		double *xCoords, *yCoords, *hCoords;
		
		//Alloc space, if needed, for the lat/long arrays
		if (latitudeArray == NULL) {
			latitudeArray = (float **)G_malloc(sizeof(float *) * yDim);
			for (k = 0; k < yDim; ++k)
				latitudeArray[k] = (float *)G_malloc(sizeof(float) * xDim);
            latMalloc = TRUE;
		}
		
		if (longitudeArray == NULL) {
			longitudeArray = (float **)G_malloc(sizeof(float *) * yDim);
			for (k = 0; k < yDim; ++k)
				longitudeArray[k] = (float *)G_malloc(sizeof(float) * xDim);
            lonMalloc = TRUE;
        }
		
		//Alloc more space to pass to the transformer
		//The transformer uses double precision, but we want floats
		xCoords = (double *)G_malloc(sizeof(double) * xDim);
		yCoords = (double *)G_malloc(sizeof(double) * xDim);
		hCoords = (double *)G_malloc(sizeof(double) * xDim);
		
		for (k = 0; k < yDim; ++k) {
			int j;
			double yCoord = oclConst->ymin + (partOff+k) * gridGeom->stepy;
			
			//Set the source coords
			for (j = 0; j < xDim; ++j) {
				xCoords[j] = oclConst->xmin + j * gridGeom->stepx;
				yCoords[j] = yCoord;
			}
			
			//Do the conversion for the row
			if (pj_do_transform(xDim, xCoords, yCoords, hCoords, &iproj, &oproj) < 0)
				G_fatal_error(_("Error in pj_do_transform"));
			
			//Read them from doubles into floats
			for (j = 0; j < xDim; ++j) {
				longitudeArray[k][j] = xCoords[j];
				latitudeArray[k][j]  = yCoords[j];
			}
		}
		
		//Free our transformer double space
		G_free(xCoords);
		G_free(yCoords);
		G_free(hCoords);
		
		//Update the 'constants'
		oclConst->latin = oclConst->longin = TRUE;
	}
	
    //Allocate and copy all the inputs
    make_hoz_mem_cl(calc, numThreads, useHorizonData(), horizonarray, &horizon_cl);
    make_input_raster_cl(calc, xDim, yDim, TRUE, 1, z, &z_cl);
    make_input_raster_cl(calc, xDim, yDim, oclConst->aspin, 2, o, &o_cl);
    make_input_raster_cl(calc, xDim, yDim, oclConst->slopein, 3, s, &s_cl);
    make_input_raster_cl(calc, xDim, yDim, oclConst->linkein, 4, li, &li_cl);
    make_input_raster_cl(calc, xDim, yDim, oclConst->albedo, 5, a, &a_cl);
    make_input_raster_cl(calc, xDim, yDim, oclConst->latin, 6, latitudeArray, &lat_cl);
    make_input_raster_cl(calc, xDim, yDim, oclConst->longin, 7, longitudeArray, &long_cl);
    make_input_raster_cl(calc, xDim, yDim, oclConst->coefbh, 8, cbhr, &cbhr_cl);
    make_input_raster_cl(calc, xDim, yDim, oclConst->coefdh, 9, cdhr, &cdhr_cl);
    
    //It's copied to the device and not needed
    if (latMalloc == TRUE) {
        for (k = 0; k < yDim; ++k)
            G_free(latitudeArray[k]);
        G_free(latitudeArray);
        oclConst->latin = FALSE;
    }
    if (lonMalloc == TRUE) {
        for (k = 0; k < yDim; ++k)
            G_free(longitudeArray[k]);
        G_free(longitudeArray);
        oclConst->longin = FALSE;
    }
    
    //Make space for the outputs
    make_output_raster_cl(calc, xDim, yDim, oclConst->incidout, 10, &lumcl_cl);
    make_output_raster_cl(calc, xDim, yDim, oclConst->beam_rad, 11, &beam_cl);
    make_output_raster_cl(calc, xDim, yDim, oclConst->glob_rad, 12, &globrad_cl);
    make_output_raster_cl(calc, xDim, yDim, oclConst->insol_time, 13, &insol_cl);
    make_output_raster_cl(calc, xDim, yDim, oclConst->diff_rad, 14, &diff_cl);
    make_output_raster_cl(calc, xDim, yDim, oclConst->refl_rad, 15, &refl_cl);
    make_min_max_cl(calc, oclConst, &min_max_cl);
    
    //Set the zmax value
    handleErr(err = clSetKernelArg(calc->calcKern, 18, sizeof(float), &(oclConst->zmax)));
    
    //Do the dirty work
    run_kern(calc, calc->calcKern, numThreads, calc->calcGroupSize, NUM_OPENCL_PARTITIONS);
    run_kern(calc, calc->consKern, calc->consGroupSize, calc->consGroupSize, 1);
    
    //Release unneeded inputs
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
    
    //Copy & release requested outputs
    copy_output_cl(calc, xDim, numCopyRows, oclConst->incidout, lumcl, lumcl_cl);
    copy_output_cl(calc, xDim, numCopyRows, oclConst->beam_rad, beam, beam_cl);
    copy_output_cl(calc, xDim, numCopyRows, oclConst->insol_time, insol, insol_cl);
    copy_output_cl(calc, xDim, numCopyRows, oclConst->diff_rad, diff, diff_cl);
    copy_output_cl(calc, xDim, numCopyRows, oclConst->refl_rad, refl, refl_cl);
    copy_output_cl(calc, xDim, numCopyRows, oclConst->glob_rad, globrad, globrad_cl);
    copy_min_max_cl(calc, oclConst, min_max_cl);
    
    return CL_SUCCESS;
}
