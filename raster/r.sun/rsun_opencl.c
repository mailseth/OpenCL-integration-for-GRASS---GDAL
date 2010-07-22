#define NUM_CONSTANTS_F 50
#define NUM_CONSTANTS_I 50


cl_device_id get_device()
{
	cl_int err = 0;
	cl_device_id device = NULL;
    
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
    size_t returned_size = 0;
    // Get some information about the returned device
    cl_char vendor_name[1024] = {0};
    cl_char device_name[1024] = {0};
    err = clGetDeviceInfo(device, CL_DEVICE_VENDOR, sizeof(vendor_name), 
                          vendor_name, &returned_size);
    err |= clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(device_name), 
                           device_name, &returned_size);
    assert(err == CL_SUCCESS);
    printf("Connecting to %s %s...\n", vendor_name, device_name);
#endif
    
    return device;
}

void make_thread_mem_cl(cl_command_queue cmd_queue, cl_context context, cl_kernel kern,
                        unsigned int numThreads, unsigned int locSize,
                        cl_mem *sunGeom_cl, cl_mem *sunVarGeom_cl, cl_mem *sunSlopeGeom_cl,
                        cl_mem *const_f_cl, cl_mem *const_i_cl)
{
    cl_int err = CL_SUCCESS;
    unsigned int sz;
    cl_mem const_f_work_cl = NULL;
    float *const_f_work = NULL;
    cl_mem const_i_work_cl = NULL;
    float *const_i_work = NULL;
    
    //Allocate space for structures
    sz = sizeof(float) * numThreads * 8;
    (*sunGeom_cl) = clCreateBuffer(context, CL_MEM_READ_WRITE, sz, NULL, &err);
    assert(err == CL_SUCCESS);
    sz = sizeof(float) * numThreads * 12;
    (*sunVarGeom_cl) = clCreateBuffer(context, CL_MEM_READ_WRITE, sz, NULL, &err);
    assert(err == CL_SUCCESS);
    sz = sizeof(float) * numThreads * 4;
    (*sunSlopeGeom_cl) = clCreateBuffer(context, CL_MEM_READ_WRITE, sz, NULL, &err);
    assert(err == CL_SUCCESS);
    
    //Set up work space for float constants
    sz = sizeof(float) * NUM_CONSTANTS_F;
    assert(sz >= 0);
    (*const_f_cl) = clCreateBuffer(context, CL_MEM_READ_ONLY, sz, NULL, &err);
    assert(err == CL_SUCCESS);
    const_f_work_cl = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR, sz, NULL, &err);
    assert(err == CL_SUCCESS);
    const_f_work = (float *)clEnqueueMapBuffer(cmd_queue, const_f_work_cl, CL_TRUE, CL_MAP_WRITE,
                                               0, sz, 0, NULL, NULL, &err);
    assert(err == CL_SUCCESS);
    
    //Copy in more vars
	const_f_work[ 0] = invScale;
	const_f_work[ 1] = pihalf;
	const_f_work[ 2] = pi2;
	const_f_work[ 3] = deg2rad;
	const_f_work[ 4] = rad2deg;
	const_f_work[ 5] = invstepx;
	const_f_work[ 6] = invstepy;
	const_f_work[ 7] = xmin;
	const_f_work[ 8] = ymin;
	const_f_work[ 9] = xmax;
	const_f_work[10] = ymax;
	const_f_work[11] = civilTime;
	const_f_work[12] = step;
	const_f_work[13] = horizonStep;
	const_f_work[14] = GridGeometry.stepx;
	const_f_work[15] = GridGeometry.stepy;
	const_f_work[16] = GridGeometry.deltx;
	const_f_work[17] = GridGeometry.delty;
	const_f_work[18] = GridGeometry.stepxy;
	const_f_work[19] = horizonInterval;
	const_f_work[20] = singleLinke;
	const_f_work[21] = singleAlbedo;
	const_f_work[22] = singleSlope;
	const_f_work[23] = singleAspect;
	const_f_work[24] = cbh;
	const_f_work[25] = cdh;
	const_f_work[26] = dist;
	const_f_work[27] = TOLER;
    const_f_work[28] = offsetx;
    const_f_work[29] = offsety;
//	const_f_work[30] = albedo_max;
//	const_f_work[31] = albedo_min;
//	const_f_work[32] = lat_max;
//	const_f_work[33] = lat_min;
//	const_f_work[34] = offsetx;
//	const_f_work[35] = offsety;
	const_f_work[36] = declination;
	const_f_work[37] = SolarRadVar.G_norm_extra;
	const_f_work[38] = timeOffset;
    const_f_work[39] = sunGeom.sindecl;
    const_f_work[40] = sunGeom.cosdecl;
    
    err = clEnqueueWriteBuffer(cmd_queue, (*const_f_cl), CL_FALSE, 0, sz,
                               (void*)const_f_work, 0, NULL, NULL);
    assert(err == CL_SUCCESS);
    
    //Set up work space for integer constants
    sz = sizeof(int) * NUM_CONSTANTS_I;
    assert(sz >= 0);
    (*const_i_cl) = clCreateBuffer(context, CL_MEM_READ_ONLY, sz, NULL, &err);
    assert(err == CL_SUCCESS);
    const_i_work_cl = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR, sz, NULL, &err);
    assert(err == CL_SUCCESS);
    const_i_work = (int *)clEnqueueMapBuffer(cmd_queue, const_i_work_cl, CL_TRUE, CL_MAP_WRITE,
                                               0, sz, 0, NULL, NULL, &err);
    assert(err == CL_SUCCESS);
    
    //Copy more stuff
	const_i_work[ 0] = n;
	const_i_work[ 1] = m;
	const_i_work[ 2] = saveMemory;
	const_i_work[ 3] = civilTimeFlag;
	const_i_work[ 4] = day;
	const_i_work[ 5] = ttime;
	const_i_work[ 6] = numPartitions;
	const_i_work[ 7] = arrayNumInt;
	const_i_work[ 8] = tim;
	const_i_work[ 9] = G_projection();
	const_i_work[10] = someRadiation;
	const_i_work[11] = numRows;
	const_i_work[12] = numThreads;
    const_i_work[13] = ll_correction;
    const_i_work[14] = (int) aspin;
    const_i_work[15] = (int) slopein;
    const_i_work[16] = (int) linkein;
    const_i_work[17] = (int) albedo;
    const_i_work[18] = (int) latin;
    const_i_work[19] = (int) longin;
    const_i_work[20] = (int) coefbh;
    const_i_work[21] = (int) coefdh;
    const_i_work[22] = (int) incidout;
    const_i_work[23] = (int) beam_rad;
    const_i_work[24] = (int) insol_time;
    const_i_work[25] = (int) diff_rad;
    const_i_work[26] = (int) refl_rad;
    const_i_work[27] = (int) glob_rad;
    const_i_work[28] = (int) useShadow();
    const_i_work[29] = (int) useHorizonData();
    
    err = clEnqueueWriteBuffer(cmd_queue, (*const_f_cl), CL_FALSE, 0, sz,
                               (void*)const_f_work, 0, NULL, NULL);
    assert(err == CL_SUCCESS);
    
    // Get all of the stuff written and allocated 
    clFinish(cmd_queue);
    
    //Clean up mem space
    if (const_f_work_cl != NULL)
        clReleaseMemObject(const_f_work_cl);
    if (const_i_work_cl != NULL)
        clReleaseMemObject(const_i_work_cl);
    
    //Set up these as arguments
    //FIXME: Best case scenario is everything is local mem instead of global, but that probably won't happen. Test, though.
    err = clSetKernelArg(kern, 0, sizeof(cl_mem), sunGeom_cl);
    assert(err == CL_SUCCESS);
    err = clSetKernelArg(kern, 1, sizeof(cl_mem), sunVarGeom_cl);
    assert(err == CL_SUCCESS);
    err = clSetKernelArg(kern, 2, sizeof(cl_mem), sunSlopeGeom_cl);
    assert(err == CL_SUCCESS);
    err = clSetKernelArg(kern, 3, sizeof(float)*locSize*8, NULL);
    assert(err == CL_SUCCESS);
    err = clSetKernelArg(kern, 4, sizeof(cl_mem), const_f_cl);
    assert(err == CL_SUCCESS);
    err = clSetKernelArg(kern, 5, sizeof(cl_mem), const_i_cl);
    assert(err == CL_SUCCESS);
}

void make_hoz_mem_cl(cl_command_queue cmd_queue, cl_context context, cl_kernel kern,
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
        assert(err == CL_SUCCESS);
        err = clEnqueueWriteBuffer(cmd_queue, (*horizon_cl), CL_TRUE, 0, sz,
                                   (void*)hozArr, 0, NULL, NULL);
        assert(err == CL_SUCCESS);
    } else {
        //Make a token cl device malloc
        (*horizon_cl) = clCreateBuffer(context, CL_MEM_READ_ONLY, 1, NULL, &err);
        assert(err == CL_SUCCESS);
    }
    
    //Set up these as arguments
    err = clSetKernelArg(kern, 6, sizeof(cl_mem), horizon_cl);
    assert(err == CL_SUCCESS);
}

void make_input_raster_cl(cl_command_queue cmd_queue, cl_context context, cl_kernel kern,
                          unsigned int x, unsigned int y, int useData, unsigned int argNum,
                          float **src_gs, cl_mem *dst_cl)
{
    int numThreads = x*y;
    
    //Set up work space
    if (useData) {
        //Token memory space
        (*dst_cl) = clCreateBuffer(context, CL_MEM_READ_ONLY, 1, NULL, &err);
        assert(err == CL_SUCCESS);
    } else {
        int i;
        
        //Allocate full buffers
        unsigned int sz = sizeof(float) * numThreads;
        assert(sz >= 0);
        (*dst_cl) = clCreateBuffer(context, CL_MEM_READ_ONLY, sz, NULL, &err);
        assert(err == CL_SUCCESS);
        cl_mem src_work_cl = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR, sz, NULL, &err);
        assert(err == CL_SUCCESS);
        float *src_work = (float *)clEnqueueMapBuffer(cmd_queue, src_work_cl, CL_TRUE, CL_MAP_WRITE,
                                                      0, sz, 0, NULL, NULL, &err);
        assert(err == CL_SUCCESS);
        
        //Copy data to buffer
        for(i = 0; i < y; ++i)
            memcpy(&(src_work[i*x]), &(src_gs[i]), sizeof(float)*x);
        
        //Copy data to host
        err = clEnqueueWriteBuffer(cmd_queue, (*dst_cl), CL_TRUE, 0, sz,
                                   (void*)src_work, 0, NULL, NULL);
        assert(err == CL_SUCCESS);
        
        //Clean up mem space
        clReleaseMemObject(src_work_cl);
    }
    
    //Set it up as an argument
    err = clSetKernelArg(kern, argNum, sizeof(cl_mem), dst_cl);
    assert(err == CL_SUCCESS);
}

void make_output_raster_cl(cl_command_queue cmd_queue, cl_context context, cl_kernel kern,
                           unsigned int x, unsigned int y, FILE *fp,
                           unsigned int argNum, cl_mem *out_cl)
{
    int numThreads = x*y;
    
    if (fp == NULL) {
        //Token memory space
        (*out_cl) = clCreateBuffer(context, CL_MEM_WRITE_ONLY, 1, NULL, &err);
        assert(err == CL_SUCCESS);
    } else {
        //Allocate mem for writing
        unsigned int sz = sizeof(float) * numThreads;
        assert(sz >= 0);
        (*out_cl) = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sz, NULL, &err);
        assert(err == CL_SUCCESS);
    }
    
    //Set it up as an argument
    err = clSetKernelArg(kern, argNum, sizeof(cl_mem), out_cl);
    assert(err == CL_SUCCESS);
}

void calculate_core_cl(int x, int y)
{
	cl_command_queue cmd_queue;
	cl_context context;
    cl_kernel kern;
	size_t groupSize;
    int numThreads = x*y;
    
    cl_device_id dev = get_device();

    cl_mem  sunGeom_cl, sunVarGeom_cl, sunSlopeGeom_cl, const_f_cl, const_i_cl,
            horizon_cl, z_cl, o_cl, s_cl, li_cl, a_cl, lat_cl, long_cl, cbhr_cl, cdhr_cl,
            lumcl_cl, beam_cl, globrad_cl, insol_cl, diff_cl, refl_cl;
    
    // Now create a context to perform our calculation with the specified device 
    context = clCreateContext(0, 1, &dev, NULL, NULL, &err);
    assert(err == CL_SUCCESS);
    
    // And also a command queue for the context
    cmd_queue = clCreateCommandQueue(context, dev, 0, NULL);
	kern = get_kernel("calculate", context, dev);
    
    //What's the recommended group size?
	clGetKernelWorkGroupInfo(kern, dev, CL_KERNEL_WORK_GROUP_SIZE,
							 sizeof(size_t), &groupSize, NULL);
#ifndef NDEBUG
	printf("Recommended Size: %lu\n", group_size);
#endif
    
    //Allocate and copy all the inputs
    make_thread_mem_cl(cmd_queue, context, kern, numThreads, groupSize,
                       &sunGeom_cl, &sunVarGeom_cl, &sunSlopeGeom_cl,
                       &const_f_cl, &const_i_cl);
    
    make_hoz_mem_cl(cmd_queue, context, kern, numThreads, useHorizonData(), horizonarray, &horizon_cl);
    make_input_raster_cl(cmd_queue, context, kern, x, y, 1, 7, z, &z_cl);
    make_input_raster_cl(cmd_queue, context, kern, x, y, aspin == NULL, 8, o, &o_cl);
    make_input_raster_cl(cmd_queue, context, kern, x, y, slopein == NULL, 9, s, &s_cl);
    make_input_raster_cl(cmd_queue, context, kern, x, y, linkein == NULL, 10, li, &li_cl);
    make_input_raster_cl(cmd_queue, context, kern, x, y, albedo == NULL, 11, a, &a_cl);
    make_input_raster_cl(cmd_queue, context, kern, x, y, latin == NULL, 12, latitudeArray, &lat_cl);
    make_input_raster_cl(cmd_queue, context, kern, x, y, longin == NULL, 13, longitudeArray, &long_cl);
    make_input_raster_cl(cmd_queue, context, kern, x, y, coefbh == NULL, 14, cbhr, &cbhr_cl);
    make_input_raster_cl(cmd_queue, context, kern, x, y, coefdh == NULL, 15, cdhr, &cdhr_cl);
    
    //Make space for the outputs
    make_output_raster_cl(cmd_queue, context, kern, x, y, incidout, 16, &lumcl_cl);
    make_output_raster_cl(cmd_queue, context, kern, x, y, beam_rad, 17, &beam_cl);
    make_output_raster_cl(cmd_queue, context, kern, x, y, glob_rad, 18, &globrad_cl);
    make_output_raster_cl(cmd_queue, context, kern, x, y, insol_time, 19, &insol_cl);
    make_output_raster_cl(cmd_queue, context, kern, x, y, diff_rad, 20, &diff_cl);
    make_output_raster_cl(cmd_queue, context, kern, x, y, refl_rad, 21, &refl_cl);
    
    //Do the dirty work
    run_kern_cl(cmd_queue, context, kern, numThreads, groupSize);
    
    //Release unneeded inputs
    clReleaseMemObject(sunGeom_cl);
    clReleaseMemObject(sunVarGeom_cl);
    clReleaseMemObject(sunSlopeGeom_cl);
    clReleaseMemObject(const_f_cl);
    clReleaseMemObject(const_i_cl);
    
    clReleaseMemObject(horizon_cl);
    clReleaseMemObject(z_cl);
    clReleaseMemObject(o_cl);
    clReleaseMemObject(s_cl);
    clReleaseMemObject(li_cl);
    clReleaseMemObject(a_cl);
    clReleaseMemObject(lat_cl);
    clReleaseMemObject(long_cl);
    clReleaseMemObject(cbhr_cl);
    clReleaseMemObject(cdhr_cl);
    
    //Copy requested outputs 
    copy_output_cl(cmd_queue, context, kern, x, y, incidout, lumcl, lumcl_cl);
    copy_output_cl(cmd_queue, context, kern, x, y, beam_rad, beam, beam_cl);
    copy_output_cl(cmd_queue, context, kern, x, y, insol_time, insol, insol_cl);
    copy_output_cl(cmd_queue, context, kern, x, y, diff_rad, diff, diff_cl);
    copy_output_cl(cmd_queue, context, kern, x, y, refl_rad, refl, refl_cl);
    copy_output_cl(cmd_queue, context, kern, x, y, glob_rad, globrad, globrad_cl);
    
    //Release remaining resources
    clReleaseMemObject(lumcl_cl);
    clReleaseMemObject(beam_cl);
    clReleaseMemObject(globrad_cl);
    clReleaseMemObject(insol_cl);
    clReleaseMemObject(diff_cl);
    clReleaseMemObject(refl_cl);
    
    clReleaseKernel(kern);
    clReleaseCommandQueue(cmd_queue);
    clReleaseContext(context);
}
