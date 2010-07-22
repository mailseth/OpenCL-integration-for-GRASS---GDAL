void com_par_const(__global float *sunGeom,
                   __local float *gridGeom,
                   __constant float *const_f,
                   __constant float *const_i,
                   const float longitTime)
{
    unsigned int gid = get_global_id(0);
    unsigned int gsz = const_i[0]*const_i[1];
    float sinlat = gridGeom[6*lsz+lid];
    float coslat = gridGeom[7*lsz+lid];
    
    sunGeom[      gid] =  sinlat*const_f[40];
    sunGeom[  gsz+gid] = -coslat*const_f[39];
    sunGeom[2*gsz+gid] = const_f[40];
    float lum_C31 = sunGeom[3*gsz+gid] = coslat*const_f[40];
    float lum_C33 = sunGeom[4*gsz+gid] = sinlat*const_f[39];
    
    
    if (fabs(lum_C31) < EPS)
        return;
    
    if (const_i[3])
        sunGeom[7*gsz+gid] -= (const_f[38*gsz+gid] + longitTime) * HOURANGLE;
    
    float pom = -lum_C33 / lum_C31;
    
    if (fabs(pom) <= 1.0f) {
        pom = acos(pom) * const_f[4];
        sunGeom[5*gsz+gid] = (90.0f - pom) / 15.0f + 6.0f;
        sunGeom[6*gsz+gid] = (pom - 90.0f) / 15.0f + 18.0f;
    } else if (pom < 0.0f) {
        //Sun is ABOVE the surface during the whole day
        sunGeom[5*gsz+gid] = 0.0f;
        sunGeom[6*gsz+gid] = 24.0f;
    } else if (fabs(pom) - 1.0f <= EPS) {
        //The sun is BELOW the surface during the whole day
        sunGeom[5*gsz+gid] = 12.0f;
        sunGeom[6*gsz+gid] = 12.0f;
    }
}

void com_par(__global float *structs,
             __constant float *const_f,
             __constant float *const_i,
             float latitude,
             float longitude)
{
    unsigned int gid = get_global_id(0);
    unsigned int gsz = const_i[0]*const_i[1];
    float costimeAngle = cos(structs[7*gsz+gid]);
    
    float newLatitude, newLongitude;
    float inputAngle;
    float delt_lat, delt_lon;
    float delt_east, delt_nor;
    float delt_dist;
    
    float lum_C31 = structs[3*gsz+gid];
    float sinSolarAltitude = structs[15*gsz+gid] = lum_C31 * costimeAngle + structs[4*gsz+gid];
    
    if (fabs(lum_C31) < EPS) {
        if (fabs(sinSolarAltitude) >= EPS) {
            if (sinSolarAltitude > 0.0f) {
                structs[5*gsz+gid] = 0.0f;
                structs[6*gsz+gid] = 24.0f;
            } else {
                structs[14*gsz+gid] = 0.0f;
                structs[17*gsz+gid] = UNDEF;
                return;
            }
        } else {
            structs[5*gsz+gid] = 0.0f;
            structs[6*gsz+gid] = 24.0f;
        }
    }
    
    float solarAltitude = structs[14*gsz+gid] = asin(sinSolarAltitude);	/* vertical angle of the sun */
    
    float lum_Lx = -structs[2*gsz+gid] * sin(structs[8*gsz+gid]);
    float lum_Ly = structs[0*gsz+gid] * costimeAngle + structs[1*gsz+gid];
    float xpom = lum_Lx * lum_Lx;
    float ypom = lum_Ly * lum_Ly;
    float pom = sqrt(xpom + ypom);
    
    if (fabs(pom) > EPS) {
        sunVarGeom->solarAzimuth = lum_Ly / pom;
        sunVarGeom->solarAzimuth = acos(sunVarGeom->solarAzimuth);	/* horiz. angle of the Sun */
        /*                      solarAzimuth *= RAD; */
        if (lum_Lx < 0)
            sunVarGeom->solarAzimuth = pi2 - sunVarGeom->solarAzimuth;
    } else {
        sunVarGeom->solarAzimuth = UNDEF;
    }
    
    if (sunVarGeom->solarAzimuth < 0.5 * M_PI)
        sunVarGeom->sunAzimuthAngle = 0.5 * M_PI - sunVarGeom->solarAzimuth;
    else
        sunVarGeom->sunAzimuthAngle = 2.5 * M_PI - sunVarGeom->solarAzimuth;
    
    inputAngle = sunVarGeom->sunAzimuthAngle + pihalf;
    inputAngle = (inputAngle >= pi2) ? inputAngle - pi2 : inputAngle;
    
    delt_lat = -0.0001f * cos(inputAngle);  /* Arbitrary small distance in latitude */
    delt_lon = 0.0001f * sin(inputAngle) / cos(latitude);
    
    newLatitude = (latitude + delt_lat) * rad2deg;
    newLongitude = (longitude + delt_lon) * rad2deg;
    
    //NOTE: this might not work!! We can't call pj_do_proj() here, so
    //we're trying to make use of the calculated lat/long values that were fed
    //into the kernel
    if ((G_projection() != PROJECTION_LL)) {
        if (pj_do_proj(&newLongitude, &newLatitude, &oproj, &iproj) < 0) {
            G_fatal_error("Error in pj_do_proj");
        }
    }
    
    delt_east = newLongitude - gridGeom->xp;
    delt_nor = newLatitude - gridGeom->yp;
    
    delt_dist = sqrt(delt_east * delt_east + delt_nor * delt_nor);
    
    
    sunVarGeom->stepsinangle = gridGeom->stepxy * delt_nor / delt_dist;
    sunVarGeom->stepcosangle = gridGeom->stepxy * delt_east / delt_dist;
    
    sunVarGeom->tanSolarAltitude = tan(solarAltitude);
    
    return;
}

float where_is_point(__global float *sunVarGeom,
                     __local float *gridGeom,
                     __constant float *const_f,
                     __constant float *const_i,
                     
                     __constant float *z,
                     const float xx0,
                     const float yy0)
{
    unsigned int gid = get_global_id(0);
    unsigned int gsz = const_i[0]*const_i[1];
    
    //Offset 0.5 cell size to get the right cell i, j
    int i = (int)(xx0 * const_f[5] + const_f[28]);
    int j = (int)(yy0 * const_f[6] + const_f[29]);
    
    //Check bounds
    if (i > const_i[0] - 1 || j > const_i[1] - 1)
        //Return *something*
        return 9999999.9f;
    
    float xDiff = ((float)(i * const_f[14])) - gridGeom[4*lsz+lid];
    float yDiff = ((float)(j * const_f[15])) - gridGeom[5*lsz+lid];
    
    sunVarGeom[3*gsz+gid] = z[j*const_i[0]+i];
    
    //Used to be distance()
    if (const_i[13])
        return DEGREEINMETERS * sqrt(sunVarGeom[11*gsz+gid] * xDiff*xDiff + yDiff*yDiff);
    else
        return sqrt(xDiff*xDiff + yDiff*yDiff);
}

int searching(__global float *sunVarGeom,
              __local float *gridGeom,
              __constant float *const_f,
              __constant float *const_i,
              __constant float *z )
{
    unsigned int gid = get_global_id(0);
    unsigned int gsz = const_i[0]*const_i[1];
    int success = 0;
    
    if (sunVarGeom[3*gsz+gid] == UNDEFZ)
        return 0;
    
    float xx0 = gridGeom[2*lsz+lid] = sunVarGeom[10*gsz+gid] + gridGeom[2*lsz+lid];
    float yy0 = gridGeom[3*lsz+lid] = sunVarGeom[ 9*gsz+gid] + gridGeom[3*lsz+lid];
    
    if (   ((xx0 + (0.5f * const_f[14])) < 0.0f)
        || ((xx0 + (0.5f * const_f[14])) > const_f[16])
        || ((yy0 + (0.5f * const_f[15])) < 0.0f)
        || ((yy0 + (0.5f * const_f[15])) > const_f[17])) {
        
        success = 3;
    } else {
        success = 1;
        
        float length = where_is_point(sunVarGeom, gridGeom,
                                      const_f, const_i, z, xx0, yy0);
        float z2 = sunVarGeom[gsz+gid] +
            EARTHRADIUS * (1.0f - cos(length / EARTHRADIUS)) +
            length * sunVarGeom[6*gsz+gid];
        
        if (z2 < sunVarGeom[3*gsz+gid])
            success = 2;		/* shadow */
        if (z2 > sunVarGeom[2*gsz+gid])
            success = 3;		/* no test needed all visible */
    }
    
    if (success != 1) {
        gridGeom[2*lsz+lid] = gridGeom[4*lsz+lid];
        gridGeom[3*lsz+lid] = gridGeom[5*lsz+lid];
    }
    
    return success;
}

float lumcline2(__global float *sunGeom,
                __global float *sunVarGeom,
                __global float *sunSlopeGeom,
                __local float *gridGeom,
                __constant float *const_f,
                __constant float *const_i,
                
                __global float *horizonArr,
                __constant float *z,
                const int horizonOff)
{
    unsigned int gid = get_global_id(0);
    unsigned int gsz = const_i[0]*const_i[1];
    float s = 0.0f;
    
    sunVarGeom[gid] = 0.0f;	/* no shadow */
    
    if (const_i[28]) {
        if (const_i[29]) {
            /* Start is due east, sungeom->timeangle = -pi/2 */
            float horizPos = sunVarGeom[8*gsz+gid] / const_f[19];
            int lowPos = (int) horizPos;
            int highPos = lowPos + 1;
            
            if (highPos == const_i[7])
                highPos = 0;
            
            float horizonHeight = const_f[0] * ((1.0f - (horizPos - lowPos)) * horizonpointer[lowPos]
                                                + (horizPos - lowPos) * horizonArr[highPos+horizonOff]);
            
            int isShadow = horizonHeight > sunVarGeom[5*gsz+gid];
            sunVarGeom[gid] = (float) isShadow;
            
            if (!isShadow)
                s = sunSlopeGeom[gsz+gid] * cos(-sunGeom[7*gsz+gid] - sunSlopeGeom[gid])
                    + sunSlopeGeom[2*gsz+gid];	/* Jenco */
        } else {
            int r;
            while ((r = searching(sunVarGeom, gridGeom, const_f, const_i, z)) == 1) {}
            
            if (r == 2)
                sunVarGeom[gid] = 1.0f;	/* shadow */
            else
                s = sunSlopeGeom[gsz+gid] * cos(-sunGeom[7*gsz+gid] - sunSlopeGeom[gid])
                    + sunSlopeGeom[2*gsz+gid];	/* Jenco */
        }
    } else {
        s = sunSlopeGeom[gsz+gid] * cos(-sunGeom[7*gsz+gid] - sunSlopeGeom[gid])
            + sunSlopeGeom[2*gsz+gid];	/* Jenco */
    }
    
    if (s < 0.0f)
        return 0.0f;
    else
        return s;
}

float brad(__global float *sunVarGeom,
           __global float *sunSlopeGeom,
           __constant float *const_f,
           __constant float *const_i,
           
           __constant float *s,
           __constant float *li,
           __global float *cbhr,
           const float sh, float *bh)
{
    unsigned int gid = get_global_id(0);
    unsigned int gsz = const_i[0]*const_i[1];
    float solarAltitude = sunVarGeom[4*gsz+gid];
    float h0refract = solarAltitude + 0.061359f *
        (0.1594f + solarAltitude * (1.123f + 0.065656f * solarAltitude)) /
        (1.0f + solarAltitude * (28.9344f + 277.3971f * solarAltitude));
    float opticalAirMass = exp(-sunVarGeom[gsz+gid] / 8434.5f) / (sin(h0refract) +
                        0.50572f * pow(h0refract * rad2deg + 6.07995f, -1.6364f));
    float rayl, cbh, slope, linke;
    
    if(const_i[15] == NULL)
        slope = const_f[22];
    else
        slope = s[gid] * const_f[3];
    
    if(const_i[16] == NULL)
        linke = const_f[20];
    else
        linke = li[gid];
    
    if(const_i[20] == NULL)
        cbh = 1.0f;
    else
        cbh = cbhr[gid];
    
    if (opticalAirMass <= 20.0f)
        rayl = 1.0f / (6.6296f + opticalAirMass * (1.7513f + opticalAirMass *
                (-0.1202f + opticalAirMass * (0.0065f - opticalAirMass * 0.00013f))));
    else
        rayl = 1.0f / (10.4f + 0.718f * opticalAirMass);
    
    float sinSolarAltitude = sunVarGeom[5*gsz+gid];
    *bh = cbh * const_f[37] * sinSolarAltitude *
                    exp(-rayl * opticalAirMass * 0.8662f * linke);
    
    if (sunSlopeGeom[3*gsz+gid] != UNDEF && slope != 0.0f)
        return *bh * sh / sinSolarAltitude;
    else
        return *bh;
}

float drad(__global float *sunVarGeom,
           __global float *sunSlopeGeom,
           __constant float *const_f,
           __constant float *const_i,
           
           __constant float *s,
           __constant float *li,
           __global float *a,
           __global float *cdhr,
           
           const float sh,
           const float bh,
           const float *rr)
{
    unsigned int gid = get_global_id(0);
    unsigned int gsz = const_i[0]*const_i[1];
    float A1, gh, fg, slope, linke;
    float sinSolarAltitude = sunVarGeom[5*gsz+gid];
    
    if(const_i[15] == NULL)
        slope = const_f[22];
    else
        slope = s[gid] * const_f[3];
    
    if(const_i[16] == NULL)
        linke = const_f[20];
    else
        linke = li[gid];
    
    float cosslope = cos(slope);
    float sinslope = sin(slope);
    float cdh;
    
    float tn = -0.015843f + linke * (0.030543f + 0.0003797f * linke);
    float A1b = 0.26463f + linke * (-0.061581f + 0.0031408f * linke);
    
    if(const_i[21] == NULL)
        cdh = 1.0f;
    else
        cdh = cdhr[gid];
    
    if (A1b * tn < 0.0022f)
        A1 = 0.0022f / tn;
    else
        A1 = A1b;

    float dh = (A1 + (2.04020f + linke * (0.018945f - 0.011161f * linke)) * sinSolarAltitude +
                (-1.3025f + linke * (0.039231f + 0.0085079f * linke)) * sinSolarAltitude * sinSolarAltitude) *
                cdh * const_f[37] * tn;
    
    if (sunSlopeGeom[3*gsz+gid] != UNDEF && slope != 0.0f) {
        float sinHalfSlope = sin(0.5f * slope);
        float solarAltitude = sunVarGeom[4*gsz+gid];
        float fg = sinslope - slope * cosslope - M_PI * sinHalfSlope * sinHalfSlope;
        float r_sky = (1.0f + cosslope) * 0.5f;
        float kb = bh / (const_f[37] * sinSolarAltitude);
        float fx = 0.0f;
        float alb;
        
        if(const_i[17] == NULL)
            alb = const_f[21];
        else
            alb = a[gid];
        
        if (sunVarGeom[gid] > 0.5 || sh <= 0.0f)
            fx = r_sky + fg * 0.252271f;
        else if (solarAltitude >= 0.1f)
            fx = ((0.00263f - kb * (0.712f + 0.6883f * kb)) * fg + r_sky) *
                (1.0f - kb) + kb * sh / sinSolarAltitude;
        else if (solarAltitude < 0.1f) {
            float a_ln = sunVarGeom[7*gsz+gid] - sunSlopeGeom[3*gsz+gid];
            
            if (a_ln > M_PI)
                a_ln -= const_f[3];
            else if (a_ln < -M_PI)
                a_ln += const_f[3];
            
            fx = ((0.00263f - 0.712f * kb - 0.6883f * kb * kb) * fg + r_sky) *
                (1.0f - kb) + kb * sinslope * cos(a_ln) /
                (0.1f - 0.008f * solarAltitude);
        }
        
        /* refl. rad */
        *rr = alb * (bh + dh) * (1.0f - cosslope) * 0.5f;
        return dh * fx;
    } else {	/* plane */
        *rr = 0.0f;
        return dh;
    }
}

void joules2(__global float *sunGeom,
             __global float *sunVarGeom,
             __global float *sunSlopeGeom,
             __local float *gridGeom,
             __constant float *const_f,
             __constant float *const_i,
             
             __global float *horizonArr,
             __constant float *z,
             __constant float *s,
             __constant float *li,
             __global float *a,
             __global float *cbhr,
             __global float *cdhr,
             
             __global float *beam,
             __global float *insol,
             __global float *diff,
             __global float *refl,
             __global float *globrad,
             
             const int horizonOff,
             const float latitude,
             const float longitude)
{
    unsigned int gid = get_global_id(0);
    unsigned int gsz = const_i[0]*const_i[1];
    float dra;
    float firstTime;
    
    //Doubles so summation works better (shouldn't slow much)
    double beam_e = 0.0;
    double diff_e = 0.0;
    double refl_e = 0.0;
    double insol_t = 0.0;
    int insol_count = 0;
    
    com_par(sunGeom, sunVarGeom, gridGeom, const_f, const_i, latitude, longitude);
    
    if (const_i[5] != NULL) {		/*irradiance */
        float s0 = lumcline2(sunGeom, suvVarGeom, sunSlopeGeom, gridGeom,
                             const_f, const_f, horizonArr, z, horizonOff);
        
        if (sunVarGeom[4*gsz+gid] > 0.0f) {
            float bh;
            if (sunVarGeom[gid] < 0.5f && s0 > 0.0f) {
                beam_e = brad(sunVarGeom, sunSlopeGeom,
                              const_f, const_i, cbhr, s, s0, &bh);	/* beam radiation */
            } else {
                beam_e = 0.0f;
                bh = 0.0f;
            }
            
            float rr = 0.0f;
            
            if ((const_i[25] != NULL) || (const_i[27] != NULL))
                diff_e = drad(sunVarGeom, sunSlopeGeom, const_f, const_i, a, cdhr, s0, bh, &rr);	/* diffuse rad. */;
            if ((const_i[26] != NULL) || (const_i[27] != NULL)) {
                if ((const_i[25] == NULL) && (const_i[27] == NULL))
                    drad(sunVarGeom, sunSlopeGeom, const_f, const_i, a, cdhr, s0, bh, &rr);
                refl_e = rr;	/* reflected rad. */
            }
        }			/* solarAltitude */
    } else {
        /* all-day radiation */
        float sunrise_time = sunGeom[5*gsz+gid];
        int srStepNo = (int)(sunrise_time / const_f[12]);
        float lastAngle = (sunGeom[6*gsz+gid] - 12.0f) * HOURANGLE;
        float firstTime;
        
        if ((sunrise_time - srStepNo * const_f[12]) > 0.5f * const_f[12])
            firstTime = (srStepNo + 1.5f) * const_f[12];
        else
            firstTime = (srStepNo + 0.5f) * const_f[12];
        
        float timeAngle = (firstTime - 12.0f) * HOURANGLE;
        
        do {
            com_par(sunGeom, sunVarGeom, gridGeom, const_f, const_i, latitude, longitude);
            float s0 = lumcline2(sunGeom, sunVarGeom, sunSlopeGeom, gridGeom,
                                 const_f, const_i, horizonArr, z, horizonOff);
            
            if (sunVarGeom[4*gsz+gid] > 0.0f) {
                if (sunVarGeom[gid] < 0.5f && s0 > 0.0f) {
                    ++insol_count;
                    beam_e += const_f[12] * brad(sunVarGeom, sunSlopeGeom,
                                                 const_f, const_i, s, cbhr, s0, &bh);
                } else {
                    bh = 0.0f;
                }
                
                float rr = 0.0f;
                if ((const_i[25] != NULL) || (const_i[27] != NULL))
                    diff_e += const_f[12] * drad(sunVarGeom, sunSlopeGeom,
                                                 const_f, const_i, a, cdhr, s0, bh, &rr);
                if ((const_i[26] != NULL) || (const_i[27] != NULL)) {
                    if ((const_i[25] == NULL) && (const_i[27] == NULL))
                        drad(sunVarGeom, sunSlopeGeom, const_f, const_i, a, cdhr, s0, bh, &rr);
                    refl_e += const_f[12] * rr;
                }
            }			/* illuminated */
            
            timeAngle += const_f[12] * HOURANGLE;
        } while (timeAngle > lastAngle); /* we've got the sunset */
        
        sunGeom[7*gsz+gid] = timeAngle;
    }				/* all-day radiation */
    
    //Only apply values to where they're wanted
    if(const_i[23] != NULL)
        beam[gid] = (float)beam_e;
    if(const_i[24] != NULL)
       insol[gid] = const_f[12]*insol_count;
    if(const_i[25] != NULL)
        diff[gid] = (float)diff_e;
    if(const_i[26] != NULL)
        refl[gid] = (float)refl_e;
    if(const_i[27] != NULL)
        globrad[gid] = (float)(beam_e + diff_e + refl_e)
}

__kernel calculate(__global float *sunGeom,
                   __global float *sunVarGeom,
                   __global float *sunSlopeGeom,
                   __local float *gridGeom,
                   __constant float *const_f,
                   __constant float *const_i,
                   
                   __global float *horizonArr,
                   __constant float *z,
                   __global float *o,
                   __constant float *s,
                   __constant float *li,
                   __global float *a,
                   __constant float *latitudeArray,
                   __constant float *longitudeArray,
                   __global float *cbhr,
                   __global float *cdhr,

                   __global float *lumcl,
                   __global float *beam,
                   __global float *globrad,
                   __global float *insol,
                   __global float *diff,
                   __global float *refl )
{
    unsigned int gid = get_global_id(0);
    unsigned int gsz = const_i[0]*const_i[1];
    unsigned int lid = get_local_id(0);
    unsigned int lsz = get_local_size(0);
    float longitTime = 0.0f;
    float o_orig;
    
    //Don't overrun arrays
    if (gid >= gsz)
        return;
    
    if (const_i[3])
        longitTime = -longitudeArray[gid] / 15.0f;
    
    gridGeom[4*lsz+lid] = gridGeom[2*lsz+lid] = (float)(gid / const_i[1]) *const_f[14];
    gridGeom[5*lsz+lid] = gridGeom[3*lsz+lid] = (float)(gid % const_i[1]) *const_f[15];
    
    gridGeom[    lid] = const_f[7] + gridGeom[2*lsz+lid];
    gridGeom[lsz+lid] = const_f[8] + gridGeom[3*lsz+lid];
    
    if (const_i[13]) {
        float coslat = cos(const_f[3] * gridGeom[lsz+lid]);
        sunVarGeom[11*gsz+gid] = coslat * coslat;
    }
    
    float z1 = sunVarGeom[gsz+gid] = sunVarGeom[3*gsz+gid] = z[gid];
    
    if (z1 == UNDEFZ)
        return;
    
    float latitude, longitude, aspect, slope;
    
    if (const_i[14] != NULL) {
        if (o[gid] != 0.0f)
            aspect = sunSlopeGeom[3*gsz+gid] = o[gid] * const_f[3];
        else
            aspect = sunSlopeGeom[3*gsz+gid] = UNDEF;
    } else {
        aspect = sunSlopeGeom[3*gsz+gid] = const_f[23];
    }
    
    if (const_i[18] != NULL)
        latitude = latitudeArray[gid]*const_f[3];
    
    if (const_i[19] != NULL)
        longitude = longitudeArray[gid]*const_f[3];
    
    if (const_i[9] == PROJECTION_LL) {		/* ll projection */
        longitude = gridGeom[lid]*const_f[3];
        latitude = gridGeom[lsz+lid]*const_f[3];
    }
    
    if (const_i[15] == NULL)
        slope = const_f[22];
    else
        slope = s[gid];
        
    float cos_u = cos(const_f[1] - slope);	/* = sin(slope) */
    float sin_u = sin(const_f[1] - slope);	/* = cos(slope) */
    float cos_v = cos(const_f[1] + aspect);
    float sin_v = sin(const_f[1] + aspect);
    
    if (const_i[5] != NULL)
        sunGeom[7*gsz+gid] = const_i[8];
    
    float geom_sinlat = gridGeom[6*lsz+lid] = sin(-latitude);
    float geom_coslat = gridGeom[7*lsz+lid] = cos(-latitude);
    float sin_phi_l = -geom_coslat * cos_u * sin_v + geom_sinlat * sin_u;
    
    sunSlopeGeom[      gid] = atan(-cos_u * cos_v / (geom_sinlat * cos_u * sin_v + geom_coslat * sin_u));
    sunSlopeGeom[  gsz+gid] = cos(asin(sin_phi_l)) * const_f[40];
    sunSlopeGeom[2*gsz+gid] = sin_phi_l * const_f[39];
    
    if ((const_i[22] != NULL) || someRadiation)
        com_par_const(sunGeo, gridGeom, const_f, const_i, longitTime);
    
    if (const_i[22] != NULL) {
        com_par(sunGeom, sunVarGeom, gridGeom,
                const_f, const_i, latitude, longitude);
        float lum = lumcline2(sunGeom, sunVarGeom, sunSlopeGeom, gridGeom,
                              const_f, const_f, horizonArr, z, gid*const_i[7]);
        
        if (lum > 0.0f) {
            lum = rad2deg * asin(lum);
            lumcl[gid] = (float)lum;
        } else {
            lumcl[gid] = UNDEFZ;
        }
    }

    if (someRadiation) {
        joules2(sunGeom, sunVarGeom, sunSlopeGeom, gridGeom,
                const_f, const_i, horizonArr, z, s, li, a, cbhr, cdhr,
                beam, insol, diff, refl, globrad,
                gid*const_i[7], latitude, longitude);
    }
}