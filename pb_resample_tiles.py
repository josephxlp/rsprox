#!/usr/bin/env python
# coding: utf-8

import os 
import time 
from glob import glob
from upaths import (D12PATH,D30PATH,DPATH,varname_list,Xlist,
                    OPEN_TOPOGRAPHY_DPATH,RESAMPLE_VAR_ENGING,RESAMPLE_VAR_SPECTIAL_ENGING)
from concurrent.futures import ProcessPoolExecutor
import rasterio
from rasterio.enums import Resampling

def list_base_files(base_dpath, varname):
    bpattern = f"{base_dpath}/*/{varname}/*tif"
    bfiles = glob(bpattern)
    print(len(bfiles))
    return bfiles

def resample_raster(fipath, bipath, fopath, algo=Resampling.nearest):

    if not os.path.isfile(fopath):
        with rasterio.open(fipath) as src, rasterio.open(bipath) as ref:
            input_data = src.read(1)
            transform = ref.transform
            width = ref.width
            height = ref.height

            crs = src.crs
            dtype = src.dtypes[0]

            resampled_data = src.read(1,out_shape=(height, width),resampling=algo)
            # change this to match data type 

            # Create a new raster to write the resampled data with the properties from the input and reference
            with rasterio.open(fopath,'w',driver='GTiff',count=1,dtype=dtype,
                crs=crs,transform=transform,width=width,height=height) as dst:
                dst.write(resampled_data, 1)
    else:
        print('already created')


def filter_files_by_endingwith(files, var_ending):
    filtered_files = [f for f in files if any(f.endswith(ending) for ending in var_ending)]
    print(f"Filtered files count: {len(filtered_files)}/{len(files)}")
    return filtered_files

def process_basefile(basefile,D12PATH,DXPATH, RESAMPLE_VAR_ENGING, RESAMPLE_VAR_SPECTIAL_ENGING):
    tilename = basefile.split('/')[-3]
    print(tilename)
    t12path = os.path.join(D12PATH, tilename)
    tXdpath = os.path.join(DXPATH, tilename)
    os.makedirs(tXdpath, exist_ok=True)
    t12files = glob(f'{t12path}/*.tif')
    t12files = filter_files_by_endingwith(t12files, RESAMPLE_VAR_ENGING)
    #print(len(t12files))

    txfiles = [os.path.join(tXdpath,os.path.basename(i)) for i in t12files]
    #special_endings = ['multi_ESAWC.tif', 'tdem_DEM__Mw.tif', 'LWM.tif']

    for idx in range(len(t12files)):
        fipath = t12files[idx]
        if any(fipath.endswith(ending) for ending in RESAMPLE_VAR_SPECTIAL_ENGING):  # Fixed instruction
            print(f'running special_endings at Resampling.nearest {os.path.basename(fipath)}')
            resample_raster(fipath=fipath, bipath=basefile, fopath=txfiles[idx], algo=Resampling.nearest)
        else:
            print('running all_other_endings at Resampling.bilinear')
            resample_raster(fipath=fipath, bipath=basefile, fopath=txfiles[idx], algo=Resampling.bilinear)


if __name__ == '__main__':
    tilenames = os.listdir(D12PATH)
    with ProcessPoolExecutor(17) as PEX:
        for i, varname in enumerate(varname_list):
            #if i > 0: break
            DXPATH = f'{DPATH}{Xlist[i]}'
            print(f'{varname}::{DXPATH}')
            basefiles = list_base_files(OPEN_TOPOGRAPHY_DPATH,varname)
            for j, basefile in enumerate(basefiles):
                #if j > 0: break
                ti = time.perf_counter()
                #process_basefile(basefile,D12PATH,DXPATH, RESAMPLE_VAR_ENGING, RESAMPLE_VAR_SPECTIAL_ENGING)
                PEX.submit(process_basefile,basefile,D12PATH,DXPATH, RESAMPLE_VAR_ENGING, RESAMPLE_VAR_SPECTIAL_ENGING)
                tf = time.perf_counter() - ti 
                print(f'[INFO] ::: {tf/60} min(s)')


    tb = time.perf_counter() - ti 
    print(f'[INFO] ::: {tb/60} min(s)')

