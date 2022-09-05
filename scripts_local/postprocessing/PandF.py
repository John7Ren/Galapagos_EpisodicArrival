from netCDF4 import Dataset
import numpy as np
import xarray as xr
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import matplotlib.pyplot as plt
import numpy as np
import math
from datetime import timedelta
from datetime import datetime
from operator import attrgetter
import xarray as xr
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation
from matplotlib import colors,dates
import matplotlib.gridspec as gridspec
from glob import glob
import cmocean.cm as cmo
from matplotlib.path import Path
import matplotlib.patches as patches
import pandas as pd
import sys
import os
import matplotlib.ticker as ticker
from windrose import WindroseAxes
from matplotlib.projections import register_projection
from matplotlib.projections import polar
import collections
#%% Function to generate data
def CalculateZeta(U,V):
    C = 40000e3 # equator circumference
    dc = C/360
    Zeta = V.differentiate('x')/(V.nav_lon.differentiate('x')*dc) - U.differentiate('y')/(U.nav_lat.differentiate('y')*dc)
    Zeta = Zeta.assign_coords(nav_lat=U.nav_lat)
    Zeta = Zeta.assign_coords(nav_lon=U.nav_lon)
    
    return Zeta

def FindField(lon_l, lat_l, ds):
    nav_lon = ds.nav_lon
    nav_lat = ds.nav_lat
    idx_lon = np.abs(nav_lon-lon_l).argmin(axis=1)
    idx_lat = np.abs(nav_lat-lat_l).argmin(axis=0)
    # if len(np.unique(idx_lon)) > 1 or len(np.unique(idx_lat)) > 1:
    #     print(f'multiple corresponding coordinates exist: idx_lon = {np.unique(idx_lon)}; idx_lat = {np.unique(idx_lat)}')
    #     print(f'the frequencies of lons: {np.trim_zeros(np.unique(np.bincount(idx_lon)))}; the frequencies of lats: {np.trim_zeros(np.unique(np.bincount(idx_lat)))}')
    idx_lon = np.bincount(idx_lon).argmax() # considering multiple coordinates exist, we choose the most frequent idx
    idx_lat = np.bincount(idx_lat).argmax()
    var = ds[idx_lat,idx_lon].item()
    
    return var

def Density_landmask(lons,lats,Lons,Lats):
    bins = [Lons,Lats]
    index = ~np.isnan(lons) & ~np.isnan(lats) # some spots are not nan but strangely big or small
    lons = lons[index]
    lats = lats[index]
    H,xe,ye = np.histogram2d(lons, lats, bins=bins)
    xb = ( xe[:-1] + xe[1:] ) / 2
    yb = ( ye[:-1] + ye[1:] ) / 2
        
    return H,xb,yb

# Draw boxes
def DrawCluster(lon, lat, radius):
    '''
    [0,1]-----------[1,1]
      |               |
      |   [lon,lat]   |
      |               |
    [0,0]-----------[1,0]  
    '''
    verts = [
             (lon-radius, lat-radius), # [0,0]
             (lon-radius, lat+radius), # [0,1]
             (lon+radius, lat+radius), # [1,1]
             (lon+radius, lat-radius), # [1,0]
             (lon-radius, lat-radius)  # [0,0]
             ]
    codes = [
             Path.MOVETO,
             Path.LINETO,
             Path.LINETO,
             Path.LINETO,
             Path.CLOSEPOLY
             ]
    
    path = Path(verts,codes)
    
    return path

def DrawRegion(box):
    '''
    [0,1]-----------[1,1]
      |               |
      |   [lon,lat]   |
      |               |
    [0,0]-----------[1,0]  
    '''
    if box == 'PNJ':
        verts = [
                 (-87, 1), # [0,0]
                 (-80, 8), # [0,1]
                 (-77, 8), # [1,1]
                 (-84, 1), # [1,0]
                 (-87, 1)  # [0,0]
                 ]
    elif box == 'PNJtilt':
        verts = [
                 (-87, 1),      # [0,0]
                 (-80, 8),      # [0,1]
                 (-78.5, 6.5),  # [1,1]
                 (-85.5, -0.5), # [1,0]
                 (-87, 1)       # [0,0]
                 ]
    elif box == 'PNJtilt_wide':
        verts = [
                 (-90, 1.),      # [0,0]
                 (-81.5, 9.5),      # [0,1]
                 (-78.5, 6.5),  # [1,1]
                 (-87, -2.), # [1,0]
                 (-90, 1.)       # [0,0]
                 ]
    elif box == 'SEC':
        verts = [
                 (-92, -2), # [0,0]
                 (-92, 2), # [0,1]
                 (-85, 2), # [1,1]
                 (-85, -2), # [1,0]
                 (-92, -2)  # [0,0]
                 ]
    elif box == 'Gala':
        verts = [
                 (-91.8, -1.4), # [0,0]
                 (-91.8, 0.7), # [0,1]
                 (-89, 0.7), # [1,1]
                 (-89, -1.4), # [1,0]
                 (-91.8, -1.4)  # [0,0]
                 ]
    elif box == 'NECC':
        verts = [
                 (-100, 6), # [0,0]
                 (-100, 9), # [0,1]
                 (-90, 9), # [1,1]
                 (-90, 6), # [1,0]
                 (-100, 6)  # [0,0]
                 ]
    elif box == 'PB':
        verts = [
                 (-81, -2), # [0,0]
                 (-81, 9), # [0,1]
                 (-77, 9), # [1,1]
                 (-77, -2), # [1,0]
                 (-81, -2)  # [0,0]
                 ]
    elif box == 'PBSR':
        verts = [
                 (-90, 0), # [0,0]
                 (-90, 10), # [0,1]
                 (-73, 10), # [1,1]
                 (-73, 0), # [1,0]
                 (-90, 0)  # [0,0]
                 ]
    elif box == 'NR':
        verts = [
                 (-102, 8), # [0,0]
                 (-102, 18), # [0,1]
                 (-84, 18), # [1,1]
                 (-84, 8), # [1,0]
                 (-102, 8)  # [0,0]
                 ]
    elif box == 'NHCS':
        verts = [
                 (-90, -20), # [0,0]
                 (-90, 0), # [0,1]
                 (-70, 0), # [1,1]
                 (-70, -20), # [1,0]
                 (-90, -20)  # [0,0]
                 ]
    elif box == 'HC':
        verts = [
                 (-90, -6), # [0,0]
                 (-90, 0), # [0,1]
                 (-80, 0), # [1,1]
                 (-80, -6), # [1,0]
                 (-90, -6)  # [0,0]
                 ]
        
    codes = [
             Path.MOVETO,
             Path.LINETO,
             Path.LINETO,
             Path.LINETO,
             Path.CLOSEPOLY
             ]
    
    path = Path(verts,codes)
    
    return path

def createlevels(H, nloop):
    per = 100
    levels = np.unique(np.percentile(H,np.linspace(0,100,per+1)))
    nlevels = len(levels)
    loop = 0
    while nlevels<100:
        per = per*2
        levels = np.unique(np.percentile(H,np.linspace(0,100,per+1)))
        nlevels = len(levels)
        loop += 1
        if loop > nloop:
            print(f'loop:{loop}')
            break
    # levels = np.delete(levels,0)
    ticks = np.percentile(levels,np.linspace(0,100,6))
    # round the digits of the ticks
    for i in range(len(ticks)):
        if ticks[i] != 0:
            digits = 10**(math.floor(np.log10(ticks[i]))-1)
            ticks[i] = round(ticks[i]/digits) * digits
    
    return levels,ticks

def tksfmt(x, pos):
    a, b = '{:.0e}'.format(x).split('e')
    b = int(b)
    return r'${} \times 10^{{{}}}$'.format(a, b)


def GenerateTrajData(simyears, backtrack, lorenz, simdays=729, releasedays=365, dailyrelease=274):
    '''
    Parameters
    ----------
    simyears : array,
        the start years of the simulations
    backtrack : list or bool
        theoratically start from 0 (the arriving time of the parcel) to
        the time I want to stop backtracking;
        backtrack[0] -> the start time
        backtrack[1] -> the end time, if backtrack[1] == 999, it means the generated data covers all the time from
        arriving time to the releasing time.
    lorenz : bool,
        whether to operate on Lorenz or Local.
    simdays : int, optional
        The default is 729.
    releasedays : int, optional
        The default is 365.
    dailyrelease : int, optional
        The default is 274.

    Returns
    -------
    None.

    '''
    if lorenz:
        input_dir = '/storage/home/9703243/galapagos/trajectories/arriving_gala/'
        output_dir = f'/storage/home/9703243/galapagos/trajectories/{backtrack[0]}-{backtrack[1]}/'
        os.makedirs(os.path.dirname(output_dir),exist_ok=True)
        priors = pd.read_csv('/storage/home/9703243/galapagos/scripts/priors_river_inputs_25N25S_v0.csv',index_col=0)
    else:
        input_dir = '/Users/renjiongqiu/Documents_local/Thesis/visualization/TrajAnalysis/trajectories/arriving_gala/'
        output_dir = f'/Users/renjiongqiu/Documents_local/Thesis/visualization/TrajAnalysis/trajectories/{backtrack[0]}-{backtrack[1]}_uniq/'
        os.makedirs(os.path.dirname(output_dir),exist_ok=True)
        priors = pd.read_csv('/Users/renjiongqiu/Documents_local/Thesis/data/priors_river_inputs_25N25S.csv',index_col=0)
    
    galapagos_extent = [-91.8, -89, -1.4, 0.7]

    for yr in simyears:
        secondyear = yr + 1
        
        fname = f'_{yr}-{secondyear}_run{simdays}d_release{releasedays}d_{dailyrelease}par_beaching'
        
        sources = list(priors.index)
    
        Lon_in = np.load(f'{input_dir}Lon{fname}.npy',allow_pickle=True)
        Lat_in = np.load(f'{input_dir}Lat{fname}.npy',allow_pickle=True)
        Time_in = np.load(f'{input_dir}Time{fname}.npy',allow_pickle=True)
        Age_in = np.load(f'{input_dir}Age{fname}.npy',allow_pickle=True)
        print(f'load data: {yr}')
        
        Lon_out = {}
        Lat_out = {}
        Time_out = {}
        Age_out = {}
    
        for iloc,loc in enumerate(sources):
            lons = Lon_in[iloc][0]
            lats = Lat_in[iloc][0]
            age = Age_in[iloc][0]
            time = Time_in[iloc][0]
            print(f'generate arrays: {loc}')
            
                
            # Get the mask of gala region
            # Note that gala_traj and gala_obsv are both 1D and are spatially correlated to present a specific point
            gala_traj,gala_obsv = np.where( (lons>=galapagos_extent[0]) & (lons<=galapagos_extent[1]) & \
                                            (lats>=galapagos_extent[2]) & (lats<=galapagos_extent[3]) )
            
            # Select the trajectories that passed Gala
            # 1st contact
            traj_idx = np.unique(gala_traj)
            obsv_idx = np.zeros_like(traj_idx)
            
            for i,idx in enumerate(traj_idx):
                obsv_idx[i] = gala_obsv[np.where(gala_traj==idx)].min() # the observation that par arrives
            
            if len(obsv_idx)>0:
                if backtrack[1] == 999:
                    Lon = np.nan * np.ones((len(traj_idx),max(obsv_idx)))
                    Lat = np.nan * np.ones((len(traj_idx),max(obsv_idx)))
                    Time = np.nan * np.ones((len(traj_idx),max(obsv_idx)))
                    Age = np.nan * np.ones((len(traj_idx),max(obsv_idx)))
                    
                    obsv_ed = 0 # backtrack to the releasing time
                    
                    for i,idx in enumerate(traj_idx):
                        obsv_st = obsv_idx[i]-backtrack[0]
                        length = obsv_st - obsv_ed
                        coords = np.empty((length,),dtype=object)
                        coords[[i for i in range(length)]] = [i for i in zip(lons[idx,obsv_ed:obsv_st],lats[idx,obsv_ed:obsv_st])]
                        _,idx_uniq = np.unique(coords,return_index=True)
                        
                        length = len(idx_uniq)
                        Lon[i,:length] = lons[idx,obsv_ed:obsv_st][idx_uniq]
                        Lat[i,:length] = lats[idx,obsv_ed:obsv_st][idx_uniq]
                        Time[i,:length] = time[idx,obsv_ed:obsv_st][idx_uniq]
                        Age[i,:length] = age[idx,obsv_ed:obsv_st][idx_uniq]
                        
                elif backtrack[1] != 999:
                    Lon = np.nan * np.ones((len(traj_idx),backtrack[1]-backtrack[0]))
                    Lat = np.nan * np.ones((len(traj_idx),backtrack[1]-backtrack[0]))
                    Time = np.nan * np.ones((len(traj_idx),backtrack[1]-backtrack[0]))
                    Age = np.nan * np.ones((len(traj_idx),backtrack[1]-backtrack[0]))
                    
                    for i,idx in enumerate(traj_idx):
                        obsv_ed = max(0,obsv_idx[i]-backtrack[1])
                        obsv_st = obsv_idx[i]-backtrack[0]
                        if obsv_st < 0:
                            print('the start point of tracking is too early, this is meaningless')
                        length = obsv_st - obsv_ed
                        coords = np.empty((length,),dtype=object)
                        coords[[i for i in range(length)]] = [i for i in zip(lons[idx,obsv_ed:obsv_st],lats[idx,obsv_ed:obsv_st])]
                        _,idx_uniq = np.unique(coords,return_index=True)
                        
                        length = len(idx_uniq)
                        Lon[i,:length] = lons[idx,obsv_ed:obsv_st][idx_uniq]
                        Lat[i,:length] = lats[idx,obsv_ed:obsv_st][idx_uniq]
                        Time[i,:length] = time[idx,obsv_ed:obsv_st][idx_uniq]
                        Age[i,:length] = age[idx,obsv_ed:obsv_st][idx_uniq]
            
            else:
                Lon = lons[traj_idx,:]
                Lat = lats[traj_idx,:]
                Time = time[traj_idx,:]
                Age = age[traj_idx,:]
            
            Lon_out[loc] = Lon
            Lat_out[loc] = Lat
            Time_out[loc] = Time
            Age_out[loc] = Age
            
            print(f'finish the selection: {loc}')
        
        suffix = f'{backtrack[0]}-{backtrack[1]}d_before'
        
        # Save to npy
        Lon_npy = pd.DataFrame.from_dict(Lon_out,orient='index')
        Lat_npy = pd.DataFrame.from_dict(Lat_out,orient='index')
        Time_npy = pd.DataFrame.from_dict(Time_out,orient='index')
        Age_npy = pd.DataFrame.from_dict(Age_out,orient='index')
        np.save(f'{output_dir}Lon{fname}_{suffix}.npy',Lon_npy,allow_pickle=True)
        np.save(f'{output_dir}Lat{fname}_{suffix}.npy',Lat_npy,allow_pickle=True)
        np.save(f'{output_dir}Time{fname}_{suffix}.npy',Time_npy,allow_pickle=True)
        np.save(f'{output_dir}Age{fname}_{suffix}.npy',Age_npy,allow_pickle=True)
        print(f'finish the saving: {fname}_{suffix} \n')

def GenerateFieldData(simyears, backtrack, simdays=729, releasedays=365, dailyrelease=274):
    input_dir = f'/storage/home/9703243/galapagos/trajectories/{backtrack[0]}-{backtrack[1]}/'
    output_dir = f'/storage/home/9703243/galapagos/fields/{backtrack[0]}-{backtrack[1]}/'
    os.makedirs(os.path.dirname(output_dir),exist_ok=True)
    priors = pd.read_csv('/storage/home/9703243/galapagos/scripts/priors_river_inputs_25N25S_v0.csv',index_col=0)
    data_path = "/storage/shared/oceanparcels/input_data/MOi/psy4v3r1/" # for field data
    
    # [-125,-65,-25,25] the same index as the landmask
    xslice = slice(2066,2665)
    yslice = slice(1185,1804)
    
    suffix = f'{backtrack[0]}-{backtrack[1]}d_before'
    
    for yr in simyears:
    
        secondyear = yr + 1
        fname = f'_{yr}-{secondyear}_run{simdays}d_release{releasedays}d_{dailyrelease}par_beaching'
        
        sources = list(priors.index)
    
        Lon_in = np.load(f'{input_dir}Lon{fname}_{suffix}.npy',allow_pickle=True)
        Lat_in = np.load(f'{input_dir}Lat{fname}_{suffix}.npy',allow_pickle=True)
        Time_in = np.load(f'{input_dir}Time{fname}_{suffix}.npy',allow_pickle=True)
        Age_in = np.load(f'{input_dir}Age{fname}_{suffix}.npy',allow_pickle=True)
        print(f'load data: {yr}')
        
        SSH_out = {}
        Zeta_out = {}
        Theta_out = {}
        SPD_out = {}
        U_out = {}
        V_out = {}
        Time_out = {}
        Age_out = {}
        Lon_out = {}
        Lat_out = {}
    
        for iloc,loc in enumerate(sources):
            lons = Lon_in[iloc][0]
            lats = Lat_in[iloc][0]
            time = Time_in[iloc][0]
            age = Age_in[iloc][0]
            print(f'generate arrays at: {loc}')
            
            # create the timeline
            time = time.astype('datetime64[ns]').astype('datetime64[D]') # convert the datatype from 'datetime64[ns]' to 'datetime64[D]'
            timeline = np.arange(np.nanmin(time),np.nanmax(time)+1,1)
            print(f'from {timeline[0]} to {timeline[-1]}')
            
            SSH_loc = np.array([])
            Zeta_loc = np.array([])
            Theta_loc = np.array([])
            SPD_loc = np.array([])
            U_loc = np.array([])
            V_loc = np.array([])
            Age_loc = np.array([])
            Time_loc = np.array([],dtype='datetime64[D]')
            Lon_loc = np.array([])
            Lat_loc = np.array([])
            
            for t in timeline:
                # find the corresponding lons and lats in trajectories
                print(f'calculating t = {t}')
                idx = np.where(time==t)
                lons_t = lons[idx]
                lats_t = lats[idx]
                age_t = age[idx]
                
                # load the field data
                ufiles = glob(data_path+'psy*U_'+f'{t}*')
                vfiles = glob(data_path+'psy*V_'+f'{t}*')
                hfiles = glob(data_path+'psy*2D_'+f'{t}*')
            
                U = xr.load_dataset(ufiles[0]).vozocrtx.sel(y=yslice,x=xslice).isel(deptht=0)
                V = xr.load_dataset(vfiles[0]).vomecrty.sel(y=yslice,x=xslice).isel(deptht=0)
                SSH = xr.load_dataset(hfiles[0]).sossheig.sel(y=yslice,x=xslice).isel(deptht=0)
                Zeta = CalculateZeta(U,V)
                
                # find the corresponding field data
                if len(lons_t) == len(lats_t):
                    for l in range(len(lons_t)):
                        lon_l = lons_t[l]
                        lat_l = lats_t[l]
                        age_l = age_t[l]
                        # print(f'the coordinate I am looking for is: ({lon_l},{lat_l}) in (lon,lat)')
                        u = FindField(lon_l,lat_l,U) # find the corresponding u
                        v = FindField(lon_l,lat_l,V) # find the corresponding v
                        ssh = FindField(lon_l,lat_l,SSH) # find the corresponding ssh
                        zeta = FindField(lon_l,lat_l,Zeta) # find the corresponding zeta
                        
                        # calculate the relevant variables
                        spd = np.hypot(u,v)
                        theta = np.array(np.arccos(u/spd)*np.sign(v))
                        if theta < 0:
                            theta += 2*np.pi
                        theta = np.rad2deg(theta)
                        
                        # append the value to the array
                        SSH_loc = np.append(SSH_loc,ssh)
                        Zeta_loc = np.append(Zeta_loc,zeta)
                        Theta_loc = np.append(Theta_loc,theta)
                        SPD_loc = np.append(SPD_loc,spd)
                        U_loc = np.append(U_loc,u)
                        V_loc = np.append(V_loc,v)
                        Age_loc = np.append(Age_loc,age_l)
                        Time_loc = np.append(Time_loc,t)
                        Lon_loc = np.append(Lon_loc,lon_l)
                        Lat_loc = np.append(Lat_loc,lat_l)
            
            SSH_out[loc] = SSH_loc
            Zeta_out[loc] = Zeta_loc
            Theta_out[loc] = Theta_loc
            SPD_out[loc] = SPD_loc
            U_out[loc] = U_loc
            V_out[loc] = V_loc
            Time_out[loc] = Time_loc
            Age_out[loc] = Age_loc
            Lon_out[loc] = Lon_loc
            Lat_out[loc] = Lat_loc
            
            # print(f'finish the selection: {loc} \n\n\n')
        
        suffix = f'{backtrack[0]}-{backtrack[1]}d_before'
        
        # Save to npy
        SSH_npy = pd.DataFrame.from_dict(SSH_out,orient='index')
        Zeta_npy = pd.DataFrame.from_dict(Zeta_out,orient='index')
        Theta_npy = pd.DataFrame.from_dict(Theta_out,orient='index')
        SPD_npy = pd.DataFrame.from_dict(SPD_out,orient='index')
        U_npy = pd.DataFrame.from_dict(U_out,orient='index')
        V_npy = pd.DataFrame.from_dict(V_out,orient='index')
        Time_npy = pd.DataFrame.from_dict(Time_out,orient='index')
        Age_npy = pd.DataFrame.from_dict(Age_out,orient='index')
        Lon_npy = pd.DataFrame.from_dict(Lon_out,orient='index')
        Lat_npy = pd.DataFrame.from_dict(Lat_out,orient='index')
        np.save(f'{output_dir}SSH{fname}_{suffix}.npy',SSH_npy,allow_pickle=True)
        np.save(f'{output_dir}Zeta{fname}_{suffix}.npy',Zeta_npy,allow_pickle=True)
        np.save(f'{output_dir}Theta{fname}_{suffix}.npy',Theta_npy,allow_pickle=True)
        np.save(f'{output_dir}SPD{fname}_{suffix}.npy',SPD_npy,allow_pickle=True)
        np.save(f'{output_dir}U{fname}_{suffix}.npy',U_npy,allow_pickle=True)
        np.save(f'{output_dir}V{fname}_{suffix}.npy',V_npy,allow_pickle=True)
        np.save(f'{output_dir}Time{fname}_{suffix}.npy',Time_npy,allow_pickle=True)
        np.save(f'{output_dir}Age{fname}_{suffix}.npy',Age_npy,allow_pickle=True)
        np.save(f'{output_dir}Lon{fname}_{suffix}.npy',Lon_npy,allow_pickle=True)
        np.save(f'{output_dir}Lat{fname}_{suffix}.npy',Lat_npy,allow_pickle=True)
        print(f'finish the saving: {fname}_{suffix} \n')
        
def GetFieldsDate(simyears, backtrack, simdays=729, releasedays=365, dailyrelease=274):
    input_dir = f'/Users/renjiongqiu/Documents_local/Thesis/visualization/TrajAnalysis/trajectories/{backtrack[0]}-{backtrack[1]}/'
    output_dir = f'/Users/renjiongqiu/Documents_local/Thesis/visualization/TrajAnalysis/fields/{backtrack[0]}-{backtrack[1]}/'
    os.makedirs(os.path.dirname(output_dir),exist_ok=True)
    priors = pd.read_csv('/Users/renjiongqiu/Documents_local/Thesis/data/priors_river_inputs_25N25S.csv',index_col=0)
    
    suffix = f'{backtrack[0]}-{backtrack[1]}d_before'
    
    for yr in simyears:
    
        secondyear = yr + 1
        fname = f'_{yr}-{secondyear}_run{simdays}d_release{releasedays}d_{dailyrelease}par_beaching'
        
        sources = list(priors.index)

        Time_in = np.load(f'{input_dir}Time{fname}_{suffix}.npy',allow_pickle=True)
        Age_in = np.load(f'{input_dir}Age{fname}_{suffix}.npy',allow_pickle=True)
        Lon_in = np.load(f'{input_dir}Lon{fname}_{suffix}.npy',allow_pickle=True)
        Lat_in = np.load(f'{input_dir}Lat{fname}_{suffix}.npy',allow_pickle=True)
        print(f'load data: {yr}')
        
        Time_out = {}
        Age_out = {}
        Lon_out = {}
        Lat_out = {}
    
        for iloc,loc in enumerate(sources):
            time = Time_in[iloc][0]
            age = Age_in[iloc][0]
            lons = Lon_in[iloc][0]
            lats = Lat_in[iloc][0]
            print(f'generate arrays at: {loc}')
            
            # create the timeline
            time = time.astype('datetime64[ns]').astype('datetime64[D]') # convert the datatype from 'datetime64[ns]' to 'datetime64[D]'
            timeline = np.arange(np.nanmin(time),np.nanmax(time)+1,1)
            print(f'from {timeline[0]} to {timeline[-1]}')
            
            Age_loc = np.array([])
            Time_loc = np.array([],dtype='datetime64[D]')
            Lon_loc = np.array([])
            Lat_loc = np.array([])
            
            for t in timeline:
                # find the corresponding lons and lats in trajectories
                print(f'calculating t = {t}')
                idx = np.where(time==t)
                age_t = age[idx]
                lons_t = lons[idx]
                lats_t = lats[idx]
                
                for l in range(len(age_t)):
                    age_l = age_t[l]
                    lon_l = lons_t[l]
                    lat_l = lats_t[l]
                    
                    Age_loc = np.append(Age_loc,age_l)
                    Time_loc = np.append(Time_loc,t)
                    Lon_loc = np.append(Lon_loc,lon_l)
                    Lat_loc = np.append(Lat_loc,lat_l)
            
            Time_out[loc] = Time_loc
            Age_out[loc] = Age_loc
            Lon_out[loc] = Lon_loc
            Lat_out[loc] = Lat_loc
            
            # print(f'finish the selection: {loc} \n\n\n')
        
        # Save to npy
        Time_npy = pd.DataFrame.from_dict(Time_out,orient='index')
        Age_npy = pd.DataFrame.from_dict(Age_out,orient='index')
        Lon_npy = pd.DataFrame.from_dict(Lon_out,orient='index')
        Lat_npy = pd.DataFrame.from_dict(Lat_out,orient='index')
        np.save(f'{output_dir}Time{fname}_{suffix}.npy',Time_npy,allow_pickle=True)
        np.save(f'{output_dir}Age{fname}_{suffix}.npy',Age_npy,allow_pickle=True)
        np.save(f'{output_dir}Lon{fname}_{suffix}.npy',Lon_npy,allow_pickle=True)
        np.save(f'{output_dir}Lat{fname}_{suffix}.npy',Lat_npy,allow_pickle=True)
        print(f'finish the saving: {fname}_{suffix} \n')
        
def NormFactor(simyears, backtrack, simdays=729, releasedays=365, dailyrelease=274):
    input_dir = f'/Users/renjiongqiu/Documents_local/Thesis/visualization/TrajAnalysis/trajectories/{backtrack[0]}-{backtrack[1]}/'
    output_dir = f'/Users/renjiongqiu/Documents_local/Thesis/visualization/TrajAnalysis/trajectories/{backtrack[0]}-{backtrack[1]}/'
    os.makedirs(os.path.dirname(output_dir),exist_ok=True)
    priors = pd.read_csv('/Users/renjiongqiu/Documents_local/Thesis/data/priors_river_inputs_25N25S.csv',index_col=0)
    
    Lon = {}
    Lat = {}
    sources = list(priors.index)
    
    for yr in simyears:
        secondyear = yr + 1
        
        fname = f'_{yr}-{secondyear}_run{simdays}d_release{releasedays}d_{dailyrelease}par_beaching_{backtrack[0]}-{backtrack[1]}d_before'
        
        Lon[yr] = np.load(f'{input_dir}Lon{fname}.npy',allow_pickle=True)
        Lat[yr] = np.load(f'{input_dir}Lat{fname}.npy',allow_pickle=True)
        
        Norm_out = {}
        
        for j,loc in enumerate(sources):
            lons = Lon[yr][j][0]
            lats = Lat[yr][j][0]
            Ni = lons.size - ( np.isnan(lons) & np.isnan(lats) ).sum() # calculate the normalizing factor
                
            Norm_out[loc] = Ni
        
        Norm_npy = pd.DataFrame.from_dict(Norm_out,orient='index')
        np.save(f'{output_dir}Norm{fname}.npy',Norm_npy,allow_pickle=True)
        
def GenerateBacktrackDF(simyears, backtrack, simdays=729, releasedays=365, dailyrelease=274):
    input_dir = f'/Users/renjiongqiu/Documents_local/Thesis/visualization/TrajAnalysis/trajectories/{backtrack[0]}-{backtrack[1]}/'
    output_dir = f'/Users/renjiongqiu/Documents_local/Thesis/visualization/TrajAnalysis/trajectories/{backtrack[0]}-{backtrack[1]}/'
    os.makedirs(os.path.dirname(output_dir),exist_ok=True)
    priors = pd.read_csv('/Users/renjiongqiu/Documents_local/Thesis/data/priors_river_inputs_25N25S.csv',index_col=0)
    sources = list(priors.index)
    northern_sources = ['Acapulco', 'SalinaCruz', 'Tecojate', 'LaColorada', 'Fonseca', 'Chira']
    pb_sources = ['Panama', 'Cacique','Esmeraldas']
    # nhcs_sources = ['Guayaquil','Parachique','Lima']
    
    PB_extent = [-90, -73, 0, 9]
    HC_extent = [-90, -75, -6, 0]
    
    # initialize
    onshore = {}
    inPB = {}
    trapPB = {}
    inHC = {}
    trapHC = {}
    #--------------------
    Ntrj_dict = {}
    onShore_N_dict = {}
    
    Noccur_dict = {}
    inPB_N_dict = {}
    inHC_N_dict = {}
    
    NoccurPB_dict = {}
    NoccurHC_dict = {}
    trapPB_N_dict = {}
    trapHC_N_dict = {}
    #--------------------
    Ntrj_dict['NS'] = 0
    onShore_N_dict['NS'] = 0
    
    Noccur_dict['NS'] = 0
    inPB_N_dict['NS'] = 0
    inHC_N_dict['NS'] = 0
    
    NoccurPB_dict['NS'] = 0
    NoccurHC_dict['NS'] = 0
    trapPB_N_dict['NS'] = 0
    trapHC_N_dict['NS'] = 0
    #--------------------
    Ntrj_dict['PB'] = 0
    onShore_N_dict['PB'] = 0
    
    Noccur_dict['PB'] = 0
    inPB_N_dict['PB'] = 0
    inHC_N_dict['PB'] = 0
    
    NoccurPB_dict['PB'] = 0
    NoccurHC_dict['PB'] = 0
    trapPB_N_dict['PB'] = 0
    trapHC_N_dict['PB'] = 0
    #--------------------
    for loc in sources:
        if loc not in northern_sources+pb_sources:
            Ntrj_dict[loc] = 0
            onShore_N_dict[loc] = 0
            
            Noccur_dict[loc] = 0
            inPB_N_dict[loc] = 0
            inHC_N_dict[loc] = 0
            
            NoccurPB_dict[loc] = 0
            NoccurHC_dict[loc] = 0
            trapPB_N_dict[loc] = 0
            trapHC_N_dict[loc] = 0
    
    for yr in simyears:
        secondyear = yr + 1
        
        fname = f'_{yr}-{secondyear}_run{simdays}d_release{releasedays}d_{dailyrelease}par_beaching_{backtrack[0]}-{backtrack[1]}d_before'
        
        Lon = np.load(f'{input_dir}Lon{fname}.npy',allow_pickle=True)
        Lat = np.load(f'{input_dir}Lat{fname}.npy',allow_pickle=True)
        
        for j,loc in enumerate(sources):
            lons = Lon[j][0]
            lats = Lat[j][0]
            nanmask = np.isnan(lons) & np.isnan(lats)
            
            # calculate the onshore fraction of traj
            Ntrj = nanmask.shape[0] # total number of trajs arriving in the Gala
            offShore_N = (nanmask.sum(axis=1) == 0).sum() # first I find the number of nan in each traj. if there is no nan in this traj, then the particle does not ashore after the backtrack[1] days
            onShore_N = Ntrj - offShore_N
            
            # calculate the in zone fraction of occurence
            Noccur = lons.size - nanmask.sum() # this is the norm_factor
            
            inPB_mask = (lons > PB_extent[0]) & (lons < PB_extent[1]) &\
                   (lats > PB_extent[2]) & (lats < PB_extent[3])
                   
            inHC_mask = (lons > HC_extent[0]) & (lons < HC_extent[1]) &\
                   (lats > HC_extent[2]) & (lats < HC_extent[3])
            
            inPB_N = inPB_mask.sum()
            inHC_N = inHC_mask.sum()
            
            # calculate the trapped-in-the-zone fraction of occurence
            # PB
            inPB_lons = lons[inPB_mask]
            inPB_lats = lats[inPB_mask]
            NoccurPB = inPB_lons.size
            
            coords = np.empty((NoccurPB,),dtype=object)
            coords[[i for i in range(NoccurPB)]] = [i for i in zip(inPB_lons,inPB_lats)]
            
            trapPB_N = NoccurPB - np.unique(coords).size
            
            # HC
            inHC_lons = lons[inHC_mask]
            inHC_lats = lats[inHC_mask]
            NoccurHC = inHC_lons.size
            
            coords = np.empty((NoccurHC,),dtype=object)
            coords[[i for i in range(NoccurHC)]] = [i for i in zip(inHC_lons,inHC_lats)]
            
            trapHC_N = NoccurHC - np.unique(coords).size
            
            # add up
            if loc in northern_sources:
                Ntrj_dict['NS'] += Ntrj
                onShore_N_dict['NS'] += onShore_N
                
                Noccur_dict['NS'] += Noccur
                inPB_N_dict['NS'] += inPB_N
                inHC_N_dict['NS'] += inHC_N
                
                NoccurPB_dict['NS'] += NoccurPB
                NoccurHC_dict['NS'] += NoccurHC
                trapPB_N_dict['NS'] += trapPB_N
                trapHC_N_dict['NS'] += trapHC_N
                
            elif loc in pb_sources:
                Ntrj_dict['PB'] += Ntrj
                onShore_N_dict['PB'] += onShore_N
                
                Noccur_dict['PB'] += Noccur
                inPB_N_dict['PB'] += inPB_N
                inHC_N_dict['PB'] += inHC_N
                
                NoccurPB_dict['PB'] += NoccurPB
                NoccurHC_dict['PB'] += NoccurHC
                trapPB_N_dict['PB'] += trapPB_N
                trapHC_N_dict['PB'] += trapHC_N
            else:
                Ntrj_dict[loc] += Ntrj
                onShore_N_dict[loc] += onShore_N
                
                Noccur_dict[loc] += Noccur
                inPB_N_dict[loc] += inPB_N
                inHC_N_dict[loc] += inHC_N
                
                NoccurPB_dict[loc] += NoccurPB
                NoccurHC_dict[loc] += NoccurHC
                trapPB_N_dict[loc] += trapPB_N
                trapHC_N_dict[loc] += trapHC_N
    
    onshore['NS'] = onShore_N_dict['NS'] / Ntrj_dict['NS']
    inPB['NS'] = inPB_N_dict['NS'] / Noccur_dict['NS']
    inHC['NS'] = inHC_N_dict['NS'] / Noccur_dict['NS']
    trapPB['NS'] = trapPB_N_dict['NS'] / Noccur_dict['NS']
    trapHC['NS'] = trapHC_N_dict['NS'] / Noccur_dict['NS']
    
    onshore['PB'] = onShore_N_dict['PB'] / Ntrj_dict['PB']
    inPB['PB'] = inPB_N_dict['PB'] / Noccur_dict['PB']
    inHC['PB'] = inHC_N_dict['PB'] / Noccur_dict['PB']
    trapPB['PB'] = trapPB_N_dict['PB'] / Noccur_dict['PB']
    trapHC['PB'] = trapHC_N_dict['PB'] / Noccur_dict['PB']
    
    for loc in sources:
        if loc not in northern_sources+pb_sources:
            onshore[loc] = onShore_N_dict[loc] / Ntrj_dict[loc]
            inPB[loc] = inPB_N_dict[loc] / Noccur_dict[loc]
            inHC[loc] = inHC_N_dict[loc] / Noccur_dict[loc]
            trapPB[loc] = trapPB_N_dict[loc] / Noccur_dict[loc]
            trapHC[loc] = trapHC_N_dict[loc] / Noccur_dict[loc]
    
    onshore_df = pd.DataFrame.from_dict(onshore,orient='index',columns=['onshore'])
    inPB_df = pd.DataFrame.from_dict(inPB,orient='index',columns=['inPB'])
    trapPB_df = pd.DataFrame.from_dict(trapPB,orient='index',columns=['trapPB'])
    inHC_df = pd.DataFrame.from_dict(inHC,orient='index',columns=['inHC'])
    trapHC_df = pd.DataFrame.from_dict(trapHC,orient='index',columns=['trapHC'])
    
    backtrack_df = pd.concat([onshore_df,inPB_df,trapPB_df,inHC_df,trapHC_df],axis=1)
    backtrack_df.to_csv(f'{output_dir}backtrack.csv')
    return backtrack_df

def DensityMap(simyears, backtrack, lorenz, normalize, simdays=729, releasedays=365, dailyrelease=274):
    normtrack = [0,999]
    # normtrack = backtrack
    if lorenz:
        input_dir = f'/storage/home/9703243/galapagos/trajectories/{backtrack[0]}-{backtrack[1]}/'
        output_dir = '/storage/home/9703243/galapagos/trajectories/'
        os.makedirs(os.path.dirname(output_dir),exist_ok=True)
        priors = pd.read_csv('/storage/home/9703243/galapagos/scripts/priors_river_inputs_25N25S_v0.csv',index_col=0)
        river_sources = np.load('/storage/home/9703243/galapagos/scripts/river_sources_25N25S_v0.npy',allow_pickle=True).item()
        landmask_path = "/storage/shared/oceanparcels/input_data/MOi/psy4v3r1/psy4v3r1-daily_2D_2019-10-25.nc"
    else:
        input_dir = f'/Users/renjiongqiu/Documents_local/Thesis/visualization/TrajAnalysis/trajectories/{backtrack[0]}-{backtrack[1]}/'
        output_dir = '/Users/renjiongqiu/Documents_local/Thesis/visualization/TrajAnalysis/trajectories/'
        os.makedirs(os.path.dirname(output_dir),exist_ok=True)
        priors = pd.read_csv('/Users/renjiongqiu/Documents_local/Thesis/data/priors_river_inputs_25N25S.csv',index_col=0)
        river_sources = np.load('/Users/renjiongqiu/Documents_local/Thesis/data/river_sources_25N25S_v0.npy',allow_pickle=True).item()
        landmask_path = "/Users/renjiongqiu/Documents_local/Thesis/data/sources/psy4v3r1-daily_2D_2019-10-25.nc"
        Ninput_dir = f'/Users/renjiongqiu/Documents_local/Thesis/visualization/TrajAnalysis/trajectories/{normtrack[0]}-{normtrack[1]}/'
        
    galapagos_extent = [-91.8, -89, -1.4, 0.7]
    
    sources = list(priors.index)
    northern_sources = ['Acapulco', 'SalinaCruz', 'Tecojate', 'LaColorada', 'Fonseca', 'Chira']
    pb_sources = ['Panama', 'Cacique','Esmeraldas']
    nhcs_sources = ['Guayaquil','Parachique','Lima']
    
    study_region = (-115,-65,-25,25)

    boxlist = ['SEC','PNJtilt','NECC','PBSR','HC']
    
    Lon = {}
    Lat = {}
    Time = {}
    Age = {}
    N_in = {}
    
    for yr in simyears:
        secondyear = yr + 1
        
        fname = f'_{yr}-{secondyear}_run{simdays}d_release{releasedays}d_{dailyrelease}par_beaching_{backtrack[0]}-{backtrack[1]}d_before'
        Nfname = f'_{yr}-{secondyear}_run{simdays}d_release{releasedays}d_{dailyrelease}par_beaching_{normtrack[0]}-{normtrack[1]}d_before'
        
        Lon[yr] = np.load(f'{input_dir}Lon{fname}.npy',allow_pickle=True)
        Lat[yr] = np.load(f'{input_dir}Lat{fname}.npy',allow_pickle=True)
        Time[yr] = np.load(f'{input_dir}Time{fname}.npy',allow_pickle=True)
        Age[yr] = np.load(f'{input_dir}Age{fname}.npy',allow_pickle=True)
        N_in[yr] = np.load(f'{Ninput_dir}Norm{Nfname}.npy',allow_pickle=True)
        print(f'load data: {yr}')

    ds = xr.load_dataset(landmask_path)
    study_region = (-115, -65, -25, 25)
    indices = {'lat': range(1185, 1804), 'lon': range(2066, 2665)} # the landmask should be one grid less

    landmask = ds.variables['sossheig'][0, indices['lat'], indices['lon']]
    landmask = np.ma.masked_invalid(landmask)
    landmask = landmask.mask.astype('int')
    landmask = np.where(landmask==1,np.nan,landmask)
    Lons_nemo = ds['nav_lon'][0,indices['lon']] # unpacks the tuple
    Lats_nemo = ds['nav_lat'][indices['lat'],0]
    X, Y = np.meshgrid(Lons_nemo, Lats_nemo)
    galamask = np.zeros_like(X)
    galamask = np.where( (X>=galapagos_extent[0]) & (X<=galapagos_extent[1]) & \
                         (Y>=galapagos_extent[2]) & (Y<=galapagos_extent[3]),  \
                          np.nan, galamask) # set the gala region to nan

    indices = {'lat': range(1184, 1804), 'lon': range(2065, 2665)}
    Lons_nemo = ds['nav_lon'][0,indices['lon']] # unpacks the tuple
    Lats_nemo = ds['nav_lat'][indices['lat'],0]

    ylen = len(Lats_nemo) - 1
    xlen = len(Lons_nemo) - 1
    Hplt = {} # 'NS', 'Guayaquil', 'Parachique', 'Lima'
    Nplt = {}
    NPplt = {}
    Hplt['NS'] = np.zeros((xlen,ylen))
    Hplt['PB'] = np.zeros((xlen,ylen))
    Hplt['HC'] = np.zeros((xlen,ylen))
    Nplt['HC'] = 0
    Nplt['NS'] = 0
    Nplt['PB'] = 0
    NPplt['HC'] = 0
    NPplt['NS'] = 0
    NPplt['PB'] = 0

    # make the plotting dataset
    for j,loc in enumerate(sources):
        H = np.zeros((xlen,ylen))
        N = 0 # initialize the calculating factor
        NP = 0
        for yr in simyears:
            lons = Lon[yr][j][0]
            lats = Lat[yr][j][0]
            Ni = N_in[yr][j][0]
            NPi = len(Lon[yr][j][0])
            # Hi,xb,yb = Density(lons,lats,focus_region)[0:3]
            Hi,xb,yb = Density_landmask(lons,lats,Lons_nemo,Lats_nemo)
            N += Ni
            NP += NPi
            H += Hi
        
        if loc in northern_sources:
            Hplt['NS'] += H
            Nplt['NS'] += N
            NPplt['NS'] += NP
        elif loc in pb_sources:
            Hplt['PB'] += H
            Nplt['PB'] += N
            NPplt['PB'] += NP
        elif loc in nhcs_sources:
            Hplt['HC'] += H
            Nplt['HC'] += N
            NPplt['HC'] += NP
        print(f'complete the calculation: {loc}')
    if normalize == 'occurrence':
        Hplt['NS'] = Hplt['NS'] / Nplt['NS']
        Hplt['PB'] = Hplt['PB'] / Nplt['PB']
        Hplt['HC'] = Hplt['HC'] / Nplt['HC']
    elif normalize == 'arriving':
        Hplt['NS'] = Hplt['NS'] / NPplt['NS']
        Hplt['PB'] = Hplt['PB'] / NPplt['PB']
        Hplt['HC'] = Hplt['HC'] / NPplt['HC']

    sumNS = Hplt['NS'].sum()
    sumPB = Hplt['PB'].sum()
    sumHC = Hplt['HC'].sum()
    print(f'check normalization: sumNS = {sumNS}, sumPB = {sumPB}, sumHC = {sumHC}\n the sum not 1.0 is because some obsv leak out of the region')
    # return xb,yb,Hplt

    fig = plt.figure(figsize=(12*3,15),constrained_layout=True)
    gs = gridspec.GridSpec(2,3, height_ratios=[14,1],wspace=0.05)
    ttl = dict(fontsize=32)
    lbm = dict(fontsize=26)
    tkm = dict(labelsize=22)
    
    crt = 200
    clft = 50
    cmlen = 50
    dcm = int((crt-clft)/cmlen)
    upper = cm.jet(np.arange(256))[cmlen:,:]
    lower = cm.Blues(np.arange(256))[clft:crt:dcm]
    # combine parts of colormap
    cmap = np.vstack(( lower, upper ))
    
    # convert to matplotlib colormap
    cmap = colors.ListedColormap(cmap, name='myColorMap', N=cmap.shape[0])
    
    suffix = f'{backtrack[1]}d before arriving'
    if backtrack[1] == 999:
        suffix = 'before arriving'
    for i in range(3):
        if i == 0:
            H = Hplt['NS']
            sourcelist = northern_sources
            boxlist = ['Gala','PBSR','PNJtilt_wide']
            title = f'The NR Sources: {suffix}'
        elif i == 1:
            H = Hplt['PB']
            sourcelist = pb_sources
            boxlist = ['Gala','PBSR','PNJtilt_wide']
            title = f'The PBSR Sources: {suffix}'
        elif i == 2:
            H = Hplt['HC']
            sourcelist = nhcs_sources
            boxlist = ['Gala','NHCS','HC']
            title = f'The NHCS Sources: {suffix}'
        # levels = np.array([10],dtype='uint64') ** (np.arange(int(math.ceil(np.log10(np.nanmin(H))))+1,int(math.ceil(np.log10(np.nanmax(H))))-1,1))
        levels = np.array([10],dtype='uint64') ** (np.arange(-8,-3.1,0.1))
        # levels = np.array([10],dtype='uint64') ** (np.arange(-3,-1.1,0.1))
        # ticks = np.percentile(levels,np.linspace(0,100,5))
        ticks = np.array([10],dtype='uint64') ** (np.arange(-8,-2,1))
        # levels, ticks = createlevels(H, nloop=1000)
        norm = colors.BoundaryNorm(levels,256)
        # norm = colors.BoundaryNorm(levels,235)
        ax = fig.add_subplot(gs[0,i],projection=ccrs.PlateCarree())
        H = H + landmask.T + galamask.T
        cf = ax.contourf(xb,yb,H.T,transform=ccrs.PlateCarree(),cmap=cmap,levels=levels,norm=norm,extend='both')
        ax.coastlines(resolution='50m')
        ax.add_feature(cfeature.NaturalEarthFeature('physical', 'land', '50m', edgecolor='face', facecolor='silver'),zorder=-1)
        ax.add_feature(cfeature.BORDERS)
        ax.set_title(title,**ttl,pad=12)
        gl = ax.gridlines(draw_labels=False,linewidth=1.,color='w',alpha=0.5,linestyle='--')
        gl.xlabels_bottom = True
        gl.xformatter = LONGITUDE_FORMATTER
        if i == 0:
            gl.ylabels_left = True
            gl.yformatter = LATITUDE_FORMATTER
        gl.xlabel_style = {'size':22}
        gl.ylabel_style = {'size':22}
        ax.set_extent(study_region,crs=ccrs.PlateCarree())
        # cb = plt.colorbar(cf,fraction=0.06, pad=0.04, ticks=ticks, format=ticker.FuncFormatter(tksfmt))
        # cb.ax.tick_params(**tkm)
        # cb.formatter.set_powerlimits((0,0))
        # cb.formatter.set_useMathText(True)
        # cb.ax.set_yticklabels(range(0,7))
        
        # draw box
        for box in boxlist:
            path = DrawRegion(box)
            patchR = patches.PathPatch(path,ec='k',fc='none',lw=2)
            ax.add_patch(patchR)
        
        # draw sources
        for loc in sourcelist:
            river_lons = river_sources[loc][1]
            river_lats = river_sources[loc][0]
            path = DrawCluster(river_lons, river_lats, 1)
            patchS = patches.PathPatch(path,ec='k',fc='none',lw=2,zorder=9)
            ax.add_patch(patchS)
        
        print(f'complete subplot: {i}')
    
    cbaxes = fig.add_axes([0.175, 0.16, 0.7, 0.03])
    cb = plt.colorbar(cf, cax=cbaxes,orientation='horizontal', ticks=ticks, format=ticker.FuncFormatter(tksfmt))
    cb.ax.tick_params(**tkm)
    
    plt.tight_layout()
    plt.savefig(output_dir+f'Density_{simyears[0]}-{simyears[-1]}yearintegrated_{suffix}.png')

def FieldScatter(simyears, backtrack, Xstr, Ystr, simdays=729, releasedays=365, dailyrelease=274):
    input_dir = f'/Users/renjiongqiu/Documents_local/Thesis/visualization/TrajAnalysis/fields/{backtrack[0]}-{backtrack[1]}/'
    output_dir = '/Users/renjiongqiu/Documents_local/Thesis/visualization/TrajAnalysis/fields/'
    os.makedirs(os.path.dirname(output_dir),exist_ok=True)
    priors = pd.read_csv('/Users/renjiongqiu/Documents_local/Thesis/data/priors_river_inputs_25N25S.csv',index_col=0)
    
    sources = list(priors.index)
    northern_sources = ['Acapulco', 'SalinaCruz', 'Tecojate', 'LaColorada', 'Fonseca', 'Chira', 'Panama', 'Cacique', 'Esmeraldas']
    
    # initialization
    Xplt = {}
    Yplt = {}
    Xplt['NS'] = np.array([])
    Yplt['NS'] = np.array([])
    for loc in sources:
        if loc not in northern_sources:
            Xplt[loc] = np.array([])
            Yplt[loc] = np.array([])
    
    for yr in simyears:
        secondyear = yr + 1
        
        fname = f'_{yr}-{secondyear}_run{simdays}d_release{releasedays}d_{dailyrelease}par_beaching_{backtrack[0]}-{backtrack[1]}d_before'
        
        X = np.load(f'{input_dir}{Xstr}{fname}.npy',allow_pickle=True)
        Y = np.load(f'{input_dir}{Ystr}{fname}.npy',allow_pickle=True)
        print(f'load data: {yr}')
        
        for iloc,loc in enumerate(sources):
            if loc in northern_sources:
                idx_notnan = ~np.isnan(X[iloc,:])
                Xplt['NS'] = np.append(Xplt['NS'],X[iloc,:][idx_notnan])
                idx_notnan = ~np.isnan(Y[iloc,:])
                Yplt['NS'] = np.append(Yplt['NS'],Y[iloc,:][idx_notnan])
            else:
                idx_notnan = ~np.isnan(X[iloc,:])
                Xplt[loc] = np.append(Xplt[loc],X[iloc,:][idx_notnan])
                idx_notnan = ~np.isnan(Y[iloc,:])
                Yplt[loc] = np.append(Yplt[loc],Y[iloc,:][idx_notnan])
    # return Xplt, Yplt
    fig = plt.figure(figsize=(8*2,6*2),constrained_layout=True)
    suffix = f'{backtrack[0]}-{backtrack[1]}d before'
    for i in range(4):
        if i == 0:
            X = Xplt['NS']
            Y = Yplt['NS']
            title = f'The Northern Sources: {suffix}'
        elif i == 1:
            X = Xplt['Guayaquil']
            Y = Yplt['Guayaquil']
            title = f'Guayaquil: {suffix}'
        elif i == 2:
            X = Xplt['Parachique']
            Y = Yplt['Parachique']
            title = f'Parachique: {suffix}'
        elif i == 3:
            X = Xplt['Lima']
            Y = Yplt['Lima']
            title = f'Lima: {suffix}'
            
        ax = fig.add_subplot(2,2,i+1)
        ax.scatter(X,Y)
        ax.set_title(title,fontsize=16)
        ax.set_xlabel(f'{Xstr}',fontsize=12)
        ax.set_ylabel(f'{Ystr}',fontsize=12)

def FieldHistogram(simyears, backtrack, normalize, Xstr, Ystr, simdays=729, releasedays=365, dailyrelease=274):
    input_dir = f'/Users/renjiongqiu/Documents_local/Thesis/visualization/TrajAnalysis/fields/{backtrack[0]}-{backtrack[1]}/'
    output_dir = '/Users/renjiongqiu/Documents_local/Thesis/visualization/TrajAnalysis/fields/'
    os.makedirs(os.path.dirname(output_dir),exist_ok=True)
    priors = pd.read_csv('/Users/renjiongqiu/Documents_local/Thesis/data/priors_river_inputs_25N25S.csv',index_col=0)
    Ninput_dir = '/Users/renjiongqiu/Documents_local/Thesis/visualization/TrajAnalysis/trajectories/0-999/'
    
    sources = list(priors.index)
    northern_sources = ['Acapulco', 'SalinaCruz', 'Tecojate', 'LaColorada', 'Fonseca', 'Chira']
    pb_sources = ['Panama', 'Cacique','Esmeraldas']
    nhcs_sources = ['Guayaquil','Parachique','Lima']
    
    X = {}
    Y = {}
    N_in = {}
    
    for iyr,yr in enumerate(simyears):
        secondyear = yr + 1
        
        fname = f'_{yr}-{secondyear}_run{simdays}d_release{releasedays}d_{dailyrelease}par_beaching_{backtrack[0]}-{backtrack[1]}d_before'
        Nfname = f'_{yr}-{secondyear}_run{simdays}d_release{releasedays}d_{dailyrelease}par_beaching_0-999d_before'
        
        X[yr] = np.load(f'{input_dir}{Xstr}{fname}.npy',allow_pickle=True)
        Y[yr] = np.load(f'{input_dir}{Ystr}{fname}.npy',allow_pickle=True)
        N_in[yr] = np.load(f'{Ninput_dir}Norm{Nfname}.npy',allow_pickle=True)
        print(f'load data: {yr}')
        # return X,Y
        
        if iyr == 0:
            xlim_max = np.nanmax(X[yr])
            xlim_min = np.nanmin(X[yr])
            
            ylim_max = np.nanmax(Y[yr])
            ylim_min = np.nanmin(Y[yr])
        else:
            xlim_max = np.nanmax([xlim_max, np.nanmax(X[yr])])
            xlim_min = np.nanmin([xlim_min, np.nanmin(X[yr])])
            
            ylim_max = np.nanmax([ylim_max, np.nanmax(Y[yr])])
            ylim_min = np.nanmin([ylim_min, np.nanmin(Y[yr])])
        
        plt_res = 1
        plt_size = 100
        if Xstr == 'Time':
            xline = np.arange(np.nanmin(X[yr]),np.nanmax(X[yr]),np.timedelta64(1,'D'))
            if Ystr == 'Lat':
                yline = np.arange(ylim_min,ylim_max,plt_res)
            else:
                yline = np.linspace(ylim_min,ylim_max,plt_size)
        if Ystr == 'Time':
            if Xstr == 'Lon':
                xline = np.arange(xlim_min,xlim_max,plt_res)
            else:
                xline = np.linspace(xlim_min,xlim_max,plt_size)
            yline = np.arange(np.nanmin(Y[yr]),np.nanmax(Y[yr]),np.timedelta64(1,'D'))
        if Ystr != 'Time' and Xstr != 'Time':
            xline = np.linspace(xlim_min,xlim_max,plt_size)
            yline = np.linspace(ylim_min,ylim_max,plt_size)
        
    bins = [ xline, yline ]
    # return X,Y

    xlen = len(bins[0]) - 1
    ylen = len(bins[1]) - 1
    Hplt = {} # 'NS', 'Guayaquil', 'Parachique', 'Lima'
    Nplt = {}
    Hplt['NS'] = np.zeros((xlen,ylen))
    Hplt['PB'] = np.zeros((xlen,ylen))
    Hplt['HC'] = np.zeros((xlen,ylen))
    Nplt['HC'] = 0
    Nplt['NS'] = 0
    Nplt['PB'] = 0
        
    for yr in simyears:
        for iloc,loc in enumerate(sources):
            x = X[yr][iloc,:]
            y = Y[yr][iloc,:]
            Ni = N_in[yr][iloc][0]
            idx_notnan = ~np.isnan(x) & ~np.isnan(y)
            x_notnan = x[idx_notnan]
            y_notnan = y[idx_notnan]
            
            if len(x_notnan) != len(y_notnan):
                print(f'{yr}-{loc}')
            
            Hi,xe,ye = np.histogram2d(x_notnan, y_notnan, bins=bins)
            if Xstr == 'Time':
                xb = ( ( xe[:-1].astype(int) + xe[1:].astype(int) ) / 2 ).astype('datetime64[ns]').astype('datetime64[D]')
                yb = ( ye[:-1] + ye[1:] ) / 2
            if Ystr == 'Time':
                xb = ( xe[:-1] + xe[1:] ) / 2 
                yb = ( ( ye[:-1].astype(int) + ye[1:].astype(int) ) / 2 ).astype('datetime64[ns]').astype('datetime64[D]')
            if Ystr != 'Time' and Xstr != 'Time':
                xb = ( xe[:-1] + xe[1:] ) / 2
                yb = ( ye[:-1] + ye[1:] ) / 2
            
            if loc in northern_sources:
                Hplt['NS'] += Hi
                Nplt['NS'] += Ni
            elif loc in pb_sources:
                Hplt['PB'] += Hi  
                Nplt['PB'] += Ni
            elif loc in nhcs_sources:
                Hplt['HC'] += Hi  
                Nplt['HC'] += Ni
                
    # normalization
    if normalize == 'arriving':
        Hplt['NS'] = Hplt['NS'] / Nplt['NS']
        Hplt['PB'] = Hplt['PB'] / Nplt['PB']
        Hplt['HC'] = Hplt['HC'] / Nplt['HC']
                
    fig = plt.figure(figsize=(8*3,6*2),constrained_layout=True)
    cmap = 'jet' # options: 'viridis', 'jet'
    suffix = f'{backtrack[0]}-{backtrack[1]}d before'
    if backtrack[1] == 999:
        suffix = 'before arriving in the Galapagos'
    for i in range(3):
        if i == 0:
            H = Hplt['NS']
            title = f'The Northern Sources: {suffix}'
        elif i == 1:
            H = Hplt['PB']
            title = f'Sources in the Panama Bight: {suffix}'
        elif i == 2:
            H = Hplt['HC']
            title = f'Sources in the NHCS: {suffix}'
        
        # levels, ticks = createlevels(H, nloop=10)
        levels = np.array([10],dtype='uint64') ** (np.arange(-8,-2.1,0.1))
        ticks = np.array([10],dtype='uint64') ** (np.arange(-8,-1,1))
        norm = colors.BoundaryNorm(levels,256)
        ax = fig.add_subplot(2,3,i+1)
        cf = ax.contourf(xb,yb,H.T,cmap=cmap,levels=levels,norm=norm,extend='both')
        
        ax.set_xlabel(f'{Xstr}',fontsize=12)
        ax.set_ylabel(f'{Ystr}',fontsize=12)
        ax.set_title(title,fontsize=16)
        plt.colorbar(cf,fraction=0.06, pad=0.04, ticks=ticks, format=ticker.FuncFormatter(tksfmt))
        
        if Ystr == 'Zeta':
            ax.set_ylim([-2e-5,2e-5])
            ax.set_xlim([xb[300],xb[-1]])
        elif Ystr == 'Theta':
            ax.set_ylim([90,270])
        elif Ystr == 'Time':
            hfmt = dates.DateFormatter('%b-%d')
            ax.yaxis.set_major_formatter(hfmt)
        
        if Xstr == 'Lon':
            ax.set_xlim([-90,-77])
        elif Xstr == 'Time':
            hfmt = dates.DateFormatter('%b-%d')
            ax.xaxis.set_major_formatter(hfmt)
            plt.setp(ax.get_xticklabels(), rotation=45, ha='center')
        
    plt.savefig(output_dir+f'{Xstr}-{Ystr}_{normalize}-normed.png')

def HovDiagram(simyears, backtrack, normalize, Xstr, Ystr, simdays=729, releasedays=365, dailyrelease=274):
    input_dir = f'/Users/renjiongqiu/Documents_local/Thesis/visualization/TrajAnalysis/trajectories/{backtrack[0]}-{backtrack[1]}/'
    output_dir = '/Users/renjiongqiu/Documents_local/Thesis/visualization/TrajAnalysis/trajectories/'
    os.makedirs(os.path.dirname(output_dir),exist_ok=True)
    priors = pd.read_csv('/Users/renjiongqiu/Documents_local/Thesis/data/priors_river_inputs_25N25S.csv',index_col=0)
    landmask_path = "/Users/renjiongqiu/Documents_local/Thesis/data/sources/psy4v3r1-daily_2D_2019-10-25.nc"
    Ninput_dir = '/Users/renjiongqiu/Documents_local/Thesis/visualization/TrajAnalysis/trajectories/0-999/'
    
    sources = list(priors.index)
    northern_sources = ['Acapulco', 'SalinaCruz', 'Tecojate', 'LaColorada', 'Fonseca', 'Chira']
    pb_sources = ['Panama', 'Cacique','Esmeraldas']
    nhcs_sources = ['Guayaquil','Parachique','Lima']
    
    X = {}
    Y = {}
    N_in = {}
    xline = {}
    yline = {}
    
    if Xstr == 'Time':
        xlen = 731 - 1
        ylen = 51 - 1
    if Ystr == 'Time':
        xlen = 51 - 1
        ylen = 731 - 1
    Hplt = {} # 'NS', 'Guayaquil', 'Parachique', 'Lima'
    Nplt = {}
    Hplt['NS'] = np.zeros((xlen,ylen))
    Hplt['PB'] = np.zeros((xlen,ylen))
    Nplt['NS'] = 0
    Nplt['PB'] = 0
    
    for loc in sources:
        if loc not in northern_sources+pb_sources:
            Hplt[loc] = np.zeros((xlen,ylen))
            Nplt[loc] = 0
        
    for iyr,yr in enumerate(simyears):
        secondyear = yr + 1
        
        fname = f'_{yr}-{secondyear}_run{simdays}d_release{releasedays}d_{dailyrelease}par_beaching_{backtrack[0]}-{backtrack[1]}d_before'
        Nfname = f'_{yr}-{secondyear}_run{simdays}d_release{releasedays}d_{dailyrelease}par_beaching_0-999d_before'
        
        X = np.load(f'{input_dir}{Xstr}{fname}.npy',allow_pickle=True)
        Y = np.load(f'{input_dir}{Ystr}{fname}.npy',allow_pickle=True)
        N_in = np.load(f'{Ninput_dir}Norm{Nfname}.npy',allow_pickle=True)
        print(f'load data: {yr}')
        '''
        ds = xr.load_dataset(landmask_path)
        study_region = (-115, -65, -25, 25)
        
        indices = {'lat': range(1184, 1804), 'lon': range(2065, 2665)}
        Lons_nemo = ds['nav_lon'][0,indices['lon']] # unpacks the tuple
        Lats_nemo = ds['nav_lat'][indices['lat'],0]
        '''
        date_st = np.datetime64(f'{yr}-01-01')
        if (yr%4 == 0) or ( (yr+1)%4 == 0 ):
            date_ed = np.datetime64(f'{yr+2}-01-01')
        else:
            date_ed = np.datetime64(f'{yr+2}-01-02')
        plt_res = 1
        
        if Xstr == 'Time':
            xline = np.arange(date_st,date_ed,np.timedelta64(1,'D')).astype('datetime64[ns]')
            # yline = Lats_nemo
            yline = np.arange(-25,26,plt_res)
        elif Ystr == 'Time':
            # xline = Lons_nemo
            xline = np.arange(-115,-64,plt_res)
            yline = np.arange(date_st,date_ed,np.timedelta64(1,'D')).astype('datetime64[ns]')
    
        bins = [ xline, yline ]

        for iloc,loc in enumerate(sources):
            if Xstr == 'Time':
                x = X[iloc][0].astype('datetime64[ns]')
                y = Y[iloc][0]
            elif Ystr == 'Time':
                x = X[iloc][0]
                y = Y[iloc][0].astype('datetime64[ns]')
            Ni = N_in[iloc][0]
            idx_notnan = ~np.isnan(x) & ~np.isnan(y)
            x_notnan = x[idx_notnan]
            y_notnan = y[idx_notnan]
            
            if len(x_notnan) != len(y_notnan):
                print(f'error: {yr}-{loc}')
            
            Hi,xe,ye = np.histogram2d(x_notnan, y_notnan, bins=bins)
            if Xstr == 'Time':
                xb = ( ( xe[:-1].astype(int) + xe[1:].astype(int) ) / 2 ).astype('datetime64[D]')
                yb = ( ye[:-1] + ye[1:] ) / 2
            elif Ystr == 'Time':
                xb = ( xe[:-1] + xe[1:] ) / 2 
                yb = ( ( ye[:-1].astype(int) + ye[1:].astype(int) ) / 2 ).astype('datetime64[D]')
            else:
                xb = ( xe[:-1] + xe[1:] ) / 2
                yb = ( ye[:-1] + ye[1:] ) / 2
            
            if loc in northern_sources:
                Hplt['NS'] += Hi
                Nplt['NS'] += Ni
            elif loc in pb_sources:
                Hplt['PB'] += Hi  
                Nplt['PB'] += Ni
            else:
                Hplt[loc] += Hi
                Nplt[loc] += Ni
                
    # normalization
    if normalize == 'arriving':
        Hplt['NS'] = Hplt['NS'] / Nplt['NS']
        Hplt['PB'] = Hplt['PB'] / Nplt['PB']
        print('sumNS = '+ str(Hplt['NS'].sum()))
        print('sumPB = '+ str(Hplt['PB'].sum()))
        for loc in sources:
            if loc not in northern_sources+pb_sources:
                Hplt[loc] = Hplt[loc] / Nplt[loc]
                print(f'sum{loc} = '+ str(Hplt[loc].sum()))

    fig = plt.figure(figsize=(8*3,6*2),constrained_layout=True)
    cmap = 'jet' # options: 'viridis', 'jet'
    suffix = f'{backtrack[0]}-{backtrack[1]}d before'
    if backtrack[1] == 999:
        suffix = 'before arriving in the Galapagos'
    for i in range(6):
        if i == 0:
            H = Hplt['NS']
            title = f'The Northern Sources: {suffix}'
        elif i == 1:
            H = Hplt['PB']
            title = f'Sources in the Panama Bight: {suffix}'
        elif i == 2:
            H = Hplt['Esmeraldas']
            title = f'Esmeraldas: {suffix}'
        elif i == 3:
            H = Hplt['Guayaquil']
            title = f'Guayaquil: {suffix}'
        elif i == 4:
            H = Hplt['Parachique']
            title = f'Parachique: {suffix}'
        elif i == 5:
            H = Hplt['Lima']
            title = f'Lima: {suffix}'
        # levels, ticks = createlevels(H, nloop=10)
        levels = np.array([10],dtype='uint64') ** (np.arange(-8,-2.1,0.1))
        ticks = np.array([10],dtype='uint64') ** (np.arange(-8,-1,1))
        norm = colors.BoundaryNorm(levels,256)
        ax = fig.add_subplot(2,3,i+1)
        cf = ax.contourf(xb,yb,H.T,cmap=cmap,levels=levels,norm=norm,extend='both')
        
        # ax.set_xlabel(f'{Xstr}',fontsize=12)
        # ax.set_ylabel(f'{Ystr}',fontsize=12)
        ax.set_title(title,fontsize=16)
        plt.colorbar(cf,fraction=0.06, pad=0.04, ticks=ticks, format=ticker.FuncFormatter(tksfmt))
        
        if Ystr == 'Time':
            hfmt = dates.DateFormatter('%b-%d')
            ax.yaxis.set_major_formatter(hfmt)
        
        if Xstr == 'Lon':
            ax.set_xlim([-90,-77])
        elif Xstr == 'Time':
            hfmt = dates.DateFormatter('%b-%d')
            ax.xaxis.set_major_formatter(hfmt)
            plt.setp(ax.get_xticklabels(), rotation=45, ha='center')
        
    plt.savefig(output_dir+f'{Xstr}-{Ystr}_{normalize}-normed.png')
        
def ThetaSPDwindrose(simyears, backtrack, Xstr, Ystr, simdays=729, releasedays=365, dailyrelease=274):
    input_dir = f'/Users/renjiongqiu/Documents_local/Thesis/visualization/TrajAnalysis/fields/{backtrack[0]}-{backtrack[1]}/'
    output_dir = '/Users/renjiongqiu/Documents_local/Thesis/visualization/TrajAnalysis/fields/'
    os.makedirs(os.path.dirname(output_dir),exist_ok=True)
    priors = pd.read_csv('/Users/renjiongqiu/Documents_local/Thesis/data/priors_river_inputs_25N25S.csv',index_col=0)
    
    sources = list(priors.index)
    northern_sources = ['Acapulco', 'SalinaCruz', 'Tecojate', 'LaColorada', 'Fonseca', 'Chira']
    pb_sources = ['Panama', 'Cacique','Esmeraldas']
    nhcs_sources = ['Guayaquil','Parachique','Lima']
    
    X = {}
    Y = {}
    
    for iyr,yr in enumerate(simyears):
        secondyear = yr + 1
        
        fname = f'_{yr}-{secondyear}_run{simdays}d_release{releasedays}d_{dailyrelease}par_beaching_{backtrack[0]}-{backtrack[1]}d_before'
        
        X[yr] = np.load(f'{input_dir}{Xstr}{fname}.npy',allow_pickle=True) 
        Y[yr] = 450 - np.load(f'{input_dir}{Ystr}{fname}.npy',allow_pickle=True) # weird mistakes for windrose package, all the angles are mirrored at 225 degree
        print(f'load data: {yr}')
        
    Xplt = {}
    Yplt = {}
    Xplt['NS'] = np.array([])
    Xplt['PB'] = np.array([])
    Xplt['HC'] = np.array([])
    Yplt['HC'] = np.array([])
    Yplt['NS'] = np.array([])
    Yplt['PB'] = np.array([])
    
    ttl = dict(fontsize=24)
    lbm = dict(fontsize=20)
    tkm = dict(labelsize=18)
    
    for yr in simyears:
        for iloc,loc in enumerate(sources):
            x = X[yr][iloc,:]
            y = Y[yr][iloc,:]
            idx_notnan = ~np.isnan(x) & ~np.isnan(y)
            x_notnan = x[idx_notnan]
            y_notnan = y[idx_notnan]
            
            if len(x_notnan) != len(y_notnan):
                print(f'{yr}-{loc}')
            
            if loc in northern_sources:
                Xplt['NS'] = np.append(Xplt['NS'],x_notnan)
                Yplt['NS'] = np.append(Yplt['NS'],y_notnan)
            elif loc in pb_sources:
                Xplt['PB'] = np.append(Xplt['PB'],x_notnan)
                Yplt['PB'] = np.append(Yplt['PB'],y_notnan)  
            elif loc in nhcs_sources:
                Xplt['HC'] = np.append(Xplt['HC'],x_notnan)
                Yplt['HC'] = np.append(Yplt['HC'],y_notnan)  
                
    fig = plt.figure(figsize=(8*3,8),constrained_layout=True)
    cmap = cmo.speed # options: 'viridis', 'jet'
    suffix = f'{backtrack[1]}d before arriving'
    bins = [0.,0.25,0.5,1.]
    for i in range(3):
        if i == 0:
            x,y = Xplt['NS'],Yplt['NS']
            title = f'The NR Sources: {suffix}'
        elif i == 1:
            x,y = Xplt['PB'],Yplt['PB']
            title = f'The PBSR Sources: {suffix}'
        elif i == 2:
            x,y = Xplt['HC'],Yplt['HC']
            title = f'The NHCS Sources: {suffix}'
        # axi = fig.add_subplot(3,2,i+1)
        rect=[0.3*(i%3)+0.05,0.1,0.3,0.8] 
        wa=WindroseAxes(fig, rect)
        fig.add_axes(wa)
        # wa.contourf(y, x, bins=np.linspace(0,1,10),cmap=cm.jet, normed=True)
        cont = True
        if cont == True:
            suffix = 'contourf'
            wa.contourf(y, x, nsector=36, bins=bins, cmap=cmap, normed=True)
            # wa.contour(y, x, bins=np.arange(0,0.61,0.3), colors='black')
        else:
            suffix = ''
            wa.bar(y, x, nsector=36, bins=bins, cmap=cmap, normed=True, opening=0.8)
        if i == 2:
            legend = wa.set_legend(title='Current speed [$m\cdot s^{-1}$]',bbox_to_anchor=(-0.2,-0.05,1.2,0.5), loc='lower right',mode='expand',ncol=5,title_fontsize=20)
        wa.set_title(title,**ttl)
        wa.set_ylim([0,12])
        wa.set_yticks(np.arange(2,13,2))
        wa.set_yticklabels(['2%','4%','6%','8%','10%','12%'])
        wa.tick_params('both',**tkm)
        
    plt.setp(legend.get_texts(), fontsize=18)
    for it,txt in enumerate(legend.get_texts()):
        if it<len(bins)-1:
            vst = bins[it]
            ved = bins[it+1]
            # txt.set_text('{:.1f}-{:.1f}'.format(vst,ved))
            txt.set_text(f'{vst:.2f} - {ved:.2f}')
        else:
            txt.set_text(f'>{bins[-1]:.2f}')
    plt.tight_layout()
    plt.savefig(output_dir+f'windrose-{suffix}-{simyears[0]}_{simyears[-1]}.png')
            
#%% Main
lorenz = False

if lorenz == False:
    action = 'windrose'
    backtrack0 = 0
    backtrack1 = 40
    # start_year = 2015
    # end_year = 2016
    normalize = 'occurrence' # 'total', 'arriving' 'occurrence'
    for yr in np.arange(2007,2021):
        start_year = yr
        end_year = yr+1
        simyears = np.arange(start_year,end_year)
        backtrack = [backtrack0,backtrack1]
        ThetaSPDwindrose(simyears, backtrack, Xstr='SPD', Ystr='Theta')

elif lorenz == True:
    action = sys.argv[1]
    backtrack0 = int(sys.argv[2])
    backtrack1 = int(sys.argv[3])
    start_year = int(sys.argv[4])
    end_year = int(sys.argv[5])

print(f'{action}\n {backtrack0}\n {backtrack1}\n {start_year}\n {end_year}\n')

simyears = np.arange(start_year,end_year)
# simyears = [2007,2008,2009,2010,2011,2012,2013,2015,2016]
backtrack = [backtrack0,backtrack1]
# action = 'density_map' # ['generate_traj_data','density_map','generate_field_data']
if action == 'generate_traj_data':
    GenerateTrajData(simyears, backtrack, lorenz)
elif action == 'generate_field_data':
    GenerateFieldData(simyears, backtrack)
elif action == 'get_fields_date':
    GetFieldsDate(simyears, backtrack)
elif action == 'generate_norm_factor':
    NormFactor(simyears, backtrack)
elif action == 'evaluate_backtrack_day':
    backtrack_df = GenerateBacktrackDF(simyears, backtrack)
elif action == 'density_map':
    DensityMap(simyears, backtrack, lorenz, normalize)
elif action == 'field_scatter':
    FieldScatter(simyears, backtrack, Xstr='SPD', Ystr='Theta')
elif action == 'field_histogram':
    # options: ['SSH','Zeta','SPD','Theta','U','V','Time','Age','Lon','Lat']
    FieldHistogram(simyears, backtrack, normalize, Xstr='Theta', Ystr='SPD')
elif action == 'hovmoller_diagram':
    HovDiagram(simyears, backtrack, normalize, Xstr='Time', Ystr='Lat')
elif action == 'windrose':
    ThetaSPDwindrose(simyears, backtrack, Xstr='SPD', Ystr='Theta')
