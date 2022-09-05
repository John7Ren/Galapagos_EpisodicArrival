from parcels import FieldSet, Field, ParticleSet, Variable, JITParticle, AdvectionRK4, plotTrajectoriesFile, ScipyParticle
import numpy as np
import math
from datetime import timedelta
from datetime import datetime
from operator import attrgetter
import xarray as xr
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib import colors
import pandas as pd
import os
from matplotlib.path import Path
import matplotlib.patches as patches
import matplotlib.ticker as ticker
#%% Function to generate data
'''
Loading the data
'''
def TrajAnalysis(simyear, arrivingbefore, backtrack, beaching=True, simdays=729, releasedays=365, dailyrelease=274):
    out_fname = f'/Users/renjiongqiu/Documents_local/Thesis/visualization/TrajAnalysis/{simyear}/'
    os.makedirs(os.path.dirname(out_fname),exist_ok=True)
    secondyear = simyear + 1
    fname = f'_{simyear}-{secondyear}_run{simdays}d_release{releasedays}d_{dailyrelease}par'
    if beaching:
        fname = fname+'_beaching'
        input_fname = f'/Volumes/John_HardDrive/Thesis/sim_results/{simyear}/with_beaching/'
    else:
        fname = fname
        input_fname = f'/Volumes/John_HardDrive/Thesis/sim_results/{simyear}/no_beaching/'
    
    
    # Loading priors. Computed with release_points.py script.
    priors = pd.read_csv('/Users/renjiongqiu/Documents_local/Thesis/data/priors_river_inputs_25N25S.csv',
                          index_col=0)
    sources = list(priors.index)
    # sources = ['Cacique','Chira','Fonseca','Guayaquil','LaColorada','Lima','Panama','Tecojate','Acapulco','Esmeraldas','SanBlas']
    # sources = ['Parachique']
    
    galapagos_extent = [-91.8, -89, -1.4, 0.7]
    
    Lon_dict = {}
    Lat_dict = {}
    Time_dict = {}
    Age_dict = {}
    
    for loc in sources:
        ds = xr.open_dataset(input_fname+f'{loc}'+fname+'.nc')
        AR = {}
        AR['lon'] = np.array(ds.lon)
        AR['lat'] = np.array(ds.lat)
        AR['time'] = np.array(ds.time)
        AR['age'] = np.array(ds.age)
        print(f'load output: {loc}')
        
            
        # Get the mask of gala region
        # Note that gala_traj and gala_obsv are both 1D and are spatially correlated to present a specific point
        gala_traj,gala_obsv = np.where( (AR['lon']>=galapagos_extent[0]) & (AR['lon']<=galapagos_extent[1]) & \
                                        (AR['lat']>=galapagos_extent[2]) & (AR['lat']<=galapagos_extent[3]) )
        
        # Select the trajectories that passed Gala
        # 1st contact
        traj_idx = np.unique(gala_traj)
        obsv_idx = np.zeros_like(traj_idx)
        
        for i,idx in enumerate(traj_idx):
            obsv_idx[i] = gala_obsv[np.where(gala_traj==idx)].min()
        
        if arrivingbefore:
            if len(obsv_idx)>0:
                if backtrack[1] == 999:
                    Lon = np.nan * np.ones((len(traj_idx),max(obsv_idx)))
                    Lat = np.nan * np.ones((len(traj_idx),max(obsv_idx)))
                    Time = np.nan * np.ones((len(traj_idx),max(obsv_idx)))
                    Age = np.nan * np.ones((len(traj_idx),max(obsv_idx)))
                    
                    obsv_st = 0
                    
                    for i,idx in enumerate(traj_idx):
                        length = obsv_idx[i] - obsv_st
                        Lon[i,:length] = AR['lon'][idx,obsv_st:obsv_idx[i]]
                        Lat[i,:length] = AR['lat'][idx,obsv_st:obsv_idx[i]]
                        Time[i,:length] = AR['time'][idx,obsv_st:obsv_idx[i]]
                        Age[i,:length] = AR['age'][idx,obsv_st:obsv_idx[i]]
                        
                elif backtrack[1] < 999:
                    Lon = np.nan * np.ones((len(traj_idx),backtrack[1]))
                    Lat = np.nan * np.ones((len(traj_idx),backtrack[1]))
                    Time = np.nan * np.ones((len(traj_idx),backtrack[1]))
                    Age = np.nan * np.ones((len(traj_idx),backtrack[1]))
                    
                    for i,idx in enumerate(traj_idx):
                        obsv_st = max(0,obsv_idx[i]-backtrack[1])
                        length = obsv_idx[i] - obsv_st
                        Lon[i,:length] = AR['lon'][idx,obsv_st:obsv_idx[i]]
                        Lat[i,:length] = AR['lat'][idx,obsv_st:obsv_idx[i]]
                        Time[i,:length] = AR['time'][idx,obsv_st:obsv_idx[i]]
                        Age[i,:length] = AR['age'][idx,obsv_st:obsv_idx[i]]
            
            else:
                Lon = AR['lon'][traj_idx,:]
                Lat = AR['lat'][traj_idx,:]
                Time = AR['time'][traj_idx,:]
                Age = AR['age'][traj_idx,:]
            
            suffix = '_{backtrack[0]}-{backtrack[1]}d_before'
        
        elif not arrivingbefore:
            Lon = AR['lon'][traj_idx,:]
            Lat = AR['lat'][traj_idx,:]
            Time = AR['time'][traj_idx,:]
            Age = AR['age'][traj_idx,:]
        
        Lon_dict[loc] = Lon
        Lat_dict[loc] = Lat
        Time_dict[loc] = Time
        Age_dict[loc] = Age
        
        print(f'process data done: {loc}')

    # Save to npy
    Lon_npy = pd.DataFrame.from_dict(Lon_dict,orient='index')
    Lat_npy = pd.DataFrame.from_dict(Lat_dict,orient='index')
    Time_npy = pd.DataFrame.from_dict(Time_dict,orient='index')
    Age_npy = pd.DataFrame.from_dict(Age_dict,orient='index')
    np.save(out_fname+'Lon'+fname+f'{suffix}.npy',Lon_npy,allow_pickle=True)
    np.save(out_fname+'Lat'+fname+f'{suffix}.npy',Lat_npy,allow_pickle=True)
    np.save(out_fname+'Time'+fname+f'{suffix}.npy',Time_npy,allow_pickle=True)
    np.save(out_fname+'Age'+fname+f'{suffix}.npy',Age_npy,allow_pickle=True)
    print('generate: '+fname)

def Density(lons,lats,focus_region,r=1/12):
    per = 100 # the number of levels for the color bar
    bins = [np.arange(focus_region[0],focus_region[1],r),np.arange(focus_region[2],focus_region[3],r)]
    index = ~np.isnan(lons) & ~np.isnan(lats) # some spots are not nan but strangely big or small
    lons = lons[index]
    lats = lats[index]
    H,xe,ye = np.histogram2d(lons, lats, bins=bins)
    xb = ( xe[:-1] + xe[1:] ) / 2
    yb = ( ye[:-1] + ye[1:] ) / 2
    if lons.shape[0] != 0:
        levels = np.unique(np.percentile(H,np.linspace(0,100,per+1)))
        nlevels = len(levels)
        loop = 0
        while nlevels<4:
            per = per*2
            levels = np.unique(np.percentile(H,np.linspace(0,100,per+1)))
            nlevels = len(levels)
            loop += 1
            if loop >100:
                print('loop:100')
                break
        # levels = np.delete(levels,0)
        ticks = np.floor(np.percentile(levels,np.linspace(0,100,5)))
        levels = np.where(levels==0,0.5,levels)
    else:
        levels = [0.5,1]
        ticks = [0,1]
    
    return H,xb,yb,levels,ticks

def Density_landmask(lons,lats,Lons,Lats):
    per = 100
    bins = [Lons,Lats]
    index = ~np.isnan(lons) & ~np.isnan(lats) # some spots are not nan but strangely big or small
    lons = lons[index]
    lats = lats[index]
    H,xe,ye = np.histogram2d(lons, lats, bins=bins)
    xb = ( xe[:-1] + xe[1:] ) / 2
    yb = ( ye[:-1] + ye[1:] ) / 2
    if lons.shape[0] != 0:
        levels = np.unique(np.percentile(H,np.linspace(0,100,per+1)))
        nlevels = len(levels)
        loop = 0
        while nlevels<4:
            per = per*2
            levels = np.unique(np.percentile(H,np.linspace(0,100,per+1)))
            nlevels = len(levels)
            loop += 1
            if loop >100:
                print('loop:100')
                break
        # levels = np.delete(levels,0)
        ticks = np.floor(np.percentile(levels,np.linspace(0,100,5)))
        levels = np.where(levels==0,0.5,levels)
    else:
        levels = [0.5,1]
        ticks = [0,1]
        
    return H,xb,yb,levels,ticks
    
# Select only the part before arriving in the Galapagos
def beforeArriving(lons,lats):
    galapagos_extent = [-91.8, -89, -1.4, 0.7]
    
    gala_traj,gala_obsv = np.where( (lons>=galapagos_extent[0]) & (lons<=galapagos_extent[1]) & \
                                    (lats>=galapagos_extent[2]) & (lats<=galapagos_extent[3]))
    traj_idx = np.unique(gala_traj)
    obsv_idx = np.zeros_like(traj_idx)
    
    for i,idx in enumerate(traj_idx):
        obsv_idx[i] = gala_obsv[np.where(gala_traj==idx)].min()
        
    if len(obsv_idx)>0:
        lons_before = np.nan * np.ones((len(traj_idx),max(obsv_idx)))
        lats_before = np.nan * np.ones((len(traj_idx),max(obsv_idx)))
    
        for i,idx in enumerate(traj_idx):
            lons_before[i,:obsv_idx[i]] = lons[idx,:obsv_idx[i]]
            lats_before[i,:obsv_idx[i]] = lats[idx,:obsv_idx[i]]
    
    else:
        lons_before = lons
        lats_before = lats
        
    return lons_before, lats_before

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
    elif box == 'PB_selfdef':
        verts = [
                 (-88, -1), # [0,0]
                 (-88, 9), # [0,1]
                 (-77, 9), # [1,1]
                 (-77, -1), # [1,0]
                 (-88, -1)  # [0,0]
                 ]
    elif box == 'NHCS':
        verts = [
                 (-88, -1), # [0,0]
                 (-88, 9), # [0,1]
                 (-77, 9), # [1,1]
                 (-77, -1), # [1,0]
                 (-88, -1)  # [0,0]
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
def add_text(ax):
    priors = pd.read_csv('/Users/renjiongqiu/Documents_local/Thesis/data/priors_river_inputs_25N25S.csv',
                          index_col=0)
    sources = list(priors.index)
    river_sources = np.load('/Users/renjiongqiu/Documents_local/Thesis/data/river_sources_25N25S_v0.npy',allow_pickle=True).item()
    northern_sources = ['Acapulco', 'SalinaCruz', 'Tecojate', 'LaColorada', 'Fonseca', 'Chira']
    panama_bight = ['Panama', 'Cacique']
    nhcs_sources = ['Guayaquil','Parachique','Lima']
    
    galapagos_extent = [-91.8, -89, -1.4, 0.7]
    text_dev = [0.03,0.1]
    for loc in sources:
        river_lons = river_sources[loc][1]
        river_lats = river_sources[loc][0]
        text_lon = (river_lons+115)/50+text_dev[1]
        text_lat = (river_lats+25)/50+text_dev[0]
        text_gala_lon = (galapagos_extent[0]+115)/50-0.05
        text_gala_lat = (galapagos_extent[2]+25)/50-0.02
        if loc == 'LaColorada' or loc == 'SalinaCruz':
            text_lon = (river_lons+115)/50+text_dev[1]
            text_lat = (river_lats+25)/50+text_dev[0]
    
        ax.text(text_lon,text_lat,f'{loc}',ha='center',va='center',fontsize=18,transform=ax.transAxes)
    ax.text(text_gala_lon,text_gala_lat+0.1,'Galapagos Archipelago',ha='center',va='center',fontsize=18,transform=ax.transAxes)


#%% Generate data
# Generate distribution array and save in .npy
simyears = np.arange(2018,2019)
for yr in simyears:
    TrajAnalysis(simyear=yr, arrivingbefore=False, backtrack=[0,999])
    print(f'{yr}')
#%% Basic settings
simyears = np.arange(2007,2021)
# simyears=[2012]
arrivingbefore = True
backtrack = [0,999]
lorenz = False

if lorenz == False:
    output_fname = '/Users/renjiongqiu/Documents_local/Thesis/visualization/visual_outputs/TrajAnalysis/trajectories/'
    priors = pd.read_csv('/Users/renjiongqiu/Documents_local/Thesis/data/priors_river_inputs_25N25S.csv',
                          index_col=0)
    sources = list(priors.index)
    river_sources = np.load('/Users/renjiongqiu/Documents_local/Thesis/data/river_sources_25N25S_v0.npy',allow_pickle=True).item()
elif lorenz == True:
    output_fname = '/storage/home/9703243/galapagos/outputs/results/'
    priors = pd.read_csv('/storage/home/9703243/galapagos/scripts/priors_river_inputs_v0.csv',
                          index_col=0)
    sources = list(priors.index)
    river_sources = np.load('/storage/home/9703243/galapagos/scripts/river_sources_25N25S_v0.npy',allow_pickle=True).item()


northern_sources = ['Acapulco', 'SalinaCruz', 'Tecojate', 'LaColorada', 'Fonseca', 'Chira']
# sources = ['Panama']

simdays = 729
releasedays = 365
dailyrelease = 274

study_region = (-115,-65,-25,25)
focus_region = (-115,-65,-25,25)

number_sources = len(sources)
number_years = len(simyears)
boxlist = ['SEC','PNJtilt','NECC','PB']
#%% Load the data
Lon = {}
Lat = {}
Time = {}
Age = {}

for yr in simyears:
    secondyear = yr + 1
    yrstr = str(yr)
    
    if lorenz == False:
        input_fname = f'/Users/renjiongqiu/Documents_local/Thesis/visualization/TrajAnalysis/trajectories/'
    elif lorenz == True:
        input_fname = f'/storage/home/9703243/galapagos/trajectories_gala/{yr}/'
        
    fname = f'_{yr}-{secondyear}_run{simdays}d_release{releasedays}d_{dailyrelease}par_beaching'
        
    if arrivingbefore:
        if backtrack[1] > 0:
            fname = fname+f'_{backtrack}d-before'
        elif not backtrack:
            fname = fname+'_before'
            input_fname = f'/Users/renjiongqiu/Documents_local/Thesis/visualization/TrajAnalysis/trajectories/arriving_gala/'
    else:
        fname = fname
    
    Lon[yrstr] = np.load(input_fname+'Lon'+fname+'.npy',allow_pickle=True)
    Lat[yrstr] = np.load(input_fname+'Lat'+fname+'.npy',allow_pickle=True)
    Time[yrstr] = np.load(input_fname+'Time'+fname+'.npy',allow_pickle=True)
    Age[yrstr] = np.load(input_fname+'Age'+fname+'.npy',allow_pickle=True)

#%% Trajectories - by years
for year in simyears:
    yrstr = str(year)
    out_fname = output_fname+f'by_year/{year}/'
    os.makedirs(os.path.dirname(out_fname),exist_ok=True)
    
    rows = math.ceil(number_sources/3)
    cols = min(3,rows)
    rows = max(3,rows)
    
    # fig,ax = plt.subplots(rows,cols,figsize=(8*cols,4.5*rows))
    fig = plt.figure(figsize=(8*cols,6*rows))
    
    for i in range(rows*cols):
        ax = fig.add_subplot(rows,cols,i+1,projection=ccrs.PlateCarree())
        if i < number_sources:
            loc = sources[i]
            river_lons = river_sources[loc][1]
            river_lats = river_sources[loc][0]
            
            lons = Lon[yrstr][i][0]
            lats = Lat[yrstr][i][0]
            H,xb,yb,levels,ticks = Density(lons,lats,focus_region)
            norm = colors.BoundaryNorm(levels,256)
            if (H!=0).any(): 
                cf = ax.contourf(xb,yb,H.T,levels=levels,transform=ccrs.PlateCarree(),norm=norm)
                ax.coastlines(resolution='50m')
                ax.add_feature(cfeature.LAND)
                ax.add_feature(cfeature.RIVERS)
                ax.add_feature(cfeature.LAKES)
                ax.add_feature(cfeature.BORDERS)
                ax.set_title(f'{sources[i]}',fontsize=16)
                ax.gridlines(draw_labels=True,linewidth=0.5,color='k',alpha=0.5,linestyle='--')
                ax.set_extent(study_region,crs=ccrs.PlateCarree())
                cb = plt.colorbar(cf,fraction=0.04, pad=0.06,ticks=levels)
                # cb.ax.set_yticklabels(range(0,7))
                
                # draw sources
                path = DrawCluster(river_lons, river_lats, 1)
                patchS = patches.PathPatch(path,ec='k',fc='none',lw=2)
                ax.add_patch(patchS)
               
                # draw boxes
                for box in boxlist:
                    path = DrawRegion(box)
                    patchR = patches.PathPatch(path,ec='k',fc='none',lw=2)
                    ax.add_patch(patchR)
                    
            elif (H==0).all():
                cf = ax.contourf(xb,yb,H.T,cmap='Blues',levels=[1,10],transform=ccrs.PlateCarree(),norm=norm)
                ax.coastlines(resolution='50m')
                ax.add_feature(cfeature.LAND)
                ax.add_feature(cfeature.RIVERS)
                ax.add_feature(cfeature.LAKES)
                ax.add_feature(cfeature.BORDERS)
                ax.set_title(f'{sources[i]}',fontsize=16)
                ax.gridlines(draw_labels=True,linewidth=0.5,color='k',alpha=0.5,linestyle='--')
                ax.set_extent(study_region,crs=ccrs.PlateCarree())
                cb = plt.colorbar(cf,fraction=0.04, pad=0.06,ticks=[1,10])
        elif i >= number_sources:
            fig.delaxes(ax)
    
    fig.suptitle(f'{year}',fontsize=18,fontweight='bold')
    
    plt.tight_layout(pad=2,w_pad=0.1)
    plt.savefig(out_fname+f'Density_{year}-{year+1}.png')

#%% Trajectories - by location
for j,loc in enumerate(sources):
    out_fname = output_fname+f'by_location/{loc}/'
    os.makedirs(os.path.dirname(out_fname),exist_ok=True)
    
    river_lons = river_sources[loc][1]
    river_lats = river_sources[loc][0]
    
    rows = math.ceil(number_years/3)
    cols = min(3,rows)
    rows = max(3,rows)
    
    # fig,ax = plt.subplots(rows,cols,figsize=(8*cols,4.5*rows))
    fig = plt.figure(figsize=(8*cols,6*rows))
    
    for i in range(rows*cols):
        ax = fig.add_subplot(rows,cols,i+1,projection=ccrs.PlateCarree())
        if i < number_years:
            yrstr = str(simyears[i])
            lons = Lon[yrstr][j][0]
            lats = Lat[yrstr][j][0]
            H,xb,yb,levels,ticks = Density(lons,lats,focus_region)
            norm = colors.BoundaryNorm(levels,256)
            if (H!=0).any(): 
                cf = ax.contourf(xb,yb,H.T,levels=levels,transform=ccrs.PlateCarree(),norm=norm)
                ax.coastlines(resolution='50m')
                ax.add_feature(cfeature.LAND)
                ax.add_feature(cfeature.RIVERS)
                ax.add_feature(cfeature.LAKES)
                ax.add_feature(cfeature.BORDERS)
                ax.set_title(f'{yrstr}',fontsize=16)
                ax.gridlines(draw_labels=True,linewidth=0.5,color='k',alpha=0.5,linestyle='--')
                ax.set_extent(study_region,crs=ccrs.PlateCarree())
                cb = plt.colorbar(cf,fraction=0.04, pad=0.06,ticks=ticks)
                # cb.ax.set_yticklabels(range(0,7))
                
                # draw sources
                path = DrawCluster(river_lons, river_lats, 1)
                patchS = patches.PathPatch(path,ec='k',fc='none',lw=2)
                ax.add_patch(patchS)
               
                # draw boxes
                for box in boxlist:
                    path = DrawRegion(box)
                    patchR = patches.PathPatch(path,ec='k',fc='none',lw=2)
                    ax.add_patch(patchR)
                
            elif (H==0).all():
                cf = ax.contourf(xb,yb,H.T,levels=[1,10],transform=ccrs.PlateCarree(),norm=norm)
                ax.coastlines(resolution='50m')
                ax.add_feature(cfeature.LAND)
                ax.add_feature(cfeature.RIVERS)
                ax.add_feature(cfeature.LAKES)
                ax.add_feature(cfeature.BORDERS)
                ax.set_title(f'{yrstr}',fontsize=16)
                ax.gridlines(draw_labels=True,linewidth=0.5,color='k',alpha=0.5,linestyle='--')
                ax.set_extent(study_region,crs=ccrs.PlateCarree())
                cb = plt.colorbar(cf,fraction=0.04, pad=0.06,ticks=[1,10])
        elif i >= number_years:
            fig.delaxes(ax)
    
    fig.suptitle(f'{loc}',fontsize=18,fontweight='bold')
    
    plt.tight_layout(pad=2,w_pad=0.1)
    plt.savefig(out_fname+f'Density_{loc}_{simyears[0]}-{simyears[-1]}-v2.png')

#%% Trajectories - by location (temporal averaged)
# create landmask
file_path = "/Users/renjiongqiu/Documents_local/Thesis/data/sources/psy4v3r1-daily_2D_2019-10-25.nc"
ds = xr.load_dataset(file_path)
galapagos_extent = [-91.8, -89, -1.4, 0.7]
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

norm = True
N = 100010
r = 1/12
# ylen = int((focus_region[3] - focus_region[2]) / r - 1)
# xlen = int((focus_region[1] - focus_region[0]) / r - 1)
ylen = len(Lats_nemo) - 1
xlen = len(Lons_nemo) - 1
out_fname = output_fname
Hplot = {} # 'NS', 'Guayaquil', 'Parachique', 'Lima'
Hplot['NS'] = np.zeros((xlen,ylen))

# make the plotting dataset
for j,loc in enumerate(sources):
    H = np.zeros((xlen,ylen))
    for i in range(number_years):
        yrstr = str(simyears[i])
        lons = Lon[yrstr][j][0]
        lats = Lat[yrstr][j][0]
        # Hi,xb,yb = Density(lons,lats,focus_region)[0:3]
        Hi,xb,yb = Density_landmask(lons,lats,Lons_nemo,Lats_nemo)[0:3]
        H += Hi
    
    if norm:
        H = H / (number_years*N)
    
    if loc in northern_sources:
        Hplot['NS'] += H
    else:
        Hplot[loc] = H
if norm:
    Hplot['NS'] = Hplot['NS'] / len(northern_sources)
#%% trajectories before Gala
if arrivingbefore:
    if backtrack > 0:
        suffix = f'{backtrack}d-before'
    elif not backtrack:
        suffix = 'before'
else:
    suffix = ''

fig = plt.figure(figsize=(8*2,6*2),constrained_layout=True)
cmap = 'jet' # options: 'viridis', 'jet'
tks = np.zeros((4,6))
for i in range(4):
    if i == 0:
        H = Hplot['NS']
        sourcelist = northern_sources
        boxlist = ['Gala','PB_selfdef','PNJtilt']
        title = f'The Northern Sources - {suffix}'
    elif i == 1:
        H = Hplot['Guayaquil']
        sourcelist = ['Guayaquil']
        boxlist = ['Gala']
        title = f'Guayaquil - {suffix}'
    elif i == 2:
        H = Hplot['Parachique']
        sourcelist = ['Parachique']
        boxlist = ['Gala']
        title = f'Parachique - {suffix}'
    elif i == 3:
        H = Hplot['Lima']
        sourcelist = ['Lima']
        boxlist = ['Gala']
        title = f'Lima - {suffix}'
    # levels = 10 ** (np.arange(0,int(math.ceil(np.log10(np.nanmax(H))))-1,-1))
    # ticks = np.percentile(levels,np.linspace(0,100,5))
    levels, ticks = createlevels(H, nloop=1000)
    norm = colors.BoundaryNorm(levels,256)
    ax = fig.add_subplot(2,2,i+1,projection=ccrs.PlateCarree())
    H = H + landmask.T + galamask.T
    cf = ax.contourf(xb,yb,H.T,transform=ccrs.PlateCarree(),cmap=cmap,levels=levels,norm=norm,extend='max')
    ax.coastlines(resolution='50m')
    ax.add_feature(cfeature.LAND)
    ax.add_feature(cfeature.BORDERS)
    ax.set_title(title,fontsize=16)
    ax.gridlines(draw_labels=True,linewidth=0.5,color='w',alpha=0.5,linestyle='--')
    ax.set_extent(study_region,crs=ccrs.PlateCarree())
    cb = plt.colorbar(cf,fraction=0.06, pad=0.04, ticks=ticks, format=ticker.FuncFormatter(tksfmt))
    # cb.formatter.set_powerlimits((0,0))
    # cb.formatter.set_useMathText(True)
    # cb.ax.set_yticklabels(range(0,7))
    
    # draw box
    for box in boxlist:
        path = DrawRegion(box)
        patchR = patches.PathPatch(path,ec='k',fc='none',lw=1.5)
        ax.add_patch(patchR)
    
    # draw sources
    for loc in sourcelist:
        river_lons = river_sources[loc][1]
        river_lats = river_sources[loc][0]
        path = DrawCluster(river_lons, river_lats, 1)
        patchS = patches.PathPatch(path,ec='dimgray',fc='none',lw=1.5)
        ax.add_patch(patchS)

plt.savefig(out_fname+f'Density_yearintegrated_{suffix}.png')

#%% test on one location
simyear = 2016
beaching=True
simdays=729
releasedays=365
dailyrelease=274
secondyear = yr + 1
fname = f'_{yr}-{secondyear}_run{simdays}d_release{releasedays}d_{dailyrelease}par_beaching'
yrstr = str(simyear)
input_fname = f'/Users/renjiongqiu/Documents_local/Thesis/visualization/TrajAnalysis/{simyear}/'
i = 3

study_region = (-115,-65,-25,25)

bins = [np.arange(-110,-75,0.1),np.arange(-10,20,0.1)]
lons = Lon[yrstr][i][0]
lats = Lat[yrstr][i][0]
H,xb,yb,levels,ticks = Density(lons,lats,focus_region)
norm = colors.BoundaryNorm(levels,256)

fig = plt.figure(figsize=(12,14))
ax = fig.add_subplot(projection=ccrs.PlateCarree())
cf = ax.contourf(xb,yb,H.T,cmap='Blues',levels=levels,transform=ccrs.PlateCarree(),norm=norm)
ax.coastlines(resolution='50m')
ax.add_feature(cfeature.LAND)
ax.add_feature(cfeature.RIVERS)
ax.add_feature(cfeature.LAKES)
ax.add_feature(cfeature.BORDERS)
# ax.set_title(f'{loc} 2yr-simulation particle density in the 2nd year',fontsize=16)
ax.gridlines(draw_labels=True,linewidth=0.5,color='k',alpha=0.5,linestyle='--')
ax.set_extent(study_region,crs=ccrs.PlateCarree())
cb = plt.colorbar(cf,fraction=0.04, pad=0.06,ticks=ticks)
# cb.ax.set_yticklabels(range(0,7))

#%% Draw region map
priors = pd.read_csv('/Users/renjiongqiu/Documents_local/Thesis/data/priors_river_inputs_25N25S.csv',
                      index_col=0)
sources = list(priors.index)
river_sources = np.load('/Users/renjiongqiu/Documents_local/Thesis/data/river_sources_25N25S_v0.npy',allow_pickle=True).item()
boxlist = ['Gala']
northern_sources = ['Acapulco', 'SalinaCruz', 'Tecojate', 'LaColorada', 'Fonseca', 'Chira']
panama_bight = ['Panama', 'Cacique']
nhcs_sources = ['Guayaquil','Parachique','Lima']

study_region = (-115,-65,-25,25)

fig = plt.figure(figsize=(12,14))
ax = fig.add_subplot(projection=ccrs.PlateCarree())
ax.coastlines(resolution='50m')
ax.add_feature(cfeature.LAND)
ax.add_feature(cfeature.BORDERS)
ax.gridlines(draw_labels=True,linewidth=0.5,color='k',alpha=0.5,linestyle='--')
ax.set_extent(study_region,crs=ccrs.PlateCarree())
# add_text(ax)
# draw box

for box in boxlist:
    path = DrawRegion(box)
    patchR = patches.PathPatch(path,ec='k',fc='none',lw=1.5)
    ax.add_patch(patchR)

    
# draw sources
for loc in sources:
    if loc in northern_sources:
        color = 'tab:red'
    elif loc in panama_bight:
        color = 'tab:green'
    elif loc in nhcs_sources:
        color = 'tab:blue'
    river_lons = river_sources[loc][1]
    river_lats = river_sources[loc][0]
    path = DrawCluster(river_lons, river_lats, 1)
    patchS = patches.PathPatch(path,ec=color,fc='none',lw=1.5)
    ax.add_patch(patchS)
    
plt.savefig('/Users/renjiongqiu/Documents_local/Thesis/results/source_map.png')