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
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from matplotlib import colors,dates,cm
import pandas as pd
import matplotlib.ticker as mticker
import calendar
import matplotlib.gridspec as gridspec
from matplotlib.path import Path
import matplotlib.patches as patches
#%% Function to generate data
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
'''
Loading the data
'''
def TimeAnalysis(simyear, beaching=True, simdays=729, releasedays=365, dailyrelease=274):
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
    # sources = ['Panama']
    
    galapagos_extent = [-91.8, -89, -1.4, 0.7]
    
    age_all = {}
    age_1st = {}
    time_all = {}
    time_1st = {}
    time_leaving = {}
    time_releasing = {}
    
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
        
        # All contacts
        age_all[loc] = AR['age'][gala_traj,gala_obsv]
        time_all[loc] = AR['time'][gala_traj,gala_obsv]
        
        # 1st contact
        traj_idx = np.unique(gala_traj)
        obsv_idx = np.zeros_like(traj_idx)
        
        for i,idx in enumerate(traj_idx):
            obsv_idx[i] = gala_obsv[np.where(gala_traj==idx)].min()

        age_1st[loc] = AR['age'][traj_idx,obsv_idx]
        time_1st[loc] = AR['time'][traj_idx,obsv_idx]
        
        # Leaving time
        time_leaving[loc] = AR['time'][traj_idx,0]
        # Remarks: age_1st and time_leaving have the same length with each position
        # in the array corresponding to the same particle (traj_idx)
        
        # releasing methods
        time_releasing[loc] = AR['time'][:,0]
        print(f'process data done: {loc}')
                
        
    # Save to npy
    Time_all = pd.DataFrame.from_dict(time_all,orient='index')
    Time_1st = pd.DataFrame.from_dict(time_1st,orient='index')
    Time_leaving = pd.DataFrame.from_dict(time_leaving,orient='index')
    Age_all = pd.DataFrame.from_dict(age_all,orient='index')
    Age_1st = pd.DataFrame.from_dict(age_1st,orient='index')
    Time = pd.DataFrame.from_dict(time_releasing,orient='index')
    np.save('/Users/renjiongqiu/Documents_local/Thesis/visualization/TimeAnalysis/ReleasingTime'+fname+'.npy',Time,allow_pickle=True)
    np.save('/Users/renjiongqiu/Documents_local/Thesis/visualization/TimeAnalysis/ArrivingTime_all'+fname+'.npy',Time_all,allow_pickle=True)
    np.save('/Users/renjiongqiu/Documents_local/Thesis/visualization/TimeAnalysis/ArrivingTime_1st'+fname+'.npy',Time_1st,allow_pickle=True)
    np.save('/Users/renjiongqiu/Documents_local/Thesis/visualization/TimeAnalysis/LeavingTime'+fname+'.npy',Time_leaving,allow_pickle=True)
    np.save('/Users/renjiongqiu/Documents_local/Thesis/visualization/TimeAnalysis/Age_all'+fname+'.npy',Age_all,allow_pickle=True)
    np.save('/Users/renjiongqiu/Documents_local/Thesis/visualization/TimeAnalysis/Age_1st'+fname+'.npy',Age_1st,allow_pickle=True)
    print('generate: '+fname)

#%% Generate data
# Generate distribution array and save in .npy
TimeAnalysis(simyear=2016, beaching=True)
#%% Basic settings
output_fname = '/Users/renjiongqiu/Documents_local/Thesis/visualization/visual_outputs/'
priors = pd.read_csv('/Users/renjiongqiu/Documents_local/Thesis/data/priors_river_inputs_25N25S.csv',
                      index_col=0)
sources = list(priors.index)
# sources = ['Acapulco','SalinaCruz','Tecojate','LaColorada','Fonseca','Chira','Panama','Esmeraldas','Guayaquil','Parachique']
# sources = ['Panama']

simyears = np.arange(2007,2021)
# simyears=[2008,2015]
beaching=True
simdays = 729
releasedays = 365
dailyrelease = 274
    
firstcontact = True

number_sources = len(sources)

#%% Load the data
Time_all = {}
Time_1st = {}
Time_leaving = {}
Age_all = {}
Age_1st = {}
Time_releasing = {}

for yr in simyears:
    secondyear = yr + 1
    fname = f'_{yr}-{secondyear}_run{simdays}d_release{releasedays}d_{dailyrelease}par'
    
    if beaching:
        fname = fname+'_beaching'
    else:
        fname = fname
    
    yrstr = str(yr)
    
    Time_all[yrstr] = np.load('/Users/renjiongqiu/Documents_local/Thesis/visualization/TimeAnalysis/ArrivingTime_all'+fname+'.npy',
                             allow_pickle=True)
    Time_1st[yrstr] = np.load('/Users/renjiongqiu/Documents_local/Thesis/visualization/TimeAnalysis/ArrivingTime_1st'+fname+'.npy',
                             allow_pickle=True)
    Time_leaving[yrstr] = np.load('/Users/renjiongqiu/Documents_local/Thesis/visualization/TimeAnalysis/LeavingTime'+fname+'.npy',
                             allow_pickle=True)
    Age_all[yrstr] = np.load('/Users/renjiongqiu/Documents_local/Thesis/visualization/TimeAnalysis/Age_all'+fname+'.npy',
                             allow_pickle=True)
    Age_1st[yrstr] = np.load('/Users/renjiongqiu/Documents_local/Thesis/visualization/TimeAnalysis/Age_1st'+fname+'.npy',
                         allow_pickle=True)
    # Time_releasing[yrstr] = np.load('/Users/renjiongqiu/Documents_local/Thesis/visualization/ReleasingTime'+fname+'.npy',
    #                          allow_pickle=True)

#%% Time distribution - Arriving
# simyears = [2019]
H_arriving = {}
norm = {}
lines = {}

checkfullyear = True
inday = False
normalization = 'byArrivingParcels' # options: 'byArrivingParcels', 'byTotalParcels'
N = 100010
subplots = False
northern_sources = ['Acapulco', 'SalinaCruz', 'Tecojate', 'LaColorada', 'Fonseca', 'Chira']
pb_sources = ['Panama', 'Cacique','Esmeraldas']
nhcs_sources = ['Guayaquil','Parachique','Lima']

if firstcontact:
    Time_arriving = Time_1st
    suffix = '_1st'
else:
    Time_arriving = Time_1st
    suffix = '_all'

dt = 7
for yr in simyears:
    if checkfullyear:
        if yr%4!=0 and (yr+1)%4!=0:
            yrstr = str(yr)
            start_time = np.datetime64(yrstr+'-01-02')
            end_time = np.datetime64(str(yr+2)+'-01-02')
            timerange = np.arange(start_time,end_time,dt)
            bins = len(timerange)-1
            H_time_arriving = np.zeros((number_sources,bins))
            xbar = np.empty(bins,dtype='datetime64[D]')
        elif yr%4==0 or (yr+1)%4==0:
            yrstr = str(yr)
            start_time = np.datetime64(yrstr+'-01-02')
            end_time = np.datetime64(str(yr+2)+'-01-01')
            timerange = np.arange(start_time,end_time,dt)
            bins = len(timerange)-1
            H_time_arriving = np.zeros((number_sources,bins))
            xbar = np.empty(bins,dtype='datetime64[D]')
    else:
        start_time = np.datetime64(f'{yr+1}'+'-01-01')
        end_time = np.datetime64(f'{yr+1}'+'-03-01')
        timerange = np.arange(start_time,end_time,1)
        bins = len(timerange)-1
        H_time_arriving = np.zeros((number_sources,bins))
        xbar = np.empty(bins,dtype='datetime64[D]')
    for i in range(number_sources):
        index = ~np.isnan(Time_arriving[yrstr][i,:])
        H_time_arriving[i,:],edge = np.histogram(Time_arriving[yrstr][i,:][index].astype('datetime64[D]').astype(int),bins=timerange.astype(int))
        x = ((edge[:-1] + edge[1:]) / 2)
        xbar[:] = x.astype('datetime64[D]')
    H_arriving[yrstr] = H_time_arriving
    norm[yrstr] = H_time_arriving.sum(axis=1) # the total number of arriving Gala par from one source from one year's of simulation
    
if normalization == 'byArrivingParcels': 
    for yr in simyears:
        yrstr = str(yr)
        l = H_arriving[yrstr].shape[1]
        H_arriving[yrstr] = H_arriving[yrstr] / np.tile(H_arriving[yrstr].sum(axis=1),(l,1)).T * 100
elif normalization == 'byTotalParcels': 
    for yr in simyears:
        yrstr = str(yr)
        H_arriving[yrstr] = H_arriving[yrstr] / N * 100
    
if inday:
    timelimit = np.timedelta64(bins+1,'D').astype(int)
    timeline = np.arange(0,timelimit-1,1)
    xticks = np.arange(0, timelimit+1, 60)
else:
    timelimit = end_time
    timeline = xbar
    xticks = np.arange(start_time,end_time,len(timeline)//12)

arr_rate = {}
for iloc,loc in enumerate(sources):
    arr_rate[loc] = np.zeros_like(timeline,dtype=float)
    for iyr,yr in enumerate(simyears):
        yrstr = str(yr)
        arr_rate[loc] += H_arriving[yrstr][iloc,:]
    arr_rate[loc] = arr_rate[loc]/len(simyears)

alfa = 0.8
river_sources = np.load('/Users/renjiongqiu/Documents_local/Thesis/data/river_sources_25N25S_v0.npy',allow_pickle=True).item()
northern_sources = ['Acapulco', 'SalinaCruz', 'Tecojate', 'LaColorada', 'Fonseca', 'Chira']
pb_sources = ['Panama', 'Cacique','Esmeraldas']
nhcs_sources = ['Guayaquil','Parachique','Lima']
study_region = (-115, -65, -25, 25)

start_time = np.datetime64('2020-01-01')
end_time = np.datetime64('2021-12-31')
dtplt = np.timedelta64(2,'M')
xticks = np.arange(start_time.astype('datetime64[M]'),end_time.astype('datetime64[M]')+dtplt,dtplt).astype('datetime64[D]')
width = dt
cols = 4
rows = math.ceil(number_sources/cols)
ttl = dict(fontsize=30)
lbm = dict(fontsize=26)
tkm = dict(labelsize=22)

color = cm.tab20( np.linspace(0,20,len(sources)).astype(int) )
if subplots:
    fig,axs = plt.subplots(rows,cols,figsize=(16*cols,9*rows))
    for j,yr in enumerate(simyears):
        yrstr = str(yr)
        for iloc,loc in enumerate(sources):
            ax = axs[iloc//4,iloc%4]
            if j == 0:
                ax.set_title(f'{loc}',**ttl)
            ax.bar(timeline[:],H_arriving[yrstr][iloc,:],width=width,color=color[iloc])
            ax.set_xlim([xticks[0],xticks[-1]])
            ax.set_ylim([0,100])
            
            ax.set_xticks(xticks.astype(int))
            hfmt = dates.DateFormatter('%b-%d')
            ax.xaxis.set_major_formatter(hfmt)
            plt.setp(ax.get_xticklabels(), rotation=45, ha='center')
            ax.tick_params('both',**tkm)
            ax.set_xlabel('Arriving date',**lbm)
            ax.set_ylabel('Fraction [%]',**lbm)
        
    # fig.suptitle('The arriving patterns for each source',fontsize=26)
    plt.tight_layout()
    plt.savefig(output_fname+f'Arriving_{simyears[0]}-{simyears[-1]}{suffix}_subplots.png')
else:
    fig = plt.figure(figsize=(22,18),constrained_layout=True)
    gs = gridspec.GridSpec(3,2,width_ratios=[16,6])
    axs = np.empty((3,2),dtype=object)
    for row in range(3):
        for col in range(2):
            if col == 0:
                ax = fig.add_subplot(gs[row,col])
                ax.set_xlim([xticks[0],xticks[-1]])
                ax.set_ylim([0,20])
                ax.set_xticks(xticks.astype(int))
                ax.tick_params('both',**tkm)
                # ax.set_xlabel('Arriving date',**lbm)
                ax.set_ylabel('Fraction [%]',**lbm)
                hfmt = dates.DateFormatter('%B')
                ax.xaxis.set_major_formatter(hfmt)
                plt.setp(ax.get_xticklabels(), rotation=45, ha='center')
                ax.grid()
                if row == 0:
                    ax.set_title('The Northern Region sources',**ttl)
                    ax.xaxis.set_ticklabels([])
                elif row == 1:
                    ax.set_title('The Panama Bight and Surrounding Region sources',**ttl)
                    ax.xaxis.set_ticklabels([])
                elif row == 2:
                    ax.set_title('The North Humboldt Current System sources',**ttl)
                    ax.set_xlabel('Arriving date',**lbm)
                axs[row,col] = ax
                
            elif col == 1:
                ax = fig.add_subplot(gs[row,col],projection=ccrs.PlateCarree())
                ax.coastlines(resolution='50m')
                ax.add_feature(cfeature.NaturalEarthFeature('physical', 'land', '50m', edgecolor='face', facecolor='silver'),zorder=-1)
                ax.add_feature(cfeature.NaturalEarthFeature('physical', 'ocean', '50m', edgecolor='face', facecolor='lightskyblue'),zorder=-1)
                ax.add_feature(cfeature.BORDERS)
                
                gl = ax.gridlines(draw_labels=True,linewidth=1.,color='k',alpha=0.5,linestyle='--')
                gl.xlabels_bottom = False
                gl.xlabels_top = False
                gl.ylabels_right = True
                gl.ylabels_left = False
                gl.yformatter = LATITUDE_FORMATTER
                if row == 0:
                    gl.xlabels_top = True
                    gl.xformatter = LONGITUDE_FORMATTER
                elif row == 2:
                    gl.xlabels_bottom = True
                    gl.xformatter = LONGITUDE_FORMATTER
                gl.xlabel_style = {'size':22}
                gl.ylabel_style = {'size':22}
                
                if row == 0:
                    sourcelist = northern_sources
                    boxlist = ['Gala','NR']
                elif row == 1:
                    sourcelist = pb_sources
                    boxlist = ['Gala','PBSR']
                elif row == 2:
                    sourcelist = nhcs_sources
                    boxlist = ['Gala','NHCS']
                # draw sources
                for loc in sourcelist:
                    river_lons = river_sources[loc][1]
                    river_lats = river_sources[loc][0]
                    path = DrawCluster(river_lons, river_lats, 1)
                    patchS = patches.PathPatch(path,ec='tab:red',fc='none',lw=2)
                    ax.add_patch(patchS)
                for box in boxlist:
                    path = DrawRegion(box)
                    patchR = patches.PathPatch(path,ec='k',fc='none',lw=2)
                    ax.add_patch(patchR)
                ax.set_extent(study_region,crs=ccrs.PlateCarree())
    
    for j,yr in enumerate(simyears):
        yrstr = str(yr)
        # plot
        for iloc,loc in enumerate(sources):
            if loc in northern_sources:
                ax = axs[0,0]
                if j == 0:
                    ax.bar(timeline[:],arr_rate[loc],width=width,color=color[iloc],label=f'{loc}',alpha=alfa)
                else:
                    ax.bar(timeline[:],arr_rate[loc],width=width,color=color[iloc],alpha=alfa)
                # ax.get_xaxis.set_visible(False)
            elif loc in pb_sources:
                ax = axs[1,0]
                if j == 0:
                    ax.bar(timeline[:],arr_rate[loc],width=width,color=color[iloc],label=f'{loc}',alpha=alfa)
                else:
                    ax.bar(timeline[:],arr_rate[loc],width=width,color=color[iloc],alpha=alfa)
                # ax.get_xaxis.set_visible(False)
            elif loc in nhcs_sources:
                ax = axs[2,0]
                if j == 0:
                    ax.bar(timeline[:],arr_rate[loc],width=width,color=color[iloc],label=f'{loc}',alpha=alfa)
                else:
                    ax.bar(timeline[:],arr_rate[loc],width=width,color=color[iloc],alpha=alfa)
                # ax.get_xaxis.set_visible(True)
            ax.legend(loc='upper left',**lbm)
            
    plt.tight_layout()
    # ax.legend(loc='upper right',bbox_to_anchor=(1.22, 1.5))
    plt.savefig(output_fname+f'Arriving_{simyears[0]}-{simyears[-1]}{suffix}_{normalization}_dt{dt}.png')
    
#%% Age distribution
bins = 729
H_age = {}

if firstcontact:
    Age_arriving = Age_1st
    suffix = '_1st'
else:
    Age_arriving = Age_all
    suffix = '_all'

agelimit = np.timedelta64(730,'D')
agerange = np.arange(0,agelimit,1)
xticks = np.arange(0, agelimit+1, 60)

for yr in simyears:
    yrstr = str(yr)
    H_age_arriving = np.zeros((number_sources,bins))
    xbar = np.empty(bins,dtype='timedelta64[D]')
    for i in range(number_sources):
        index = ~np.isnan(Age_arriving[yrstr][i,:])
        H_age_arriving[i,:],edge = np.histogram(Age_arriving[yrstr][i,:][index]\
                                   .astype('timedelta64[s]').astype('timedelta64[D]').astype(int),\
                                   bins=agerange.astype(int))
        x = ((edge[:-1] + edge[1:]) / 2)
        xbar[:] = x.astype('timedelta64[D]')
    H_age[yrstr] = H_age_arriving
    
width = 1
rows = math.ceil(number_sources/3)
cols = min(3,rows)
rows = max(3,rows)

color = cm.tab20( np.linspace(0,20,len(simyears)).astype(int) )
fig,ax = plt.subplots(rows,cols,figsize=(8*cols,4.5*rows))
for j,yr in enumerate(simyears):
    yrstr = str(yr)
    for i in range(rows*cols):
        if i < number_sources:
            ax[i//cols,i%cols].bar(xbar[:],H_age[yrstr][i,:],width=width,color=color[j],label=f'{yr}-{yr+1}')
            ax[i//cols,i%cols].set_title(sources[i],fontsize=14,fontweight='bold')
            ax[i//cols,i%cols].set_xlim([0,agelimit.astype(int)])
            ax[i//cols,i%cols].tick_params('both',labelsize=12)
            ax[i//cols,i%cols].set_xlabel('Age [d]',fontsize=14)
            ax[i//cols,i%cols].set_ylabel('Number of particles',fontsize=14)
            ax[i//cols,i%cols].legend()
            
            ax[i//cols,i%cols].set_xticks(xticks.astype(int))
            plt.setp(ax[i//cols,i%cols].get_xticklabels(), rotation=45, ha='center')
    
        elif i >= number_sources and j == 0:
            fig.delaxes(ax[i//cols,i%cols])
            
    
# fig.suptitle('The age distribution of particles arriving at Galapagos'+fname+' contact',fontsize=18,fontweight='bold')
plt.tight_layout()
plt.savefig(output_fname+f'Age_{simyears[0]}-{simyears[-1]}'+suffix+'.png')
#%% Time distribution - Leaving
bins = 729
H_leaving = {}

timelimit = np.timedelta64(730,'D')
timeline = np.arange(0,timelimit-1,1)
xticks = np.arange(0, timelimit+1, 60)

for yr in simyears:
    if yr%4!=0 and (yr+1)%4!=0:
        yrstr = str(yr)
        H_time_leaving = np.zeros((number_sources,bins))
        xbar = np.empty(bins,dtype='datetime64[D]')
        start_time = np.datetime64(yrstr+'-01-02')
        end_time = np.datetime64(str(yr+2)+'-01-02')
        timerange = np.arange(start_time,end_time,1)
    elif yr%4==0 or (yr+1)%4==0:
        yrstr = str(yr)
        H_time_leaving = np.zeros((number_sources,bins))
        xbar = np.empty(bins,dtype='datetime64[D]')
        start_time = np.datetime64(yrstr+'-01-02')
        end_time = np.datetime64(str(yr+2)+'-01-01')
        timerange = np.arange(start_time,end_time,1)
    for i in range(number_sources):
        index = ~np.isnan(Time_leaving[yrstr][i,:])
        H_time_leaving[i,:],edge = np.histogram(Time_leaving[yrstr][i,:][index].astype('datetime64[D]').astype(int),bins=timerange.astype(int))
        x = ((edge[:-1] + edge[1:]) / 2)
        xbar[:] = x.astype('datetime64[D]')
    H_leaving[yrstr] = H_time_leaving
    
width = 1
rows = math.ceil(number_sources/3)
cols = min(3,rows)
rows = max(3,rows)

orders = [4,3,2,1]
colors = ['tab:blue','tab:orange','tab:red','tab:green','tab:purple','tab:brown','tab:pink','tab:gray','tab:olive','tab:cyan']
fig,ax = plt.subplots(rows,cols,figsize=(8*cols,4.5*rows))
for j,yr in enumerate(simyears):
    yrstr = str(yr)
    for i in range(rows*cols):
        if i < number_sources:
            ax[i//cols,i%cols].bar(timeline[:],H_leaving[yrstr][i,:],width=width,color=colors[j],label=yrstr,zorder=orders[j])
            ax[i//cols,i%cols].set_title(sources[i],fontsize=14,fontweight='bold')
            ax[i//cols,i%cols].set_xlim([0,timelimit.astype(int)])
            ax[i//cols,i%cols].tick_params('both',labelsize=12)
            ax[i//cols,i%cols].set_xlabel('Time passed [d]',fontsize=14)
            ax[i//cols,i%cols].set_ylabel('Number of particles',fontsize=14)
            ax[i//cols,i%cols].legend()
            
            ax[i//cols,i%cols].set_xticks(xticks.astype(int))
            plt.setp(ax[i//cols,i%cols].get_xticklabels(), rotation=45, ha='center')
        
        elif i >= number_sources and j == 0:
            fig.delaxes(ax[i//cols,i%cols])
    
# fig.suptitle('The leaving time distribution of particles arriving at Galapagos'+fname+' contact',fontsize=18,fontweight='bold')

plt.tight_layout()
plt.savefig(output_fname+f'Leaving_{simyears[0]}-{simyears[-1]}'+suffix+'.pdf')
#%% Age accumulation
dt = np.timedelta64(1,'D')
start_time = np.datetime64('2018-01-02')
end_time = np.datetime64('2019-12-24')
timerange = np.arange(start_time,end_time,dt)
agelimit = np.timedelta64(len(timerange),'D')
agerange = np.arange(0,agelimit,dt)
xticks = np.arange(0, agelimit+1, 60)

bins = len(agerange)-1
H_age = np.zeros((number_sources,bins))
xbar = np.empty(bins,dtype='timedelta64[D]')

for i in range(number_sources):
    index = ~np.isnan(Age_all[i,:])
    H_age[i,:],edge = np.histogram(Age_all[i,:][index]\
                               .astype('timedelta64[s]').astype('timedelta64[D]').astype(int),\
                               bins=agerange.astype(int))
    x = ((edge[:-1] + edge[1:]) / 2)
    xbar[:] = x.astype('timedelta64[D]').astype(int)
    
S_age = H_age.cumsum(axis=1)

width = 1
rows = math.ceil(number_sources/3)
cols = min(3,rows)
rows = max(3,rows)

fig,ax = plt.subplots(rows,cols,figsize=(8*cols,4.5*rows))
for i in range(rows*cols):
    if i < number_sources:
        ax[i//cols,i%cols].plot(xbar[:],S_age[i,:],color='tab:orange')
        ax[i//cols,i%cols].set_title(sources[i],fontsize=14,fontweight='bold')
        ax[i//cols,i%cols].set_xlim([0,agelimit.astype(int)])
        ax[i//cols,i%cols].tick_params('both',labelsize=12)
        ax[i//cols,i%cols].set_xlabel('Age [d]',fontsize=14)
        ax[i//cols,i%cols].set_ylabel('Number of particles',fontsize=14)
        
        ax[i//cols,i%cols].set_xticks(xticks.astype(int))
        plt.setp(ax[i//cols,i%cols].get_xticklabels(), rotation=45, ha='center')
    
    else:
        fig.delaxes(ax[i//cols,i%cols])
        
plt.tight_layout()

plt.savefig(output_fname+'Age_accumulation.pdf')