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
from matplotlib.ticker import FormatStrFormatter
import cmocean.cm as cmo
from matplotlib import cm
from matplotlib.animation import FuncAnimation
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from matplotlib import colors,dates
import seaborn as sns
import matplotlib.gridspec as gridspec
from matplotlib.path import Path
import matplotlib.patches as patches
import pandas as pd
import os
import pickle
import numpy.ma as ma
from scipy.signal import correlate2d,correlate,convolve2d
#%% function to count the parcels
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
def behaviormode(lattrj, lontrj, loctrj, timetrj, region, behavior):
    head = np.array([])
    tail = np.array([])
    # timetrj and lattrj have the same shape
    if region == 'PNJtilt':
        if behavior == 'holdthehead':
            entrance = 5
            exit = 2
            length = 30
            
            # hold the head
            indx0 = np.where(lattrj>entrance )
            time0 = timetrj[indx0] # the time of the start lats that is higher than 7N
            # fix the length
            for t1 in time0:
                t2 = t1 + np.timedelta64(length,'D')
                indx1 = np.where(timetrj==t1 )[0][0]
                indx2 = np.where(timetrj<=t2 )[0][-1]
            # hold the tail
                if indx1 != indx2:
                    if np.nanmin(lattrj[indx1:indx2])<exit:
                        head = indx1
                        tail = indx2
                        break
                    
        elif behavior == 'holdthehead_tilt':
            b0 = -76
            loc1 = 144 # the corresponding b1 is 2N
            length = 30
            
            # hold the head
            indx0 = np.where(lattrj > -lontrj + b0 )
            time0 = timetrj[indx0] # the time of the start lats that is higher than 7N
            # fix the length
            for t1 in time0:
                t2 = t1 + np.timedelta64(length,'D')
                indx1 = np.where(timetrj==t1 )[0][0]
                indx2 = np.where(timetrj<=t2 )[0][-1]
            # hold the tail
                if indx1 != indx2:
                    if np.nanmax(loctrj[indx1:indx2]) > loc1:
                        head = indx1
                        tail = indx2
                        break
                        
        elif behavior == 'holdthetail':
            entrance = 5
            exit = 2
            length = 30
            
            # hold the tail
            indx0 = np.where(lattrj<exit )
            time0 = timetrj[indx0] # the time of the start lats that is higher than 7N
            # fix the length
            for t1 in time0:
                t2 = t1 - np.timedelta64(length,'D')
                indx1 = np.where(timetrj==t1 )[0][0]
                indx2 = np.where(timetrj>=t2 )[0][0]
            # hold the tail
                if indx1 != indx2:
                    if np.nanmax(lattrj[indx2:indx1])>entrance:
                        head = indx2
                        tail = indx1
                        break
        
            
    elif region == 'SEC':
        entrance = -87
        exit = -90
        length = 15
        
        # hold the head
        indx0 = np.where(lontrj>entrance )
        time0 = timetrj[indx0] # the time of the start lons that is easter than 7N
        # fix the length
        for t1 in time0:
            t2 = t1 + np.timedelta64(length,'D')
            indx1 = np.where(timetrj==t1 )[0][0]
            indx2 = np.where(timetrj<=t2 )[0][-1]
        # hold the tail
            if indx1 != indx2:
                if np.nanmin(lontrj[indx1:indx2])<exit:
                    head = indx1
                    tail = indx2
                    break
                
    elif region == 'NECC':
        entrance = -98
        exit = -92
        length = 30
        
        # hold the head
        indx0 = np.where(lontrj<entrance )
        time0 = timetrj[indx0] # the time of the start lons that is easter than 7N
        # fix the length
        for t1 in time0:
            t2 = t1 + np.timedelta64(length,'D')
            indx1 = np.where(timetrj==t1 )[0][0]
            indx2 = np.where(timetrj<=t2 )[0][-1]
        # hold the tail
            if indx1 != indx2:
                if np.nanmax(lontrj[indx1:indx2])>exit:
                    head = indx1
                    tail = indx2
                    break
    
    elif region == 'HC':
        entrance = -82
        exit = -88
        length = 40
        
        # hold the head
        indx0 = np.where(lontrj>entrance )
        time0 = timetrj[indx0] # the time of the start lons that is easter than -81
        # fix the length
        for t1 in time0:
            t2 = t1 + np.timedelta64(length,'D')
            indx1 = np.where(timetrj==t1 )[0][0]
            indx2 = np.where(timetrj<=t2 )[0][-1] # -1 is used to index the last observation
        # hold the tail
            if indx1 != indx2:
                if np.nanmin(lontrj[indx1:indx2])<exit:
                    head = indx1
                    tail = indx2
                    break
    
    return head, tail

def tubeselect(lons, lats, time, region, behavior):
    
    # 1) - create a new tilted dimension 'locs'
    locs = np.zeros_like(lats)
    if region == 'PNJtilt':
        r = 1/12
        b0 = -72 # the top limit ot intercept
        b1 = -86 # the lower limit of intercept
        bn0 = b0
        bn1 = bn0 - r
        n = 1
        # convert into new dimension
        while bn1>=b1:
            mask = (lats > (lons+85)) & (lats < (lons+88)) & (lats > (-lons+bn1)) & (lats < (-lons+bn0))
            indx = np.where(mask)
            locs[indx] = n
            
            bn1 -= r
            bn0 -= r
            n += 1
    
    latstube = {}
    lonstube = {}
    timetube = {}
    # 2) - deal with one thread at a time
    for trj in range(len(lats)):
        
        lattrj = lats[trj,:]
        loctrj = locs[trj,:]
        lontrj = lons[trj,:]
        timetrj = time[trj,:]
        
        # 2.1) - select the region
        if region == 'PB':
            regionsel = True
        elif region == 'PNJ':
            regionsel = (lattrj > lontrj + 85) & (lattrj < lontrj + 88)
        elif region == 'PNJtilt':
            regionsel = (lattrj > lontrj + 85) & (lattrj < lontrj + 88) & (lattrj > (-lontrj+b1)) & (lattrj < (-lontrj+b0))
        elif region == 'SEC':
            regionsel = ( lontrj < -85 ) & ( lontrj > -92 ) & ( lattrj > -2 ) & (lattrj < 2)
        elif region == 'NECC':
            regionsel = ( lontrj < -90 ) & ( lontrj > -100 ) & ( lattrj > 6 ) & (lattrj < 9)
        elif region == 'HC':
            regionsel = ( lontrj < -80 ) & ( lontrj > -92 ) & ( lattrj > -10 ) & (lattrj < 0)
        indx = np.where(regionsel)
        
        latsel = lattrj[indx]
        lonsel = lontrj[indx]
        locsel = loctrj[indx]
        timesel = timetrj[indx]
        '''
        # 2.2) select out the part that actually travel the tube
        head, tail = behaviormode(latsel,lonsel,locsel,timesel,region,behavior)
        
        # 2.3) - put the selected thread into a dict
        if head>0:
            if region == 'PNJ':
                latstube[str(trj)] = latsel[head:tail]
            elif region == 'PNJtilt':
                latstube[str(trj)] = locsel[head:tail]
            elif region in ['SEC','NECC','HC']:
                lonstube[str(trj)] = lonsel[head:tail]
            timetube[str(trj)] = timesel[head:tail]
        '''
        latstube[str(trj)] = locsel
        lonstube[str(trj)] = lonsel
        timetube[str(trj)] = timesel
        
    return latstube, lonstube, timetube
#%% function to calculate different numbers
pd.set_option('display.max_columns', None) # show all the columns of the dataframe

def calculate(region, loadyears):
    priors = pd.read_csv('/Users/renjiongqiu/Documents_local/Thesis/data/priors_river_inputs_25N25S.csv',
                          index_col=0)
    sources = list(priors.index)
    
    beaching = True
    simdays = 729
    releasedays = 365
    dailyrelease = 274
    
    Lats = {}
    Lons = {}
    Time = {}
    LatsG = {}
    LonsG = {}
    TimeG = {}
    
    for year in loadyears:
        # yrstr = str(year)
        secondyear = year + 1
        fname = f'_{year}-{secondyear}_run{simdays}d_release{releasedays}d_{dailyrelease}par'
        Ginput_fname = f'/Users/renjiongqiu/Documents_local/Thesis/visualization/TrajAnalysis/trajectories/arriving_gala/'
        
        if beaching:
            fname = fname+'_beaching'
        else:
            fname = fname
        
        load_fname = f'/Users/renjiongqiu/Documents_local/Thesis/visualization/quantification/{region}/{year}/'
        
        if region in ['PNJ','PNJtilt','PNJtilt_N5N2']:
            with open(load_fname+'Lats'+fname+'.pickle', 'rb') as handle:
                Lats[year] = pickle.load(handle)
            with open(load_fname+'Time'+fname+'.pickle', 'rb') as handle:
                Time[year] = pickle.load(handle)
            
        elif region in ['SEC','NECC','HC']:
            with open(load_fname+'Lons'+fname+'.pickle', 'rb') as handle:
                Lons[year] = pickle.load(handle)
            with open(load_fname+'Time'+fname+'.pickle', 'rb') as handle:
                Time[year] = pickle.load(handle)
        
        LatsG[year] = np.load(Ginput_fname+'Lat'+fname+'.npy',allow_pickle=True)
        LonsG[year] = np.load(Ginput_fname+'Lon'+fname+'.npy',allow_pickle=True)
        TimeG[year] = np.load(Ginput_fname+'Time'+fname+'.npy',allow_pickle=True)
    # Calculate the percentage of the PNJ-tube parcels arriving in Gala
    percentT = {}
    percentG = {}
    whole = {}
    whole_norm = {}
    gala = {}
    gala_whole = {}
    
    for year in loadyears:
        # yrstr = str(year)
        percentT[year] = []
        percentG[year] = []
        whole[year] = []
        whole_norm[year] = []
        gala[year] = []
        gala_whole[year] = []
        for iloc,loc in enumerate(Time[year]['whole'].keys()):
            whole[year].append( len(Time[year]['whole'][loc]) )
            whole_norm[year].append( len(Time[year]['whole'][loc]) / (releasedays*dailyrelease)*100 )  # normalize the data
            gala[year].append( len(Time[year]['gala'][loc]) )
            gala_whole[year].append( len(TimeG[year][iloc][0]) )
            if len(Time[year]['whole'][loc]) > 0 and len(Time[year]['gala'][loc]) > 0:
                percentT[year].append( len(Time[year]['gala'][loc]) / len(Time[year]['whole'][loc])*100 )
                percentG[year].append( len(Time[year]['gala'][loc]) / len(TimeG[year][iloc][0])*100 )
            elif len(Time[year]['whole'][loc]) == 0 or len(Time[year]['gala'][loc]) == 0:
                percentT[year].append(0)
                percentG[year].append(0)
    
    # put the result into a data frame
    percentT_df = pd.DataFrame.from_dict(percentT, orient='index', columns=Time[2019]['whole'].keys())
    percentG_df = pd.DataFrame.from_dict(percentG, orient='index', columns=Time[2019]['whole'].keys())
    whole_df = pd.DataFrame.from_dict(whole, orient='index', columns=Time[2019]['whole'].keys())
    whole_norm_df = pd.DataFrame.from_dict(whole_norm, orient='index', columns=Time[2019]['whole'].keys())
    gala_tube_df = pd.DataFrame.from_dict(gala, orient='index', columns=Time[2019]['gala'].keys())
    gala_whole_df = pd.DataFrame.from_dict(gala_whole, orient='index', columns=Time[2019]['gala'].keys())
    gala_whole_norm_df = gala_whole_df / (releasedays*dailyrelease)*100
    
    return percentT_df, percentG_df, whole_df, whole_norm_df, gala_tube_df, gala_whole_df, gala_whole_norm_df,Lats,Lons,Time

#%% calculate all
region = 'PNJtilt_N5N2'
loadyears = np.arange(2007,2021)

percentT_df, percentG_df, whole_df, whole_norm_df, gala_tube_df, gala_whole_df, \
    gala_whole_norm_df,Lats,Lons,Time = calculate(region, loadyears)


#%% generate particle behavior analysis data
# Selection --------------------------------------------------------------------------------------------------------
priors = pd.read_csv('/Users/renjiongqiu/Documents_local/Thesis/data/priors_river_inputs_25N25S.csv',
                      index_col=0)
sources = list(priors.index)

region = 'PNJtilt'
behavior = 'holdthehead'

Lats = {}
Lons = {}
Locs = {}
Time = {}
Lats['whole'] = {}
Lats['gala'] = {}
Lons['whole'] = {}
Lons['gala'] = {}
Locs['whole'] = {}
Locs['gala'] = {}
Time['whole'] = {}
Time['gala'] = {}

# year = 2017
yearlist = np.arange(2007,2021)
for year in yearlist:
    yrstr = str(year)
    secondyear = year + 1
    
    beaching = True
    simdays = 729
    releasedays = 365
    dailyrelease = 274
    
    fname = f'_{year}-{secondyear}_run{simdays}d_release{releasedays}d_{dailyrelease}par'
    Ginput_fname = '/Users/renjiongqiu/Documents_local/Thesis/visualization/TrajAnalysis/trajectories/0-999/'
    
    if beaching:
        fname = fname+'_beaching'
        Winput_fname = f'/Volumes/John_HardDrive/Thesis/sim_results/{year}/with_beaching/'
    else:
        fname = fname
        Winput_fname = f'/Volumes/John_HardDrive/Thesis/sim_results/{year}/no_beaching/'
    
    for loc in sources:
        iloc = np.where(np.array(sources)==loc)[0][0]
        # Load the Gala trajectories
        lonsG = np.load(f'{Ginput_fname}Lon{fname}_0-999d_before.npy',allow_pickle=True)[iloc][0]
        latsG = np.load(f'{Ginput_fname}Lat{fname}_0-999d_before.npy',allow_pickle=True)[iloc][0]
        timeG = np.load(f'{Ginput_fname}Time{fname}_0-999d_before.npy',allow_pickle=True)[iloc][0].astype('datetime64[ns]')
        
        # Load the Whole trajectories
        ds = xr.open_dataset(Winput_fname+f'{loc}'+fname+'.nc')
        lonsW = np.array(ds.lon)
        latsW = np.array(ds.lat)
        timeW = np.array(ds.time)
        
        # Select the trajs
        resultsW = tubeselect(lonsW, latsW, timeW, region, behavior)
        resultsG = tubeselect(lonsG, latsG, timeG, region, behavior)
        if region in ['PNJtilt','PNJ']:
            Lats['whole'][loc], Time['whole'][loc] = resultsW[0], resultsW[2]
            Lats['gala'][loc], Time['gala'][loc] = resultsG[0], resultsG[2]
        elif region in ['SEC','NECC']:
            Lons['whole'][loc], Time['whole'][loc] = resultsW[1], resultsW[2]
            Lons['gala'][loc], Time['gala'][loc] = resultsG[1], resultsG[2]
    
    # save the ds
    out_fname = f'/Users/renjiongqiu/Documents_local/Thesis/visualization/quantification/{region}/{year}/'
    os.makedirs(os.path.dirname(out_fname),exist_ok=True)
    
    if region in ['PNJ','PNJtilt']:
        with open(out_fname+'Lats'+fname+'.pickle', 'wb') as handle:
            pickle.dump(Lats, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open(out_fname+'Time'+fname+'.pickle', 'wb') as handle:
            pickle.dump(Time, handle, protocol=pickle.HIGHEST_PROTOCOL)
    elif region in ['SEC','NECC']:
        with open(out_fname+'Lons'+fname+'.pickle', 'wb') as handle:
            pickle.dump(Lons, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open(out_fname+'Time'+fname+'.pickle', 'wb') as handle:
            pickle.dump(Time, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
#%% generate the data (simple version)
priors = pd.read_csv('/Users/renjiongqiu/Documents_local/Thesis/data/priors_river_inputs_25N25S.csv',
                      index_col=0)
sources = list(priors.index)

region = 'HC'
behavior = 'nobehavior'

Lats = {}
Lons = {}
Locs = {}
Time = {}

# year = 2017
yearlist = np.arange(2007,2021)
for year in yearlist:
    yrstr = str(year)
    secondyear = year + 1
    
    beaching = True
    simdays = 729
    releasedays = 365
    dailyrelease = 274
    
    fname = f'_{year}-{secondyear}_run{simdays}d_release{releasedays}d_{dailyrelease}par_beaching'
    Ginput_fname = '/Users/renjiongqiu/Documents_local/Thesis/visualization/TrajAnalysis/trajectories/0-999/'
    
    suffix = '_0-999d_before' # '_0-999d_before'
    
    for loc in sources:
        iloc = np.where(np.array(sources)==loc)[0][0]
        # Load the Gala trajectories
        lonsG = np.load(f'{Ginput_fname}Lon{fname}{suffix}.npy',allow_pickle=True)[iloc][0]
        latsG = np.load(f'{Ginput_fname}Lat{fname}{suffix}.npy',allow_pickle=True)[iloc][0]
        timeG = np.load(f'{Ginput_fname}Time{fname}{suffix}.npy',allow_pickle=True)[iloc][0].astype('datetime64[ns]')
        
        # Select the trajs
        resultsG = tubeselect(lonsG, latsG, timeG, region, behavior)
        if region in ['PNJtilt','PNJ']:
            Lats[loc], Time[loc] = resultsG[0], resultsG[2]
        elif region in ['SEC','NECC','HC']:
            Lons[loc], Time[loc] = resultsG[1], resultsG[2]
    
    # save the ds
    out_fname = f'/Users/renjiongqiu/Documents_local/Thesis/visualization/quantification/{behavior}/{region}_simple_0-999/{year}/'
    os.makedirs(os.path.dirname(out_fname),exist_ok=True)
    
    if region in ['PNJ','PNJtilt']:
        with open(out_fname+'Lats'+fname+'.pickle', 'wb') as handle:
            pickle.dump(Lats, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open(out_fname+'Time'+fname+'.pickle', 'wb') as handle:
            pickle.dump(Time, handle, protocol=pickle.HIGHEST_PROTOCOL)
    elif region in ['SEC','NECC','HC']:
        with open(out_fname+'Lons'+fname+'.pickle', 'wb') as handle:
            pickle.dump(Lons, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open(out_fname+'Time'+fname+'.pickle', 'wb') as handle:
            pickle.dump(Time, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    with open(f'{out_fname}readme.txt', 'w') as f:
        f.write(f' region = {region}{suffix}\n behavior = {behavior}')
#%% quantification integrated through all years and separate years
priors = pd.read_csv('/Users/renjiongqiu/Documents_local/Thesis/data/priors_river_inputs_25N25S.csv',
                      index_col=0)
sources = list(priors.index)
northern_sources = ['Acapulco', 'SalinaCruz', 'Tecojate', 'LaColorada', 'Fonseca', 'Chira']
pb_sources = ['Panama', 'Cacique','Esmeraldas']
nhcs_sources = ['Guayaquil','Parachique','Lima']
loadyears = np.arange(2007,2021)

beaching = True
simdays = 729
releasedays = 365
dailyrelease = 274
region_pnj = 'PNJtilt_simple_0-999'
region_hc = 'HC_simple_arriving_gala'

gala = {}
gala_year = {}
galapnj = {}
percent_galapnj = {}
galahc = {}
percent_galahc = {}
percent_galapnj_year = {}
percent_galahc_year = {}

for loc in sources:
    gala[loc] = 0
    galapnj[loc] = 0
    galahc[loc] = 0

for year in loadyears:
    percent_galapnj_year[year] = {}
    percent_galahc_year[year] = {}
    gala_year[year] = {}
    
    # yrstr = str(year)
    secondyear = year + 1
    fname = f'_{year}-{secondyear}_run{simdays}d_release{releasedays}d_{dailyrelease}par_beaching'
    Ginput_fname = f'/Users/renjiongqiu/Documents_local/Thesis/visualization/TrajAnalysis/trajectories/arriving_gala/'
    
    load_fname_pnj = f'/Users/renjiongqiu/Documents_local/Thesis/visualization/quantification/{region_pnj}/{year}/'
    # load_fname_pnj = f'/Users/renjiongqiu/Documents_local/Thesis/visualization/quantification/holdthehead/PNJtilt_simple_0-999_N5N2/{year}/'
    # load_fname_pnj = f'/Users/renjiongqiu/Documents_local/Thesis/visualization/quantification/PNJtilt_simple_arriving_gala/{year}/'
    load_fname_hc = f'/Users/renjiongqiu/Documents_local/Thesis/visualization/quantification/{region_hc}/{year}/'
    
    if region_pnj == 'PNJtilt_N7N2' or region_pnj == 'PNJtilt_N5N2':
        with open(f'{load_fname_pnj}Time{fname}.pickle', 'rb') as handle:
                Time_galapnj = pickle.load(handle)['gala']
    else:
        with open(f'{load_fname_pnj}Time{fname}.pickle', 'rb') as handle:
                Time_galapnj = pickle.load(handle)
    
    with open(f'{load_fname_hc}Time{fname}.pickle', 'rb') as handle:
            Time_galahc = pickle.load(handle)
    
    Time_gala = np.load(f'{Ginput_fname}Time{fname}.npy',allow_pickle=True)

    for iloc,loc in enumerate(sources):
        
        gala_year[year][loc] = len(Time_gala[iloc][0]) / 100010 * 100
        
        # if len(Time_galapnj['gala'][loc]) > 0:
        #     galapnj[loc] += len(Time_galapnj['gala'][loc])
            
        if len(Time_galapnj[loc]) > 0:
            galapnj[loc] += len(Time_galapnj[loc])
        
        if len(Time_galahc[loc]) > 0:
            galahc[loc] += len(Time_galahc[loc])
        
        if len(Time_gala[iloc][0]) > 0:
            gala[loc] += len(Time_gala[iloc][0])
        
        if len(Time_galapnj[loc]) > 0 and len(Time_gala[iloc][0]) > 0:
            percent_galapnj_year[year][loc] = len(Time_galapnj[loc]) / len(Time_gala[iloc][0]) * 100
        else:
            percent_galapnj_year[year][loc] = 0
        
        # if len(Time_galapnj[loc]) > 0 and len(Time_gala[iloc][0]) > 0:
        #     percent_galapnj_year[year][loc] = len(Time_galapnj[loc]) / len(Time_gala[iloc][0]) * 100
        # else:
        #     percent_galapnj_year[year][loc] = 0
            
        if len(Time_galahc[loc]) > 0 and len(Time_gala[iloc][0]) > 0:
            percent_galahc_year[year][loc] = len(Time_galahc[loc]) / len(Time_gala[iloc][0]) * 100
        else:
            percent_galahc_year[year][loc] = 0
            
for loc in sources:
    percent_galapnj[loc] = galapnj[loc] / gala[loc] * 100
    percent_galahc[loc] = galahc[loc] / gala[loc] * 100

# put the result into a data frame
percent_galapnj_df = pd.DataFrame.from_dict(percent_galapnj, orient='index', columns=['PNJ'])
percent_galahc_df = pd.DataFrame.from_dict(percent_galahc, orient='index', columns=['HC'])
percent_total_df = pd.concat([percent_galahc_df,percent_galapnj_df],axis=1)
percent_galapnj_year_df = pd.DataFrame.from_dict(percent_galapnj_year, orient='index')
percent_galahc_year_df = pd.DataFrame.from_dict(percent_galahc_year, orient='index')
percent_gala_year_df = pd.DataFrame.from_dict(gala_year, orient='index')


#%% Calculate the dataframe
subplots = False

region = 'PNJtilt_N5N2'
loadyears = np.arange(2007,2021)

whole_norm_df = calculate(region, loadyears)[3]
whole_df = calculate(region, loadyears)[2]
gala_tube_df = calculate(region, loadyears)[4]
gala_whole_df = calculate(region, loadyears)[5]
gala_whole_norm_df = calculate(region, loadyears)[6]

# Get the likelihood
likelihood = gala_whole_norm_df.mean()

# Get the priors
priors = pd.read_csv('/Users/renjiongqiu/Documents_local/Thesis/data/priors_river_inputs_25N25S.csv',
                      index_col=0)
sources = list(priors.index)
priors['priors'] = priors['priors']*100 # convert the values to %
priors.drop('merged_rivers',axis=1,inplace=True) # drop the 'merged_rivers' column

# Get the posterior
posterior = priors['priors'] * likelihood / 100
posterior = posterior / posterior.sum() * 100

# Make the Bayesian framework
bayesian = pd.concat([priors,likelihood,posterior],axis=1)
bayesian.rename({0:'likelihood',1:'posterior'},axis='columns',inplace=True)

# Calculate the others
others = pd.DataFrame(data=[[100-bayesian['priors'].sum(),0,0]], columns=['priors','likelihood','posterior'], index=['Others'])
bayesian = pd.concat([bayesian,others],axis=0)
bayesian.drop(index='Others',inplace=True)
#%%
plot_type = 'priors_horiz' # typelist = ['arrivingGala','arrivingGala_mean','bayesian','bay_horiz']

pltdir = '/Users/renjiongqiu/Documents_local/Thesis/visualization/quantification/'
# Create the color map
color = cm.tab20( np.linspace(0,20,len(loadyears)).astype(int) )
# just plot
# make labels
mylabels = np.empty(len(loadyears),dtype=object)
for i,yr in enumerate(loadyears):
    mylabels[i] = f'{yr}-{yr+1}'
    
if subplots:
    n = 5
    fig = plt.figure(figsize=(18,n*5), constrained_layout=True)
    gs = gridspec.GridSpec(n,1,hspace=0.5)
    
    for i in range(n+1):
        ax.legend(loc='upper right', bbox_to_anchor=(1.1,1.))
        ax.grid()
        if i == 0:
            # all pass the tube - normed
            ax = fig.add_subplot(gs[i])
            whole_norm_df.transpose().plot.bar(width=0.7,ax=ax,rot=45,color=color,legend=False)
            ax.set_title(f'All the tube-transported parcels released from the sources - normed - {region}',fontsize=16)
            ax.set_ylim([0,50])
            ax.set_ylabel('fraction [%]',fontsize=12)
        elif i == 1:
            # all pass the tube - in number
            ax = fig.add_subplot(gs[i])
            whole_df.transpose().plot.bar(width=0.7,ax=ax,rot=45,color=color,legend=False)
            ax.set_title(f'All the tube-transported parcels released from the sources - in number - {region}',fontsize=16)
            ax.set_ylim([0,50000])
            ax.set_ylabel('number',fontsize=12)
        elif i == 2:
            # gala + tube
            ax = fig.add_subplot(gs[i])
            gala_tube_df.transpose().plot.bar(width=0.7,ax=ax,rot=45,color=color,legend=False)
            ax.set_title(f'The parcels that both travel through tube and end up in Galapagos - {region}',fontsize=16)
            ax.set_ylim([0,20000])
            ax.set_ylabel('number',fontsize=12)
        elif i == 3:
            # all end up in gala
            ax = fig.add_subplot(gs[i])
            gala_whole_df.transpose().plot.bar(width=0.7,ax=ax,rot=45,color=color,legend=False)
            ax.set_title(f'All the parcels that end up in Galapagos - in number - {region}',fontsize=16)
            ax.set_ylim([0,40000])
            ax.set_ylabel('number',fontsize=12)
        elif i == 4:
            # all end up in gala
            ax = fig.add_subplot(gs[i])
            gala_whole_norm_df.transpose().plot.bar(width=0.7,ax=ax,rot=45,color=color,legend=False)
            ax.set_title(f'All the parcels that end up in Galapagos - normed - {region}',fontsize=16)
            ax.set_ylim([0,50])
            ax.set_ylabel('fraction [%]',fontsize=12)
    
    # fig.suptitle(f'{region}',fontsize=18,fontweight='bold')
    # fig.set_tight_layout(tight={'rect':(0,0,1,0.2)})
    # plt.tight_layout(rect=(1,1,1,1))
    
    plt.savefig(pltdir+f'quantification2_{region}.png')
    
else:
    if plot_type == 'arrivingGala':
        # all end up in gala
        fig,ax = plt.subplots(figsize=(18,6))
        gala_whole_norm_df.transpose().plot.bar(width=0.7,ax=ax,rot=45,color=color,legend=False)
        ax.set_title(f'All the parcels that arrive in Galapagos',fontsize=16)
        # ax.set_ylim([0,50])
        ax.set_ylabel('fraction [%]',fontsize=12)
        ax.grid()
        ax.legend(labels=mylabels,loc='upper right', bbox_to_anchor=(1.1,0.85))
        
        # make a second spines (yaxis)
        ax2 = ax.twinx()
        ax2.spines['left'].set_position(('axes',-0.05))
        
        # ax2.set_ylim([0,50005])
        ax2.set_ylabel('number of parcels',fontsize=12)
        ax2.yaxis.set_label_position('left')
        ax2.yaxis.set_ticks_position('left')
        
        # plt.savefig(pltdir+f'quantification2_{plot_type}.png')
    elif plot_type == 'arrivingGala_mean':
        mean = gala_whole_norm_df.mean()
        std = gala_whole_norm_df.std()
        
        lbm = dict(fontsize=16)
        tkm = dict(labelsize=12)
        # all end up in gala
        fig,ax = plt.subplots(figsize=(16,12))
        # mean.plot.bar(width=0.7,ax=ax,yerr=std,rot=45,color='tab:blue',legend=False)
        bayesian['likelihood'].plot.bar(width=0.7,ax=ax,rot=45,color='tab:blue',legend=False)
        
        # ax.errorbar(std.index, mean, yerr=std)
        ax.set_title('Arriving rate for different sourced particles (averaged from all the simulations)',fontsize=20)
        ax.set_ylim([0,30])
        ax.set_ylabel('fraction [%]',**lbm)
        ax.tick_params(axis='both',**tkm)
        ax.grid()
        
        # make a second spines (yaxis)
        ax2 = ax.twinx()
        ax2.spines['left'].set_position(('axes',-0.07))
        
        ax2.set_ylim([0,30005])
        ax2.set_ylabel('number of parcels',**lbm)
        ax2.yaxis.set_label_position('left')
        ax2.yaxis.set_ticks_position('left')
        ax2.tick_params(axis='y',**tkm)
        ax2.ticklabel_format(axis='y',style='sci',scilimits=[0,5],useMathText=True)
        
        # add label for each bar
        for container in ax.containers:
            ax.bar_label(container, fmt='%.2f', padding=0.1, fontsize=12)
        
        plt.savefig(pltdir+f'quantification2_{plot_type}.png')
    
    elif plot_type == 'bayesian':
        showing = 'likelihood' # ['priors','posterior','posterior_zoomin','likelihood']
        N = 100010
        
        lbm = dict(fontsize=16)
        tkm = dict(labelsize=12)
        # all end up in gala
        fig,ax = plt.subplots(figsize=(16,12))
        
        if showing == 'priors':
            bayesian['priors'].plot.bar(width=0.7,ax=ax,rot=45,color='silver',legend=False)
            ylim = 60
        elif showing == 'posterior' or showing == 'posterior_zoomin':
            bayesian['posterior'].plot.bar(width=0.7,ax=ax,rot=45,color='black',legend=False,zorder=3)
            bayesian['priors'].plot.bar(width=0.7,ax=ax,rot=45,color='silver',legend=False)
            if showing == 'posterior':
                ylim = 60
            elif showing == 'posterior_zoomin':
                ylim = 7
        elif showing == 'likelihood':
            bayesian['likelihood'].plot.bar(width=0.7,ax=ax,rot=45,color='dimgray',legend=False,zorder=3)
            ylim = 30
            
            # make a second spines (yaxis) only applied for 'likelihood'
            ax2 = ax.twinx()
            ax2.spines['left'].set_position(('axes',-0.07))
            
            ax2.set_ylim([0,ylim/100*N])
            ax2.set_ylabel('number of parcels',**lbm)
            ax2.yaxis.set_label_position('left')
            ax2.yaxis.set_ticks_position('left')
            ax2.tick_params(axis='y',**tkm)
            ax2.ticklabel_format(axis='y',style='sci',scilimits=[0,6],useMathText=True)
            
        ax.set_title(f'{showing}',fontsize=20)
        ax.set_ylim([0,ylim])
        ax.set_ylabel('fraction [%]',**lbm)
        ax.tick_params(axis='both',**tkm)
        ax.grid()
        
        # add label for each bar
        ax.bar_label(ax.containers[0], fmt='%.2f', padding=0.1, fontsize=12)
        
        plt.savefig(pltdir+f'quantification2_{plot_type}-{showing}.png')
    
    elif plot_type == 'bay_horiz':
        lbm = dict(fontsize=18)
        tkm = dict(labelsize=14)
        grdl = dict(linestyle='--',alpha=0.5,zorder=-1)
        
        fig,ax = plt.subplots(figsize=(16,8))
        ax2 = ax.twiny()
        ax.set_zorder(ax2.get_zorder()+1)
        ax.patch.set_visible(False)
        
        # plotting
        bayesian['likelihood'].plot.barh(width=0.7,ax=ax2,color='tab:blue',legend=False)
        bayesian['posterior'].plot.barh(width=0.7,ax=ax,color='tab:orange',legend=False)
        
        # ax for posterior
        ax.invert_yaxis()
        ax.set_title('Attribution (Posterior)',fontsize=20,x=0.75)
        ax.set_xlim([-105,100])
        ax.set_xticks(np.arange(0,101,10))
        ax.axvline(x=0.,c='k',linewidth=2,zorder=9)
        ax.bar_label(ax.containers[0], fmt='%.2f%%', padding=5, fontsize=14)
        ax.set_xlabel('Probability [%]',**lbm)
        ax.set_ylabel('Sources (from north to south)',labelpad=12,**lbm)
        ax.tick_params(axis='both',**tkm)
        ax.grid(**grdl)
        ax.arrow(-127,1,0,10,clip_on=False,width=0.1,head_length=0.8,head_width=1.5,length_includes_head=True,color='k')
        # ax2 for likelihood
        ax2.invert_xaxis()
        ax2.set_title('Arriving rate (Likelihood)',fontsize=20,x=0.25)
        ax2.set_xlim([30,-31.5])
        ax2.set_xticks(np.arange(0,30.1,5))
        ax2.axvline(x=0.,c='k',linewidth=2,zorder=9)
        ax2.bar_label(ax2.containers[0], fmt='%.2f%%', padding=-55, fontsize=14)
        ax2.xaxis.set_ticks_position('bottom')
        ax2.tick_params(axis='both',**tkm)
        ax2.grid(**grdl)
        
        plt.tight_layout()
        plt.savefig(pltdir+f'likelihood-posterior.png')
    
    elif plot_type == 'priors_horiz':
        lbm = dict(fontsize=18)
        tkm = dict(labelsize=14)
        grdl = dict(linestyle='--',alpha=0.5,zorder=-1)
        
        fig,ax = plt.subplots(figsize=(12,9))
        
        # plotting
        bayesian['priors'].plot.barh(width=0.7,ax=ax,color='silver',legend=False)
        
        # ax for posterior
        ax.invert_yaxis()
        ax.set_title('Riverine plastic inputs (Priors)',fontsize=20)
        ax.set_xlim([0,70])
        ax.set_xticks(np.arange(0,71,10))
        # ax.axvline(x=0.,c='k',linewidth=2,zorder=9)
        ax.bar_label(ax.containers[0], fmt='%.2f%%', padding=5, fontsize=14)
        ax.set_xlabel('Probability [%]',**lbm)
        ax.set_ylabel('Sources (from north to south)',labelpad=12,**lbm)
        ax.tick_params(axis='both',**tkm)
        ax.grid(**grdl)
        ax.arrow(-9.5,1,0,10,clip_on=False,width=0.1,head_length=0.8,head_width=1.5,length_includes_head=True,color='k')
        
        plt.tight_layout()
        plt.savefig(pltdir+f'priors.png')


#%% load simple data
simdays = 729
releasedays = 365
dailyrelease = 274

Lats = {}
Lons = {}
Time = {}
region = 'PNJtilt_simple'
loadyears = np.arange(2007,2021)
for year in loadyears:
    # yrstr = str(year)
    secondyear = year + 1
    fname = f'_{year}-{secondyear}_run{simdays}d_release{releasedays}d_{dailyrelease}par_beaching'
    
    load_fname = f'/Users/renjiongqiu/Documents_local/Thesis/visualization/quantification/{region}/{year}/'
    
    if region in ['PNJ','PNJtilt','PNJtilt_N7N2','PNJtilt_simple']:
        with open(load_fname+'Lats'+fname+'.pickle', 'rb') as handle:
            Lats[year] = pickle.load(handle)
        with open(load_fname+'Time'+fname+'.pickle', 'rb') as handle:
            Time[year] = pickle.load(handle)
        
    elif region in ['SEC','NECC','HC']:
        with open(load_fname+'Lons'+fname+'.pickle', 'rb') as handle:
            Lons[year] = pickle.load(handle)
        with open(load_fname+'Time'+fname+'.pickle', 'rb') as handle:
            Time[year] = pickle.load(handle)

#%% Combine arriving time
simdays = 729
releasedays = 365
dailyrelease = 274
Time_arriving = {}
Harriving = {}
Narriving = {}
Arriving_Hplt = {}
Arriving_df = {}
Arriving_Hplt['NS'] = np.array([])
Arriving_Hplt['PB'] = np.array([])
Arriving_Hplt['HC'] = np.array([])
loadyears = np.arange(2007,2021)
for iyr,yr in enumerate(loadyears):
    
    secondyear = yr + 1
    fname = f'_{yr}-{secondyear}_run{simdays}d_release{releasedays}d_{dailyrelease}par_beaching'
    Time_1st = np.load(f'/Users/renjiongqiu/Documents_local/Thesis/visualization/TimeAnalysis/ArrivingTime_1st{fname}.npy',
                             allow_pickle=True)
    
    load_fname = f'/Users/renjiongqiu/Documents_local/Thesis/visualization/quantification/holdthehead/PNJtilt_simple_0-999_N5N2/{yr}/'
    # load_fname = f'/Users/renjiongqiu/Documents_local/Thesis/visualization/quantification/PNJtilt_nobehavior/{yr}/'
    
    with open(load_fname+'Time'+fname+'.pickle', 'rb') as handle:
        Time_pnj = pickle.load(handle)
    
    load_fname = f'/Users/renjiongqiu/Documents_local/Thesis/visualization/quantification/HC_simple_arriving_gala/{yr}/'
    # load_fname = f'/Users/renjiongqiu/Documents_local/Thesis/visualization/quantification/HC_nobehavior/{yr}/'
    
    with open(load_fname+'Time'+fname+'.pickle', 'rb') as handle:
        Time_hc = pickle.load(handle)
    
    dt = 7
    st = np.datetime64(f'{yr}-01-01','ns')
    ed = np.datetime64(f'{yr+2}-01-01','ns')
    timeline = np.arange(st,ed,np.timedelta64(dt,'D'))
    time_bin = np.append(st-np.timedelta64(dt,'D'),timeline)
    # make the histogram
    Harriving['NS'] = 0
    Harriving['PB'] = 0
    Harriving['HC'] = 0
    
    Narriving['NS'] = 0
    Narriving['PB'] = 0
    Narriving['HC'] = 0
    
    Hpnj_NS = 0
    Hpnj_PB = 0
    Hhc_HC = 0
    
    Npnj_NS = 0
    Npnj_PB = 0
    Nhc_HC = 0
    
    Time_pnj_arr = np.array([],dtype='datetime64[ns]')
    Time_hc_arr = np.array([],dtype='datetime64[ns]')
    
    # dict -> array
    for iloc,loc in enumerate(sources):
        # arriving date
        Time_1st_loc = Time_1st[iloc,:]
        idxnan = np.isnan(Time_1st_loc)
        Time_arriving = Time_1st_loc[~idxnan]
        H, _ = np.histogram(Time_arriving, bins=time_bin)
        N = H.sum()
        # pb&ns
        if loc in northern_sources:
            Harriving['NS'] += H
            Narriving['NS'] += N
        elif loc in pb_sources:
            Harriving['PB'] += H
            Narriving['PB'] += N
        # hc
        elif loc in nhcs_sources:
            Harriving['HC'] += H
            Narriving['HC'] += N
            
        '''
        where I stop: should I use histogram or histogram2d?
        - Use histogram
        '''
        if loc in northern_sources+pb_sources:
            # pnj
            Time_dict_pnj = Time_pnj[loc].values()
            length = np.shape([*Time_dict_pnj])[0]
            for l in range(length):
                Time_pnj_arr = np.append(Time_pnj_arr,[*Time_dict_pnj][l])
            H, _ = np.histogram(Time_pnj_arr, bins=time_bin)
            N = H.sum()
            
            if loc in northern_sources:
                Hpnj_NS += H
                Npnj_NS += N
            elif loc in pb_sources:
                Hpnj_PB += H
                Npnj_PB += N
        
        elif loc in nhcs_sources:
            # hc
            Time_dict_hc = Time_hc[loc].values()
            length = np.shape([*Time_dict_hc])[0]
            for l in range(length):
                Time_hc_arr = np.append(Time_hc_arr,[*Time_dict_hc][l])
            H, _ = np.histogram(Time_hc_arr, bins=time_bin)
            N = H.sum()
            Hhc_HC += H
            Nhc_HC += N
        
    Arriving_Hplt['NS'] = Harriving['NS']/Narriving['NS']
    Arriving_Hplt['PB'] = Harriving['PB']/Narriving['PB']
    Arriving_Hplt['HC'] = Harriving['HC']/Narriving['HC']
    PNJ_NS = Hpnj_NS/Npnj_NS
    PNJ_PB = Hpnj_PB/Npnj_PB
    HC_HC = Hhc_HC/Nhc_HC
    
    Arrival_NS_da = xr.DataArray(
        data = Arriving_Hplt['NS'],
        dims = ['time_counter'],
        coords = [timeline]
        )
    Arrival_PB_da = xr.DataArray(
        data = Arriving_Hplt['PB'],
        dims = ['time_counter'],
        coords = [timeline]
        )
    Arrival_HC_da = xr.DataArray(
        data = Arriving_Hplt['HC'],
        dims = ['time_counter'],
        coords = [timeline]
        )
    
    PNJ_NS_da = xr.DataArray(
        data = PNJ_NS,
        dims = 'time_counter',
        coords = [timeline])
    PNJ_PB_da = xr.DataArray(
        data = PNJ_PB,
        dims = 'time_counter',
        coords = [timeline])
    HC_HC_da = xr.DataArray(
        data = HC_HC,
        dims = 'time_counter',
        coords = [timeline])
    
    Arriving_ds[yr] = xr.Dataset(
        data_vars = {'Arrival_NS':Arrival_NS_da,
                     'Arrival_PB':Arrival_PB_da,
                     'Arrival_HC':Arrival_HC_da,
                     'PNJ_NS':PNJ_NS_da,
                     'PNJ_PB':PNJ_PB_da,
                     'HC_HC':HC_HC_da}
        )
    print(f'{yr}')
    
#%% save arriving dataset
save = False
if save:
    simyears = np.arange(2007,2021)
    outdir = f'/Users/renjiongqiu/Documents_local/Thesis/visualization/quantification/hov/when/'
    os.makedirs(os.path.dirname(outdir),exist_ok=True)
    for yr in simyears:
        Arriving_ds[yr].to_netcdf(f'{outdir}Arriving_ds_{yr}-dt{dt}.nc')
#%% load arriving dataset
dt = 7
Arriving_ds = {}
outdir = f'/Users/renjiongqiu/Documents_local/Thesis/visualization/quantification/hov/when/'
simyears = np.arange(2007,2021)
for yr in simyears:
    Arriving_ds[yr] = xr.load_dataset(f'{outdir}Arriving_ds_{yr}-dt{dt}.nc')
#%% load fieldhov data
load = True
if load:
    outdir = '/Users/renjiongqiu/Documents_local/Thesis/visualization/quantification/hov/'
    spd_pnj = xr.load_dataset(f'{outdir}PNJhovdata.nc').spd
    theta_pnj = xr.load_dataset(f'{outdir}PNJhovdata.nc').theta
    spd_hc = xr.load_dataset(f'{outdir}HChovdata.nc').spd
    theta_hc = xr.load_dataset(f'{outdir}HChovdata.nc').theta

# mask the field data
mask_pnj = (theta_pnj>160) & (theta_pnj<250) & (spd_pnj>0.3) # PNJ mask
mask_hc = (theta_hc>90) & (theta_hc<270) & (spd_hc>0.) # PNJ mask
# mask_pnj = True
# mask_hc = True
spd_pnj_mask = spd_pnj.where(mask_pnj,0)
spd_hc_mask = spd_hc.where(~spd_hc.isnull(),0).where(mask_hc,0)

# calculate the mean and bin the data
spd_pnjplt = {}
spd_hcplt = {}
spd_pnjmean = spd_pnj_mask.mean(dim='lats')
spd_hcmean = spd_hc_mask.mean(dim='x')
for yr in simyears:
    st = np.datetime64(f'{yr}-01-01','ns')
    ed = np.datetime64(f'{yr+2}-01-01','ns')
    dt = np.timedelta64(7,'D')
    timeline = np.arange(st,ed,dt)
    spd_pnjplt[yr] = np.zeros_like(timeline,dtype='float')
    spd_hcplt[yr] = np.zeros_like(timeline,dtype='float')
    for it,t in enumerate(timeline):
        t += np.timedelta64(12,'h')
        spd_pnjplt[yr][it] = spd_pnjmean.loc[t:t+dt-1].mean().data
        spd_hcplt[yr][it] = spd_hcmean.loc[t:t+dt-1].mean().data
#%% plot - arrival, pnj&hc, speed
simyears = np.arange(2007,2021)
output_fname = '/Users/renjiongqiu/Documents_local/Thesis/visualization/visual_outputs/'
alfa = 0.9
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
ttl = dict(fontsize=30)
lbm = dict(fontsize=26)
tkm = dict(labelsize=22)

color = cm.tab20( np.linspace(0,20,len(sources)).astype(int) )

fig = plt.figure(figsize=(22,18),constrained_layout=True)
gs = gridspec.GridSpec(3,2,width_ratios=[16,6])
axs = np.empty((3,2),dtype=object)
axs2 = np.empty((3,2),dtype=object)
for row in range(3):
    for col in range(2):
        if col == 0:
            ax = fig.add_subplot(gs[row,col])
            ax.set_xlim([xticks[0],xticks[-1]])
            ax.set_ylim([0,100])
            ax.set_xticks(xticks.astype(int))
            ax.tick_params('both',**tkm)
            # ax.set_xlabel('Arriving date',**lbm)
            ax.set_ylabel('Fraction [%]',**lbm)
            hfmt = dates.DateFormatter('%B')
            ax.xaxis.set_major_formatter(hfmt)
            plt.setp(ax.get_xticklabels(), rotation=45, ha='center')
            ax.grid()
            
            axs[row,col] = ax
            
            ax2 = ax.twinx()
            ax2.set_zorder(ax.get_zorder()-1)
            ax.patch.set_visible(False)
            ax2.set_xlim([xticks[0],xticks[-1]])
            ax2.set_ylim([0,1.0])
            ax2.set_xticks(xticks.astype(int))
            ax2.tick_params('both',**tkm)
            # ax.set_xlabel('Arriving date',**lbm)
            ax2.set_ylabel('Speed [$m\cdot s^{-1}$]',**lbm)
            hfmt = dates.DateFormatter('%B')
            ax2.xaxis.set_major_formatter(hfmt)
            plt.setp(ax2.get_xticklabels(), rotation=45, ha='center')
            if row == 0:
                ax2.set_title('The Northern Region sources',**ttl)
                ax2.xaxis.set_ticklabels([])
            elif row == 1:
                ax2.set_title('The Panama Bight and Surrounding Region sources',**ttl)
                ax2.xaxis.set_ticklabels([])
            elif row == 2:
                ax2.set_title('The North Humboldt Current System sources',**ttl)
                ax2.set_xlabel('Arriving date',**lbm)
            axs2[row,col] = ax2
            
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

# bar
for iyr,yr in enumerate(simyears):
    clspeed = 'lightskyblue'
    ax = axs[0,0]
    ax2 = axs2[0,0]
    if iyr == 0:
        st = np.datetime64(f'{2020}-01-01','ns')
        ed = np.datetime64(f'{2022}-01-01','ns')
        dt = np.timedelta64(7,'D')
        timeline = np.arange(st,ed,dt).astype('datetime64[D]')
        hd1 = ax.bar(timeline,Arriving_ds[yr].Arrival_NS.data*100,width=dt,color='tab:green',label='Arriving Galápagos',alpha=alfa)
        hd2 = ax.bar(timeline,Arriving_ds[yr].PNJ_NS.data*100,width=dt,color='tab:orange',label='PNJ transported',alpha=alfa,zorder=5)
        hd3 = ax2.bar(timeline,spd_pnjplt[yr],width=dt,color=clspeed,label='Speed of PNJ')
    else:
        ax.bar(timeline,Arriving_ds[yr].Arrival_NS.data*100,width=dt,color='tab:green')
        ax.bar(timeline,Arriving_ds[yr].PNJ_NS.data*100,width=dt,color='tab:orange',zorder=5)
        ax2.bar(timeline,spd_pnjplt[yr],width=dt,color=clspeed)
        # ax.get_xaxis.set_visible(False)
        ax.legend(handles=[hd1,hd2,hd3],loc='upper left',**lbm)

    ax = axs[1,0]
    ax2 = axs2[1,0]
    if iyr == 0:
        st = np.datetime64(f'{2020}-01-01','ns')
        ed = np.datetime64(f'{2022}-01-01','ns')
        dt = np.timedelta64(7,'D')
        timeline = np.arange(st,ed,dt).astype('datetime64[D]')
        hd1 = ax.bar(timeline,Arriving_ds[yr].Arrival_PB.data*100,width=dt,color='tab:green',label='Arriving Galápagos',alpha=alfa)
        hd2 = ax.bar(timeline,Arriving_ds[yr].PNJ_PB.data*100,width=dt,color='tab:orange',label='PNJ transported',alpha=alfa,zorder=5)
        hd3 = ax2.bar(timeline,spd_pnjplt[yr],width=dt,color=clspeed,label='Speed of PNJ')
    else:
        ax.bar(timeline,Arriving_ds[yr].Arrival_PB.data*100,width=dt,color='tab:green')
        ax.bar(timeline,Arriving_ds[yr].PNJ_PB.data*100,width=dt,color='tab:orange',zorder=5)
        ax2.bar(timeline,spd_pnjplt[yr],width=dt,color=clspeed)
        # ax.get_xaxis.set_visible(False)
        ax.legend(handles=[hd1,hd2,hd3],loc='upper left',**lbm)

    ax = axs[2,0]
    ax2 = axs2[2,0]
    if iyr == 0:
        st = np.datetime64(f'{2020}-01-01','ns')
        ed = np.datetime64(f'{2022}-01-01','ns')
        dt = np.timedelta64(7,'D')
        timeline = np.arange(st,ed,dt).astype('datetime64[D]')
        hd1 = ax.bar(timeline,Arriving_ds[yr].Arrival_HC.data*100,width=dt,color='tab:green',label='Arriving Galápagos',alpha=alfa)
        hd2 = ax.bar(timeline,Arriving_ds[yr].HC_HC.data,width=dt,color='tab:orange',label='HC transported',alpha=alfa,zorder=5)
        hd3 = ax2.bar(timeline,spd_hcplt[yr],width=dt,color=clspeed,label='Speed of HC')
    else:
        ax.bar(timeline,Arriving_ds[yr].Arrival_HC.data*100,width=dt,color='tab:green')
        ax.bar(timeline,Arriving_ds[yr].HC_HC.data*100,width=dt,color='tab:orange',zorder=5)
        ax2.bar(timeline,spd_hcplt[yr],width=dt,color=clspeed)
        # ax.get_xaxis.set_visible(True)
        ax.legend(handles=[hd1,hd2,hd3],loc='upper left',**lbm)
        
plt.tight_layout()
# ax.legend(loc='upper right',bbox_to_anchor=(1.22, 1.5))
plt.savefig(output_fname+f'Arriving_Trans_Speed_dt{dt}.png')
'''
# plot
for iyr,yr in enumerate(simyears):
    clspeed = 'lightskyblue'
    ax = axs[0,0]
    ax2 = axs2[0,0]
    if iyr == 0:
        st = np.datetime64(f'{2020}-01-01','ns')
        ed = np.datetime64(f'{2022}-01-01','ns')
        dt = np.timedelta64(7,'D')
        timeline = np.arange(st,ed,dt).astype('datetime64[D]')
        hd1 = ax.scatter(timeline,Arriving_ds[yr].Arrival_NS.data*100, color='tab:green',label='Arriving Galápagos',alpha=alfa)
        hd2 = ax.scatter(timeline,Arriving_ds[yr].PNJ_NS.data*100, color='tab:orange',label='PNJ transported',alpha=alfa,zorder=5)
        hd3 = ax2.scatter(timeline,spd_pnjplt[yr], color=clspeed,label='Speed of PNJ')
        # ax.legend(handles=[hd1,hd2,hd3],loc='upper left',**lbm)
    else:
        ax.scatter(timeline,Arriving_ds[yr].Arrival_NS.data*100, color='tab:green')
        ax.scatter(timeline,Arriving_ds[yr].PNJ_NS.data*100, color='tab:orange',zorder=5)
        ax2.scatter(timeline,spd_pnjplt[yr], color=clspeed)
        # ax.get_xaxis.set_visible(False)

    ax = axs[1,0]
    ax2 = axs2[1,0]
    if iyr == 0:
        st = np.datetime64(f'{2020}-01-01','ns')
        ed = np.datetime64(f'{2022}-01-01','ns')
        dt = np.timedelta64(7,'D')
        timeline = np.arange(st,ed,dt).astype('datetime64[D]')
        hd1 = ax.scatter(timeline,Arriving_ds[yr].Arrival_PB.data*100, color='tab:green',label='Arriving Galápagos',alpha=alfa)
        hd2 = ax.scatter(timeline,Arriving_ds[yr].PNJ_PB.data*100, color='tab:orange',label='PNJ transported',alpha=alfa,zorder=5)
        hd3 = ax2.scatter(timeline,spd_pnjplt[yr], color=clspeed,label='Speed of PNJ')
    else:
        ax.scatter(timeline,Arriving_ds[yr].Arrival_PB.data*100, color='tab:green')
        ax.scatter(timeline,Arriving_ds[yr].PNJ_PB.data*100, color='tab:orange',zorder=5)
        ax2.scatter(timeline,spd_pnjplt[yr], color=clspeed)
        # ax.get_xaxis.set_visible(False)
        # ax.legend(handles=[hd1,hd2,hd3],loc='upper left',**lbm)

    ax = axs[2,0]
    ax2 = axs2[2,0]
    if iyr == 0:
        st = np.datetime64(f'{2020}-01-01','ns')
        ed = np.datetime64(f'{2022}-01-01','ns')
        dt = np.timedelta64(7,'D')
        timeline = np.arange(st,ed,dt).astype('datetime64[D]')
        hd1 = ax.scatter(timeline,Arriving_ds[yr].Arrival_HC.data*100, color='tab:green',label='Arriving Galápagos',alpha=alfa)
        hd2 = ax.scatter(timeline,Arriving_ds[yr].HC_HC.data, color='tab:orange',label='HC transported',alpha=alfa,zorder=5)
        hd3 = ax2.scatter(timeline,spd_hcplt[yr], color=clspeed,label='Speed of HC')
    else:
        ax.scatter(timeline,Arriving_ds[yr].Arrival_HC.data*100, color='tab:green')
        ax.scatter(timeline,Arriving_ds[yr].HC_HC.data*100, color='tab:orange',zorder=5)
        ax2.scatter(timeline,spd_hcplt[yr], color=clspeed)
        # ax.get_xaxis.set_visible(True)
        # ax.legend(handles=[hd1,hd2,hd3],loc='upper left',**lbm)
        
plt.tight_layout()
# ax.legend(loc='upper right',bbox_to_anchor=(1.22, 1.5))
plt.savefig(output_fname+f'Arriving_Trans_Speed_dt{dt}.png')
'''
