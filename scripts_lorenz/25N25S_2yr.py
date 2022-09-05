from parcels import FieldSet, Field, ParticleSet, Variable, JITParticle, AdvectionRK4, plotTrajectoriesFile, ScipyParticle, ErrorCode, ParcelsRandom
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
from glob import glob
import sys
import os
from matplotlib.path import Path
import matplotlib.patches as patches

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

###############################################################################
# Simulation parameters
###############################################################################
simyear = sys.argv[1]
loc = sys.argv[2]
zarr = eval(sys.argv[3])

secondyear = str(int(simyear)+1)
start_time = datetime.strptime(simyear+'-01-01 00:00:00', '%Y-%m-%d %H:%M:%S')
# end_time = '2020-01-01'
simdays = 729 # days
releasedays = 365 # days
dailyrelease = 274
N = dailyrelease*releasedays # Number of particles released

resusTime = 171
shoreTime = 10

fname = f'/storage/home/9703243/galapagos/outputs/{simyear}/sim_results/'
outfname = f'{loc}_{simyear}-{secondyear}_run{simdays}d_release{releasedays}d_{dailyrelease}par_beaching'
os.makedirs(os.path.dirname(fname),exist_ok=True)

###############################################################################
# Import and Set Releasing information
###############################################################################
# Load the releasing information
release_positions = np.load(f'release_positions_25N25S_{N}releasing_v0.npy',allow_pickle=True).item()
# river_sources = np.load('/Users/renjiongqiu/Documents/Thesis/data/river_sources_25N25S.npy',allow_pickle=True).item()
print('load the releasing information: done')

# Set the releasing location
lons = release_positions[loc]['X_bin'].values
lats = release_positions[loc]['Y_bin'].values

# Set the releasing date
np.random.seed(0)
n_points = release_positions[loc].shape[0]
if n_points != N:
    print(f'importing {n_points} points, while expecting {N} points')

dates = np.empty(n_points, dtype='O')
for i in range(releasedays):
    for j in range(dailyrelease):
        dates[i*dailyrelease+j] = start_time + timedelta( days=i,
                                                         hours=np.random.randint(0,24) # smaller than 24
                                                         )
#%%
###############################################################################
# Import FieldSet
###############################################################################
data_path = '/storage/shared/oceanparcels/input_data/MOi/psy4v3r1/'

ufiles1 = glob(data_path+'psy*U_'+simyear+'*')
ufiles2 = glob(data_path+'psy*U_'+secondyear+'*')
ufiles = sorted(ufiles1+ufiles2)

vfiles1 = glob(data_path+'psy*V_'+simyear+'*')
vfiles2 = glob(data_path+'psy*V_'+secondyear+'*')
vfiles = sorted(vfiles1+vfiles2)

wfiles1 = glob(data_path+'psy*W_'+simyear+'*')
wfiles2 = glob(data_path+'psy*W_'+secondyear+'*')
wfiles = sorted(wfiles1+wfiles2)

mesh_mask = '/storage/shared/oceanparcels/input_data/MOi/domain_ORCA0083-N006/coordinates.nc'

filenames = {'U': {'lon':mesh_mask, 'lat':mesh_mask, 'depth':wfiles[0], 'data':ufiles},
             'V': {'lon':mesh_mask, 'lat':mesh_mask, 'depth':wfiles[0], 'data':vfiles},
             'W': {'lon':mesh_mask, 'lat':mesh_mask, 'depth':wfiles[0], 'data':wfiles}}
variables = {'U': 'vozocrtx',
             'V': 'vomecrty',
             'W': 'vovecrtz'}
dimensions = {'U': {'lon':'glamf', 'lat':'gphif', 'depth':'depthw', 'time':'time_counter'},
              'V': {'lon':'glamf', 'lat':'gphif', 'depth':'depthw', 'time':'time_counter'},
              'W': {'lon':'glamf', 'lat':'gphif', 'depth':'depthw', 'time':'time_counter'}}
fieldset = FieldSet.from_nemo(filenames,variables,dimensions,allow_time_extrapolation=True)

# Set the initial location to release the particles
fU = fieldset.U
fieldset.computeTimeChunk(fU.grid.time[0],1) # `fU.grid.time` gives the start and end time (of the day?)

print('loading fieldset: done')

# add fieldset
coastal_fields = xr.load_dataset('coastal_fields_whole.nc')
distance = coastal_fields.distance2shore.values
fieldset.add_field(Field('distance2shore', distance,
                         lon=fU.grid.lon, lat=fU.grid.lat,
                         mesh='spherical'),)
print('adding fieldset: done')


#%%
def beach(particle, fieldset, time):

    if particle.beach == 0:
        dist = fieldset.distance2shore[time, particle.depth, particle.lat,
                                       particle.lon]
        if dist < 10:
            beach_prob = math.exp(-particle.dt/(particle.coastPar*86400.))
            if ParcelsRandom.random(0., 1.) > beach_prob:
                particle.beach = 1
    # Now the part where we build in the resuspension
    elif particle.beach == 1:
        resus_prob = math.exp(-particle.dt/(particle.resus_t*86400.))
        if ParcelsRandom.random(0., 1.) > resus_prob:
            particle.beach = 0

def SampleGalapagos(fieldset, particle, time):
    if fieldset.galapagosmask[time, particle.depth, particle.lon, particle.lat] == 1:
        particle.visitedgalapagos = 1

# Save the age of the particles
def Age(fieldset, particle, time):
    particle.age = particle.age + math.fabs(particle.dt)
    
# Define Galapagos particles
class GalapagosParticle(JITParticle):
    age = Variable('age', initial=0.)
    beach = Variable('beach', dtype=np.int32, initial=0)
    resus_t = Variable('resus_t', dtype=np.float32, initial=resusTime, to_write=False)
    coastPar = Variable('coastPar', dtype=np.float32, initial=shoreTime, to_write=False)
    # visitedgalapagos = Variable('visitedgalapagos', initial=0.)

def delete_particle(particle, fieldset, time):
    """Kernel for deleting particles if they are out of bounds."""
    # print(f'lon:{fieldset.grid.lon}, lat:{fieldset.grid.lat}') 
    print(particle.lon, particle.lat, particle.depth)
    particle.delete()

# Set ParticleSet and output_file
pset = ParticleSet.from_list(fieldset=fieldset,
                             pclass=GalapagosParticle,
                             lon=lons,
                             lat=lats,
                             time=dates)
if zarr == True:
    output_file = pset.ParticleFile(name=fname+outfname+'.zarr', outputdt=timedelta(days=1))
elif zarr == False:
    output_file = pset.ParticleFile(name=fname+outfname+'.nc', outputdt=timedelta(days=1))
print('particle set: done')

# kernels = AdvectionRK4 + pset.Kernel(Age) + pset.Kernel(SampleGalapagos)
kernels = AdvectionRK4 + pset.Kernel(Age)
# Execute pset
pset.execute(kernels,
             runtime=timedelta(days=simdays),
             dt=timedelta(hours=1),
             output_file=output_file,
             recovery={ErrorCode.ErrorOutOfBounds: delete_particle})

output_file.export()
print('simulation: done')

#%% Visualization
river_sources = np.load('/storage/home/9703243/galapagos/scripts/river_sources_25N25S_v0.npy',allow_pickle=True).item()
study_region = (-115,-65,-25,25)
fps = 30

if zarr == True:
    ds = xr.open_zarr(fname+outfname+'.zarr')
elif zarr == False:
    ds = xr.open_dataset(fname+outfname+'.nc')
AR = {}
AR['lon'] = np.array(ds.lon[:,:])
AR['lat'] = np.array(ds.lat[:,:])
AR['time'] = np.array(ds.time)
AR['age'] = np.array(ds.age)
# AR['visitedgalapagos'] = np.array(ds.visitedgalapagos)
print('load output: done')
'''
bins = [np.arange(-115,-65,0.5),np.arange(-25,25,0.5)]
outputdt = timedelta(days=1)
timerange = np.arange(np.nanmin(AR['time']), np.nanmax(AR['time'])+np.timedelta64(outputdt), np.timedelta64(outputdt))
time_traj = np.where(AR['time']>timerange[364])[0]
time_obsv = np.where(AR['time']>timerange[364])[1]

lon_2nd = AR['lon'][time_traj,time_obsv]
lat_2nd = AR['lat'][time_traj,time_obsv]

H2,xe,ye = np.histogram2d(lon_2nd, lat_2nd, bins=bins)
# H0 = H/365
# plot contourf
xb2 = ( xe[:-1] + xe[1:] ) / 2
yb2 = ( ye[:-1] + ye[1:] ) / 2

levels = [10**i for i in range(0,7)]
fig = plt.figure(figsize=(12,14))
ax = fig.add_subplot(projection=ccrs.PlateCarree())
cf = ax.contourf(xb2,yb2,H2.T,levels=levels,transform=ccrs.PlateCarree(),norm=colors.LogNorm())
ax.scatter(river_sources[loc][1],river_sources[loc][0],s=150,color='red')
ax.coastlines(resolution='50m')
ax.add_feature(cfeature.LAND)
ax.add_feature(cfeature.RIVERS)
ax.add_feature(cfeature.LAKES)
ax.add_feature(cfeature.BORDERS)
ax.set_title(f'{loc} 2yr-simulation particle density in the 2nd year',fontsize=16)
ax.gridlines(draw_labels=True,linewidth=0.5,color='k',alpha=0.5,linestyle='--')
ax.set_extent(study_region,crs=ccrs.PlateCarree())
cb = plt.colorbar(cf,fraction=0.04, pad=0.06,ticks=levels)
cb.ax.set_yticklabels(range(0,7))
plt.savefig(fname+outfname+'.pdf')

print('plot contourf: done')

'''
#%%
# Make animation
anim_fname = f'/storage/home/9703243/galapagos/outputs/{simyear}/animations/'
os.makedirs(os.path.dirname(anim_fname),exist_ok=True)
text_dev = [0.035,0.07]
outputdt = timedelta(days=1)

river_lons = river_sources[loc][1]
river_lats = river_sources[loc][0]
path = DrawCluster(river_lons, river_lats, 1)
patch = patches.PathPatch(path,ec='k',fc='none',lw=2)
text_lon = (river_lons+115)/50-text_dev[1]
text_lat = (river_lats+25)/50-text_dev[0]
if loc == 'LaColorada' or loc == 'SalinaCruz':
    text_lon = (river_lons+115)/50+text_dev[1]
    text_lat = (river_lats+25)/50+text_dev[0]

galapagos_extent = [-91.8, -89, -1.4, 0.7]
galabox = patches.Polygon([[galapagos_extent[0],galapagos_extent[2]],
                       [galapagos_extent[1],galapagos_extent[2]],
                       [galapagos_extent[1],galapagos_extent[3]],
                       [galapagos_extent[0],galapagos_extent[3]]],
                      ec='k',fc='none',lw=2)  

# doutputdt = timedelta(days=5)
dt = np.timedelta64(outputdt,'D').astype(int)
timerange = np.arange(np.nanmin(AR['time']), np.nanmax(AR['time'])+np.timedelta64(outputdt), np.timedelta64(outputdt))
tplot = np.arange(np.nanmin(AR['time'].astype('datetime64[D]')), np.nanmax(AR['time'].astype('datetime64[D]'))+2*dt, dt)
t = str(tplot[0])

fig = plt.figure(figsize=(12,12))
ax = fig.add_subplot(projection=ccrs.PlateCarree())
scatter = ax.scatter(AR['lon'][:,0],AR['lat'][:,0],transform=ccrs.PlateCarree(),color='orange')
# ax.scatter(river_sources[loc][1],river_sources[loc][0],s=150,color='black')
ax.add_patch(patch)
plt.text(text_lon,text_lat,f'{loc}',ha='center',va='center',fontsize=20,transform=ax.transAxes,fontweight='bold')

ax.set_title(f'Particles from {loc} at t = {t}',fontsize=16)
ax.coastlines(resolution='50m')
ax.add_feature(cfeature.LAND)
ax.add_feature(cfeature.RIVERS)
ax.add_feature(cfeature.LAKES)
ax.add_feature(cfeature.BORDERS)
ax.add_feature(cfeature.OCEAN)
ax.gridlines(draw_labels=True,linewidth=0.5,color='k',alpha=0.5,linestyle='--')
ax.set_extent(study_region,crs=ccrs.PlateCarree())
ax.add_patch(galabox)

def animate(i):
    time_traj = np.where(AR['time'] == timerange[i])[0]
    time_obsv = np.where(AR['time'] == timerange[i])[1]
    t = str(tplot[i])
    
    ax.set_title(f'Particles from {loc} at t = {t}',fontsize=16)
    scatter.set_offsets(np.c_[AR['lon'][time_traj,time_obsv], AR['lat'][time_traj,time_obsv]])
    # ax.add_patch(galabox)
    
anim = FuncAnimation(fig, animate, frames=len(timerange), interval=100)
anim.save(anim_fname+outfname+'.gif', writer='imagemagick', fps=fps)
print(f'save animation_{loc}_{simyear}: done')