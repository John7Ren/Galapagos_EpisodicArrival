"""
This script is based on my collegue Claudio M. Pierard's code. 
The study region is adjusted to the region in my work.
A new feature of identifying the west coast of the land is added.

It uses the Meijer2021_midpoint_emissions GIS dataset.
"""
import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd


def haversine_distance_two(point_A, point_B):
    """Calculates the great circle distance between two points
    on the Earth.

    Parameters
    ----------
    point_A: tuple
        containing the (latitude, longitude) in decimal degrees coordinates of
        point A.
    point_B: tuple
        containing the (latitude, longitude) in decimal degrees coordinates of
        point B.

    Returns
    -------
    km: float
        the distance in km between point A and point B
    """
    lat1, lon1 = point_A
    lat2, lon2 = point_B
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2

    c = 2 * np.arcsin(np.sqrt(a))
    km = 6367 * c
    return km


def region_filters(DF, lon_min, lon_max, lat_min, lat_max, shapefile=False):
    """Takes a Dataframe with the all the rivers information and filters the data
    from the rivers in an specific region.

    Parameters
    ----------
    DF: Dataframe
        is the River_sources dataframe.
    lat_min, lat_max, lon_min, lon_max: float, float, float float
        domain limits.
    shapefile: bool, optional
        True when dealing with a geopandas Dataframe.

    Returns
    -------
    new_DF: Dataframe
        a pandas Dataframe rivers in the especific region.
    """
    if shapefile:
        X = DF.geometry.x
        Y = DF.geometry.y
    else:
        X = DF['X']
        Y = DF['Y']

    mask = (X <= lon_max) & (X > lon_min) & (Y <= lat_max) & (Y > lat_min)

    new_DF = DF[mask]
    return new_DF


def nearest_coastal_cell(latidute, longitude, coord_lat, coord_lon):
    """Function to find the index of the closest point to a certain lon/lat value.

    Parameters
    ----------
    latidute: 1D array
        the latitude 1D array of the grid.
    longitude: 1D array
        the longitude 1D array of the grid.
    coord_lat: float
        latitudinal coordinate of a point.
    coord_lon: float
        longitudinal coordinate of a point.

    Returns
    -------
    index: int array
        The index of the cell from the latidute and longitude arrays. 1 index
        for both arrays.
    """

    distance = np.sqrt((longitude-coord_lon)**2 + (latidute-coord_lat)**2)
    index = distance.argmin()

    return index


def convert_geopandas2pandas(geoDF):
    '''Replaces the geometry column with a X and Y columns
    There no built-in function for this in geopandas!

    Parameters
    ----------
    geoDF: Dataframe
        a GeoPandas Dataframe with geometry column.

    Returns
    -------
    geoDF: Dataframe
        a pandas Dataframe with X, Y coordinates for each point.
    '''

    L = len(geoDF)
    coord = np.zeros((L, 2))
    coord[:, 0] = geoDF.geometry.x
    coord[:, 1] = geoDF.geometry.y
    aux = pd.DataFrame(coord, columns=['X', 'Y'])
    geoDF.drop(columns=['geometry'], inplace=True)
    geoDF = pd.concat([geoDF, aux], axis=1)

    return geoDF


def rivers2coastalgrid(DF, coastal_fields, westcoastYES=True):
    """Takes the river locations in the riverine plastic discharge dataset and
    it bins it into the coastal_cells of the model to be used.

    Parameters
    ----------
    DF: Dataframe
        Pandas dataframe with the riverine plastic discharge and locations.
    coastal_fields: xarray Dataset
        must be generated with landmask.py and contains all the coastal fields
        of the velocity fields to be used. e.g. SMOC dataset.

    Returns
    -------
    new_DF: Dataframe
        a pandas Dataframe with the binned rivers into the coastal_cells.
    """
    N = len(DF)
    coast = coastal_fields.coast.values
    westcoast = coastal_fields.westcoast.values
    lats = coastal_fields.lat.values
    lons = coastal_fields.lon.values
    if westcoastYES:
        iy_coast, ix_coast = np.where(westcoast == 1)
    else:
        iy_coast, ix_coast = np.where(coast == 1)
    lat_coast = lats[iy_coast]
    lon_coast = lons[ix_coast]

    new_coordinates = np.zeros((N, 2))

    for i in range(N):
        x_lon = DF.iloc[i].X
        x_lat = DF.iloc[i].Y

        n_index = nearest_coastal_cell(lat_coast, lon_coast, x_lat, x_lon)
        new_coordinates[i, :] = (lon_coast[n_index], lat_coast[n_index])

    aux = pd.DataFrame(new_coordinates, columns=['X_bin', 'Y_bin'],
                       index=DF.index)
    new_DF = pd.concat([DF, aux], axis=1)

    counts = new_DF.groupby(['X_bin', 'Y_bin']).count().loc[:, 'X'].values     # After converting a finner-resolution data to a bigger-res data some X and Y become identical
    new_DF = new_DF.groupby(['X_bin', 'Y_bin']).sum()
    new_DF['merged_rivers'] = counts
    new_DF.reset_index(inplace=True)
    new_DF.drop(labels=['X', 'Y'], axis=1, inplace=True)

    return new_DF


def center_of_mass(DF):
    """Computes the center of mass from a river dataframe.
    Warning: Only works with Meijer dataset.
    the returned coords are not on the grid cells, they are just numbers

    Parameters
    ----------
    DF: Dataframe
        The file location of the spreadsheet
    coastal_fields : bool, optional
        A flag used to print the columns to the console (default is
        False)

    Returns
    -------
    new_DF: Dataframe
        a pandas Dataframe with the binned rivers into the coastal_cells.

    """
    x = DF.X_bin
    y = DF.Y_bin
    m = DF.dots_exten  # this is so annoying, only for Meijer.
    M = m.sum()
    return sum(m*y)/M, sum(m*x)/M


def rivers_per_location(DF, loc_coords, radius, binned=False, tolerance=0.1):
    """It clusters the rivers in a square with sides 2*radius. The clustering
    is done iteratively using the center of mass.

    Parameters
    ----------
    DF: Dataframe
        the pandas Dataframe with data River_sources.
    loc_coords: tuple
        containing the location coordinates as in (lat, lon).
    radius: float
        the radius in degrees around loc_coords.
    binned: bool, optional
        default to False. True if the Dataframe is binned using
        rivers2coastalgrid.
    tolerance: float, optional
        the tolerance in km to stop the iterations.

    Returns
    -------
    mask: list
        contains the index with the rivers around loc_coords within radius.
    CM: tuple
        a tuple with the (lat, lon) coordinates of the center of mass.
    """
    if binned:
        _label = '_bin'

    else:
        _label = ''

    x_col = f'X{_label}' # f-string, variable inside {}
    y_col = f'Y{_label}'

    lat, lon = loc_coords
    mask = (DF[x_col] <= lon + radius) & (DF[x_col] > lon - radius) & \
        (DF[y_col] <= lat + radius) & (DF[y_col] > lat - radius)
    CM = center_of_mass(DF[mask])
    dist = haversine_distance_two((lat, lon), CM)

    while dist > tolerance:
        lat, lon = CM
        mask = (DF[x_col] <= lon + radius) & (DF[x_col] > lon - radius) & \
            (DF[y_col] <= lat + radius) & (DF[y_col] > lat - radius)
        CM = center_of_mass(DF[mask])
        dist = haversine_distance_two((lat, lon), CM)

    loc_df = DF[mask]
    p = pd.DataFrame({'p': loc_df['dots_exten']/loc_df['dots_exten'].sum()})
    loc_df = loc_df.drop(['dots_exten'], axis=1)
    loc_df = pd.concat([loc_df, p], axis=1)
    loc_df.reset_index(inplace=True)

    return mask, CM

#%%

###############################################################################
# Parameters
###############################################################################
r = 1  # radius for clusters.
N = 274*365  # Number of particles realesed per source.
# Ecuador_region = (-85, -75, -5, 5)  # the region to study
# dataname = '25N25S'
dataname = '25N25S'

if dataname == '35N35S':
    study_region = (-125, -55, -35, 35)
    westcoastYES = False
elif dataname == '25N25S':
    study_region = (-115, -65, -25, 25)
    westcoastYES = True

save_priors = True  # True for saving the priors.

###############################################################################
# Load all requiered data
###############################################################################

# the coastal fields dataset.
coastal_fields = xr.load_dataset('/Users/renjiongqiu/Documents_local/Thesis/data/sources/coastal_fields_'+dataname+'.nc')
coast = coastal_fields.coast.values
westcoast = coastal_fields.westcoast.values
lats = coastal_fields.lat.values
lons = coastal_fields.lon.values
X = coastal_fields.lon_mesh
Y = coastal_fields.lat_mesh

# reshape the coastal fields in 1D arrays
if westcoastYES:
    iy_coast, ix_coast = np.where(westcoast == 1)
else:
    iy_coast, ix_coast = np.where(coast == 1)
lat_coast = lats[iy_coast]
lon_coast = lons[ix_coast]


# Read the GIS Shapefile from Meijer
path = '/Users/renjiongqiu/Documents_local/Thesis/data/sources/Meijer2021_midpoint_emissions/Meijer2021_midpoint_emissions.shp'
river_discharge = convert_geopandas2pandas(gpd.read_file(path))
river_discharge = region_filters(river_discharge, *study_region)
river_discharge = rivers2coastalgrid(river_discharge, coastal_fields, westcoastYES)

# compute total discharged plastic in Ecuador
total_plastic = river_discharge['dots_exten'].sum()

# sort the rivers by discharge from large to small.
river_discharge = river_discharge.sort_values(['dots_exten'], ascending=False)
river_discharge.reset_index(inplace=True, drop=True)

# define the cluster river locations (by eye)
cluster_locations = {
                     # Mexico
                     # 'SanBlas': (21.53, -105.3), # upper Mexico, should include river de santiago and 'Vallarta': (20.65, -105.3),
                     # 'Lazaro': (17.94, -102.1)
                     'Acapulco': (16.83, -99.38), # middle Mexico
                     'SalinaCruz': (16.13, -94.50),
                     # Guatemala
                     'Tecojate': (14.00, -91.43), 
                     #ElSalvador
                     'LaColorada': (13.40, -89.30),
                     # Nicaraagua
                     'Fonseca': (12.85, -87.20), # 
                     # Costa Rica
                     'Chira': (10.08, -85.10),
                     # Panama
                     'Panama': (8.70, -79.10),  
                     # Colombia
                     'Cacique': (6.65, -77.42),
                     # Ecuador
                     'Esmeraldas': (0.9760, -79.65),
                     'Guayaquil': (-2.210, -79.92),
                     # Peru
                     'Parachique': (-5.36, -80.86),
                     'Lima': (-11.95, -77.14)
                    }

# Move them into the coastal cells. Maybe this is not necessary.
grid_cluster_centers = {}
for loc in cluster_locations:

    indx = nearest_coastal_cell(lat_coast, lon_coast, *cluster_locations[loc])
    grid_cluster_centers[loc] = (lat_coast[indx], lon_coast[indx])

#%%
###############################################################################
# Generate the Clusters, release points and priors
###############################################################################
release_points = {}
priors = {}

cluster_percent = 0  # counter for the percentege of plastic in the clusters.
merged_rivers = 0  # counter for the number or rivers in all the clusters.
unclustered_mask = 0  # Empty value for the mask for unclustered rivers

for i, loc in enumerate(cluster_locations):
    print(loc)
    # get the local DF for cluster
    mask, _CM = rivers_per_location(river_discharge, cluster_locations[loc],
                                    r, binned=True)
    unclustered_mask = unclustered_mask | mask  # True | False = True
    loc_df = river_discharge[mask]

    # number of rivers merged
    number_rivers = loc_df['merged_rivers'].sum()
    merged_rivers += number_rivers

    # compute prior for cluster
    loc_percent = loc_df['dots_exten'].sum()/total_plastic
    priors[loc] = [loc_percent, number_rivers]
    cluster_percent += loc_df['dots_exten'].sum()/total_plastic

    # compute the weights for each river within the cluster
    p = pd.DataFrame({'p': loc_df['dots_exten']/loc_df['dots_exten'].sum()})

    loc_df = loc_df.drop(['dots_exten'], axis=1)  # droppin this, dont need it
    loc_df = pd.concat([loc_df, p], axis=1)
    loc_df.reset_index(drop=True, inplace=True)

    # IMPORTANT STEP. Samples randomly the locations of the rivers N times
    # according to the weigths 'p'. This creates the initital conditions for
    # the experiment.
    release_points[loc] = loc_df.sample(n=N, replace=True, weights='p', random_state=1)

priors = pd.DataFrame.from_dict(priors,orient='index',columns=['priors','merged_rivers'])
#%% -- Save the stuff

np.save('/Users/renjiongqiu/Documents_local/Thesis/data/river_sources_'+dataname+'.npy', grid_cluster_centers, allow_pickle=True)
np.save('/Users/renjiongqiu/Documents_local/Thesis/data/release_positions_'+dataname+f'_{N}releasing.npy', release_points, allow_pickle=True)
# np.save('/Users/renjiongqiu/Documents/Thesis/data/river_discharge_25N25S.npy', river_discharge, allow_pickle=True)
river_discharge.to_csv('/Users/renjiongqiu/Documents_local/Thesis/data/river_discharge_'+dataname+'.csv')
if save_priors:
    priors.to_csv('/Users/renjiongqiu/Documents_local/Thesis/data/priors_river_inputs_'+dataname+'.csv')

#%% Plot the origin Meijer data

###############################################################################
# Visualization
###############################################################################
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt

study_region = (-125, -55, -35, 35)
# the coastal fields dataset.
coastal_fields = xr.load_dataset('/Users/renjiongqiu/Documents_local/Thesis/data/sources/coastal_fields_25N25S.nc')
coast = coastal_fields.coast.values
lats = coastal_fields.lat.values
lons = coastal_fields.lon.values
X = coastal_fields.lon_mesh
Y = coastal_fields.lat_mesh

# reshape the coastal fields in 1D arrays
iy_coast, ix_coast = np.where(westcoast == 1)
lat_coast = lats[iy_coast]
lon_coast = lons[ix_coast]


# Read the GIS Shapefile from Meijer
path = '/Users/renjiongqiu/Documents_local/Thesis/data/sources/Meijer2021_midpoint_emissions/Meijer2021_midpoint_emissions.shp'
river_discharge = convert_geopandas2pandas(gpd.read_file(path))
river_discharge = region_filters(river_discharge, *study_region)
river_discharge = rivers2coastalgrid(river_discharge, coastal_fields)

# Visualization
output_fname = '/Users/renjiongqiu/Documents_local/Thesis/visualization/visual_outputs/'
scale = 0.1*river_discharge.dots_exten
fig = plt.figure(figsize=(12,12))
ax = fig.add_subplot(projection=ccrs.PlateCarree())
ax.coastlines(resolution='50m')
ax.add_feature(cfeature.LAND)
ax.add_feature(cfeature.RIVERS)
ax.add_feature(cfeature.LAKES)
ax.add_feature(cfeature.BORDERS)
ax.add_feature(cfeature.OCEAN)
# ax.scatter(loc_df['X_bin'],loc_df['Y_bin'])
ax.scatter(river_discharge.X_bin,river_discharge.Y_bin,c='orange')
ax.gridlines(draw_labels=True,linewidth=0.5,color='k',alpha=0.5,linestyle='--')
ax.set_extent(study_region,crs=ccrs.PlateCarree())
ax.set_title('Map of data from Meijer et al. 2021 (west coast 25N-25S)', fontsize=20)
# ax.set_extent((-140,-60,-45,30),crs=ccrs.PlateCarree()) # Releasing location
plt.tight_layout(pad=2)
# plt.savefig(output_fname+'Meijer_map-2.pdf')
