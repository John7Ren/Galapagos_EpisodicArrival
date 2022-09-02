"""
This script is based on my collegue Claudio M. Pierard's code. 
The study region is adjusted to the region in my work.
A new feature of identifying the west coast of the land is added.
"""

from netCDF4 import Dataset
import numpy as np
import xarray as xr


def make_landmask(path2output, indices):
    """Returns landmask where land = 1 and ocean = 0.
    Warning: teste for NEMO 2D, 'sossheig' = sea surface height

    Parameters
    ----------
    path2output: string
        the path to an output of a model (netcdf file expected).
    indices: dictionary
        a dictionary such as {'lat': slice(1, 900), 'lon': slice(1284, 2460)}.

    Returns
    -------
    landmask: array
        2D array containing the landmask. Were the landcells are 1 and the
        ocean cells are 0.
    """
    datafile = Dataset(path2output)
    landmask = datafile.variables['sossheig'][0, indices['lat'], indices['lon']]
    landmask = np.ma.masked_invalid(landmask)
    landmask = landmask.mask.astype('int')

    return landmask


def get_coastal_cells(landmask):
    """Function that detects the coastal cells, i.e. the ocean cells directly
    next to land. Computes the Laplacian of landmask.

    Parameters
    ----------
    landmask: array
        the land mask built using `make_landmask` function , where landcell = 1
                and oceancell = 0.

    Returns
    -------
    coastal: array
        2D array array containing the coastal cells, the coastal cells are
        equal to one, and the rest is zero.
    """
    mask_lap = np.roll(landmask, -1, axis=0) + np.roll(landmask, 1, axis=0)
    mask_lap += np.roll(landmask, -1, axis=1) + np.roll(landmask, 1, axis=1)
    mask_lap -= 4*landmask
    mask_lap[0,:] = 0
    mask_lap[-1,:] = 0
    mask_lap[:,0] = 0
    mask_lap[:,-1] = 0
    # coastal = np.ma.masked_array(landmask, mask_lap > 0)
    coastal = np.ma.masked_array(mask_lap > 0)
    coastal = coastal.data.astype('int')

    return coastal

def peripheral(coastal_cells,loc_y,loc_x):
    """Function that detects 

    Parameters
    ----------
    coastal_cells: Array
        from the functino 'get_coastal_cells'. Only the coastal cells are valued 1,
        with the other cells valued 0.
        
    loc_y, loc_x: int
        the y,x coordinates of the selected cell.

    Returns
    -------
    locs: tuple
        The peripheral coordinates of the selected cell. (loc_y,loc_x)
    """
    ymax = np.shape(coastal_cells)[0]
    xmax = np.shape(coastal_cells)[1]
    locs = []
    r = 1
    for i in range(-r,r+1):
        for j in range(-r,r+1):
            if (loc_y+i<ymax) & (loc_y+i>=0) & (loc_x+j<xmax) & (loc_x+j>=0):
                locs.append( (loc_y+i,loc_x+j) )
    locs.remove( (loc_y,loc_x) )
    
    return locs
    

def get_westcoast_cells(coastal_cells,X,Y,lon,lat):
    '''
    Careful that this function can so far be used at this 'study_region',
    because one needs to manually find the 'head of the thread'
    
    Parameters
    ----------
    coastal_cells : 2D array
        The coastal mask using 'get_coastal_cells' function, only coastal cells = 1.
    X : meshgrid
    Y : meshgrid
    lon : float
        The starting lon.
    lat : float
        The starting lat.

    Returns
    -------
    westcoast :  2D array
        The west coastal mask with west coastal cells = 1.
    '''
    
    loc_y, loc_x = np.where( (X==lon).astype(int) * (Y==lat).astype(int) == 1 )
    loc_x = loc_x.item()
    loc_y = loc_y.item()
    westcoast_locs = [(loc_y,loc_x)]
    # loc_pre = (loc_y,loc_x)
    times = 0
    order = [ [2,3,4], [1,0,5], [8,7,6] ]
    order = np.array(order)
    while (loc_y>1 and loc_x<np.shape(X)[1]-1):
        locs = peripheral(coastal_cells,loc_y,loc_x)
        # print('locs='+str(locs))
        loc_iscoast = []
        loc_order = []
        for l in locs:
            if coastal_cells[l] == 1:
                loc_iscoast.append(l)
                ly = 1 - (l[0] - loc_y)
                lx = l[1] - loc_x + 1
                loc_order.append(order[(ly,lx)])
        if len(loc_iscoast) >= 1:
            distance = []
            repeat = []
            for i,l in enumerate(loc_iscoast):
                if l not in westcoast_locs:
                    distance.append( np.sqrt( (l[0]-loc_y)**2 + (l[1]-loc_x)**2 ) )
                elif l in westcoast_locs:
                    distance.append( 10*np.sqrt( (l[0]-loc_y)**2 + (l[1]-loc_x)**2 ) )
                    loc_order[i] = loc_order[i] * 10
                    
                    for j,lj in enumerate(westcoast_locs):
                        if lj == l:
                            repeat.append(j)
                else:
                    print('unexpected')
            
            distance = np.array(distance)
            loc_isshort = np.where( distance == distance.min() )[0]
            if loc_isshort.size > 1:
                if len(repeat) < len(loc_iscoast):
                    index = np.array(loc_order).argmin()
                elif len(repeat) == len(loc_iscoast):
                    index = np.array(repeat).argmin()
            elif loc_isshort.size == 1:
                index = loc_isshort[0]
                
            loc_y = loc_iscoast[index][0]
            loc_x = loc_iscoast[index][1]
        else:
            print('cannot find peripheral cells')
        westcoast_locs.append((loc_y,loc_x))
        times += 1
        if times == 1500:
            break
    
    westcoast = np.zeros_like(coastal_cells)

    for l in westcoast_locs:
        westcoast[l] = 1 
    
    return westcoast


def get_coastal_nodes(landmask):
    mask_lap = np.roll(landmask, -1, axis=0) + np.roll(landmask, 1, axis=0)
    mask_lap += np.roll(landmask, -1, axis=1) + np.roll(landmask, 1, axis=1)
    mask_lap -= 4*landmask
    mask_lap[0,:] = 0
    mask_lap[-1,:] = 0
    mask_lap[:,0] = 0
    mask_lap[:,-1] = 0
    ci = np.ma.masked_array(mask_lap == 1)
    ci = ci.data.astype('int')
    
    return ci
    
def get_coastal_nodes_diagonal(landmask):
    mask_lap = np.roll(landmask, -1, axis=0) + np.roll(landmask, 1, axis=0)
    mask_lap += np.roll(landmask, -1, axis=1) + np.roll(landmask, 1, axis=1)
    mask_lap -= 4*landmask
    mask_lap[0,:] = 0
    mask_lap[-1,:] = 0
    mask_lap[:,0] = 0
    mask_lap[:,-1] = 0
    ci_d = np.ma.masked_array(mask_lap == 2)
    ci_d = ci_d.data.astype('int')
    
    return ci_d

def get_coastal_nodes_3sides(landmask):
    mask_lap = np.roll(landmask, -1, axis=0) + np.roll(landmask, 1, axis=0)
    mask_lap += np.roll(landmask, -1, axis=1) + np.roll(landmask, 1, axis=1)
    mask_lap -= 4*landmask
    mask_lap[0,:] = 0
    mask_lap[-1,:] = 0
    mask_lap[:,0] = 0
    mask_lap[:,-1] = 0
    ci_tr = np.ma.masked_array(mask_lap == 3)
    ci_tr = ci_tr.data.astype('int')
    
    return ci_tr

def get_shore_cells(landmask):
    """Function that detects the shore cells, i.e. the land cells directly
    next to the ocean. Computes the Laplacian of landmask.

    Parameters
    ----------
    landmask: array
        the land mask built using `make_landmask`, where land cell = 1
        and ocean cell = 0.

    Returns
    -------
    shore: array
        2D array array containing the shore cells, the shore cells are
        equal to one, and the rest is zero.
    """
    mask_lap = np.roll(landmask, -1, axis=0) + np.roll(landmask, 1, axis=0)
    mask_lap += np.roll(landmask, -1, axis=1) + np.roll(landmask, 1, axis=1)
    mask_lap -= 4*landmask
    # shore = np.ma.masked_array(landmask, mask_lap < 0)
    shore = np.ma.masked_array(mask_lap < 0)
    shore = shore.data.astype('int')

    return shore


def create_border_current(landmask, double_cell=False):
    """Function that creates a border current 1 m/s away from the shore.

    Parameters
    ----------
    landmask: array
        the land mask built using `make_landmask`.
    double_cell: bool, optional
        True for if you want a double cell border velocity. Default set to
        False.

    Returns
    -------
    v_x, v_y: array, array
        two 2D arrays, one for each component of the velocity.
    """
    shore = get_shore_cells(landmask)
    coastal = get_coastal_cells(landmask)
    Ly = np.roll(landmask, -1, axis=0) - np.roll(landmask, 1, axis=0)
    Lx = np.roll(landmask, -1, axis=1) - np.roll(landmask, 1, axis=1)

    if double_cell:
        v_x = abs(Lx)*(coastal+shore)
        v_y = abs(Ly)*(coastal+shore)
    else:
        v_x = abs(Lx)*(shore)
        v_y = abs(Ly)*(shore)

    magnitude = np.sqrt(v_y**2 + v_x**2)
    # the coastal cells between land create a problem. Magnitude there is zero
    # I force it to be 1 to avoid problems when normalizing.
    ny, nx = np.where(magnitude == 0)
    magnitude[ny, nx] = 1

    v_x = v_x/magnitude
    v_y = v_y/magnitude

    return v_x, v_y


def distance_to_shore(landmask, dx=1):
    """Function that computes the distance to the shore. It is based in the
    the `get_coastal_cells` algorithm.

    Parameters
    ----------
    landmask: array
        the land mask built using `make_landmask` function.
    dx: float, optional
        the grid cell dimesion. This is a crude approximation of the real
        distance (be careful). Default set to 1.

    Returns
    -------
    distance: array
        2D array containing the distances from shore.
    """
    # ci = get_coastal_cells(landmask)
    # landmask_i = landmask + ci
    # dist = ci
    # i = 0
    #
    # while i < dist.max():
    #     ci = get_coastal_cells(landmask_i)
    #     landmask_i += ci
    #     dist += ci*(i+2)
    #     i += 1
    #
    # distance = dist*dx
    # return distance
    ci = get_coastal_nodes(landmask)  # direct neighbours
    dist = ci*dx                     # 1 dx away

    ci_d = get_coastal_nodes_diagonal(landmask)  # diagonal neighbours
    dist_d = ci_d*np.sqrt(2*dx**2)/2.       # sqrt(2) dx away
    
    ci_tr = get_coastal_nodes_3sides(landmask)  # diagonal neighbours
    dist_d = ci_tr*dx       # sqrt(2) dx away

    return dist+dist_d+ci_tr


def generate_dataset(path2output, indices, output_path):
    """Creates a netCDF file with all the fields needed to run
    SAG_experiment.py.

    Parameters
    ----------
    - path2output: string
        is the path to an output of a model (netcdf file expected).
    - indices: dictionary
        a dictionary such as {'lat': slice(1, 900), 'lon': slice(1284, 2460)}.
    output_path: string
        is the output path and name of the netCDF file.

    Returns
    -------
    None: it saves the file to output_path.
    """
    model = xr.load_dataset(path2output)
    lons = model['nav_lon'][0,indices['lon']]  # * unpacks the tuple
    lats = model['nav_lat'][indices['lat'],0]
    X, Y = np.meshgrid(lons, lats)

    landmask = make_landmask(path2output, indices)
    coastal_cells = get_coastal_cells(landmask)
    
    lon = X[-2,83]
    lat = Y[-2,83]
    west_coast = get_westcoast_cells(coastal_cells,X,Y,lon,lat)
    
    shore_cells = get_shore_cells(landmask)
    coastal_u, coastal_v = create_border_current(landmask)
    distance2shore = distance_to_shore(landmask, dx=8.33)  # km, the origin code is 9.26, but should be 100/12 = 8.33 

    ds = xr.Dataset(
        data_vars=dict(
            landmask=(["y", "x"], landmask),
            coast=(["y", "x"], coastal_cells),
            westcoast=(["y", "x"], west_coast),
            shore=(["y", "x"], shore_cells),
            coastal_u=(["y", "x"], coastal_u),
            coastal_v=(["y", "x"], coastal_v),
            distance2shore=(["y", "x"], distance2shore),
            lat_mesh=(["y", "x"], Y),
            lon_mesh=(["y", "x"], X),),

        coords=dict(lon=(["x"], lons.values),
                    lat=(["y"], lats.values),),

        attrs=dict(description="setup files for SAG_experiment.py.",
                   index_lat=(indices['lat'].start, indices['lat'].stop),
                   index_lon=(indices['lon'].start, indices['lon'].stop)))

    ds.to_netcdf(output_path)

#%%
###############################################################################
# Getting my data saved for simulations
###############################################################################
'''
Ecuador - 
indices = {'lat': range(1434, 1554), 'lon': range(2425, 2545)}
Study region -
indices = {'lat': range(1184, 1804), 'lon': range(2065, 2665)}
'''

dataname = '25N25S'

if dataname == '35N35S':
    study_region = (-125, -55, -35, 35)
    indices = {'lat': range(1045, 1960), 'lon': range(1945, 2785)}
elif dataname == '25N25S':
    study_region = (-115, -65, -25, 25)
    indices = {'lat': range(1184, 1804), 'lon': range(2065, 2665)}
elif dataname == 'whole':
    indices = {'lat': range(3059), 'lon': range(4322)}
    
print('Generating coastal_fields_'+dataname+'.nc')

file_path = "/Users/renjiongqiu/Documents_local/Thesis/data/sources/psy4v3r1-daily_2D_2019-10-25.nc"
ds = xr.load_dataset(file_path)
outfile = '/Users/renjiongqiu/Documents_local/Thesis/data/sources/coastal_fields_'+dataname+'.nc'

generate_dataset(file_path, indices, outfile)