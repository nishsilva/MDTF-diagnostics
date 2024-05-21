import os
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs  # Assuming ccrs is Cartopy's Coordinate Reference Systems
import xesmf as xe
import numpy as np

print("Libs imported!")

# Make sure these environment variables are correctly set
input_path = os.environ["ZOS_FILE"]
zos_var_name = os.environ["zos_var"]
time_coord_name = os.environ["time_coord"]  # Replace 'time_coord' with the correct env var name

############
#model data#
############

model_dataset = xr.open_dataset(input_path)
zos_data = model_dataset[[zos_var_name]]
model_mean_zos = zos_data.mean(dim=time_coord_name)

##########
#obs data#
##########

input_path = "{OBS_DATA}/dtu22mdt.nc".format(**os.environ)

obs_dataset = xr.open_dataset(input_path)    # command to load the netcdf file
dtu_mdt = obs_dataset['mdt']

#####################
#regridding function#
#####################

## a regriding function that automatically identifies the var name

def regrid_to_gr(ds, var, delta_lat, delta_lon, w, e, s, n, method):
    if isinstance(var, str):
        dr = ds[var].squeeze()
        original_var_name = var  # Store the original variable name
    else:
        dr = ds.squeeze()
        original_var_name = list(ds.data_vars)[0]  # Assume the first variable if not specified
    
    # Define the output grid
    ds_out = xr.Dataset({
        'latitude': (['latitude'], np.arange(s, n + delta_lat, delta_lat)),
        'longitude': (['longitude'], np.arange(w, e + delta_lon, delta_lon)),
    })
    
    # Create the regridder and apply it
    regridder = xe.Regridder(ds, ds_out, method) 
    dr_out = regridder(dr)
    
    # Rename the regridded variable
    renamed_dr_out = dr_out.rename({original_var_name: f"{original_var_name}_rg"})
    
    return renamed_dr_out


#########################################
# determine the grid specs for the model#
#########################################

#latidude spacing
delta_lat = (zos_data['latitude'].max() - zos_data['latitude'].min())/(zos_data['latitude'].count()-1.)

#longitude spacing
delta_lon = (zos_data['longitude'].max() - zos_data['longitude'].min())/(zos_data['longitude'].count()-1.)

w = zos_data['longitude'].min()
e = zos_data['longitude'].max()
s = zos_data['latitude'].min()
n = zos_data['latitude'].max()


#############
#Saving Data#
#############

WORK_DIR = os.environ['WORK_DIR']
#out_dir = os.path.join(WORK_DIR, "model")
#assert os.path.isdir(out_dir), f'{out_dir} not found'

out_path = "{WORK_DIR}/obs/netCDF/SL_obs.nc".format(**os.environ)
obs_dataset.to_netcdf(out_path)  # Saving mean instead of 'mdt'

out_path = "{WORK_DIR}/model/netCDF/SL_model.nc".format(**os.environ)
model_mean_zos.to_netcdf(out_path)  # Saving mean instead of 'mdt'


print(model_mean_zos)

# regriddgin obs data
ds_obs_dtu_rg = regrid_to_gr(obs_dataset, obs_dataset.mdt, delta_lat, delta_lon, w, e, s, n, 'bilinear')

# regridding model data as well
#it appears that if we do not regrid both the model and obs data it will not provide a 2D map potential dataset once the error calculation is done.

model_mean_zos_rg = regrid_to_gr(model_mean_zos, model_mean_zos[zos_var_name], delta_lat, delta_lon, w, e, s, n, 'bilinear')

########################
#Diagnsotic Calculation#
########################

# Define 'error' appropriately based on your requirement
error = ds_obs_dtu_rg.mdt_rg - model_mean_zos_rg.zos_rg

#######################
#Saving Regridded Data#
#######################

out_path = "{WORK_DIR}/obs/netCDF/SL_obs_rg.nc".format(**os.environ)
ds_obs_dtu_rg.to_netcdf(out_path)  # Saving mean instead of 'mdt'

out_path = "{WORK_DIR}/model/netCDF/SL_model_rg.nc".format(**os.environ)
model_mean_zos_rg.to_netcdf(out_path)  # Saving mean instead of 'mdt'

out_path = "{WORK_DIR}/model/netCDF/SL_error.nc".format(**os.environ)
error.to_netcdf(out_path)  # Saving mean instead of 'mdt'

print("test")
print(error)

##################
#Plotting Figures#
##################

#plot model mean

fig = plt.figure(figsize=(12, 16), tight_layout=True)
ax = fig.add_subplot(3, 1, 1, projection=ccrs.Robinson())
model_mean_zos.zos.plot(ax=ax) 

plot_path = "{WORK_DIR}/model.png".format(model_or_obs="model", **os.environ)
plt.savefig(plot_path, bbox_inches='tight')

# plot error

fig = plt.figure(figsize=(12, 16), tight_layout=True)
ax = fig.add_subplot(3, 1, 1, projection=ccrs.Robinson())
error.plot(ax=ax) 

plot_path = "{WORK_DIR}/error.png".format(model_or_obs="model", **os.environ)
plt.savefig(plot_path, bbox_inches='tight')

def plot_and_save_figure(model_or_obs, title_string, dataset):
    # initialize the plot
    plt.figure(figsize=(12,6))
    plot_axes = plt.subplot(1,1,1)
    # actually plot the data (makes a lat-lon colormap)
    dataset.plot(ax=plot_axes)
    plot_axes.set_title(title_string)

    # build the base file path
    base_plot_path = "{WORK_DIR}/{model_or_obs}/PS/SL_trial_{model_or_obs}_plot".format(
        model_or_obs=model_or_obs, **os.environ
    )

    # check if the file exists and find a new name if necessary
    count = 0
    plot_path = base_plot_path + ".eps"
    while os.path.exists(plot_path):
        count += 1
        plot_path = base_plot_path + str(count) + ".eps"

    # save the plot with the new file name
    plt.savefig(plot_path, bbox_inches='tight')

# Usage example
# plot_and_save_figure("model", "Title Here", dataset)


# set an informative title using info about the analysis set in env vars
title_string = "{CASENAME}: Trial {zos_var} ({FIRSTYR}-{LASTYR})".format(**os.environ)
# Plot the model data:
plot_and_save_figure("model", title_string, model_mean_zos.zos)

plot_and_save_figure("model", title_string, error)

title_string = "{CASENAME}: Trial {zos_var} ({FIRSTYR}-{LASTYR} Regridded)".format(**os.environ)
plot_and_save_figure("obs", title_string, ds_obs_dtu_rg.mdt_rg)