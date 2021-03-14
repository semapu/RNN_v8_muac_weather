"""
Author: Sergi Mas-Pujol
Last update: 30/11/2020


This file can be used to plot a specific weather feature from the API:
https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-pressure-levels?tab=form

The live calls to the API are very slow. Therefore, I will used the pre-downloaded files.
Saved at './Export_weather_information'.

Inside the grb file the order is:
    1) Weather features,
    2) Days inside the month,
    3) Timestamps for each day.

Steps:
1) Given a air traffic sector and a valid month, load the grib file
2) Select the information for the given day (all features and timestamps)
3) Iterate de day by hours
    1) Extract the lat/lons from the border of the region
    2) Create the mesh given the previously extract lat/lons
    3) Create the Basemap
    4) Plot the data from the .grib file into the Basemap
"""

import numpy as np
import pygrib
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap, shiftgrid
from utils_airspaceConfiguration_aircraftCharacteristics import extractNames_primitiveSectors_fromRNESTname, \
                                                                extract_coordinates_primitive_sectors, \
                                                                compute_central_coordinates_sector

# Dictionary defining all the available weather features
weather_features = {
    'divergence': 0,
    'fraction_of_cloud_cover': 1,
    'geopotential': 2,
    'ozone_mass_mixing_ratio': 3,
    'potential_vorticity': 4,
    'relative_humidity': 5,
    'specific_cloud_ice_water_content': 6,
    'specific_cloud_liquid_water_content': 7,
    'specific_humidity': 8,
    'specific_rain_water_content': 9,
    'specific_snow_water_content': 10,
    'temperature': 11,
    'u_component_of_wind': 12,
    'v_component_of_wind': 13,
    'vertical_velocity': 14,
    'vorticity': 15,
}


FLIGHT_LEVEL_hPa: int = 250
NUM_WEATHER_FEATURES: int = 16  # 7
weather_feature_study = 'divergence'  # It must be one of the option in the dictionary 'weather_features'
idx_weather_feature_study = weather_features[weather_feature_study]

sector_name: str = 'D6WH'
month: str = '08'  # Options: {06, 07, 08, 09}
day: int = 22  # Options: [1, 30]

# Obtained the coordinates for the Basemap used to plot the information
primitive_sectors, volumes = extractNames_primitiveSectors_fromRNESTname(sector_name)
lons_primitive_regions, lats_primitive_regions = extract_coordinates_primitive_sectors(primitive_sectors, volumes)
lon_centre, lat_centre = compute_central_coordinates_sector(lons_primitive_regions, lats_primitive_regions)


def main():
    # Iterate over all the weather features
    for key in weather_features:
        weather_feature_study = key
        idx_weather_feature_study = weather_features[weather_feature_study]

        # Defin ing the final layout
        fig, axes = plt.subplots(nrows=4, ncols=6, figsize=(15, 15))
        fig.suptitle(key, fontsize=12)

        # Load the corresponding grib file
        grib = pygrib.open('./Export_weather_information/' + sector_name + '_' + month + '_250.grib')

        grb = grib.select()[NUM_WEATHER_FEATURES * 24 * (day - 1):
                            NUM_WEATHER_FEATURES * 24 * (day - 1) + NUM_WEATHER_FEATURES * 24]

        for i, ax in zip(range(idx_weather_feature_study, 24*NUM_WEATHER_FEATURES, NUM_WEATHER_FEATURES),
                         axes.flat):

            print('feature:', grb[i]['parameterName'])
            print('month: ', grb[i]['month'])
            print('day:', grb[i]['day'])
            print('hour:', grb[i]['hour'])

            lons = np.linspace(float(grb[i]['longitudeOfFirstGridPointInDegrees']),
                               float(grb[i]['longitudeOfLastGridPointInDegrees']), int(grb[i]['Ni']))
            lats = np.linspace(float(grb[i]['latitudeOfFirstGridPointInDegrees']),
                               float(grb[i]['latitudeOfLastGridPointInDegrees']), int(grb[i]['Nj']))

            data = grb[i].values
            grid_lon, grid_lat = np.meshgrid(lons, lats)  # regularly spaced 2D grid

            # Creating the Basemap fot the given air space sector
            m = Basemap(width=200000, height=200000,  # width/height in meters
                        resolution='l', projection='tmerc',
                        lon_0=lon_centre, lat_0=lat_centre, ax=ax)  # Central coordinated of the map

            x, y = m(grid_lon, grid_lat)

            cs = m.pcolormesh(x, y, data, shading='flat', cmap=plt.cm.gist_stern_r)

            ax.title.set_text(str(grb[i]['hour']))
            fig.colorbar(cs, ax=ax)

        plt.tight_layout()
        fig.savefig('./Export_weather_information/Images_weather_features/' + str(day) + '_' + month + '_' + sector_name
                    + '_' + weather_feature_study + '.png')
        # plt.show()


# def main():
#
#     # Defin ing the final layout
#     fig, axes = plt.subplots(nrows=4, ncols=6, figsize=(15, 15))
#     fig.suptitle(weather_feature_study, fontsize=12)
#
#     # Load the corresponding grib file
#     grib = pygrib.open('./Export_weather_information/' + sector_name + '_' + month + '_250.grib')
#
#     grb = grib.select()[NUM_WEATHER_FEATURES * 24 * (day - 1):
#                         NUM_WEATHER_FEATURES * 24 * (day - 1) + NUM_WEATHER_FEATURES * 24]
#
#     for i, ax in zip(range(idx_weather_feature_study, 24*NUM_WEATHER_FEATURES, NUM_WEATHER_FEATURES),
#                      axes.flat):
#
#         print('feature:', grb[i]['parameterName'])
#         print('month: ', grb[i]['month'])
#         print('day:', grb[i]['day'])
#         print('hour:', grb[i]['hour'])
#
#         lons = np.linspace(float(grb[i]['longitudeOfFirstGridPointInDegrees']),
#                            float(grb[i]['longitudeOfLastGridPointInDegrees']), int(grb[i]['Ni']))
#         lats = np.linspace(float(grb[i]['latitudeOfFirstGridPointInDegrees']),
#                            float(grb[i]['latitudeOfLastGridPointInDegrees']), int(grb[i]['Nj']))
#
#         data = grb[i].values
#         grid_lon, grid_lat = np.meshgrid(lons, lats)  # regularly spaced 2D grid
#
#         # Creating the Basemap fot the given air space sector
#         m = Basemap(width=200000, height=200000,  # width/height in meters
#                     resolution='l', projection='tmerc',
#                     lon_0=lon_centre, lat_0=lat_centre, ax=ax)  # Central coordinated of the map
#
#         x, y = m(grid_lon, grid_lat)
#
#         cs = m.pcolormesh(x, y, data, shading='flat', cmap=plt.cm.gist_stern_r)
#
#         ax.title.set_text(str(grb[i]['hour']))
#         fig.colorbar(cs, ax=ax)
#
#     plt.tight_layout()
#     # fig.savefig('./Export_weather_information/Images_weather_features/' + str(day) + '_' + month + '_' + sector_name
#     #             + '_' + weather_feature_study + '.png')
#     plt.show()


if __name__ == "__main__":
    main()  # It does not need to be a main(). It can be any function, or multiple functions
