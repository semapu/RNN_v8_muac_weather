"""
Author: Sergi Mas Pujol
Last update: 10/11/2020

Python version: 3.6
"""

import pygrib  # Available in the environment PhD_GPU
import sys


def extract_weather_information_from_list_days_and_timestamps(list_days,
                                                              start_time_samples, end_time_samples,
                                                              traffic_volume,
                                                              num_weather_features: int,
                                                              num_metric_per_weather_feature: int,
                                                              X):
    """

    Args:
        list_days: [List] List of days from which extract the features in format DD/MM/YYYY
        start_time_samples: [List] List of timestamps from which extract the features in format HHMMSS
        end_time_samples: [List] List of timestamps from which extract the features in format HHMMSS
        traffic_volume: [List] List of volumes from which extract the features in format XXXX
        num_weather_features: [int] Number of weather's features in the grib files
        num_metric_per_weather_feature: [int] Number of variables for each weather feature {min, avg, max}
        X: [List-List] Final variable where the information will be save

    Returns:
        X: [List-List] Final variable where the information has been save according to the input days and timestamps
    """

    # Identify if "traffic_volume" is a unique a string or a list of strings
    # If it is a unique String -> Create a list repeating the given value
    if isinstance(traffic_volume, str):
        traffic_volume = [traffic_volume] * list_days.shape[0]

    # Read the first required grib file
    current_sector_name = traffic_volume[0]
    current_month = list_days[0].split('/')[1]
    grib = pygrib.open('./Export_weather_information/' + current_sector_name + '_' + current_month + '_250.grib')
    # grib = pygrib.open('./Export_weather_information_all_features/' + current_sector_name + '_' + current_month + '_250.grib')

    counter_day = 0
    for day, start, end, sector_name in zip(list_days, start_time_samples, end_time_samples, traffic_volume):
        print('day: {day} | start:{start} | TV: {tv}'.format(day=day, start=start, tv=sector_name))

        # If either the month or the sector has changed, load the new grib file
        # Else, continue using the already loaded grib file
        next_month = day.split('/')[1]
        if next_month != current_month or current_sector_name != sector_name:
            current_sector_name = sector_name
            current_month = next_month

            # Load and open the grib file
            grib = pygrib.open('./Export_weather_information/' + current_sector_name + '_' + current_month + '_250.grib')
            # grib = pygrib.open('./Export_weather_information_all_features/' + current_sector_name + '_' + current_month + '_250.grib')

        # Extract the required features
        day_int_format = int(day.split('/')[0])
        timestamp = int(start[0:2])

        grb = grib.select()[num_weather_features * 24 * (day_int_format - 1) + num_weather_features * timestamp:
                            num_weather_features * 24 * (day_int_format - 1) + num_weather_features * timestamp + num_weather_features]

        for i in range(num_weather_features):
            # print('month: ', grb[i]['month'])
            # print('day:', grb[i]['day'])
            # print('hour:', grb[i]['hour'])
            # print('parameterName: ', grb[i]['parameterName'])

            # Sanity check to ensure the correct features are being extracted
            if int(grb[i]['month']) == int(current_month) and int(grb[i]['day']) == day_int_format and int(grb[i]['hour']) == timestamp:
                X[counter_day, 0:30, num_metric_per_weather_feature * i + 0] = grb[i]['minimum']
                X[counter_day, 0:30, num_metric_per_weather_feature * i + 1] = grb[i]['average']
                X[counter_day, 0:30, num_metric_per_weather_feature * i + 2] = grb[i]['maximum']
            else:
                sys.exit('Error between the extracted weather feature and the sample information')

        counter_day += 1

    return X
