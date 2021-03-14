"""
Author: Sergi Mas Pujol
Last update: 24/11/2020

Python version: 3.6 
"""


import pandas as pd
import math
import numpy as np
import sys
from matplotlib.patches import Polygon

from datetime import datetime

import utils_timeProcessing as timeProcessing

from utils_timeProcessing import substract_minutes_given_HHSSMM


def extract_list_traffic_volumes_with_regulations(traffic_volumes):
    """
    Usage:
        extract_list_traffic_volumes_with_regulations(REGULATIONS.iloc[:]["traffic_volume"].values)

    Description:
        This function return the a list with the traffic volumes in a dataframe.
        The dataframe must belong to the file: DATEX_DATEY_REGULATIONS.csv

    Args:
        traffic_volumes: [list-str] List of string containing the full name of the different TVs

    Returns:
        unique_volumes: [list] List the salt 4 letters of the TV in the input list
    """

    unique_volumes = set()

    for volume in traffic_volumes:
        unique_volumes.add(volume[4:8])

    return list(unique_volumes)


def compute_coordinates_subregion_extraction_weather_features(lon_centre, lat_centre, width_region_meters,
                                                              height_region_meters):
    """
    Returns the north, west, south, east coordinates of a sub-region in decimal degrees.
    It was developed to work with the dataset ERA5 from ECWMF.

    Note: 1 degree is (more or less) equivalent to 100km or 100.000 meters

    Args:
        lon_centre: [float] Central longitude in decimal hours for the sub-region
        lat_centre: [float] Central latitude in decimal hours for the sub-region
        width_region_meters: [int] Width of the sub-region in meters
        height_region_meters: [int] Height of the sub-region in meters

    Returns:
        north: [float] North coordinate in decimal hours
        west: [float] West coordinate in decimal hours
        south: [float] South coordinate in decimal hours
        east: [float] East coordinate in decimal hours
    """

    # Compute the increment/decrease of the width from the central coordinates
    half_width_region_degrees = int((width_region_meters / 100000) / 2)

    # Compute the increment/decrease of the hight from the central coordinates
    half_hight_region_degrees = int((height_region_meters / 100000) / 2)

    # Compute the final coordinates for the sub-region
    north = lat_centre + half_hight_region_degrees
    south = lat_centre - half_hight_region_degrees
    west = lon_centre - half_width_region_degrees
    east = lon_centre + half_width_region_degrees

    return north, west, south, east


def compute_central_coordinates_sector(lons_primitive_regions, lats_primitive_regions):
    """
    Given a list with all the coordinates required to define a sector, the function extracts the central coordinates.
    These central coordinates are computed sorting all the coordinates, and then extracting the values in the middle.

    Args:
        lons_primitive_regions: [List] List of longitudes
        lats_primitive_regions: [List] List of latitudes

    Returns:
        lon_centre: [int] Central longitude for the sector
        lat_centre: [int] Central latitude for the sector

    """

    # Sort the lon/lats to fins the central point
    lons_primitive_regions.sort()
    lats_primitive_regions.sort()

    # Fins the coordinates from the central point
    lon_centre = lons_primitive_regions[int(np.floor(len(lons_primitive_regions) / 2))]
    lat_centre = lats_primitive_regions[int(np.floor(len(lats_primitive_regions) / 2))]

    return lon_centre, lat_centre


def extract_coordinates_primitive_sectors(primitiveSectors, volumes):
    """
    Function which extracts the coordinates of the given primitive volumes.

    Args:
        primitiveSectors: [str-vector] Names of the primitive sector which compose the sector of study
        volumes: [str-vector] Names of theextractNames_primitiveSectors_fromRNESTname regions that are used to describe the primitive regions

    Returns:
        lons_primitiveRegions: [List] Coordinates in decimal hours to define the primitive volumes
        lats_primitiveRegions: [List] Coordinates in decimal hours to define the primitive volumes
    """

    """ Loading the required files """
    # Path to the files requires for extracting the information
    airspace_file = "./sectorsTAGS/Airspace_1906.spc"
    sectors_are_file = "./sectorsTAGS/sectors_1906.are"

    # Read the Airspace file
    col_names = ["col1", "col2", "col3", "col4", "col5"]
    airspace_file = pd.read_table(airspace_file, sep=';', header=None, names=col_names)  # The 'C' engine is faster

    # Read the .ara file - corresponding with the airspace sectors
    sectors_are = pd.read_csv(sectors_are_file, sep=" ", header=None, engine='c',
                              low_memory=False)  # The 'C' engine is faster

    """ Compute the different patches """
    lats_primitiveRegions = []
    lons_primitiveRegions = []

    # Inside if the TV is a primitive one
    if len(primitiveSectors) == 1:

        primitiveRegion_inside_areFile = sectors_are[sectors_are.loc[:, 14] == volumes[0]]
        start_index_primitiveRegio = primitiveRegion_inside_areFile.index[0]
        num_vertices_defining_primitiveRegion = int(primitiveRegion_inside_areFile[0].values[0])

        # Extract the coordinates of each smollest region compasing the elementary sector
        for j in range(1, num_vertices_defining_primitiveRegion + 1):
            lats_primitiveRegions.append(float("{0:.2f}".format(
                sectors_are.loc[start_index_primitiveRegio + j, 0] / 60)))  # From decimal minutes to decimal hours
            lons_primitiveRegions.append(float("{0:.2f}".format(
                sectors_are.loc[start_index_primitiveRegio + j, 1] / 60)))  # From decimal minutes to decimal hours

    # Else, if the TV is a collapsed one
    else:
        for i in range(len(primitiveSectors)):

            # Iterate through the different primitive volumes
            for primitiveRegion in volumes[i]:
                primitiveRegion_inside_areFile = sectors_are[sectors_are.loc[:, 14] == primitiveRegion]
                start_index_primitiveRegio = primitiveRegion_inside_areFile.index[0]
                num_vertices_defining_primitiveRegion = int(primitiveRegion_inside_areFile[0].values[0])

                # Extract the coordinates of each smollest region compasing the elementary sector
                for j in range(1, num_vertices_defining_primitiveRegion + 1):
                    lats_primitiveRegions.append(float("{0:.2f}".format(sectors_are.loc[
                                                                            start_index_primitiveRegio + j, 0] / 60)))  # From decimal minuts to decimal hours
                    lons_primitiveRegions.append(float("{0:.2f}".format(sectors_are.loc[
                                                                            start_index_primitiveRegio + j, 1] / 60)))  # From decimal minuts to decimal hours

    return lons_primitiveRegions, lats_primitiveRegions


def extractPatches_primitiveSectors(primitiveSectors, volumes, basemap):
    """
    Function which extracts the coordinates of the given primitive volumes. 

    Input:
        * primitiveSectors[str-vector] -> Names of the primitive sector which compose the sector of study
        * volumes[str-vector] -> Names of theextractNames_primitiveSectors_fromRNESTname regions that are used to describe the primitive regions
        * basemap[Basemap] -> Where the information will be ploted
        
    Output:
        * patches[Polygon] -> Array of polygon requiered to plot the figures using "PatchCollection"
        * lons_primitiveRegions[List] -> Coordinates in decimal hours to define the primitive volumes
        * lats_primitiveRegions[List] -> Coordinates in decimal hours to define the primitive volumes
    """


    """ Loading the requiered files """
    # Path to the files requieres for extracting the informatio
    airspace_file = "./sectorsTAGS/Airspace_1906.spc"
    sectors_are_file = "./sectorsTAGS/sectors_1906.are"

    # Read the Airspace file
    col_names = ["col1", "col2", "col3", "col4", "col5"]
    airspace_file = pd.read_table(airspace_file, sep=';', header=None, names=col_names) # The 'C' engine is faster

    # Read the .ara file - corresponding with the airspace sectors
    sectors_are = pd.read_csv(sectors_are_file, sep=" ", header=None, engine='c', low_memory=False) # The 'C' engine is faster
    
    """ Compute the different patches """
    patches = []

    if(len(primitiveSectors) ==1 and len(volumes) == 1):
        # Lats/Lons vertices of each region
        lats_primitiveRegions = []
        lons_primitiveRegions = []

        primitiveRegion_inside_areFile = sectors_are[sectors_are.loc[:,14] == volumes[0]]
        start_index_primitiveRegio = primitiveRegion_inside_areFile.index[0]
        num_vertices_defining_primitiveRegion = int(primitiveRegion_inside_areFile[0].values[0])

        # Extract the coordinates of each smollest region compasing the elementary sector
        for j in range(1, num_vertices_defining_primitiveRegion+1):

            lats_primitiveRegions.append(float("{0:.2f}".format(sectors_are.loc[start_index_primitiveRegio+j,0]/60))) # From decimal minuts to decimal hours
            lons_primitiveRegions.append(float("{0:.2f}".format(sectors_are.loc[start_index_primitiveRegio+j,1]/60))) # From decimal minuts to decimal hours
        
            x, y = basemap( lons_primitiveRegions, lats_primitiveRegions )
            xy = zip(x,y)
            poly = Polygon( list(xy))
            patches.append(poly)

    else:
        for i in range(len(primitiveSectors)):
            # Lats/Lons vertices of each region
            lats_primitiveRegions = []
            lons_primitiveRegions = []

            # Iterate through the different primitive volumes
            for primitiveRegion in volumes[i]:
                primitiveRegion_inside_areFile = sectors_are[sectors_are.loc[:,14] == primitiveRegion]
                start_index_primitiveRegio = primitiveRegion_inside_areFile.index[0]
                num_vertices_defining_primitiveRegion = int(primitiveRegion_inside_areFile[0].values[0])

                # Extract the coordinates of each smollest region compasing the elementary sector
                for j in range(1, num_vertices_defining_primitiveRegion+1):

                    lats_primitiveRegions.append(float("{0:.2f}".format(sectors_are.loc[start_index_primitiveRegio+j,0]/60))) # From decimal minuts to decimal hours
                    lons_primitiveRegions.append(float("{0:.2f}".format(sectors_are.loc[start_index_primitiveRegio+j,1]/60))) # From decimal minuts to decimal hours
                
                    x, y = basemap( lons_primitiveRegions, lats_primitiveRegions )
                    xy = zip(x,y)
                    poly = Polygon( list(xy))
                    patches.append(poly)

    return patches


def extractNames_primitiveSectors_fromRNESTname(sector_study):
    """
    Fucntion to extract the names of the primitve region given the name of a sector in the AIRAC.

    Input:
        * sector_study[str] -> Last 4 digits of the sector (starting as EDYY)

    Output:
        * primitiveSectors[str-vector] -> Names of the primitive sector which compose the sector of study
        * volumes[str-vector] -> Names of the regions that are used to describe the primitive regions
    """

    """ Loading the requiered files """
    # Path to thew files requieres for ploting secotr from R-NEST
    airspace_file = "./sectorsTAGS/Airspace_1906.spc"
    # sectors_are_file = "./sectorsTAGS/sectors_1906.are"
    sectors_sls_file = "./sectorsTAGS/sectors_1906.sls"

    # Read the Airspace file
    col_names = ["col1", "col2", "col3", "col4", "col5"]
    airspace_file = pd.read_table(airspace_file, sep=';', header=None, names=col_names) # The 'C' engine is faster

    # Read the .ara and .sls files - corresponding with the airspace sectors
    # sectors_are = pd.read_csv(sectors_are_file, sep=" ", header=None, engine='c', low_memory=False) # The 'C' engine is faster
    sectors_sls = pd.read_csv(sectors_sls_file, sep=" ", header=None, engine='c', low_memory=False) # The 'C' engine is faster

    """ Extract the coordinates of the smallest regions """
    # Extract the smallest reagion that compose the sector of study
    sectors_sls_hotspotVolume = sectors_sls[sectors_sls.loc[:,0] == 'EDYY'+sector_study]
    # Extract the names of the smallest regions
    volumes = sectors_sls_hotspotVolume.loc[:,2].values

    if(len(volumes) != 0):
        print('Inside if len(volumes) = 0')
        return [sector_study], volumes
    
    # Check if the search of the volume in .sls is empty
    # If empty this implies that it is a collapsed sector
    # We have to search the smallest regions is airspace file
    # if(len(volumes) == 0):
    else:
        primitiveSectors = []
        num_elementarySector = int(airspace_file[airspace_file['col2'] == 'EDYY'+sector_study]['col5'].values[0])
        index_collapsetSector = int(airspace_file[airspace_file['col2'] == 'EDYY'+sector_study].index[0])

        for i in range(index_collapsetSector+1, index_collapsetSector+1+num_elementarySector):
            # We are only saving the final part of the sector's name
            # The elements that change between sectors
            primitiveSectors.append(airspace_file.loc[i]['col2'][4:])

        volumes = []
        for volume in primitiveSectors:
            sectors_sls_hotspotVolume = sectors_sls[sectors_sls.loc[:,0] == 'EDYY'+volume]
            volumes.append(sectors_sls_hotspotVolume.loc[:,2].values)

        return primitiveSectors, volumes


def coordsAircraft_timestamp(latBeginSeg, lonBeginSeg, latEndSeg, lonEndSeg, relativePortionTime):
    """
    Given the relative proportion of time and coordinates begin/end segment, this function computes
    the coordinates corresponding to the same proportion of space in the interval.
    
    Input:
        latBeginSeg -> Decimal minutes
        lonBeginSeg -> Decimal minutes
        latEndSeg -> Decimal minutes
        relativePortionTime -> Decimal minutes
    Output:
        lat_aircraft -> Decimal hours
        lon_aircraft -> Decimal hours
    """

    lats_diff = latEndSeg/60 - latBeginSeg/60
    lons_diff = lonEndSeg/60 - lonBeginSeg/60

    lat_aircraft = latBeginSeg/60 + lats_diff*relativePortionTime
    lon_aircraft = lonBeginSeg/60 + lons_diff*relativePortionTime
    
    return lat_aircraft, lon_aircraft


def speed_endOfLine(latBeginSeg, latEndSeg, lonBeginSeg, lonEndSeg, coords_Aircraft, speed_normalized):
    """
    Function that computes the end of the line expressing the speed of the aircraft.
    It computes the heading of the aircraft between the horizontal line and the segment. Then,
    used this angle to compute the varation of lat/lon necessary to express the speed of the aircraft,
    given the current location of the aircraft.
    
    Input:
        latBeginSeg -> Decimal minutes
        latEndSeg -> Decimal minutes
        lonBeginSeg -> Decimal minutes
        lonEndSeg -> Decimal minutes
        coords_Aircraft = (lat, lon) -> Decimal hours
        speed_normalized -> Should be between [0, 1]. But you can pass any value
    
    Output:
        newLat - > Decimal hours (current location aircraft + variation acording to speed using the heading)
        newLon -> Decimal hours (current location aircraft + variation acording to speed using the heading)
    """
    # Segment with longitude zero -> Strat/end same location
    if(latBeginSeg == latEndSeg or lonBeginSeg == lonEndSeg):
        return coords_Aircraft
    else:
        # Compute the parameters of the line that connects both coordinates 
        m = (float(latEndSeg/60 - latBeginSeg/60))/(lonEndSeg/60 - lonBeginSeg/60)
        
        # Angle (degrees) between horizontal and flight tragectory
        angle = math.degrees(math.atan(m))
        
        # Given the angle and normalized speed -> Compute the cords of "end line"
        ca = speed_normalized * math.cos(math.radians(angle))
        co = speed_normalized * math.sin(math.radians(angle))

        lat_aircraft = coords_Aircraft[0] # Latitude of the aircraft
        lon_aircraft = coords_Aircraft[1] # Longitude of the aircraft  
        
        if(latEndSeg>latBeginSeg):
            newLat = lat_aircraft + abs(co)
        else:
            newLat = lat_aircraft - abs(co)
        
        if(lonEndSeg>lonBeginSeg):
            newLon = lon_aircraft + abs(ca)
        else:
            newLon = lon_aircraft - abs(ca)
    
        return newLat,newLon


def abs_vertical_speed(FLBeginSegment, FLEndSeg, timeSegment_hours):
    """
    Compute the absolute vertical speed of an aircraft given a segment of information.

    Input:
        * FL_begin [int]-> flight level at the beginning of the segment
        * FL_end [int] -> flight level at the end of the segment
        * timeSegment_hours [float]-> interval of time during the information was recorded 
    
    Output:
        * abs_vertical_speed [float] -> Absolute vertical speed in NM/min {np.abs(FLEndSeg - FLBeginSegment) / (timeSegment_hours)*60}
    """
    
    return np.abs(FLEndSeg - FLBeginSegment) / (timeSegment_hours)*60


def extractFlight_given_timestamp(trajectoriesFile, timeStamp, FL_begin,
                                  maxValue_normalization_speed, maxValue_normalization_vertical_speed):
    """
    Load all the trajectories over MUAC in the given date and (if specified) the lower FL of interest (lower bound).
        * The date is included in the file's name

    Input:
        trajectoriesFile [str] -> Trajectory file associated with the date and timestamp of interest
        timeStamp [int] -> Format HHMMSS
        FL_begin -> The sigment where the FLBeginSegment > FL_begin will be filtered
        maxValue_normalization_speed [int/float] -> Upper bound of the normalizaation
        maxValue_normalization_vertical_speed -> Upper bound of the normalizaation

    Output:
        * flights_timestamp -> Data Frame with the filter sigments and the aditional metrics computed
    """
    allTrajectories = pd.read_csv(trajectoriesFile, sep=" ", header=None, engine='c', low_memory=False) # The 'C' engine is faster
    
    # Security check: Very all the requiered files are reachable
    if (len(allTrajectories.index.values) == 0):
        sys.exit('Missing the .are files')  

    allTrajectories.columns = ['segID', 'originFlight', 'destiFlight', 
                                'aircraftType', 'timeBeginSeg', 'timeEndSeg', 
                                'FLBeginSegment', 'FLEndSeg', 'status', 
                                'callSign', 'dateBeginSeg', 'dateEndSeg',
                                'latBeginSeg', 'lonBeginSeg', 
                                'latEndSeg', 'lonEndSeg', 
                                'flightID', 'sequence', 'segLength', 'parity/color']

    """
    Obtaining segments which contains the given timestamp
        * Return segments where the timestamp given belong to the interval
        
    Tipically, the segments have a length of 20-25 NM. And the mean speed of the aircraft is 400 NM/hour
        * Mean duration of segemnts: 400 / allTrajectories["segLength"].mean() = 0.529 hours = 3.1779 mins
        * We are going to use an interval of 1 hours (given the timestemps) for searching the segemnts 
            that contians the timestamp (enough "wid e")

    """
    # A lower bound FL was specified
    if (FL_begin == None):
        flights_timestamp = allTrajectories.loc[(allTrajectories["timeBeginSeg"] >= (timeStamp - 30000)) & 
                                                (allTrajectories["timeEndSeg"] <= (timeStamp + 30000)) &
                                                (allTrajectories["timeBeginSeg"] <= timeStamp) &
                                                (allTrajectories["timeEndSeg"] >= timeStamp)]
    # A lower bound FL was NOT specified.
    # The segment belonging to a FL lower than FL_begin will be filtered
    else:
        flights_timestamp = allTrajectories.loc[(allTrajectories["timeBeginSeg"] >= (timeStamp - 30000)) & 
                                                (allTrajectories["timeEndSeg"] <= (timeStamp + 30000)) &
                                                (allTrajectories["timeBeginSeg"] <= timeStamp) &
                                                (allTrajectories["timeEndSeg"] >= timeStamp) & 
                                                (allTrajectories["FLBeginSegment"] >= int(FL_begin))]

    timestamp_datetime_object = datetime.strptime(str(timeStamp), '%H%M%S')
    hour = str("{:02d}".format(timestamp_datetime_object.hour))
    minutes = str("{:02d}".format(timestamp_datetime_object.minute))
    seconds = str("{:02d}".format(timestamp_datetime_object.second))
    timeStamp_tail = int(substract_minutes_given_HHSSMM(str(hour + ":" + minutes + ":" + seconds), 5))

    # Computing the proportion of time between segment started and time of study
    flights_timestamp.loc[:, 'relativePortionTime'] = flights_timestamp.apply(lambda x: timeProcessing.relativePortionTime(x.timeBeginSeg, x.timeEndSeg, timeStamp), axis=1)

    flights_timestamp.loc[:, 'relativePortionTime_tail'] = flights_timestamp.apply(lambda x: timeProcessing.relativePortionTime(x.timeBeginSeg, x.timeEndSeg, timeStamp_tail), axis=1)

    # Current location of the aircraft
    # Given the relative position in the interval of time.
    # ASSUMING: constant velocity. 90% of interval time -> 90% of the segment lenght
    flights_timestamp.loc[:, 'coords_Aircraft'] = flights_timestamp.apply(lambda x: coordsAircraft_timestamp(x.latBeginSeg, x.lonBeginSeg, x.latEndSeg, x.lonEndSeg, x.relativePortionTime), axis=1)

    flights_timestamp.loc[:, 'coords_Aircraft_tail'] = flights_timestamp.apply(lambda x: coordsAircraft_timestamp(x.latBeginSeg, x.lonBeginSeg, x.latEndSeg, x.lonEndSeg, x.relativePortionTime_tail), axis=1)


    # Convert timeBeginSegment from HHMMSS to HH
    flights_timestamp.loc[:, 'timeBeginSeg_hours'] = flights_timestamp['timeBeginSeg'].apply(timeProcessing.from_HHMMSS_to_HH)
    # Convert timeEndSegment from HHMMSS to HH
    flights_timestamp.loc[:, 'timeEndSeg_hours'] = flights_timestamp['timeEndSeg'].apply(timeProcessing.from_HHMMSS_to_HH)
    # Compute time inside each segment (in HOURS)
    flights_timestamp.loc[:, 'timeSegment_hours'] = flights_timestamp.apply(lambda x: timeProcessing.intervalTime_segment(x.timeBeginSeg_hours, x.timeEndSeg_hours), axis=1)

    # Compute the speed of each aircraft along each segment
    flights_timestamp.loc[:, 'speed'] = flights_timestamp['segLength'] / flights_timestamp['timeSegment_hours']
    # By default the normalization of the speed is between [0,1]
    flights_timestamp.loc[:, 'speed_normalized'] = (flights_timestamp['speed'] - flights_timestamp['speed'].min())/(flights_timestamp['speed'].max()-flights_timestamp['speed'].min())*maxValue_normalization_speed


    flights_timestamp.loc[:, 'speed_normalized_tail'] = (flights_timestamp['speed'] - flights_timestamp['speed'].min())/(flights_timestamp['speed'].max()-flights_timestamp['speed'].min())*20



    # Express the horizontal speed (lenght of the line)
    flights_timestamp.loc[:, 'speed_endOfLine'] = flights_timestamp.apply(lambda x: speed_endOfLine(x.latBeginSeg, x.latEndSeg, x.lonBeginSeg, x.lonEndSeg, x.coords_Aircraft, x.speed_normalized), axis=1)


    flights_timestamp.loc[:, 'speed_endOfLine_tail'] = flights_timestamp.apply(lambda x: speed_endOfLine(x.latBeginSeg, x.latEndSeg, x.lonBeginSeg, x.lonEndSeg, x.coords_Aircraft_tail, x.speed_normalized), axis=1)



    # Compute the absolute vertical spped of the aircraft (Nm / min) - In reality used ft/min
    flights_timestamp.loc[:, 'abs_vertical_speed'] = flights_timestamp.apply(lambda x: abs_vertical_speed(x.FLBeginSegment, x.FLEndSeg, x.timeSegment_hours), axis=1)
    # Normalized vertical speed
    flights_timestamp.loc[:, 'abs_vertical_speed_normalized'] = (flights_timestamp['abs_vertical_speed'] - flights_timestamp['abs_vertical_speed'].min())/(flights_timestamp['abs_vertical_speed'].max()-flights_timestamp['abs_vertical_speed'].min())*maxValue_normalization_vertical_speed

    return flights_timestamp
