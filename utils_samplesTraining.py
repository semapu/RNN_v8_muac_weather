"""
Author: Sergi Mas Pujol
Last update: 29/05/2020

Python version: 3.6
"""

import pandas as pd
import sys
import numpy as np
from datetime import date
import random
from typing import List

from utils_timeProcessing import from_YYYYMMDDHHMMSS_to_HHMMSS_withTwoDots, from_DDMMYYY_to_YYYYMMDD, \
                                 substract_minutes_given_HHSSMM, add_minutes_given_HHSSMM


def compute_start_end_timestamps_from_days_with_regulations(list_dates_with_regulations,
                                                            start_regulations, end_regulations,
                                                            list_volumes_regulations,
                                                            gap_before_start_time, gap_after_start_time,
                                                            safety_windows):
    """
    Given a list of regulations, it creates new timestamps for the different days associated with and without regulations
        * The generated samples strictly belong to one of the two clases (no regulation or regulation)

    The "gap_before_start_time" and "gap_after_start_time" are used to determine the amount of time which will be
        taken into account when generation each new sample.

    Args:
        list_dates_with_regulations: complete list of all days with regulations to take into account
        start_regulations: list starting timestamps for each previous regulation
        end_regulations: list ending timestamps for each previous regulation
        list_volumes_regulations: list of volumes for each previous regulation
        gap_before_start_time: added before the timestamps
        gap_after_start_time: added after the timestamps
        safety_windows: MINUTES to subtract to the beginning and end of the intervals with regulations

    Returns:
        list_days_regulatios: list expressing the day of the following starting timestamp
        start_time_samples_regulations: list of the starting timestamps created (taking into account the
                                        gap_before_start_time and gap_after_start_time) associated with
                                        regulation
        end_time_samples_regulations: list of the ending timestamps created (taking into account the
                                      gap_before_start_time and gap_after_start_time) associated with
                                      regulation
        volumes_regulations: list of the volumes associated with each regulation
        list_days_no_regulations: list expressing the day of the following starting timestamp
        start_time_samples_no_regulations: list of the starting timestamps created (taking into account the
                                           gap_before_start_time and gap_after_start_time) associated with
                                           NO regulation
        end_time_samples_no_regulations: list of the ending timestamps created (taking into account the
                                         gap_before_start_time and gap_after_start_time) associated with
                                         NO regulation
        volumes_no_regulations: list of the volumes associated with each NO regulation
    """

    list_days_regulations = list()
    start_time_samples_regulations = list()
    end_time_samples_regulations = list()
    volumes_regulations = list()

    list_days_no_regulations = list()
    start_time_samples_no_regulations = list()
    end_time_samples_no_regulations = list()
    volumes_no_regulations = list()

    for day, start_regulation, end_regulation, volume in zip(list_dates_with_regulations, start_regulations,
                                                             end_regulations, list_volumes_regulations):

        # Iterate multiple times to extract more samples
        for _ in range(0, 20):

            # Randomly create the hour and the minutes
            hour = str("{:02d}".format(random.randint(3, 21)))  # From 3 to 21 to avoid extremes (wrap around issue)
            minute = str("{:02d}".format(random.randint(10, 50)))

            # Initialize the timestamp
            random_timestamp = hour + ":" + minute + ":" + '00'

            # Create the initial timestamp - The following function is used to convert the HH:MM:SS into an int
            start_timestamp = add_minutes_given_HHSSMM(random_timestamp, 0)

            # Add a given about to create the ending timestamp
            end_timestamp = add_minutes_given_HHSSMM(random_timestamp, (gap_before_start_time+gap_after_start_time))

            # Find all regulations for that day
            indexes_regulations_same_day = np.where(list_dates_with_regulations == day)

            timestamps_start_for_that_day_with_regulations = start_regulations[indexes_regulations_same_day]
            timestamps_end_for_that_day_with_regulations = end_regulations[indexes_regulations_same_day]
            volumes_for_that_day_with_regulations = list_volumes_regulations[indexes_regulations_same_day]

            # Flag to know if the generated value is valid for a sample with regulation
            flag_regulation = False
            # Flag to know if the timestamp created is invalid - Partially inside
            flag_partially_inside = False

            for start_reg, end_reg, vol in zip(timestamps_start_for_that_day_with_regulations,
                                               timestamps_end_for_that_day_with_regulations,
                                               volumes_for_that_day_with_regulations):

                range_values = range(start_reg + safety_windows * 60, end_reg - safety_windows * 60)

                if vol == volume:
                    # If new timestamps completely inside a regulation period -> Valid sample
                    if int(start_timestamp) in range_values and int(end_timestamp) in range_values:
                        flag_regulation = True

                    # If sample partially inside -> Ignored
                    if (int(start_timestamp) in range_values and int(end_timestamp) not in range_values) or (int(start_timestamp) not in range_values and int(end_timestamp) in range_values):
                        flag_partially_inside = True

            # According to the flag ignore the sample
            if flag_partially_inside:
                continue

            # According to the flag, save the timestamps in the correct list
            elif flag_regulation:
                list_days_regulations.append(day)
                start_time_samples_regulations.append(start_timestamp)
                end_time_samples_regulations.append(end_timestamp)
                print(str(' ' + volume[4:]))
                volumes_regulations.append(str(volume[4:-1]))

            else:
                # Randomly decide if save or nat the sample, to avoid to much values
                random_coin = random.uniform(0, 1)

                if random_coin >= 0.8:
                    list_days_no_regulations.append(day)
                    start_time_samples_no_regulations.append(start_timestamp)
                    end_time_samples_no_regulations.append(end_timestamp)
                    print(str(' ' + volume[4:]))
                    volumes_no_regulations.append(str(volume[4:-1]))
                else:
                    continue

    return [list_days_regulations, start_time_samples_regulations, end_time_samples_regulations, volumes_regulations,
            list_days_no_regulations, start_time_samples_no_regulations, end_time_samples_no_regulations, volumes_no_regulations]


def convert_to_one_hot(y: List[int], c: int) -> List:
    """
    Given the label "y", it return the same list as one-hot vectors taking into account the number of different classes.

    Args:
        y: List of labels
        c: Number of classes

    Returns:
        y_oh: One-hot vector containing the labels

    """
    y_oh = np.eye(c)[y.reshape(-1)]

    return y_oh


def read_REGULATIONS_file(path_to_file):
    """
    Function that given a path to the REGULATIONS file, it return the file (only useful information)

    Input:
        * path_to_file[str] -> Path to the TAGS file

    Output:
        * REGULATIONS[dataframe] -> DataFrame containing the regulations
    """


    REGULATIONS = pd.read_csv(path_to_file, sep="|", header=None, engine='c', low_memory=False, skiprows=[0])
    REGULATIONS.columns = ['target_date',
                    'regulation',
                    'traffic_volume_set',
                    'traffic_volume',
                    'location',
                    'location_type',
                    'start_time',
                    'end_time',
                    'cancel_time',
                    'regulation_duration_minutes',
                    'regulated_flights',
                    'regulation_reason',
                    'regulation_description']

    return REGULATIONS


def extract_hotspotsREGULATIONfile_given_sector_and_date(REGULATIONS, sectorName, date):
    """
    Funtion that given a specific sector and name, extract the CREATED hotspots in the TAGS file.

    Input:
        * REGULATIONS[Dateframe] -> DAtaframe with all the regulations in the REGULATIONS file
        * sectorName[str] -> Name sector to extract CREATED tags
        * date[str-YYYY/MM/DD] -> Date associated to the CRETAED tags

    Outout:
        * TAGS_created[dtaFrame] -> Datafram with the filterd TAGS according to the sector name and date
    """
    try:
        REGULATIONS_hotspot = REGULATIONS.loc[(REGULATIONS["target_date"] == date+' ')&
                                              (REGULATIONS["traffic_volume"] == ' MAS'+sectorName+' ') &
                                              (REGULATIONS["regulation_reason"].isin([' C-ATC Capacity ',  ' R-ATC Routeing ' ]))] 
        
        # start_time - from_YYYYMMDDHHMMSS_to_HHMMSS
        REGULATIONS_hotspot.loc[:, 'start_time_HHMMSS'] = REGULATIONS_hotspot.apply(lambda x: from_YYYYMMDDHHMMSS_to_HHMMSS_withTwoDots(x.start_time.replace(" ", "")), axis=1)
        # end_time - from_YYYYMMDDHHMMSS_to_HHMMSS
        REGULATIONS_hotspot.loc[:, 'end_time_HHMMSS'] = REGULATIONS_hotspot.apply(lambda x: from_YYYYMMDDHHMMSS_to_HHMMSS_withTwoDots(x.end_time.replace(" ", "")), axis=1)

    except ValueError:
        REGULATIONS_hotspot = pd.DataFrame()

    return REGULATIONS_hotspot


def extract_counts_excelFromStatisticsRNEST(counts_fromStats, delta_timesteps):
    """
    Given the filtered counts (FROM STATISTICS IN RNEST) for a specific day and metrics (SINGLE ROW), it groups the values according to "delta_timestep"
        * The "delta_timestep" refers to the time-windows used to group the values
    
    Extract the counts for: {Workload} - Statistics in RNEST


    Input:
        * counts_fromStats[dataFrame-singleRow] -> Row with counts associates to a given day and a single sector
        * delta_timesteps[int] -> Temporal windows used to group the values

    Output:
        * counts[np.array] -> Final grouping of the values
    """

    try:
        counts = np.zeros((int(24*60/delta_timesteps)))
    except:
        print('The "delta_timesteps must be a divider of 60"')
        sys.exit()

    numElements_toSum = delta_timesteps/5 # In the occupancy file the temporal windows is equal to 5 mins

    counter_ellementsAdded = 0
    cummulative_count = 0
    counter = 0

    for value in counts_fromStats.values[0][6:-1]:
        cummulative_count += value
        counter_ellementsAdded+=1
        
        if(counter_ellementsAdded == numElements_toSum):
            counts[counter] = cummulative_count
            
            cummulative_count = 0
            counter_ellementsAdded=0
            counter+=1

    return counts


def readAssociatedFile_fromAIRAC_givenDate(day):
    """
    Given a day, it retunrs the files associates to that day.
        * In the AIRACS there is information from 28 days
        * It returns: {capacity, occupancyCount, entryCount}

    Input:
        * day[str-DD/MM/YYYY]

    Output:
        * normal_capacity[dataFrame] -> Exported file from the AIRAC as a dataFrame
        * regulations[dataFrame] -> Exported file from the AIRAC as a dataFrame
        * oc[dataFrame] -> Exported file from the AIRAC as a dataFrame
        * workload_1_1[dataFrame] -> Workload exported from R-NEST, sliding windows = 1 and integration windows = 1
    """

    split_day = day.split('/')

    if ((date(int(split_day[2]), int(split_day[1]), int(split_day[0])) >= date(2019, 5, 23)) and
            (date(int(split_day[2]), int(split_day[1]), int(split_day[0])) <= date(2019, 6, 19))):
        normal_capacity = pd.read_csv('./Exports_RNEST/Capacity_1906.ncap', sep=";", header=None)
        regulations = pd.read_csv('./Exports_RNEST/RegPlan_1906.nreg', sep=" ", header=None)
        oc = pd.read_csv('./Exports_RNEST/oc_1906.csv', sep=";", header=None, engine='c', low_memory=False,
                         skiprows=[0, 1])
        ec_20_20 = pd.read_csv('./Exports_RNEST/ec_1906_20_20.csv', sep=";", header=None, engine='c', low_memory=False,
                               skiprows=[0, 1])
        ec_60_20 = pd.read_csv('./Exports_RNEST/ec_1906_60_20.csv', sep=";", header=None, engine='c', low_memory=False,
                               skiprows=[0, 1])
        workload_1_1 = pd.read_csv('./Exports_RNEST/complexity_workload_1906_1_1.csv', sep=";", header=None, engine='c',
                                   low_memory=False, skiprows=[0, 1])
        workload_20_20 = pd.read_csv('./Exports_RNEST/complexity_workload_1906_20_20.csv', sep=";", header=None,
                                     engine='c', low_memory=False, skiprows=[0, 1])
        num_conflicts_1_1 = pd.read_csv('./Exports_RNEST/complexity_numConflicts_1906_1_1.csv', sep=";", header=None,
                                        engine='c', low_memory=False, skiprows=[0, 1])
        climbing_1_1 = pd.read_csv('./Exports_RNEST/complexity_climbing_1906_1_1.csv', sep=";", header=None, engine='c',
                                   low_memory=False, skiprows=[0, 1])
        cruising_1_1 = pd.read_csv('./Exports_RNEST/complexity_cruising_1906_1_1.csv', sep=";", header=None, engine='c',
                                   low_memory=False, skiprows=[0, 1])
        descending_1_1 = pd.read_csv('./Exports_RNEST/complexity_descending_1906_1_1.csv', sep=";", header=None,
                                     engine='c', low_memory=False, skiprows=[0, 1])

    elif ((date(int(split_day[2]), int(split_day[1]), int(split_day[0])) >= date(2019, 6, 20)) and
          (date(int(split_day[2]), int(split_day[1]), int(split_day[0])) <= date(2019, 7, 17))):
        normal_capacity = pd.read_csv('./Exports_RNEST/Capacity_1907.ncap', sep=";", header=None)
        regulations = pd.read_csv('./Exports_RNEST/RegPlan_1907.nreg', sep=" ", header=None)
        oc = pd.read_csv('./Exports_RNEST/oc_1907.csv', sep=";", header=None, engine='c', low_memory=False,
                         skiprows=[0, 1])
        ec_20_20 = pd.read_csv('./Exports_RNEST/ec_1907_20_20.csv', sep=";", header=None, engine='c', low_memory=False,
                               skiprows=[0, 1])
        ec_60_20 = pd.read_csv('./Exports_RNEST/ec_1907_60_20.csv', sep=";", header=None, engine='c', low_memory=False,
                               skiprows=[0, 1])
        workload_1_1 = pd.read_csv('./Exports_RNEST/complexity_workload_1907_1_1.csv', sep=";", header=None, engine='c',
                                   low_memory=False, skiprows=[0, 1])
        workload_20_20 = pd.read_csv('./Exports_RNEST/complexity_workload_1907_20_20.csv', sep=";", header=None,
                                     engine='c', low_memory=False, skiprows=[0, 1])
        num_conflicts_1_1 = pd.read_csv('./Exports_RNEST/complexity_numConflicts_1907_1_1.csv', sep=";", header=None,
                                        engine='c', low_memory=False, skiprows=[0, 1])
        climbing_1_1 = pd.read_csv('./Exports_RNEST/complexity_climbing_1907_1_1.csv', sep=";", header=None, engine='c',
                                   low_memory=False, skiprows=[0, 1])
        cruising_1_1 = pd.read_csv('./Exports_RNEST/complexity_cruising_1907_1_1.csv', sep=";", header=None, engine='c',
                                   low_memory=False, skiprows=[0, 1])
        descending_1_1 = pd.read_csv('./Exports_RNEST/complexity_descending_1907_1_1.csv', sep=";", header=None,
                                     engine='c', low_memory=False, skiprows=[0, 1])

    elif ((date(int(split_day[2]), int(split_day[1]), int(split_day[0])) >= date(2019, 7, 18))
          and (date(int(split_day[2]), int(split_day[1]), int(split_day[0])) <= date(2019, 8, 14))):
        normal_capacity = pd.read_csv('./Exports_RNEST/Capacity_1908.ncap', sep=";", header=None)
        regulations = pd.read_csv('./Exports_RNEST/RegPlan_1908.nreg', sep=" ", header=None)
        oc = pd.read_csv('./Exports_RNEST/oc_1908.csv', sep=";", header=None, engine='c', low_memory=False,
                         skiprows=[0, 1])
        ec_20_20 = pd.read_csv('./Exports_RNEST/ec_1908_20_20.csv', sep=";", header=None, engine='c', low_memory=False,
                               skiprows=[0, 1])
        ec_60_20 = pd.read_csv('./Exports_RNEST/ec_1908_60_20.csv', sep=";", header=None, engine='c', low_memory=False,
                               skiprows=[0, 1])
        workload_1_1 = pd.read_csv('./Exports_RNEST/complexity_workload_1908_1_1.csv', sep=";", header=None, engine='c',
                                   low_memory=False, skiprows=[0, 1])
        workload_20_20 = pd.read_csv('./Exports_RNEST/complexity_workload_1908_20_20.csv', sep=";", header=None,
                                     engine='c', low_memory=False, skiprows=[0, 1])
        num_conflicts_1_1 = pd.read_csv('./Exports_RNEST/complexity_numConflicts_1908_1_1.csv', sep=";", header=None,
                                        engine='c', low_memory=False, skiprows=[0, 1])
        climbing_1_1 = pd.read_csv('./Exports_RNEST/complexity_climbing_1908_1_1.csv', sep=";", header=None, engine='c',
                                   low_memory=False, skiprows=[0, 1])
        cruising_1_1 = pd.read_csv('./Exports_RNEST/complexity_cruising_1908_1_1.csv', sep=";", header=None, engine='c',
                                   low_memory=False, skiprows=[0, 1])
        descending_1_1 = pd.read_csv('./Exports_RNEST/complexity_descending_1908_1_1.csv', sep=";", header=None,
                                     engine='c', low_memory=False, skiprows=[0, 1])

    elif ((date(int(split_day[2]), int(split_day[1]), int(split_day[0])) >= date(2019, 8, 15))
          and (date(int(split_day[2]), int(split_day[1]), int(split_day[0])) <= date(2019, 9, 11))):
        normal_capacity = pd.read_csv('./Exports_RNEST/Capacity_1909.ncap', sep=";", header=None)
        regulations = pd.read_csv('./Exports_RNEST/RegPlan_1909.nreg', sep=" ", header=None)
        oc = pd.read_csv('./Exports_RNEST/oc_1909.csv', sep=";", header=None, engine='c', low_memory=False,
                         skiprows=[0, 1])
        ec_20_20 = pd.read_csv('./Exports_RNEST/ec_1909_20_20.csv', sep=";", header=None, engine='c', low_memory=False,
                               skiprows=[0, 1])
        ec_60_20 = pd.read_csv('./Exports_RNEST/ec_1909_60_20.csv', sep=";", header=None, engine='c', low_memory=False,
                               skiprows=[0, 1])
        workload_1_1 = pd.read_csv('./Exports_RNEST/complexity_workload_1909_1_1.csv', sep=";", header=None, engine='c',
                                   low_memory=False, skiprows=[0, 1])
        workload_20_20 = pd.read_csv('./Exports_RNEST/complexity_workload_1909_20_20.csv', sep=";", header=None,
                                     engine='c', low_memory=False, skiprows=[0, 1])
        num_conflicts_1_1 = pd.read_csv('./Exports_RNEST/complexity_numConflicts_1909_1_1.csv', sep=";", header=None,
                                        engine='c', low_memory=False, skiprows=[0, 1])
        climbing_1_1 = pd.read_csv('./Exports_RNEST/complexity_climbing_1909_1_1.csv', sep=";", header=None, engine='c',
                                   low_memory=False, skiprows=[0, 1])
        cruising_1_1 = pd.read_csv('./Exports_RNEST/complexity_cruising_1909_1_1.csv', sep=";", header=None, engine='c',
                                   low_memory=False, skiprows=[0, 1])
        descending_1_1 = pd.read_csv('./Exports_RNEST/complexity_descending_1909_1_1.csv', sep=";", header=None,
                                     engine='c', low_memory=False, skiprows=[0, 1])

    else:
        sys.exit("The requested files do not exist for the given day: " + str(day))

    return normal_capacity, regulations, oc, ec_20_20, ec_60_20, \
           workload_1_1, workload_20_20, num_conflicts_1_1, climbing_1_1, cruising_1_1, descending_1_1


def extract_counts_excelFromExportRNEST(counts_fromExport, delta_timesteps):
    """
    Given the filtered counts for a specific day and metrics (SINGLE ROW), it groups the values according to "delta_timestep"
        * The "delta_timestep" refers to the time-windows used to group the values

    Extract the counts for: {Capacity, Occupancy count, Netry count} - Export in RNEST

    Input:
        * counts_fromExport[dataFrame-singleRow] -> Row with counts associates to a given day and a single sector
        * delta_timesteps[int] -> Temporal windows used to group the values

    Output:
        * counts[np.array] -> Final grouping of the values
    """

    try:
        counts = np.zeros((int(24*60/delta_timesteps)))
    except:
        print('The "delta_timesteps must be a divider of 60"')
        sys.exit()

    numElements_toSum = delta_timesteps/5 # In the occupancy file the temporal windows is equal to 5 mins

    counter_ellementsAdded = 0
    cummulative_count = 0
    counter = 0

    for value in counts_fromExport.values[0][5:-1]:
        cummulative_count += value
        counter_ellementsAdded += 1
        
        if(counter_ellementsAdded == numElements_toSum):
            counts[counter] = cummulative_count
            
            cummulative_count = 0
            counter_ellementsAdded=0
            counter+=1

    return counts



def extract_regulations(sectorName, dayOfStudy, regulations):
    """
    Function that given the regulations from an entire AIRAC, it filters the ones for a given sector and date
    
    Input:
        * sectorName[str] -> Name sector to extract CREATED tags
        * date[str-DD/MM/YYYY] -> Date associated to the CRETAED tags
        * regulations[dataFrame] -> All regulation from a given AIRAC

    Output:
        * intervals_regulation[list-str] -> Consecutive list of pairs {start, end}
        * capacity_dueTo_regulation[float] -> Capacities (variuations) due to hotspots respect the "normal" one
    """

    regulations_atDate = regulations[(regulations.iloc[:, 0] == from_DDMMYYY_to_YYYYMMDD(dayOfStudy)) &
                                     (regulations.iloc[:, 2] == 'MAS'+sectorName)]

    capacity_dueTo_regulation = []
    intervals_regulation = []

    # Number of regulation must be taken into account
    for index in regulations_atDate.index.astype(int):
        
        # Intervals of time where the regulations affected
        for i in range(regulations.iloc[index, 8].astype(int)):
            intervals_regulation.append(regulations.iloc[index+i+1, 0]) # Add start time
            intervals_regulation.append(regulations.iloc[index+i+1, 1]) # Add end time
            capacity_dueTo_regulation.append(regulations.iloc[index+i+1, 2])
            
    # Sanity check - Verify proportion between intervals and capacity is correct
    if len(intervals_regulation) != (len(capacity_dueTo_regulation)*2):
        sys.exit('The proportion between the number of different capacities and the intervals do not match')

    return intervals_regulation, capacity_dueTo_regulation


def extract_intervals_hotspots(TAGS_created):
    """
    Extract the intervals of the filtered hotspots

    Input:
        * TAGS_created[dataFrame] -> Dataframe from where we want to extract the intervals of time (CREATED)

    Output:
        * intervals_hotspots[list-str] -> Consecutive list of pairs {start, end}
        * hotspot_labels[float] -> Label fot the hotspots (1 == hotspot)
    """
    
    # Extract the intervals where hotspots were reported
    intervals_hotspots = []

    for start, end in zip(TAGS_created['start_time_HHMMSS'].values, TAGS_created['end_time_HHMMSS'].values):
        intervals_hotspots.append(start)
        intervals_hotspots.append(end)
        
    # hotspots reported = 1
    hotspot_labels = list(np.ones(int(len(intervals_hotspots)/2)))

    return intervals_hotspots, hotspot_labels



def extract_hotspotsTAGSfile_given_sector_and_date(TAGS, sectorName, date, checkDELETED):
    """
    Funtion that given a specific sector and name, extract the CREATED hotspots in the TAGS file.

    Input:
        * TAGS[Dateframe] -> Dtaframe withh all the tags in the TAGS file
        * sectorName[str] -> Name sector to extract CREATED tags
        * date[str-DD/MM/YYYY] -> Date associated to the CRETAED tags

    Outout:
        * TAGS_created[dtaFrame] -> Datafram with the filterd TAGS according to the sector name and date
    """
    try:
        TAGS_created = TAGS.loc[(TAGS["target_date"] == date) &
                                (TAGS["volume"] == sectorName) &
                                (TAGS["status"] == 'CREATED') &
                                (TAGS["volume"].isnull() == False)] 

        # start_time - from_YYYYMMDDHHMMSS_to_HHMMSS
        TAGS_created.loc[:, 'start_time_HHMMSS'] = TAGS_created.apply(lambda x: from_YYYYMMDDHHMMSS_to_HHMMSS_withTwoDots(x.start_time), axis=1)
        # end_time - from_YYYYMMDDHHMMSS_to_HHMMSS
        TAGS_created.loc[:, 'end_time_HHMMSS'] = TAGS_created.apply(lambda x: from_YYYYMMDDHHMMSS_to_HHMMSS_withTwoDots(x.end_time), axis=1)

        # checkDELETED = True -> Check if exist the DELETED pair for the extracted CRATED one
        if(checkDELETED == True):
            TAGS_deleted = TAGS.loc[(TAGS["target_date"] == date) &
                                    (TAGS["volume"] == sectorName) &
                                    (TAGS["status"] == 'DELETED') &
                                    (TAGS["volume"].isnull() == False)]

            # If 'start_time' is different -> NO HOTSPOT
            if(TAGS_created['start_time'].values[0] != TAGS_deleted['start_time'].values[0] or TAGS_deleted.values.shape[0] == 0):
                TAGS_created = pd.DataFrame()

    except:
        TAGS_created = pd.DataFrame()

    return TAGS_created



def addIntervals_toFinalConjunt(intervals, labels, delta_timesteps):
    """
    Function that given a list of intervals (intervals) and the label for each intevral (labels),
    add this information to "finalConjunt".
        * The seconds in the intervals are not taken into account

    Input:
        * intervals[list-str] -> Sorted sequence of pairs {start interval, end interval}
        * labels[list-int/float] -> For each "pair" in intervals, must be given a label
        * finalConjunt[list-int/float] -> Final list to add the given intervals
        * delta_timesteps[int] -> difference in MINUTES between the timestamps represented in "finalConjunt"

    Output:
        * finalConjunt[list-int/float]
    """

    # Create and empty array to store the labels of the hotspots
    try:
        finalConjunt = np.zeros((int(24*60/delta_timesteps)))
    except ValueError:
        print('The "delta_timesteps must be a divider of 60"')
        sys.exit()
        
    for start, end, hotspot_label in zip(intervals[0::2], intervals[1::2], labels):
        try:
            start_split = start.split(':')
            end_split = end.split(':')
            
            finalConjunt[int((int(start_split[0])*60 + int(start_split[1]))/delta_timesteps):
                         int((int(end_split[0])*60 + int(end_split[1]))/delta_timesteps)] = hotspot_label
        except ValueError:
            sys.exit('Check: utils_samplesTRaining/addIntervals_toFinalConjunt -> ' +
                     'Possible error between the intervals on the timestamps')
    
    return finalConjunt



def read_TAGS_file(path_to_file):
    """
    Function that given a path to the TAGS file, it return the file (only useful information)

    Input:
        * path_to_file[str] -> Path to the TAGS file

    Output:
        * TAGS[dataframe] -> DataFrame containing the TAGS
    """

    # Load the TAGS file (HOTSPOTS) - Excel
    TAGSfile = path_to_file
    TAGS = pd.read_excel(TAGSfile+".xlsx")
    # Save the XLSX file as CSV
    TAGS.to_csv(TAGSfile+".csv", index=False)
    # Read again the CSV, using as a separator the ;
    TAGS = pd.read_csv(TAGSfile+".csv", sep=";", header=None, engine='c', low_memory=False) # The 'C' engine is faster
    TAGS.columns = ['target_date', 
                    'position',
                    'sector_group',
                    'start_time',
                    'end_time',
                    'reason',
                    'subcategory',
                    'value',
                    'status',
                    'status_change_time',
                    'volume',
                    'type',]

    # Drop the first row which contanis again the labels 
    TAGS.drop(0)

    return TAGS