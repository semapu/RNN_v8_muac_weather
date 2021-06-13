"""
Author: Sergi Mas Pujol
Last update: 10/07/2020

Python version: 3.6
"""

import math
import sys
import numpy as np
import random
from typing import List
from datetime import datetime

from utils_samplesTraining import readAssociatedFile_fromAIRAC_givenDate, \
                                  extract_regulations, \
                                  addIntervals_toFinalConjunt, \
                                  add_minutes_given_HHSSMM
from utils_timeProcessing import from_HHMMSS_to_HH


def extract_days_timestamps_volumes_labels_days_with_regulations(list_dates_with_regulations: List[str],
                                                                 start_regulations: List[int],
                                                                 end_regulations: List[int],
                                                                 list_volumes_regulations: List[str],
                                                                 gap_before_start_time: int,
                                                                 gap_after_start_time: int,
                                                                 num_additional_samples_per_day: int,
                                                                 min_timestamps_with_regulations: int):
    """
    Function that given the information from the regulations recorded it computes samples from those days
    WITH and WITHOUT regulations.

    Args:
        list_dates_with_regulations: [DDMMYYYY] days with regulations
        start_regulations: [HHMMSS] starting timestamp of the regulations
        end_regulations: [HHMMSS] ending timestamp of the regulations
        list_volumes_regulations: Suffix of the traffic volume (e.g. D6WH or B3EH)
        gap_before_start_time: Additional time added to the beginning of the starting timestamp of the regulation
        gap_after_start_time: Additional time added to the end of the starting timestamp of the regulation
        num_additional_samples_per_day: How many samples are going to be generated per day
        min_timestamps_with_regulations: Minimum number of timestamps with regulations. To avoid samples, for instance,
                                         with only two timestamps with regulations over a period of 30 minutes.

    Returns:
        list_days: List with the day of the generated samples
        start_time_samples: Starting timestamp of the samples
        end_time_samples: Ending timestamp of the sample
        volumes: Volume associated to the generated sample
        labels: Labels for each sample (one label per minute)
    """

    # Initialize the final variables
    list_days = list()
    start_time_samples = list()
    end_time_samples = list()
    volumes = list()

    # Number of timestamps per sample
    num_sample_to_generate = list_dates_with_regulations.shape[0] * num_additional_samples_per_day
    # Number of labels per sample
    labels = np.zeros((num_sample_to_generate, (gap_before_start_time + gap_after_start_time), 1), dtype=int)

    # Initialize a counter to track which sample we finally have saved
    counter = 0

    for day, start_regulation, end_regulation, volume in zip(list_dates_with_regulations,
                                                             start_regulations,
                                                             end_regulations,
                                                             list_volumes_regulations):

        # Iterate multiple times to extract more samples
        for i in range(0, num_additional_samples_per_day):

            # Randomly create the hour and the minutes
            hour = str("{:02d}".format(random.randint(3, 21)))  # From 3 to 21 to avoid extremes (wrap around issue)
            minute = str("{:02d}".format(random.randint(10, 50)))

            # Initialize the timestamp
            random_timestamp = hour + ":" + minute + ":" + '00'

            # Create the initial timestamp - The following function is used to convert the HH:MM:SS into an int
            start_timestamp = add_minutes_given_HHSSMM(random_timestamp, 0)

            # Add a given about to create the ending timestamp
            end_timestamp = add_minutes_given_HHSSMM(random_timestamp, (gap_before_start_time + gap_after_start_time))

            # Find all regulations for that day
            indexes_regulations_same_day = np.where(list_dates_with_regulations == day)

            timestamps_start_for_that_day_with_regulations = start_regulations[indexes_regulations_same_day]
            timestamps_end_for_that_day_with_regulations = end_regulations[indexes_regulations_same_day]
            volumes_for_that_day_with_regulations = list_volumes_regulations[indexes_regulations_same_day]

            # Label for the entire day
            labels_entire_day = np.zeros((1, 24 * 60, 1), dtype=int)

            for start_reg, end_reg, vol in zip(timestamps_start_for_that_day_with_regulations,
                                               timestamps_end_for_that_day_with_regulations,
                                               volumes_for_that_day_with_regulations):

                # Check if the sectors volumes are coincidentg
                if vol == volume:
                    start_minutes_regulation = from_HHMMSS_to_HH(start_reg) * 60 + int(str(start_reg)[-4:-2])
                    end_minutes_regulation = from_HHMMSS_to_HH(end_reg) * 60 + int(str(end_reg)[-4:-2])

                    labels_entire_day[0][start_minutes_regulation:end_minutes_regulation] = 1

            # Extract the interval of the entire day in which we are interested
            start_minutes_timestamp = from_HHMMSS_to_HH(int(start_timestamp)) * 60 + int(str(start_timestamp)[-4:-2])
            end_minutes_timestamp = from_HHMMSS_to_HH(int(end_timestamp)) * 60 + int(str(end_timestamp)[-4:-2])

            # Check if the created sample is a negative one (no regulation)
            # If true, randomly decide if save it or not
            if np.sum(labels_entire_day[0][start_minutes_timestamp:end_minutes_timestamp]) == 0:

                random_coin = random.uniform(0, 1)

                if random_coin >= 0.85:

                    list_days.append(day)
                    start_time_samples.append(start_timestamp)
                    end_time_samples.append(end_timestamp)
                    volumes.append(str(volume[4:-1]))

                    labels[counter] = labels_entire_day[0][start_minutes_timestamp:end_minutes_timestamp]

                    counter += 1

                else:
                    continue

            # When a sample contains regulations -> Save it if a regulation is reported by more than 10 minutes
            # If less minutes with regulation, the sample is discarded
            else:

                if np.sum(labels_entire_day[0][start_minutes_timestamp:end_minutes_timestamp]) >= min_timestamps_with_regulations:

                    list_days.append(day)
                    start_time_samples.append(start_timestamp)
                    end_time_samples.append(end_timestamp)
                    volumes.append(str(volume[4:-1]))

                    labels[counter] = labels_entire_day[0][start_minutes_timestamp:end_minutes_timestamp]
                    counter += 1

                else:
                    continue

    # Only save the labels for the samples we have saved
    labels = labels[0:counter]

    return list_days, start_time_samples, end_time_samples, volumes, labels


def extract_features_from_list_days_and_timestamps(list_days,
                                                   start_time_samples, end_time_samples,
                                                   traffic_volume,
                                                   gap_before_start_time: int, gap_after_start_time: int,
                                                   X):
    """
    Given a "list_days, start_time_samples and end_time_samples", it extract the available features and save it to X.

    The "sector_name", can be given as a single String when working with a unique sector, or as a list of the different
    volumes. The order must be the same as in the "list_days, start_time_samples and end_time_samples".


    Args:
        list_days: List containing the days on which we are interested
        start_time_samples: List containing the starting timestamps for the given day
        end_time_samples: List containing the ending timestamps for the given day
        traffic_volume: Single string or List[str] of the volumes associated with the days and timestamps
        gap_before_start_time: Gap used to create "start_time_samples"
        gap_after_start_time: Gap used to create "end_time_samples"
        X: Final variable where the results will be saved.

    Returns:
        X: The initial empty object filled with the corresponding features / variables extracted.
    """

    # Identify if as a "sector_names", the function has received a string or a list of strings
    # If it is a unique String -> Create a list repeating the given value
    if isinstance(traffic_volume, str):
        traffic_volume = [traffic_volume] * list_days.shape[0]

    # Sanity check: Ensuring the list "sector_name" has the correct length
    if len(traffic_volume) != list_days.shape[0]:
        sys.exit("The length of both days opf study and volumes must be the same. "
                 "[extract_features_from_list_days_and_timestamps]")

    counter_day = 0

    for day, start, end, sector_name in zip(list_days, start_time_samples, end_time_samples, traffic_volume):
        # Read the correct files according to the date of interest
        # AIRACs only contain information from an interval of 28 days

        normal_capacity, regulations, oc, ec_20_20, ec_60_20, \
         workload_1_1, workload_20_20, num_conflicts_1_1, \
         climbing_1_1, cruising_1_1, descending_1_1 = readAssociatedFile_fromAIRAC_givenDate(day)

        # normal_capacity, regulations, oc, ec_20_20, ec_60_20, \
        #  workload_1_1, workload_20_20, \
        #  climbing_1_1, cruising_1_1, descending_1_1 = readAssociatedFile_fromAIRAC_givenDate(day)

        starting_index_inside_day = int(start[0:2]) * 60 + int(start[2:4])
        end_index_inside_day = int(end[0:2]) * 60 + int(end[2:4])

        """ Timestamp - Interval of the day """
        interval_day_associated_timestamp = math.floor(starting_index_inside_day / (gap_before_start_time + gap_after_start_time))
        X[counter_day][:][:, 0] = interval_day_associated_timestamp

        """ Day of the week """
        day_week = datetime.strptime(day, "%d/%m/%Y").weekday()
        X[counter_day][:][:, 1] = day_week

        """ Capacity """
        # Extract the regulation for the given date
        # intervals_regulation, capacity_dueTo_regulation = extract_regulations(sector_name, day, regulations)

        # Using the nominal capacity
        delta_timesteps = 1  # Difference en minutes between the timestamps in the sample
        tmp_capacity = np.zeros((int(24 * 60 / delta_timesteps)))

        # Add the regulations to the temporal capacity vector
        # tmp_capacity = addIntervals_toFinalConjunt(intervals_regulation, capacity_dueTo_regulation, 1)

        # Add the usual capacity to the temporal capacity
        tmp_capacity[tmp_capacity == 0] = normal_capacity[(normal_capacity.iloc[:, 0] == day) &
                                                          (normal_capacity.iloc[:, 1] == 'MAS' + sector_name)][4].values[0]

        X[counter_day][:][:, 2] = tmp_capacity[starting_index_inside_day:end_index_inside_day]

        """ Occupancy count """
        # Filtering the timestamps and the sector
        oc_atDate = oc[(oc.iloc[:, 0] == day) & (oc.iloc[:, 2] == 'EDYY' + sector_name)].values[0][5:-1]

        # Sanity check - Ensure only one row is obtained after the filtering
        if len(oc_atDate) == 0:
            sys.exit("No OC were found for the date: " + str(day))

        X[counter_day][:][:, 3] = oc_atDate[starting_index_inside_day:end_index_inside_day]

        """ Entry count 20/20 """
        # Filtering the datestamp and the sector
        ec_20_20_atDate = ec_20_20[(ec_20_20.iloc[:, 0] == day) & (ec_20_20.iloc[:, 2] == 'EDYY' + sector_name)].values[0][6:-1]

        for i in range(0, gap_before_start_time + gap_after_start_time):
            # ec_20_20 has a frequency of 20 minutes
            index_value = math.floor((starting_index_inside_day + i) / 20)
            X[counter_day][:][i, 4] = ec_20_20_atDate[index_value]

        # Sanity check - Ensure only one row is obtained after the filtering
        if len(ec_20_20_atDate) == 0:
            sys.exit("No EC was found for the date: " + str(day))

        """ Entry count 60/20 """
        # Filtering the datastamp and the sector
        ec_60_20_atDate = ec_60_20[(ec_60_20.iloc[:, 0] == day) & (ec_60_20.iloc[:, 2] == 'EDYY' + sector_name)].values[0][6:-1]

        for i in range(0, gap_before_start_time + gap_after_start_time):
            # ec_60_20 has a frequency of 20 minutes
            index_value = math.floor((starting_index_inside_day + i) / 20)
            X[counter_day][:][i, 5] = ec_60_20_atDate[index_value]

        # Sanity check - Ensure only one row is obtained after the filtering
        if len(ec_60_20_atDate) == 0:
            sys.exit("No EC was found for the date: " + str(day))

        """ Workload_1_1 """
        # Filtering the timestamps and the sector
        workload_at_date = workload_1_1[(workload_1_1.iloc[:, 0] == day) & (workload_1_1.iloc[:, 2] == 'EDYY' + sector_name)].values[0][6:-1]

        # Sanity check - Ensure only one row is obtained after the filtering
        if len(workload_at_date) == 0:
            sys.exit("No workload (1 1) were found for the date: " + str(day))

        X[counter_day][:][:, 6] = workload_at_date[starting_index_inside_day:end_index_inside_day]

        """ Workload_20_20 """
        # Filtering the timestamp and the sector
        workload_20_20_atDate = workload_20_20[(workload_20_20.iloc[:, 0] == day) & (workload_20_20.iloc[:, 2] == 'EDYY' + sector_name)].values[0][6:-1]

        for i in range(0, gap_before_start_time + gap_after_start_time):
            # ec_20_20 has a frequency of 20 minutes
            index_value = math.floor((starting_index_inside_day + i) / 20)
            X[counter_day][:][i, 7] = workload_20_20_atDate[index_value]

        # Sanity check - Ensure only one row is obtained after the filtering
        if len(workload_20_20_atDate) == 0:
            sys.exit("No workload (20 20) was found for the date: " + str(day))

        """ Number of conflicts """
        # Filtering the timestamps and the sector
        num_conflicts_at_date = num_conflicts_1_1[(num_conflicts_1_1.iloc[:, 0] == day) & (num_conflicts_1_1.iloc[:, 2] == 'EDYY' + sector_name)].values[0][5:-1]

        # Sanity check - Ensure only one row is obtained after the filtering
        if len(num_conflicts_at_date) == 0:
            sys.exit("No Num. conflicts were found for the date: " + str(day))

        X[counter_day][:][:, 8] = num_conflicts_at_date[starting_index_inside_day:end_index_inside_day]

        """ Climbing_1_1 """
        # Filtering the timestamps and the sector
        climbing_at_date = climbing_1_1[(climbing_1_1.iloc[:, 0] == day) & (climbing_1_1.iloc[:, 2] == 'EDYY' + sector_name)].values[0][6:-1]

        # Sanity check - Ensure only one row is obtained after the filtering
        if len(climbing_at_date) == 0:
            sys.exit("No climbing were found for the date: " + str(day))

        X[counter_day][:][:, 9] = climbing_at_date[starting_index_inside_day:end_index_inside_day]

        """ Cruising_1_1 """
        # Filtering the timestamps and the sector
        cruising_at_date = cruising_1_1[(cruising_1_1.iloc[:, 0] == day) & (cruising_1_1.iloc[:, 2] == 'EDYY' + sector_name)].values[0][6:-1]

        # Sanity check - Ensure only one row is obtained after the filtering
        if len(climbing_at_date) == 0:
            sys.exit("No cruising time were found for the date: " + str(day))

        X[counter_day][:][:, 10] = cruising_at_date[starting_index_inside_day:end_index_inside_day]

        """ Descending_1_1 """
        # Filtering the timestamps and the sector
        descending_at_date = descending_1_1[(descending_1_1.iloc[:, 0] == day) & (descending_1_1.iloc[:, 2] == 'EDYY' + sector_name)].values[0][6:-1]

        # Sanity check - Ensure only one row is obtained after the filtering
        if len(workload_at_date) == 0:
            sys.exit("No descending time were found for the date: " + str(day))

        X[counter_day][:][:, 11] = descending_at_date[starting_index_inside_day:end_index_inside_day]

        ##################################
        # Increase the counting of the day
        counter_day += 1

        # Print information to be able to track evolution of the cod3e
        print(sector_name + ' | counter: ' + str(counter_day))

    return X
