"""
Author: Sergi Mas Pujol
Last update: 04/06/2020

Python version: 3.6
"""


from datetime import datetime, date, timedelta


def from_YYYYMMDD_to_DDMMYYYY_given_separator(original_date: str, separator: str) -> str:
    """
    Change format of original_date (from YYYYMMDD to DDMMYYYY) where the separator between elements can be specified.

    Args:
        original_date: date in format YYYYMMDD
        separator: separator to use between elements

    Returns:
        new_date: new date in format DDMMYYYY using the separator
    """

    year = original_date[0:4]
    month = original_date[5:7]
    day = original_date[8:10]

    new_date = day + separator + month + separator + year

    return new_date


def add_minutes_given_HHSSMM(timestamp: str, addition: int) -> str:
    """
    Given a timestamps it add a certain amount of time.

    Args:
        timestamp: in the format %H:%M:%S
        addition: minutes that will be added to the timestamp

    Returns:
        (str) New timestamps
    """

    # Convert the timestamp into a datetime object
    datetime_object = datetime.strptime(timestamp, '%H:%M:%S')
    # Subtract X minutes to the timestamp
    gap_after_minutes = timedelta(minutes=addition)
    updated_timestamp = datetime_object + gap_after_minutes

    # Generate the final timestamp
    hours = str("{:02d}".format(updated_timestamp.hour))
    minutes = str("{:02d}".format(updated_timestamp.minute))
    seconds = str("{:02d}".format(updated_timestamp.second))

    return hours + minutes + seconds


def substract_minutes_given_HHSSMM(timestamp: str, substraction: int) -> str:
    """
    Given a timestamps it subtract a certain amount of time.

    Args:
        timestamp: in the format %H:%M:%S
        substraction: minutes that will be removed to the timestamp

    Returns:
        (str) New timestamps
    """

    # Convert the timestamp into a datetime object
    datetime_object = datetime.strptime(timestamp, '%H:%M:%S')

    # Substract X minutes to the timestamp
    gap_before_minutes = timedelta(minutes=substraction)
    updated_timestamp = datetime_object - gap_before_minutes

    # Generate the final timestamp
    hours = str("{:02d}".format(updated_timestamp.hour))
    minutes = str("{:02d}".format(updated_timestamp.minute))
    seconds = str("{:02d}".format(updated_timestamp.second))

    return hours + minutes + seconds


def listDays_betweenTwoDates(start_date, end_date, type, separator):
    """
    Generate list of dates between "start_date and "end_date"
        * The difference between dates is iqual to 1 day.
    
    Input:
        * start_date[date from datetime] -> Beginning date
        * end_date[date from datetime] -> End date
        * type[str] -> Indicates the format of the date: {DDMMYYY, YYYYMMDD}
        * separator[str] -> Indicate the "separator" used to express the date

    Output:
        * list_days[list-str] -> List of days between "start_date and "end_date"    
    """
    
    list_days =[]
    delta = end_date - start_date # as timedelta

    if(type == 'DDMMYYYY'):
        for i in range(delta.days + 1):
            date = start_date + timedelta(days=i)
            list_days.append(str("{:02d}".format(date.day))+separator+
                                 str("{:02d}".format(date.month))+separator+
                                 str(date.year))

    elif(type == 'YYYYMMDD'):
        for i in range(delta.days + 1):
            date = start_date + timedelta(days=i)
            list_days.append(str(date.year)+separator+
                                 str("{:02d}".format(date.month))+separator+
                                 str("{:02d}".format(date.day)))

    return list_days



def midPoint_betweenDates(startDate, endDate):
    """
    Given two dates in format HHMMSS computes the mid-point between them.

    Input:
        * startDate[int] -> Start date
        * endDate[int] -> Final date
    Output:
        * newTimestampts[int] -> mid-point between dates
    """

    # Convert dates from INT to STR
    end = datetime.strptime(str(int(endDate)), "%H%M%S")
    start = datetime.strptime(str(int(startDate)), "%H%M%S")

    # Compute the mip point
    midPoint = start + (end-start)/2

    # Format the new timestamp generated - Ensure right foramt
    hours = str(midPoint.hour)
    minutes = str(midPoint.minute)
    seconds = str(midPoint.second)
    
    if(len(minutes) == 1):
        minutes = "0"+ minutes
    if(len(seconds) == 1):
        seconds = "0" + seconds

    # Concatenation format: HHMMSS or HMMSS
    newTimestampts = int(hours+minutes+seconds)


    return newTimestampts



def relativePortionTime(timeBeginSeg, timeEndSeg, timestamp):
    """
    Function that given an interval of time, computed the proportion of time (%) happened between 
    the segment started and the timestamp of study.
    
    Input:
        * timeBeginSeg -> HHMMSS 
        * timeEndSeg -> HHMMSS
        * timestamp[int] -> HHMMSS
            * If it is not  INT, the code will force the format
    Output:
        + relativePosition -> % of the interval
    """
    # Maping the times
    begin = list(map(str, str(timeBeginSeg))) # list of each given number
    end = list(map(str, str(timeEndSeg)))

    timestamp = int(timestamp) # Force that the time stamp is a INT
    timestamp = list(map(str, str(timestamp))) 
    
    # Initialice the variables
    begin_secs = 0
    end_secs = 0
    timestamp_secs = 0

    # Time belonging to 00MMSS
    if(len(begin) == 4):
        begin_secs = int("".join(begin[0:2]))*60+int("".join(begin[2:4]))
    # Time belonging to 0HMMSS
    elif(len(begin) == 5): 
        begin_secs = int(begin[0])*3600+int("".join(begin[1:3]))*60+int("".join(begin[3:5]))
    # Time belonging to HHMMSS
    else:
        begin_secs = int("".join(begin[0:2]))*3600+int("".join(begin[2:4]))*60+int("".join(begin[4:6]))

    if(len(end) == 4):
        end_secs = int("".join(end[0:2]))*60+int("".join(end[2:4]))
    elif(len(end) == 5):
        end_secs = int(end[0])*3600+int("".join(end[1:3]))*60+int("".join(end[3:5]))
    else:
        end_secs = int("".join(end[0:2]))*3600+int("".join(end[2:4]))*60+int("".join(end[4:6]))

    if(len(timestamp) == 4): 
        timestamp_secs = int("".join(timestamp[0:2]))*60+int("".join(timestamp[2:4]))
    elif(len(timestamp) == 5): 
        timestamp_secs = int(timestamp[0])*3600+int("".join(timestamp[1:3]))*60+int("".join(timestamp[3:5]))
    else:
        timestamp_secs = int("".join(timestamp[0:2]))*3600+int("".join(timestamp[2:4]))*60+int("".join(timestamp[4:6]))    


    return (timestamp_secs - begin_secs)/(end_secs - begin_secs)



def from_HHMMSS_to_HH(time):
    """
    Function that convert HHMMSS into HH.
    
    Input:
        time -> int HHMMSS
    Output:
        hours -> int HH
    """
    
    time = str('{0:06}'.format(time)) # This is requiered for padding with 0s when hour from 0:00 to 9:59 am
    indexes = [iter(time)]*2
    split = [''.join(z) for z in zip(*indexes)]
    hours = int(split[0])+int(split[1])/60+int(split[2])/3600
    
    return int(hours)



def intervalTime_segment(begin, end):
    """
    Function than computes the interval of time between two given units of time.
    Both must be in HH, MM or SS. The output will have the same units.
    It takes into account if the inteval start in one day and ens in the next one.
    
    Input:
        begin -> float of HH, MM otr SS
        end -> float of HH, MM otr SS
        
    Output:
        internal -> float expressing the interval of time. In HH, MM or SS
    """
    if(end > begin): # Start end in the same day
        return (end-begin)
    else: # The end hour belong to the the next day
        return (24.0-begin+end)



def from_YYYYMMDDHHMMSS_to_HHMMSS(YYYYMMDDHHMMSS):
    """
    Function to convert timestamp Year/Month/Day/T/Hours/Minuts/Seconds/z to
    HoursMinutesSeconds(HHMMSS).
    
    Input:
        * YYYYMMDDHHMMSS -> String 
    Output:
        * HHMMSS -> Interger
    """

    # Convert the timestamp
    HHMMSS = YYYYMMDDHHMMSS[11:13]+YYYYMMDDHHMMSS[14:16]+YYYYMMDDHHMMSS[17:19]
    
    return int(HHMMSS)



def from_YYYYMMDDHHMMSS_to_HHMMSS_withTwoDots(YYYYMMDDHHMMSS):
    """
    Function to convert timestamp Year/Month/Day/T/Hours/Minuts/Seconds/Z to
    Hours:Minutes:Seconds(HH:MM:SS).
    
    Input:
        * YYYYMMDDHHMMSS[str] 
    Output:
        * HHMMSS[str]
    """
    HHMMSS = str(YYYYMMDDHHMMSS[11:13]+':'+YYYYMMDDHHMMSS[14:16]+':'+YYYYMMDDHHMMSS[17:19])
    
    return HHMMSS



def from_DDMMYYY_to_YYMMDD(date):
    """
    Function that convert a date in format DD/MM/YYYY to YYMMDD
    
    Input:
        date -> string DD/MM/YYYY
    Output:
        string YYMMDD
    """
    date_str = str('{0:10}'.format(date))
    return date_str[-2:]+date_str[3:5]+date_str[0:2]



def from_DDMMYYY_to_YYYYMMDD(date):
    """
    Function that convert a date in format DD/MM/YYYY to YYMMDD
    
    Input:
        date -> string DD/MM/YYYY
    Output:
        string YYYYMMDD
    """
    date_str = str('{0:10}'.format(date))
    return date_str[-4:]+date_str[3:5]+date_str[0:2]
