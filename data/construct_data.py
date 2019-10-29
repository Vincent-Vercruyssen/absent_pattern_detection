""" Construct the benchmark power usage data.

Author: Vincent Vercruyssen.

INSTRUCTIONS BEFORE RUNNING (READ CAREFULLY):

1. Download the UCI power consumption dataset: https://archive.ics.uci.edu/ml/datasets/individual+household+electric+power+consumption
2. Download the UCR time series datasets: https://www.cs.ucr.edu/~eamonn/time_series_data/
3. Construct .csv files for the UCR time series dataset files.
4. Extract the datasets and store them in an accessible folder.
5. Enter the correct paths to the datasets below.

"""

import sys, os, math
import random
import pickle
import pandas as pd
import numpy as np
import scipy as sp
import scipy.stats as sps
import datetime
from collections import Counter
from scipy import signal


def main():

    """
    ACTION REQUIRED: ENTER THE CORRECT PATHS
    """
    uci_dataset = '<path>'      # should be a csv file
    uci_store_path = '<path>'   # should be a csv file
    ucr_dataset = '<path>'      # should be a csv file
    
    # STEP 1: create a cleaned version of the UCI power consumption data

    # load the data
    data = pd.read_csv(uci_dataset, sep=';', na_values=['?'], infer_datetime_format=True, parse_dates=[[0, 1]],
                dtype={'Date': str, 'Time': str, 'Global_active_power': np.float64, 'Global_reactive_power': np.float64,
                'Voltage': np.float64, 'Global_intensity': np.float64, 'Sub_metering_1': np.float64, 'Sub_metering_2': np.float64,
                'Sub_metering_3': np.float64})

    # extract some signals we will need
    AP_no_heater_airco = (data.Global_active_power.values * (1000/60) - data.Sub_metering_3.values) * (60/1000)
    AP_full = data.Global_active_power.values
    part = AP_no_heater_airco[60*24*60:60*24*80].copy()
    basic_noise = part[14820:21950].copy()
    repeat_basic_noise = np.tile(basic_noise, math.ceil(len(AP_full) / len(basic_noise)))[:len(AP_full)]

    # fill the none values with noisy signal generated from the data
    ixm = np.where(np.array([1 if np.isnan(e) else 0 for e in AP_no_heater_airco]) == 1)[0]
    AP_no_heater_airco[ixm] = repeat_basic_noise[ixm]
    AP_full[ixm] = repeat_basic_noise[ixm]

    # construct the cleaned up frame
    def get_hour(x):
        return int(pd.to_datetime(x).hour)
    vfunc = np.vectorize(get_hour)

    cleaned_data = pd.DataFrame(data={'timestamp': data.Date_Time.values, 'usage': AP_full, 'noise': repeat_basic_noise})
    timestamps = data.Date_Time.values
    cleaned_data['hour'] = vfunc(timestamps)
    cleaned_data.set_index('timestamp', inplace=True)
    cleaned_data = cleaned_data[datetime.datetime(2006, 12, 17, 0, 0):]
    cleaned_data.to_csv(uci_store_path, sep=',')

    
    # STEP 2: create the synthetic data
    # Note that variations might be due to differences in the random generation of the data

    # get the patterns from the UCR dataset
    patterns = get_patterns(ucr_dataset, classes=[1.], n_samples=[20, 28])

    # generate the synthetic data with time as the descriptor variable
    generator = dataGeneratorRealistic(sampling_rate=1/300, skip_samples=5, patterns=patterns, dependency='time', avg_interarrival=24*7, time_constraint=False,
                                   breakdown_time=21, avg_interusage=2000, superimposed=False, two_months_test=True, failure_type='catastrophic')
    tr_dat, te_dat, _ = generator.generate_data()
    """
    ACTION REQUIRED: STORE THE RESULTS IN CSV FILES
        tr_dat : dataframe containing the training series
            (more training data than actually used in the experiment to allowing for retraining the model every x months)
        te_dat : dictionary containing the dataframes for each test month
    """

    # generate the synthetic data with usage as the descriptor variable
    generator = dataGeneratorRealistic(sampling_rate=1/300, skip_samples=5, patterns=patterns, dependency='usage', avg_interarrival=24*7, time_constraint=False,
                                   breakdown_time=21, avg_interusage=2000, superimposed=False, two_months_test=True, failure_type='catastrophic')
    tr_dat, te_dat, _ = generator.generate_data()
    """
    ACTION REQUIRED: STORE THE RESULTS IN CSV FILES
        tr_dat : dataframe containing the training series
            (more training data than actually used in the experiment to allowing for retraining the model every x months)
        te_dat : dictionary containing the dataframes for each test month
    """


class dataGeneratorRealistic:
    
    def __init__(self,
                 # where to get the basic signal
                 basic_signal_path='<path_to_cleaned_uci_csv_file>',
                 # one or two months of testing data
                 two_months_test=False,         # one or two months test
                 # other parameters
                 sampling_rate=1/60,            # 1 sample every 60 seconds (number of samples / number of seconds)
                 skip_samples=1,
                 patterns={},                   # possible set of patterns to add to the signal
                 dependency='time',             # 'time', 'usage', ...
                 avg_interarrival=24*5,         # average interarrival time if dependency is time: in hours
                 avg_interusage=1,              # average cumulative usage between pattern occurrences
                 time_constraint=False,         # additional time constraint on the occurrence of the pattern
                 superimposed=False,            # superimpose on the signal or replace the signal
                 failure_type='catastrophic',   # 'catastrophic', 'deterioration'
                 breakdown_time=10,             # number of days to noise after catastrophic failure
                ):
        
        """ TODO: sample rate. If the basic signal is given, this is OK. """
        
        self.basic_signal_path = str(basic_signal_path)
        self.two_months_test = bool(two_months_test)
        self.sampling_rate = float(sampling_rate)
        self.skip_samples = int(skip_samples)
        # number of training days fixed to 1 year (365 days)
        self.train_time = int(365)
        self.patterns = patterns
        self.dependency = str(dependency)
        self.avg_interarrival = float(avg_interarrival)
        self.avg_interusage = float(avg_interusage)
        self.time_constraint = bool(time_constraint)
        self.superimposed = bool(superimposed)
        self.failure_type = str(failure_type)
        self.breakdown_time = int(breakdown_time)
    
    def generate_data(self):
        
        # basic data
        basic_data = pd.read_csv(self.basic_signal_path, sep=',', infer_datetime_format=True, parse_dates=[0])
        basic_data = basic_data.iloc[15*60*24:]  # only start from january the 1st
        
        # subsample if necessary
        if self.skip_samples > 1:
            basic_data = basic_data[::self.skip_samples]
        self.day_len = int(60 * 60 * 24 * self.sampling_rate)
            
        # curtail the length to 3 years (we don't need more)
        basic_data = basic_data.iloc[:self.day_len*365*3]
        
        # add the patterns to the data
        signal_data, pattern_locs = self.generate_signal_data(basic_data, self.patterns)
        
        # generate the train and test sets
        train_data, test_data = self.generate_train_and_test_data(basic_data, signal_data, pattern_locs)
        print('Data succesfully generated!')
        
        # return the result
        return train_data, test_data, pattern_locs
    
    def generate_signal_data(self, basic_data, pattern_variations):
        """ Add pattern variations to the signal + add holidays and long weekends """
        
        # signal, noise, hours, locations of the patterns
        signal_data = basic_data.usage.values.copy()
        base_noise = basic_data.noise.values.copy()
        hour_of_the_day = basic_data.hour.values.copy()
        pattern_locs = np.zeros(len(signal_data), dtype=float)
        
        # --------
        # HOLIDAYS
        # --------
        
        """ NOTE: inspected this part visually and seems to work """
        # randomly inject 4 1-week holidays + 4 long weekends every year
        # during these holidays, there is only a base level of usage (i.e., base noise)
        week_len = 7 * self.day_len
        weekend_len = 3 * self.day_len
        year_len = 365 * self.day_len
        for i in range(3):
            # randomly pick 4 holiday weeks + 4 long weekends for the year
            holiday_weeks = np.random.choice(np.arange(0, 51, 1), 5, replace=False)
            long_weekends = np.random.choice(np.setdiff1d(np.arange(0, 51, 1), holiday_weeks), 4, replace=False)
            
            # substitute with base_noise
            for hi in holiday_weeks:
                signal_data[(i*year_len)+(hi*week_len):(i*year_len)+((hi+1)*week_len)] = base_noise[(i*year_len)+(hi*week_len):(i*year_len)+((hi+1)*week_len)]
            for wi in long_weekends:
                signal_data[(i*year_len)+((wi+1)*week_len-weekend_len):(i*year_len)+((wi+1)*week_len)] = base_noise[(i*year_len)+((wi+1)*week_len-weekend_len):(i*year_len)+((wi+1)*week_len)]
        
        # --------
        # PATTERNS
        # --------
        
        month_vals = self._increasing_month_values(basic_data.timestamp.values)
        
        # TIME-based patterns
        """ NOTE: inspected this part visually and seems to work """
        if self.dependency == 'time':
            """ MECHANISM:
                
                subsequent occurrences: interarrival time sampled from Weibull distribution
                TODO: additional restraint that the pattern can only occur between 0 and 6 in the morning
            """
            hour_len = self.day_len / 24
            last_ix = 0
            stop = False
            while not(stop):
                # sample a new interarrival time (in hours)
                """ Reducing the second parameter increases the width of the Weibull distribution """
                st = sps.exponweib.rvs(float(self.avg_interarrival), 0.5, loc=float(self.avg_interarrival), scale=1.0, size=1)
                
                # sample a pattern variation
                p_ix = np.random.randint(0, len(pattern_variations)-1)
                pvar = pattern_variations[p_ix]
                plen = len(pvar)
                
                # update index + stopping criterion
                last_ix = last_ix + int(st*hour_len)
                if last_ix + plen > len(signal_data):
                    stop = True
                    continue
                    
                # adapt the location to not overlap with the switch of the month
                if month_vals[last_ix] != month_vals[last_ix+plen]:
                    # shift to the previous month
                    for j in range(plen):
                        if month_vals[last_ix] != month_vals[last_ix+j]:
                            break
                        shift = j
                    # apply shift
                    last_ix = last_ix - shift
                    print('Shifted with', shift, 'values from', last_ix+shift, 'to', last_ix)
                
                # insert the variation
                if self.superimposed == True:
                    signal_data[last_ix:last_ix+plen] = signal_data[last_ix:last_ix+plen] + pvar
                else:
                    signal_data[last_ix:last_ix+plen] = pvar
                
                # pattern location
                pattern_locs[last_ix:last_ix+plen] = 1.0

        # USAGE-based patterns
        """ NOTE: inspected this part visually and seems to work """
        if self.dependency == 'usage':
            """ MECHANISM:
            
                subsequent occurrences: necessary usage limit sampled from Weibull distribution
                This assumes that the system only starts looking at usage when the pattern has fully occurred
                Additional restraint that the pattern can only occur between 0 and 6 in the morning
            """
            u_threshold = sps.exponweib.rvs(float(self.avg_interusage), 0.5, loc=float(self.avg_interusage), scale=1.0, size=1)
            i, last_i = 1, 0
            while i < len(signal_data):
                # cumulative usage
                cumul_usage = np.sum(signal_data[last_i:i])
                
                # crossed a threshold and is between the appropriate hours?
                insert = False
                if cumul_usage >= u_threshold:
                    if self.time_constraint:
                        if hour_of_the_day[i] <= 6 and hour_of_the_day[i] >= 0:
                            insert = True
                        else:
                            insert = False
                    else:
                        insert = True
                
                # insert the pattern + resample threshold
                if insert:
                    # sample a pattern variation
                    p_ix = np.random.randint(0, len(pattern_variations)-1)
                    pvar = pattern_variations[p_ix]
                    plen = len(pvar)
                    
                    # adapt the location to not overlap with the switch of the month
                    if month_vals[i] != month_vals[i+plen]:
                        # shift to the previous month
                        for j in range(plen):
                            if month_vals[i] != month_vals[i+j]:
                                break
                            shift = j
                        # apply shift
                        i = i - shift
                        print('Shifted with', shift, 'values from', i+shift, 'to', i)
                    
                    # insert the variation
                    if self.superimposed == True:
                        signal_data[i:i+plen] = signal_data[i:i+plen] + pvar
                    else:
                        signal_data[i:i+plen] = pvar
                        
                    # pattern location
                    pattern_locs[i:i+plen] = 1.0
                        
                    # resample the threshold
                    u_threshold = sps.exponweib.rvs(float(self.avg_interusage), 0.5, loc=float(self.avg_interusage), scale=1.0, size=1)
                    
                    # jump ahead
                    i += plen
                    last_i = i
                
                i += 1
        
        return signal_data, pattern_locs
    
    def generate_train_and_test_data(self, basic_data, signal_data, pattern_locs):
        """ Train data of 1 year + the subsequent years are test months """
        
        # --------
        # TRAINING
        # --------
        
        # use the full dataset as the training data --> in the experiments we will only use part of it!
        train_data = pd.DataFrame(data={'timestamp': basic_data.timestamp.values,
                                        'usage_m3': signal_data,
                                        'pattern': pattern_locs,
                                        'missing_label': np.zeros(len(signal_data))})
        train_data.set_index('timestamp', inplace=True)
        
        # add the month values to the training data
        train_data['month'] = self._increasing_month_values(train_data.index.values)
        
        # --------
        # TESTING
        # --------
        
        """ NOTE: visually inspected and seems ok """
        
        test_sets = {}
        
        # basic test data start after a year
        # add the clean (i.e., without pattern) data (this will be used later to replace the pattern)
        test_basic = train_data[(365 * self.day_len):].copy()
        test_basic['clean'] = basic_data[(365 * self.day_len):].usage.values
        test_basic['noise'] = basic_data[(365 * self.day_len):].noise.values
        
        # construct the test sets
        month_ix = test_basic.month.values[0]
        last_month = test_basic.month.values[-1]
        i = 0
        stop = False
        while not(stop):
            # select the next 2 months
            # first month to get the patterns, second month to get clean data
            if self.two_months_test:
                month_single = test_basic[(test_basic['month'] == month_ix) | (test_basic['month'] == month_ix+1)]
                month_double = test_basic[(test_basic['month'] == month_ix) | (test_basic['month'] == month_ix+1) | (test_basic['month'] == month_ix+2)]
            else:
                month_single = test_basic[test_basic['month'] == month_ix]
                month_double = test_basic[(test_basic['month'] == month_ix) | (test_basic['month'] == month_ix+1)]
            
            # variables
            te_times = month_double.index.values.copy()
            te_signal = month_double.usage_m3.values.copy()
            te_clean = month_double.clean.values.copy()
            te_noise = month_double.noise.values.copy()
            te_locs = month_single.pattern.values.copy()
            mpat_ranges = self._find_pattern_ranges(te_locs)
            if len(mpat_ranges) == 1:
                print('Skipped month:', month_ix)
                month_ix += 1
                continue
            
            # indices of the end of the last pattern and the missing pattern
            ix_last = mpat_ranges[-2][-1]
            miss_pat = mpat_ranges[-1]
            ix_miss = miss_pat[-1]
            
            # remove the final occurrences (and replace by clean signal)
            te_signal[ix_last:] = te_clean[ix_last:]
            
            # failure after the last pattern
            """ TODO: add other failure types """
            if self.failure_type == 'catastrophic':
                ix_fail = ix_miss + int(self.day_len * self.breakdown_time)
                te_signal[ix_fail:] = te_noise[ix_fail:]
                
            # correct the pattern locations and the missing pattern location
            te_locs = month_double.pattern.values.copy()
            te_locs[ix_last:] = 0.0
            te_miss = np.zeros(len(month_double))
            te_miss[int((miss_pat[-1] + miss_pat[0]) / 2)] = 1.0
            
            # construct the test data
            test_df = pd.DataFrame(data={'timestamp': te_times,
                                         'usage_m3': te_signal,
                                         'pattern': te_locs,
                                         'missing_label': te_miss})
            test_df.set_index('timestamp', inplace=True)
            
            # store the new_data in the appropriate location
            test_sets[i] = test_df
            
            if (month_ix == last_month - 2) or i > 15:
                stop = True
            
            month_ix += 1
            i += 1
        
        return train_data, test_sets
    
    def _increasing_month_values(self, timestamps):
        month_vals = np.array([pd.to_datetime(t).month for t in timestamps])
        new_vals = np.zeros(len(month_vals))
        c = 0
        for i, e in enumerate(month_vals):
            if i > 0:
                if prev_e != e:
                    c += 1
            new_vals[i] = c
            prev_e = e
        return new_vals
    
    def _find_pattern_ranges(self, y):
        """ Find the indices of each free-range pattern.

        :returns ranges : array of arrays
            Each array corresponds to the indices of the pattern.
        """

        ranges = []
        ix = np.where(y == 1.0)[0]
        if len(ix) > 0:
            bp, ep = ix[0], ix[0]
            for i, e in enumerate(ix):
                if i == len(ix) - 1:
                    if e - ep > 1:
                        ranges.append(np.arange(bp, ep+1, 1))
                    else:
                        ranges.append(np.arange(bp, e+1, 1))
                elif e - ep > 1:
                    ranges.append(np.arange(bp, ep+1, 1))
                    bp = e
                ep = e
        return np.array(ranges)


def read_data_from_UCR_file(file_path):
    """ Read data from UCR standard file and return as an array

    :param file_path: string
        Full file path without file name.

    :returns train_data: pd.DataFrame()
        Data from the TRAIN file, first column are the classes.
    :returns test_data: pd.DataFrame()
        Data from the TEST file, first column are the classes.
    """

    # relevant files in the path
    all_files = os.listdir(file_path)
    # all_files = [f for f in all_files if '.txt' in f]

    # for each file
    for file_name in all_files:
        if 'TRAIN' in file_name:
            train_d = pd.read_csv(os.path.join(file_path, file_name), sep=',', header=None)
        elif 'TEST' in file_name:
            test_d = pd.read_csv(os.path.join(file_path, file_name), sep=',', header=None)
        else:
            train_d, test_d = pd.DataFrame(), pd.DataFrame()
    
    all_data = pd.concat((train_d, test_d))
    
    return all_data  # train_d, test_d


def get_patterns(file_path, classes, n_samples=[100, 140], rescale_bot=[0.5, 1.0], rescale_top=[4.0, 5.0], sample_technique='fft'):
    """ Get the patterns for these classes using the given file path """
    
    # get the data
    all_data = read_data_from_UCR_file(file_path)
    
    # labels and data
    labels = all_data.iloc[:, 0].values
    ts_data = all_data.iloc[:, 1:].values
    label_types = np.unique(labels)
    
    # select the data for the classes
    class_data = np.array([])
    for c in classes:
        idx = np.where(labels == c)[0]
        l_data = ts_data[idx, :]
        if len(class_data) == 0:
            class_data = l_data
        else:
            class_data = np.vstack((class_data, l_data))
    ns, ts_length = class_data.shape
            
    # upsample/downsample
    if n_samples[0] > 0 and n_samples[1] > 0:
        patterns = {}
        for i in range(ns):
            # randomly select resample rate
            np.random.seed(GLOBAL_RANDOM_SEED)
            rr = np.random.choice(np.arange(n_samples[0], n_samples[1], 1), 1)[0]
            # resample
            if sample_technique == 'fft':
                sample = sp.signal.resample(class_data[i, :], num=rr)
            elif sample_technique == 'linear':
                pass
            patterns[i] = sample
    else:
        patterns = {i: d for i, d in enumerate(class_data)}
        
    # rescale patterns to fit between rescale range
    for i, p in patterns.items():
        np.random.seed(GLOBAL_RANDOM_SEED)
        a = np.random.uniform(rescale_bot[0], rescale_bot[1])
        b = np.random.uniform(rescale_top[0], rescale_top[1])
        p = ((b - a) * (p - min(p))) / (max(p) - min(p))
        patterns[i] = p
    
    return patterns





if __name__ == '__main__':
    main()
    