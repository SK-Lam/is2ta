## packages

from collections import defaultdict
import numpy as np
import pandas as pd
import json

from scipy.interpolate import CubicSpline


##import data 
def load_skate_json(filepath):

    with open(filepath, "r") as f:
        return json.load(f)

# load raw data
segment= load_skate_json("segments.json")
meanlines= load_skate_json("meanlines.json")



## helper functions

def skate_processing(dict):
    '''Extract geometry information from dictionaries outputed by SKATE.
    Ran on both segment and meanline json files.
    '''
    list = dict['features']

    df = pd.DataFrame(list).rename(columns={'type':'type_0'})
    details_df = df.geometry.apply(pd.Series)
    data = pd.concat([df.drop(columns = ['geometry']), details_df], axis = 1)
    
    data_clean = data.drop(columns = ['type_0', 'properties', 'type'])

    return data_clean

def get_segment_avgs(segment_data):
    '''Get segment centroids of each segment.
    
    :segment_data: skate_processing(segments.json)
    
    :return: array of [x_centroid, y_centroid] for each segment'''

    segments = segment_data['coordinates']
    return np.array([np.mean(segment, axis=0) for segment in segments])

def sze_meanline_algorithm(meanline_data):
    '''Original meanline algorithm as given by Sze. Slightly modified to be more concise but otherwise unchanged. The algorithm searches through meanlines and filters outlier slopes while saving the median slope. In then interpolates gaps in the meanlines using the median slope and median meanline interval length.
    
    :meanline_data: skate_processing(meanlines.json)

    :return: array of meanline data, each row [slope, x0, y0, xn, yn],
    median meanline slope, median meanline interval gap.
    '''
    pairs = meanline_data['coordinates']

    # convert coordinate pairs to arrays
    first_coords = np.array([pair[0] for pair in pairs])
    second_coords = np.array([pair[1] for pair in pairs])


    # list of all meanline slopes
    slopes = (second_coords[:,1] - first_coords[:, 1]) / (second_coords[:,0] - first_coords[:, 0])

    # filter out outlier slopes based on quartiles
    slope_q1 = np.quantile(slopes, 0.25)
    meanline_slope = np.median(slopes)
    slope_q3 = np.quantile(slopes, 0.75)

    # create a boolean mask of where slopes are within the desired range
    mask = (slopes >= slope_q1) & (slopes <= slope_q3)

    # apply mask to extract relevant values
    filtered_slopes = slopes[mask]
    filtered_first = first_coords[mask]
    filtered_second = second_coords[mask]

    # atack the filtered data into a single array (N, 5)
    line_data = np.column_stack([
        filtered_slopes,
        filtered_first[:, 0],  # x0
        filtered_first[:, 1],  # y0
        filtered_second[:, 0], # x1
        filtered_second[:, 1], # y1
    ])

    # grab y_coords, sorted
    y_coords = np.sort(second_coords[:,1])

    # get interval lengths between each filtered meanline
    intervals = []
    for i in range(1, len(y_coords)):
        interval = y_coords[i] - y_coords[i-1]
        intervals.append(interval)

    # median, q1, & small interval length
    interval_small = np.quantile(intervals, 0.1)
    median_interval = np.median(intervals)
    interval_q3 = np.quantile(intervals, 0.75)

    line_data = line_data[np.argsort(line_data[:, -1])]

    # create new line data based on gaps in intervals
    line_data_new = line_data.copy()
    for i in range(1, len(line_data)):
        interval = (line_data[i,-1] - line_data[i-1,-1])
        if interval >= interval_q3: # better threshold?
            num_add = round(interval/median_interval)
            for n in range(1, num_add+1):
                new_line_data = [line_data[i-1,0], line_data[i-1,1], line_data[i-1,2] + n*median_interval, line_data[i-1,3], line_data[i-1,4] + n*median_interval]
                line_data_new = np.vstack([line_data_new, new_line_data])

    line_data_new = line_data_new[np.argsort(line_data_new[:, -1])]
    return line_data_new, meanline_slope, median_interval

def get_tracelines(segment_data, meanline_array_fixed, median_interval, segment_avgs):
    '''Takes outputs from Sze's meanline algorithm to create "trace lines". These are simply meanlines that have been adjusted to more closely fit the data for the purposes of segment assignment. New lines are created by fitting lines of best fit to the closest 100 points to each meanline that have evenly spaced x-coordinates. Lines that are too close are then deleted.
    
    Since the meanlines have been refitted to the data, they are no longer accurate meanlines. Therefore, they are referred as trace lines.
    
    :segment_data: skate_processing(segments.json)
    :meanline_array_fixed: array output of sze's meanline algorithm
    :median_interval: median meanline interval gap
    :segment_avgs: array of centroids of each segment
    
    :return: array of traceline data, each row [slope, x0, y0, xn, yn]
    '''
    # x start and stop of centroids
    start = np.min(segment_avgs[:,0])
    stop = np.max(segment_avgs[:,0])

    # shift newly created lines by refitting the line to the closest 100 points to the line, with even x coordinates
    n_sample = 100
    bins_len = (stop-start)/100
    line_fitted_data = np.zeros((len(meanline_array_fixed), 2))
    for i, line in enumerate(meanline_array_fixed):
        x = np.zeros(100)
        y = np.zeros(100)
        for n in range(100):
            filter = (segment_avgs[:,0] >= n*(bins_len) + start) & (segment_avgs[:,0] <= (n+1)*(bins_len) + start)
            segment_avgs_filter = segment_avgs[filter]
            distances = np.abs((line[0]*(0 - line[1]) + line[2] + line[0]*segment_avgs_filter[:,0] - segment_avgs_filter[:,1]) / np.sqrt(1 + line[0]**2))
            closest_point_arg = np.argmin(distances)
            x[n] = segment_avgs_filter[closest_point_arg, 0]
            y[n] = segment_avgs_filter[closest_point_arg, 1]
        m, k = np.polyfit(x, y, 1)
        line_fitted_data[i] = [m,k]

    line_data_fitted = np.column_stack([
        line_fitted_data[:, 0],
        meanline_array_fixed[:, 1],
        line_fitted_data[:, 1],
        meanline_array_fixed[:, 3],
        meanline_array_fixed[:, 3] * line_fitted_data[:, 0] + line_fitted_data[:, 1]
    ])

    # delete lines that are too close, keeping the one that gives better spacing
    delete_filter = np.ones(len(line_data_fitted), dtype=bool)
    for i in range(1, len(line_data_fitted)):
        if not delete_filter[i-1]:
            previous = i-2
            current = i 
        else:
            previous = i-1
            current = i
        interval = line_data_fitted[current, -1] - line_data_fitted[previous, -1]
        if interval <= median_interval/2: # diff threshold?
            # []_spacing gives the resulting spacing difference assuming that ~[] is removed (current/previous)
            try:
                next_y = line_data_fitted[current+1, -1]
            except:
                next_y = line_data_fitted[current, -1] + median_interval
            current_spacing = np.abs((next_y - line_data_fitted[current-1, -1]) - (line_data_fitted[current, -1] - line_data_fitted[previous-1, -1]))
            previous_spacing = np.abs((next_y - line_data_fitted[previous, -1]) - (line_data_fitted[previous, -1] - line_data_fitted[previous-1, -1]))
            if current_spacing <= previous_spacing:
                delete_filter[previous] = False
            else:
                delete_filter[current] = False

    line_data_final = line_data_fitted[delete_filter]
    line_data_w_slopes = line_data_final[np.argsort(line_data_final[:, -1])]

    return line_data_w_slopes

def array_transform(segment_data, type = 'simple'):
    '''Transform segment data into array format. Segments are grouped by their corresponding trace. Each row corresponds to a new trace, with columns corresponding to y-values. Going by trace, y-values from the segments data is inputed into the array. There are two accepted types of array_tranform:
        
        "simple": If one trace has two segments that have different y-values for the same x-value, both segments are discarded. Hence, segments are "globally discarded".
        
        "complex": Rather than discarding the entire segment, x-values are looked at one at a time. If one x-value has multiple y-values, than the y-value corresponding to the segment closest to the trace is kept. Using type="complex" requires needing an 'error' column in segment_data in order to compute which segment is closest to the trace.
    
    :segment_data: must include 'trace_id', and 'error' if type = "complex"
    :type: "simple" or "complex"
    
    :return: full range of x, matrix of Y values (trace by x-value), matrix with Y values discarded due to being duplicates (each column corresponds to a list of y-values discarded for that x-value)
    '''
    all_ids = np.sort(segment_data['trace_id'].unique())
    N = len(all_ids)
    grouped = segment_data.groupby('trace_id')
    
    # map to hold x_to_ys per trace_id and accumulate global x values
    id_to_xys = {}
    global_x_set = set()
    
    for trace_id, group in grouped:
        x_to_ys = defaultdict(list)
        for segment_id, coords in zip(group.index, group['coordinates']):
            for x, y in coords:
                x_to_ys[x].append((segment_id, y))
        id_to_xys[trace_id] = x_to_ys
        global_x_set.update(x_to_ys.keys())
    
    # global x range and map x to column index
    x_full = np.arange(min(global_x_set), max(global_x_set) + 1)
    X = len(x_full)
    x_to_col_idx = {x: i for i, x in enumerate(x_full)}
    
    # initialize
    Y_main = np.full((N, X), np.nan)
    duplicate_lists = [[] for _ in range(X)]
    
    if type == 'complex':
        error_lookup = segment_data['error'].to_dict()
    
    # process each trace_id
    for row_idx, trace_id in enumerate(all_ids):
        x_to_ys = id_to_xys[trace_id]

        if type == 'simple':
            # identify x with duplicate y values (by segment_id) globally
            segment_ids_to_remove = {
                seg_id
                for pairs in x_to_ys.values() if len(pairs) > 1
                for seg_id, _ in pairs
            }
            
            # process and separate clean vs. duplicate values
            for x, pairs in x_to_ys.items():
                col_idx = x_to_col_idx[x]
                clean = [(sid, y) for sid, y in pairs if sid not in segment_ids_to_remove]
        
                if len(clean) == 1:
                    Y_main[row_idx, col_idx] = clean[0][1]
                elif len(clean) > 1:
                    for seg_id, y in clean:
                        duplicate_lists[col_idx].append(y)

        if type == 'complex':
            # no global segment removal
            for x, pairs in x_to_ys.items():
                col_idx = x_to_col_idx[x]

                if len(pairs) == 1:
                    # only one segment with this x — keep it
                    Y_main[row_idx, col_idx] = pairs[0][1]
                else:
                    # multiple segments for the same x — choose one with smallest error
                    sorted_pairs = sorted(pairs, key=lambda tup: error_lookup.get(tup[0], np.inf))
                    best_segment = sorted_pairs[0]
                    Y_main[row_idx, col_idx] = best_segment[1]

                    # add the rest to duplicates
                    for seg_id, y in sorted_pairs[1:]:
                        duplicate_lists[col_idx].append(y)
  
    # convert duplicates to matrix
    M = max(len(ys) for ys in duplicate_lists)
    Y_dups = np.full((M, X), np.nan)
    for col_idx, ys in enumerate(duplicate_lists):
        Y_dups[:len(ys), col_idx] = ys

    return x_full, Y_main, Y_dups

def spline_interpolate(array, max_gap=200):
    '''Interpolates gaps in matrix of y-values. Small gaps use spline interpolation while larger gaps are interpolated by a best fit line.

    :array: Y_main from array_transform
    :max_gap: cuttoff to determine small vs. large gaps of NA values

    :return: interpolated Y_main
    '''
    array = array.copy()
    result = np.full_like(array, np.nan, dtype=float)
    skip_mask = np.zeros_like(array, dtype=bool)

    mask = np.isfinite(array)
    result[mask] = array[mask]

    gap_start = None
    for i in range(len(array)):
        if not mask[i] and gap_start is None:
            gap_start = i
        elif mask[i] and gap_start is not None:
            gap_end = i
            gap_len = gap_end - gap_start

            if gap_len > max_gap:
                skip_mask[gap_start:gap_end] = True

            gap_start = None

    valid_idx = np.where(np.isfinite(result) & ~skip_mask)[0]
    if len(valid_idx) > 1:
        spline = CubicSpline(valid_idx, result[valid_idx], extrapolate=False)
        interp_idx = np.where(~np.isfinite(result) & ~skip_mask)[0]
        result[interp_idx] = spline(interp_idx)

    final_mask = np.isfinite(result)
    if np.sum(final_mask) >= 2:
        x_fit = np.where(final_mask)[0]
        y_fit = result[x_fit]
        m, b = np.polyfit(x_fit, y_fit, 1)
        fill_idx = np.where(~np.isfinite(result))[0]
        result[fill_idx] = m * fill_idx + b

    return result

def tick_detect(df, diffs_col='diffs', flag_col='non_outlier'):
    '''Helper function in minute_mark_func. Detects minute tick marks when there is a noticable jump in the data.'''
    # initialize column with True
    df[flag_col] = True

    # conditions
    cond_current = df[diffs_col].between(-50, -25)
    cond_next = df[diffs_col].shift(-1).between(25, 50)

    # flag where both conditions are True
    to_flag = cond_current & cond_next

    # set the flag to False where the pattern is found
    df.loc[to_flag, flag_col] = False

    return df

def minute_mark_func(segment_data, segment_avgs):
    '''Minute mark detection function. Detects outliers when there is a notable jump in the data. Jumps are then analyzed to find a common spacing: this determines the minute mark spacing, which is saved and used to further help detect minute marks.
    
    This function can be edited to be more accurate in filling in data for minute marks.
    '''
    segment_data = segment_data.copy()
    # extract first x from each list in 'coordinates'
    segment_data['x_first'] = segment_data['coordinates'].apply(lambda coords: coords[0][0])
    # extract avg y
    segment_data['y_avg'] = segment_avgs[:,1]
    
    segment_data = segment_data.sort_values(by=['final_trace_id', 'x_first'], ascending=[True, True]).reset_index(drop=True)
    
    # get differences
    differences = segment_data.groupby('final_trace_id')['y_avg'].diff()
    segment_data.loc[1:,'diffs'] = differences
    
    # identify possible minute marks
    segment_data['non_outlier'] = True
    segment_data = tick_detect(segment_data)
    segment_outliers = segment_data[~segment_data['non_outlier']]
    
    # get info from minute marks
    segment_outliers['spacing'] = segment_outliers.groupby('final_trace_id')['x_first'].diff()
    minute_spacing = segment_outliers.groupby('final_trace_id')['spacing'].median().median()
    tick_segment_len = np.quantile(segment_outliers['segment_length'], 0.75)
    
    # remove most minute marks
    segment_data_filtered = segment_data[segment_data['segment_length'] > tick_segment_len]

    return minute_spacing, segment_data_filtered, tick_segment_len

def clip(Y_main, Y_interpolate, x_full, gap_run=50, onset_run=10):
    '''Clips beginning of array. The beginning of the array usually starts at segmentation errors on the left of the image. These are removed by finding large gaps in between these segments and where the rest of the data starts by searching for large gaps of interpolated segments. Then, the matrix is aligned so they all begin at a common x-value.
    
    :gap_run: the length of interpolated gaps used to detect that the trace started at an incorrect segment.
    :onset_run: the length of non-interpolate gap proceeding the interpolate gap to determine when the actual data starts.
    
    :returns: cleaned y-value matrix, new x range
    '''
    n_rows, n_cols = Y_main.shape
    finite_mask = np.isfinite(Y_main)

    nan_mask = ~finite_mask
    gap_window = np.ones(gap_run, dtype=int)
    gap_counts = np.apply_along_axis(
        lambda row: np.convolve(row, gap_window, mode='same'),
        axis=1, arr=nan_mask.astype(int)
    )
    gap_starts = np.argmax(gap_counts >= gap_run, axis=1)
    no_gap = ~np.any(gap_counts >= gap_run, axis=1)
    gap_starts[no_gap] = 0

    start_indices = np.full(n_rows, np.nan, dtype=float)
    onset_window = np.ones(onset_run, dtype=int)

    for r in range(n_rows):
        search_start = gap_starts[r] + gap_run
        if search_start >= n_cols:
            continue
        onset_counts = np.convolve(finite_mask[r], onset_window, mode='same')
        onset_counts[:search_start] = 0
        if np.any(onset_counts >= onset_run):
            start_indices[r] = np.argmax(onset_counts >= onset_run)

    median_start = np.nanmedian(start_indices)
    start = int(np.round(median_start))

    Y_masked = Y_interpolate[:, start:]
    x = np.arange(Y_masked.shape[1]) + x_full[start]

    last_row = Y_masked[-1].copy()
    last_nan_mask = ~np.isfinite(last_row).astype(int)
    # convolve to detect "long" gaps of NaNs
    gap_counts_end = np.convolve(last_nan_mask, gap_window, mode='same')
    # find the last position where gap >= gap_run
    large_gap_indices = np.where(gap_counts_end >= gap_run)[0]
    if len(large_gap_indices) > 0:
        cutoff = large_gap_indices[-1] + 1  # everything after this becomes NaN
        Y_masked[-1, cutoff:] = np.nan
    else:
        # if no large gap, optionally NaN all the trailing part after last non-NaN
        last_valid = np.max(np.where(np.isfinite(last_row))[0]) + 1
        Y_masked[-1, last_valid:] = np.nan

    return Y_masked, x

def spike_remover(array, limit = 15):
    '''Remove spikes based on outlier second derivative values.'''
    array = array.copy()
    for i in range(1, len(array)-26):
        if np.abs(array[i]) > limit:
            sign = np.sign(array[i])
            for j in range(i+1, i+25):
                if -sign*array[j] > limit:
                    array[i:j] = None
                    break
    return array

def y_transform(x, y, angle):
    '''Adjust y-axis based on meanlines.'''
    return y - np.arctan(angle)*x
    

## Full algorithm

def it2sa(segments, meanlines):
    '''Full segmentation algorithm (IT2SA). Outputs minute marker length, x-values, and y-value matrix for each trace.'''
    ####################################################################################################
    ## Phase 1: Meanline and Traceline Creation
    '''SKATE segment and meanline data is processed and cleaned. Tracelines are computed from cleaned meanlines.'''

    segment_data = skate_processing(segments)
    meanline_data = skate_processing(meanlines)

    segment_avgs = get_segment_avgs(segment_data)

    meanline_array, median_slope, median_interval = sze_meanline_algorithm(meanline_data)
    tracelines = get_tracelines(segment_data, meanline_array, median_interval, segment_avgs)

    ####################################################################################################
    ## Phase 2: Initial Traceline Assignment and Interpolation
    '''Segments are assigned to closest traceline. Segments that are too small are filtered out, and any segments are removed if they conflict with other segments' y-values. Leftover segments are interpolated by spline interpolation (for small gaps) or by a best fit line (for large gaps)'''

    new_ids = []
    for coord in segment_avgs:
        # assigns each segment to a traceline based on which traceline the segment centroid is closest to
        distances = np.abs((tracelines[:,0]*(0 - tracelines[:, 1]) + tracelines[:, 2] + tracelines[:, 0]*coord[0] - coord[1]) / np.sqrt(1 + tracelines[:, 0]**2))
        line_num = np.argmin(distances)
        new_ids.append(line_num)

    segment_data['trace_id'] = new_ids

    # remove segments that are too small (segmentation errors, smudges, tick marks, low data availability, etc.)
    segment_data['segment_length'] = segment_data['coordinates'].apply(len)
    outlier_len = np.quantile(segment_data['segment_length'], 0.4)
    segment_data_first_filtered = segment_data[segment_data['segment_length'] > outlier_len]

    # array transform and interpolation
    x_full, Y_main_first, Y_dups_first = array_transform(segment_data_first_filtered) # Y_dups_first not needed
    Y_interpolate_first = np.apply_along_axis(lambda x: spline_interpolate(x, max_gap = 100),
                                              axis = 1,
                                              arr = Y_main_first)
    
    ####################################################################################################
    ## Phase 3: Second Traceline Assignment and Interpolation
    '''Interpolated traces are now treated as the "new traceline". The processes from Phase 2 is then repeated. Additionally, minute marks are removed prior to segment interpolation. Segment transformation and interpolation remain the same; however, instead of throwing out the entire segment if it conflicts with another segment, x values are treated case-by-case. For ones that have a duplciate y-value, the y-value on the segment closest to the traceline is selected. 
    
    This results in segments being more properly assigned, leading to fewer gaps and more accurate interpolation.
    
    This phase has room for improvement. First of all, a more accurate minute mark detection function could used, one that uses minute mark data rather than interpolating the gaps. Additionally, segment assignment in this phase uses a cost function of distnace like phase 2. A different cost function that takes into account smoothness (L1 cost on Fourier tranform, or a regularization on L2 approach) could be more accurate.'''

    segment_data['final_trace_id'] = -1 # -1 used as placeholder for unknown trace
    segment_data['error'] = np.inf
    for idx, segment_row in segment_data.iterrows():
        x = np.array(segment_row['coordinates'])[:, 0]
        y = np.array(segment_row['coordinates'])[:,1]
        x_indices = np.where(np.isin(x_full, x))[0]
        if not x_indices.size > 0:
            continue
        else:
            Y = Y_interpolate_first[:,x_indices]
            try:
                errors = np.mean(np.abs(Y - y), axis = 1)
                new_trace = np.argmin(errors)
                segment_data.loc[idx, 'final_trace_id'] = new_trace
                segment_data.loc[idx, 'error'] = errors[new_trace] 
            except:
                continue
    
    segment_data_valid_traces = segment_data[segment_data.final_trace_id != -1]
    segment_avgs_valid = segment_avgs[segment_data.final_trace_id != -1]
    minute_spacing, segment_data_filtered, tick_segment_len = minute_mark_func(segment_data_valid_traces, segment_avgs_valid)

    # remove small segments
    outlier_segment_len = np.quantile(segment_data_filtered['segment_length'], 0.25)
    segment_data_filtered_known = segment_data_filtered[segment_data_filtered['segment_length'] > outlier_segment_len]
    segment_data_filtered_unknown = segment_data_filtered[segment_data_filtered['segment_length'] <= outlier_segment_len] # not currently used
    
    x_full, Y_main, Y_dups = array_transform(segment_data_filtered_known, type = 'complex') # Y_dups currently not used
    Y_interpolate = np.apply_along_axis(lambda x: spline_interpolate(x, max_gap = 100), axis = 1, arr = Y_main)

    ####################################################################################################
    ## Phase 4: Cleaning & Spike Removal
    '''Clips beginning values to determine correct start of data. Removed large spikes caused by segmentation errors at top and bottom of image.'''

    Y_aligned, x = clip(Y_main, Y_interpolate, x_full)

    # fix y-values based on meanlines
    theta = np.arctan(median_slope)
    Y_trans = np.apply_along_axis(lambda y: y_transform(x, y, theta), axis = 1, arr = Y_aligned)

    Y_second_raw = np.apply_along_axis(lambda x: np.diff(np.diff(x)), axis=1, arr=Y_trans)
    # pad with NaNs at beginning and end
    Y_second = np.full_like(Y_trans, np.nan)
    Y_second[:, 1:-1] = Y_second_raw

    Y_to_remove = np.apply_along_axis(spike_remover, axis = 1, arr = Y_second)
    Y_cleaned = Y_trans.copy()
    Y_cleaned[np.isnan(Y_to_remove)] = np.nan
    Y_cleaned = np.apply_along_axis(lambda x: spline_interpolate(x, max_gap = 200), axis = 1, arr = Y_cleaned)

    ####################################################################################################
    ## Output

    return x_full, Y_cleaned, minute_spacing
