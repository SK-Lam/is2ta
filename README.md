# IS2TA (Iterative Segment-to-Trace Assignment)

This code contains the IS2TA algorithm and the helper functions that supply it. The main function is the `is2ta` function at the end of the code, which takes in raw segment and meanline dictionaries (from json files SKATE outputs) and returns the range of x values, the array of y values (each row being a trace) and the distance between minute marks.

## Outline

The algorithmn follows three main features:

1. Creating Tracelines (time-series close to the trace we hope to recover)
2. Assigning SKATE segments to Tracelines and Interpolating
3. Minute-Mark Detection and Data Cleaning/Transforming

Features (1) and (2) are done iteratively: once segments are assigned, tracelines are restimated. IS2TA uses two iterations.

The code lays IS2TA into four phases:

1. Meanline and Traceline Creation
2. First Traceline Assignment and Interpolation
3. Second Traceline Assignment and Interpolation
4. Cleaning

Note minute-mark detection is done in phase (3). The details on how tracelines are created, assigned, and interpolated vary between phases (2) and (3) and are detailed in the code.

## Limitations

Please note that the minute mark detection algorithm and the final "axes transform" (to account for the drum moving) have not been fully implemented and could be improved. The former removes minute marks and interpolates between rather than using them and shifting down, while the latter simply uses an arctan transformation on the median meanline slope to transform the axes.




