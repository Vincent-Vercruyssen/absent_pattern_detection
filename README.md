# Detecting Suspicious Pattern Absences in Continuous Time Series

Code for the algorithms used in *Detecting Suspicious Pattern Absences in Continuous Time Series*.

Author: Vincent Vercruyssen, 2019.

## Abstract:

*Given its large applicational potential, time series anomaly detection has become a crucial data mining task as it attempts to identify periods of a time series where there is a deviation from the expected behavior. While existing approaches focus on analyzing whether the currently observed behavior is different from the previously seen, normal behavior, the task targeted here implies that the absence of a previously observed behavior is indicative of an anomaly. In other words, a pattern that is expected to recur in the time series is absent. In real-world use cases, absent patterns can be linked to serious problems. For instance, if a scheduled, regular maintenance operation of a machine does not take place, this can be harmful to the machine at a later time. In this paper, we introduce the task of detecting when a specific pattern is absent in a real-valued time series and propose a novel technique, FZapPa, to tackle this task. FZapPa first learns to detect all occurrences of a specific pattern in the time series, then uses statistical techniques to model how the occurrences are distributed in the time series data, and finally predicts the absence of the pattern in previously unseen data. We show the effectiveness of FZapPa by evaluating it on a benchmark of real-world datasets.*

## Disclaimer:

The repository contains easy-to-use versions of the algorithms from the paper.

In addition, the repository contains the constructed power usage datasets from the paper, and the script detailing how to construct such datasets. Running the script requires some action on the users' part (i.e., filling in the correct paths, downloading the UCI and UCR data, transforming the UCR datafiles to csv files...). The UCI and UCR data are not included in this repository in order to minimize its size and because they are freely available elsewhere.

## Repository:

The repository contains the **python** code for the following algorithms:

1. FZapPa
2. MatrixProfile
3. iForest for time series
4. kNNo for time series
5. LOF for time series
6. AverageMissing
7. SSDO
8. SSAD

## Installation and usage:

To use the repository, simply download everything. The **data/** folder contains the power usage data. The **missing_patterns/** folder contains the algorithms that were used in the paper. Most of them follow the fit-predict format as popularized by scikit-learn. In order to use them, most algorithms require three inputs (besides setting the hyperparameters): the time series values, the accompanying time stamps, and a vector of 0/1 indicating for each time stamp whether user annotated a pattern or not.


