# Reliable Predictive Inference with Corrupted Data

An important factor to guarantee a responsible use of data-driven systems is that we should be able to communicate their uncertainty to decision makers. This can be accomplished by constructing prediction sets, which provide an intuitive measure of the limits of predictive performance.

This package contains a Python implementation of Privileged Conforaml Prediction (PCP) [1] methodology for constructing distribution-free prediction sets that provably achieve the desired coverage level when the observed data is corrupted. 

# Robust Conformal Prediction Using Privileged Information [1]

PCP is a method that constructs reliable prediction sets under the setting that the available data is corrupted.

[1] Shai Feldman, Yaniv Romano, [“Robust Conformal Prediction Using Privileged Information.”](???) 2024.


### Prerequisites

* python
* numpy
* scipy
* scikit-learn
* pytorch
* pandas

### Installing

The development version is available here on github:
```bash
git clone https://github.com/shai128/pcp.git
```

## Usage


## Reproducible Research

The code available under src/run_all_experiments.py in the repository replicates the experimental results in the paper.

Please refer to [notebooks/Paper figures.ipynb](notebooks/Paper-figures.ipynb) to view the results presented in the main text.

Refer to [notebooks/Supplementary figures.ipynb](notebooks/Supplementary-figures.ipynb) to view the results presented in the appendix.


### Publicly Available Datasets

* [Facebook Variant 1 and Variant 2](https://archive.ics.uci.edu/ml/datasets/Facebook+Comment+Volume+Dataset): Facebook  comment  volume  data  set.

* [Bio](https://archive.ics.uci.edu/ml/datasets/Physicochemical+Properties+of+Protein+Tertiary+Structure): Physicochemical  properties  of  protein  tertiary  structure  data  set.

* [House](https://www.kaggle.com/c/house-prices-advanced-regression-techniques): House prices.

* [Blog](https://archive.ics.uci.edu/ml/datasets/BlogFeedback): BlogFeedback data set.

* [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html): CIFAR-10.

* [CIFAR-10N](http://noisylabels.com/): CIFAR-10N.

* [CIFAR-10C](https://github.com/hendrycks/robustness): CIFAR-10C.

* [Twins](https://github.com/AMLab-Amsterdam/CEVAE/tree/master/datasets/TWINS): Twins.

* [IHDP](https://www.fredjo.com/): IHDP.
 
* [NSLM](https://github.com/grf-labs/grf/blob/master/experiments/acic18/synthetic_data.csv): NSLM.



### Data subject to copyright/usage rules

The Medical Expenditure Panel Survey (MPES) data can be downloaded using the code in the folder /get_meps_data/ under this repository. It is based on [this explanation](/get_meps_data/README.md) (code provided by [IBM's AIF360](https://github.com/IBM/AIF360)).

* [MEPS_19](https://meps.ahrq.gov/mepsweb/data_stats/download_data_files_detail.jsp?cboPufNumber=HC-181): Medical expenditure panel survey,  panel 19.


