import datetime as dt
import numpy as np
import pandas as pd
import pandas_bokeh
pandas_bokeh.output_file("data/bokeh_output.html")
import polygon_df as pdf
import polygon_ds as pds
import polygon_s3 as ps3
import bar_samples as bs
import bar_labels as bl
import bar_meta as bm
import utils_filters as ft
from utils_pickle import pickle_dump, pickle_load


# set sampling params
symbol = 'VTI'
start_date = '2020-04-14'
end_date = '2020-04-22'

thresh = {
    # mad filter
    'mad_value_winlen': 11,
    'mad_k': 22,    
    'mad_deviation_winlen': 333,     
    # jma filter
    'jma_winlen': 7,
    'jma_power': 2,
    # bar thresholds
    'renko_return': 'jma_return',
    'renko_size': 0.11,
    'renko_reveral_multiple': 2,
    'renko_range_frac': 22,
    'max_duration_td': dt.timedelta(minutes=33),
    'min_duration_td': dt.timedelta(seconds=33),
    'min_tick_count': 33,
    # label params
    'label_reward_ratios': list(np.arange(2.5, 11, 0.5)),
}


# bar workflow
date = '2020-04-12'

bar_date = bm.bar_workflow(symbol, date, thresh, add_label=True)
