from datetime import datetime, timedelta, time
import numpy as np
import pandas as pd
from polygon_s3 import fetch_date_df
from polygon_ds import get_dates_df
from utils_filters import TickRule, MADFilter, JMAFilter, jma_filter_df
from bar_samples import BarSampler
from bar_labels import label_bars



def process_bar_dates(bar_dates: list, imbalance_thresh: float=0.95) -> pd.DataFrame:
    
    results = []
    for date_d in bar_dates:
        bdf = pd.DataFrame(date_d['bars'])
        results.append({
            'date': date_d['date'], 
            'bar_count': len(date_d['bars']),
            'imbalance_thresh': bdf.volume_imbalance.quantile(q=imbalance_thresh),
            'duration_min_mean': bdf.duration_min.mean(),
            'duration_min_median': bdf.duration_min.median(),
            'price_range_mean': bdf.price_range.mean(),
            'price_range_median': bdf.price_range.median(),
            'thresh': date_d['thresh']
            })
    daily_bar_stats_df = jma_filter_df(pd.DataFrame(results), 'imbalance_thresh', length=5, power=1)
    daily_bar_stats_df.loc[:, 'imbalance_thresh_jma_lag'] = daily_bar_stats_df['imbalance_thresh_jma'].shift(1)
    daily_bar_stats_df = daily_bar_stats_df.dropna()
    
    return daily_bar_stats_df


def stacked_df_stats(stacked_df: pd.DataFrame) -> pd.DataFrame:

    bars_df = stacked_df[stacked_df['bar_trigger'] != 'gap_filler'].reset_index(drop=True)
    bars_df.loc[:, 'date'] = bars_df['close_at'].dt.date.astype('string')
    bars_df.loc[:, 'duration_min'] = bars_df['duration_td'].dt.seconds / 60
    
    dates_df = bars_df.groupby('date').agg(
        bar_count=pd.NamedAgg(column="price_close", aggfunc="count"),
        duration_min_median=pd.NamedAgg(column="duration_min", aggfunc="median"),
        jma_range_mean=pd.NamedAgg(column="jma_range", aggfunc="mean"),
        first_bar_open=pd.NamedAgg(column="open_at", aggfunc="min"),
        last_bar_close=pd.NamedAgg(column="close_at", aggfunc="max"),
    ).reset_index()

    return dates_df


def get_symbol_vol_filter(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    # get exta 10 days
    adj_start_date = (datetime.fromisoformat(start_date) - timedelta(days=10)).date().isoformat()
    # get market daily from pyarrow dataset
    df = get_dates_df(symbol='market', tick_type='daily', start_date=adj_start_date, end_date=end_date, source='local')
    df = df.loc[df['symbol'] == symbol].reset_index(drop=True)
    # range/volitiliry metric
    df.loc[:, 'range'] = df['high'] - df['low']
    df = jma_filter_df(df, col='range', winlen=5, power=1)
    df.loc[:, 'range_jma_lag'] = df['range_jma'].shift(1)
    # recent price/value metric
    df.loc[:, 'price_close_lag'] = df['close'].shift(1)
    df = jma_filter_df(df, col='vwap', winlen=7, power=1)
    df.loc[:, 'vwap_jma_lag'] = df['vwap_jma'].shift(1)
    return df.dropna().reset_index(drop=True)


def fill_gap(bar_1: dict, bar_2: dict, renko_size: float, fill_col: str) -> dict:

    num_steps = round(abs(bar_1[fill_col] - bar_2[fill_col]) / (renko_size / 2))
    fill_values = list(np.linspace(start=bar_1[fill_col], stop=bar_2[fill_col], num=num_steps))
    fill_values.insert(-1, bar_2[fill_col])
    fill_values.insert(-1, bar_2[fill_col])
    fill_dt = pd.date_range(
        start=bar_1['close_at'] + timedelta(hours=1),
        end=bar_2['open_at'] - timedelta(hours=1),
        periods=num_steps + 2,
        )
    fill_dict = {
        'bar_trigger': 'gap_filler',
        'close_at': fill_dt,
        fill_col: fill_values,
    }
    return pd.DataFrame(fill_dict).to_dict(orient='records')


def fill_gaps_dates(bar_dates: list, fill_col: str) -> pd.DataFrame:

    for idx, date in enumerate(bar_dates):
        if idx == 0:
            continue

        try:
            gap_fill = fill_gap(
                bar_1=bar_dates[idx-1]['bars'][-1],
                bar_2=bar_dates[idx]['bars'][1],
                renko_size=bar_dates[idx]['thresh']['renko_size'],
                fill_col=fill_col,
            )
            bar_dates[idx-1]['bars'] = bar_dates[idx-1]['bars'] + gap_fill
        except:
            print(date['date'])
            continue
    # build continous 'stacked' bars df
    stacked = []
    for date in bar_dates:
        stacked = stacked + date['bars']

    return pd.DataFrame(stacked)


def filter_tick(tick: dict, mad_filter: MADFilter, jma_filter: JMAFilter, 
                tick_rule: TickRule, bar_sampler: BarSampler) -> dict:
    
    irregular_conditions = [2, 5, 7, 10, 13, 15, 16, 20, 21, 22, 29, 33, 38, 52, 53]

    tick['nyc_time'] = tick['sip_dt'].tz_localize('UTC').tz_convert('America/New_York')

    mad_filter.update(next_value=tick['price'])  # update mad filter

    if tick['volume'] < 1:  # zero volume/size tick
        tick['status'] = 'zero_volume'
    elif pd.Series(tick['conditions']).isin(irregular_conditions).any():  # 'irrgular' tick condition
        tick['status'] = 'irregular_condition'
    elif abs(tick['sip_dt'] - tick['exchange_dt']) > pd.to_timedelta(2, unit='S'):  # large ts deltas
        tick['status'] = 'ts_delta'
    elif mad_filter.status != 'mad_clean':  # MAD filter outlier
        tick['status'] = 'mad_outlier'
    else:  # 'clean' tick
        tick['status'] = 'clean'
        tick['jma'] = jma_filter.update(next_value=tick['price'])  # update jma filter
        tick['side'] = tick_rule.update(next_price=tick['price'])  # update tick rule
        
        if tick['nyc_time'].to_pydatetime().time() < time(hour=9, minute=30):
            tick['status'] = 'clean_pre_market'
        elif tick['nyc_time'].hour >= 16:
            tick['status'] = 'clean_after_hours'
        else:
            tick['status'] = 'clean_open_market'
            bar_sampler.update(tick)

    tick.pop('sip_dt', None)
    tick.pop('exchange_dt', None)
    tick.pop('conditions', None)

    return tick


def build_bars(ticks_df: pd.DataFrame, thresh: dict) -> tuple:

    mad_filter = MADFilter(thresh['mad_value_winlen'], thresh['mad_deviation_winlen'], thresh['mad_k'])
    jma_filter = JMAFilter(thresh['jma_winlen'], thresh['jma_power'])
    tick_rule = TickRule()
    bar_sampler = BarSampler(thresh)
    ft_ticks = []
    for row in ticks_df.itertuples():
        tick = {
            'sip_dt': row.sip_dt,
            'exchange_dt': row.exchange_dt,
            'price': row.price,
            'volume': row.size,
            'conditions': row.conditions,
            'status': 'raw',
        }
        ft_tick = filter_tick(tick, mad_filter, jma_filter, tick_rule, bar_sampler)
        ft_ticks.append(ft_tick)

    return bar_sampler.bars, pd.DataFrame(ft_ticks)


def bar_workflow(symbol: str, date: str, thresh: dict, add_label: bool=True) -> dict:
    
    # get ticks
    ticks_df = fetch_date_df(symbol, date, tick_type='trades')

    # sample bars
    bars, ft_tdf = build_bars(ticks_df, thresh)

    if add_label:
        bars = label_bars(
            bars=bars,
            ticks_df=ft_tdf[ft_tdf['status'].str.startswith('clean_open_market')],
            risk_level=thresh['renko_size'],
            horizon_mins=thresh['max_duration_td'].total_seconds() / 60,
            reward_ratios=thresh['label_reward_ratios'],
            )

    bar_date = {
        'symbol': symbol,
        'date': date,
        'thresh': thresh,
        'bars': bars,
        'ticks_df': ft_tdf,
        }

    return bar_date
