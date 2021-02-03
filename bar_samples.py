from datetime import time
import pandas as pd
from bar_features import state_to_bar
from utils_filters import TickRule, MADFilter, JMAFilter



class BarActor:
    
    def __init__(self, thresh: dict):
        self.state = reset_state(thresh)
        self.bars = []

    def update(self, tick: dict):
        self.bars, self.state = update_bar_state(tick, self.state, self.bars, self.state['thresh'])


def reset_state(thresh: dict={}) -> dict:
    state = {}    
    state['thresh'] = thresh
    state['stat'] = {}
    # accumulators
    state['stat']['duration_td'] = None
    state['stat']['price_min'] = 10 ** 5
    state['stat']['price_max'] = 0
    state['stat']['price_range'] = 0
    state['stat']['price_return'] = 0
    state['stat']['jma_min'] = 10 ** 5
    state['stat']['jma_max'] = 0
    state['stat']['jma_range'] = 0
    state['stat']['jma_return'] = 0
    state['stat']['tick_count'] = 0
    state['stat']['volume'] = 0
    state['stat']['dollars'] = 0
    state['stat']['tick_imbalance'] = 0
    state['stat']['volume_imbalance'] = 0
    state['stat']['dollar_imbalance'] = 0
    # copy of tick events
    state['trades'] = {}
    state['trades']['date_time'] = []
    state['trades']['price'] = []
    state['trades']['volume'] = []
    state['trades']['side'] = []
    state['trades']['jma'] = []
    # trigger status
    state['trigger_yet?!'] = 'waiting'
    return state


def imbalance_net(state: dict) -> dict:
    state['stat']['tick_imbalance'] += state['trades']['side'][-1]
    state['stat']['volume_imbalance'] += (state['trades']['side'][-1] * state['trades']['volume'][-1])
    state['stat']['dollar_imbalance'] += (state['trades']['side'][-1] * state['trades']['volume'][-1] * state['trades']['price'][-1])
    return state


def imbalance_runs(state: dict) -> dict:
    if len(state['trades']['side']) >= 2:
        if state['trades']['side'][-1] == state['trades']['side'][-2]:
            state['stat']['tick_run'] += 1        
            state['stat']['volume_run'] += state['trades']['volume'][-1]
            state['stat']['dollar_run'] += state['trades']['price'][-1] * state['trades']['volume'][-1]
        else:
            state['stat']['tick_run'] = 0
            state['stat']['volume_run'] = 0
            state['stat']['dollar_run'] = 0
    
    return state


def check_bar_thresholds(state: dict) -> dict:

    def get_next_renko_thresh(renko_size: float, last_bar_return: float, reversal_multiple: float) -> tuple:
        if last_bar_return >= 0:
            thresh_renko_bull = renko_size
            thresh_renko_bear = -renko_size * reversal_multiple
        elif last_bar_return < 0:
            thresh_renko_bull = renko_size * reversal_multiple
            thresh_renko_bear = -renko_size
        return thresh_renko_bull, thresh_renko_bear

    if 'renko_size' in state['thresh']:
        try:
            state['thresh']['renko_bull'], state['thresh']['renko_bear'] = get_next_renko_thresh(
                renko_size=state['thresh']['renko_size'],
                last_bar_return=state['stat']['last_bar_return'],
                reversal_multiple=state['thresh']['renko_reveral_multiple']
            )
        except:
            state['thresh']['renko_bull'] = state['thresh']['renko_size']
            state['thresh']['renko_bear'] = -state['thresh']['renko_size']

        if state['stat'][state['thresh']['renko_return']] >= state['thresh']['renko_bull']:
            state['trigger_yet?!'] = 'renko_up'
        if state['stat'][state['thresh']['renko_return']] < state['thresh']['renko_bear']:
            state['trigger_yet?!'] = 'renko_down'

    if 'volume_imbalance' in state['thresh'] and abs(state['stat']['volume_imbalance']) >= state['thresh']['volume_imbalance']:
        state['trigger_yet?!'] = 'volume_imbalance'
    
    if 'max_duration_td' in state['thresh'] and state['stat']['duration_td'] > state['thresh']['max_duration_td']:
        state['trigger_yet?!'] = 'duration'

    # over-ride newbar trigger with 'minimum' thresholds
    if 'min_duration_td' in state['thresh'] and state['stat']['duration_td'] < state['thresh']['min_duration_td']:
        state['trigger_yet?!'] = 'waiting'

    if 'min_tick_count' in state['thresh'] and state['stat']['tick_count'] < state['thresh']['min_tick_count']:
        state['trigger_yet?!'] = 'waiting'

    return state


def update_bar_state(tick: dict, state: dict, bars: list, thresh: dict={}) -> tuple:

    state['trades']['date_time'].append(tick['date_time'])
    state['trades']['price'].append(tick['price'])
    state['trades']['jma'].append(tick['jma'])
    state['trades']['volume'].append(tick['volume'])
    state['trades']['side'].append(tick['side'])
#     if len(state['trades']['price']) >= 2:
#         tick_side = tick_rule(
#             latest_price=state['trades']['price'][-1],
#             prev_price=state['trades']['price'][-2],
#             last_side=state['trades']['side'][-1],
#             )
#     else:
#         tick_side = 0
#     state['trades']['side'].append(tick_side)
    state = imbalance_net(state)
    # state = imbalance_runs(state)
    state['stat']['duration_td'] = state['trades']['date_time'][-1] - state['trades']['date_time'][0]
    state['stat']['tick_count'] += 1
    state['stat']['volume'] += tick['volume']
    state['stat']['dollars'] += tick['price'] * tick['volume']
    # price
    state['stat']['price_min'] = tick['price'] if tick['price'] < state['stat']['price_min'] else state['stat']['price_min']
    state['stat']['price_max'] = tick['price'] if tick['price'] > state['stat']['price_max'] else state['stat']['price_max']
    state['stat']['price_range'] = state['stat']['price_max'] - state['stat']['price_min']
    state['stat']['price_return'] = tick['price'] - state['trades']['price'][0]
    state['stat']['last_bar_return'] = bars[-1]['price_return'] if len(bars) > 0 else 0
    # jma
    state['stat']['jma_min'] = tick['jma'] if tick['jma'] < state['stat']['jma_min'] else state['stat']['jma_min']
    state['stat']['jma_max'] = tick['jma'] if tick['jma'] > state['stat']['jma_max'] else state['stat']['jma_max']
    state['stat']['jma_range'] = state['stat']['jma_max'] - state['stat']['jma_min']
    state['stat']['jma_return'] = tick['jma'] - state['trades']['jma'][0]
    # check state tirggered sample threshold
    state = check_bar_thresholds(state)
    if state['trigger_yet?!'] != 'waiting':
        new_bar = state_to_bar(state)
        bars.append(new_bar)
        state = reset_state(thresh)
    
    return bars, state


def filter_tick(tick: dict, mad_filter: MADFilter, jma_filter: JMAFilter, tick_rule: TickRule) -> dict:

    tick['date_time'] = tick['sip_dt'].tz_localize('UTC').tz_convert('America/New_York')

    mad_filter.update(next_value=tick['price'])  # update mad filter
    
    irregular_conditions = [2, 5, 7, 10, 13, 15, 16, 20, 21, 22, 29, 33, 38, 52, 53]

    if tick['volume'] < 1:  # zero volume/size tick
        tick['status'] = 'zero_volume'
    elif pd.Series(tick['conditions']).isin(irregular_conditions).any():  # 'irrgular' tick condition
        tick['status'] = 'irregular_condition'
    elif abs(tick['sip_dt'] - tick['exchange_dt']) > pd.to_timedelta(2, unit='S'):  # remove large ts deltas
        tick['status'] = 'ts_delta'
    elif mad_filter.status != 'mad_clean':  # MAD filter outlier
        tick['status'] = 'mad_outlier'
    else:  # 'clean' tick
        tick['status'] = 'clean'
        tick['jma'] = jma_filter.update(next_value=tick['price'])  # update jma filter
        tick['side'] = tick_rule.update(next_price=tick['price'])  # update tick rule
        if tick['date_time'].to_pydatetime().time() < time(hour=9, minute=30):
            tick['status'] = 'clean_pre_market'
        elif tick['date_time'].hour >= 16:
            tick['status'] = 'clean_after_hours'
        else:
            tick['status'] = 'clean_open_market'

    # remove fields
    tick.pop('sip_dt', None)
    tick.pop('exchange_dt', None)
    tick.pop('conditions', None)

    return tick


def build_bars(ticks_df: pd.DataFrame, thresh: dict) -> tuple:
    
    mad_filter = MADFilter(thresh['mad_value_winlen'], thresh['mad_deviation_winlen'], thresh['mad_k'])
    jma_filter = JMAFilter(ticks_df['price'].values[0], thresh['jma_winlen'], thresh['jma_power'])
    tick_rule = TickRule()
    bar_actor = BarActor(thresh)
    ticks = []
    for t in ticks_df.itertuples():
        tick = {
            'sip_dt': t.sip_dt,
            'exchange_dt': t.exchange_dt,
            'price': t.price,
            'volume': t.size,
            'conditions': t.conditions,
            'status': 'raw',
        }
        tick = filter_tick(tick, mad_filter, jma_filter, tick_rule)

        if tick['status'] == 'clean_open_market':
        # if tick['status'].startswith('clean'):
            bar_actor.update(tick)

        ticks.append(tick)

    return bar_actor.bars, ticks
