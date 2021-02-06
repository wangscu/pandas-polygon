from bar_features import trades_to_bar, state_to_bar


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
    state['trades']['utc_dt'] = []
    state['trades']['price'] = []
    state['trades']['volume'] = []
    state['trades']['side'] = []
    state['trades']['jma'] = []
    # trigger status
    state['bar_trigger'] = 'waiting'

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
            state['bar_trigger'] = 'renko_up'
        if state['stat'][state['thresh']['renko_return']] < state['thresh']['renko_bear']:
            state['bar_trigger'] = 'renko_down'

    if 'volume_imbalance' in state['thresh'] and abs(state['stat']['volume_imbalance']) >= state['thresh']['volume_imbalance']:
        state['bar_trigger'] = 'volume_imbalance'
    
    if 'max_duration_td' in state['thresh'] and state['stat']['duration_td'] > state['thresh']['max_duration_td']:
        state['bar_trigger'] = 'duration'

    # over-ride newbar trigger with 'minimum' thresholds
    if 'min_duration_td' in state['thresh'] and state['stat']['duration_td'] < state['thresh']['min_duration_td']:
        state['bar_trigger'] = 'waiting'

    if 'min_tick_count' in state['thresh'] and state['stat']['tick_count'] < state['thresh']['min_tick_count']:
        state['bar_trigger'] = 'waiting'

    return state


def update_bar_state(tick: dict, state: dict, bars: list=[], thresh: dict={}) -> tuple:

    # append tick
    state['trades']['utc_dt'].append(tick['utc_dt'])
    state['trades']['price'].append(tick['price'])
    state['trades']['volume'].append(tick['volume'])
    state['trades']['side'].append(tick['side'])
    state['trades']['jma'].append(tick['jma'])
    # imbalances
    state['stat']['tick_imbalance'] += state['trades']['side'][-1]
    state['stat']['volume_imbalance'] += (state['trades']['side'][-1] * state['trades']['volume'][-1])
    state['stat']['dollar_imbalance'] += (state['trades']['side'][-1] * state['trades']['volume'][-1] * state['trades']['price'][-1])
    # other
    state['stat']['duration_td'] = state['trades']['utc_dt'][-1] - state['trades']['utc_dt'][0]
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

    if state['bar_trigger'] != 'waiting':
        # new_bar = trades_to_bar(state['trades'], state['bar_trigger'])
        new_bar = state_to_bar(state)
        # new_bar['state'] = state
        bars.append(new_bar)
        state = reset_state(thresh)
    else:
        new_bar = {'bar_trigger': 'waiting'}
    
    return bars, state, new_bar


class BarSampler:
    
    def __init__(self, thresh: dict):
        self.state = reset_state(thresh)
        self.bars = []

    def update(self, tick: dict) -> dict:
        self.bars, self.state, new_bar = update_bar_state(tick, self.state, self.bars, self.state['thresh'])
        return new_bar

    def reset(self):
        self.state = reset_state(self.state['thresh'])
        self.bars = []
