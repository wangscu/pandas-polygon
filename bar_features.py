from statsmodels.stats.weightstats import DescrStatsW
    

def state_to_bar(state: dict) -> dict:
    
    bar = {}
    if state['stat']['tick_count'] == 0:
        return bar

    bar['bar_trigger'] = state['trigger_yet?!']
    # time
    bar['open_at'] = state['trades']['date_time'][0]
    bar['close_at'] = state['trades']['date_time'][-1]
    bar['duration_td'] = bar['close_at'] - bar['open_at']
    # price
    bar['price_open'] = state['trades']['price'][0]
    bar['price_close'] = state['trades']['price'][-1]
    bar['price_low'] = state['stat']['price_min']
    bar['price_high'] = state['stat']['price_max']
    bar['price_range'] = state['stat']['price_range']
    bar['price_return'] = state['stat']['price_return']
    # volume weighted price
    dsw = DescrStatsW(data=state['trades']['price'], weights=state['trades']['volume'])
    qtiles = dsw.quantile(probs=[0.1, 0.5, 0.9]).values
    bar['price_wq10'] = qtiles[0]
    bar['price_wq50'] = qtiles[1]
    bar['price_wq90'] = qtiles[2]
    bar['price_wq_range'] = bar['price_wq90'] - bar['price_wq10']
    bar['price_wmean'] = dsw.mean
    bar['price_wstd'] = dsw.std
    # jma
    bar['jma_open'] = state['trades']['jma'][0]
    bar['jma_close'] = state['trades']['jma'][-1]
    bar['jma_low'] = state['stat']['jma_min']
    bar['jma_high'] = state['stat']['jma_max']
    bar['jma_range'] = state['stat']['jma_range']
    bar['jma_return'] = state['stat']['jma_return']
    # volume weighted jma
    dsw = DescrStatsW(data=state['trades']['jma'], weights=state['trades']['volume'])
    qtiles = dsw.quantile(probs=[0.1, 0.5, 0.9]).values
    bar['jma_wq10'] = qtiles[0]
    bar['jma_wq50'] = qtiles[1]
    bar['jma_wq90'] = qtiles[2]
    bar['jma_wq_range'] = bar['jma_wq90'] - bar['jma_wq10']
    bar['jma_wmean'] = dsw.mean
    bar['jma_wstd'] = dsw.std
    # tick/vol/dollar/imbalance
    bar['tick_count'] = state['stat']['tick_count']
    bar['volume'] = state['stat']['volume']
    bar['dollars'] = state['stat']['dollars']
    bar['tick_imbalance'] = state['stat']['tick_imbalance']
    bar['volume_imbalance'] = state['stat']['volume_imbalance']
    bar['dollar_imbalance'] = state['stat']['dollar_imbalance']

    return bar


def trades_to_bar(trades: dict, bar_trigger: str) -> dict:
    
    bar = {}
    if state['stat']['tick_count'] == 0:
        return bar

    bar['bar_trigger'] = bar_trigger
    # time
    bar['open_at'] = trades['date_time'][0]
    bar['close_at'] = trades['date_time'][-1]
    bar['duration_td'] = bar['close_at'] - bar['open_at']
    # price
    bar['price_open'] = trades['price'][0]
    bar['price_close'] = trades['price'][-1]
    bar['price_low'] = min(trades['price'])
    bar['price_high'] = max(trades['price'])
    bar['price_range'] = bar['price_high'] - bar['price_low']
    bar['price_return'] = bar['price_close'] - bar['price_close']
    # volume weighted price
    dsw = DescrStatsW(data=trades['price'], weights=trades['volume'])
    qtiles = dsw.quantile(probs=[0.1, 0.5, 0.9]).values
    bar['price_wq10'] = qtiles[0]
    bar['price_wq50'] = qtiles[1]
    bar['price_wq90'] = qtiles[2]
    bar['price_wq_range'] = bar['price_wq90'] - bar['price_wq10']
    bar['price_wmean'] = dsw.mean
    bar['price_wstd'] = dsw.std
    # jma
    bar['jma_open'] = trades['jma'][0]
    bar['jma_close'] = trades['jma'][-1]
    bar['jma_low'] = min(trades['jma'])
    bar['jma_high'] = max(trades['jma'])
    bar['jma_range'] = bar['jma_high'] - bar['jma_low']
    bar['jma_return'] = bar['jma_close'] - bar['jma_open']
    # volume weighted jma
    dsw = DescrStatsW(data=trades['jma'], weights=state['trades']['volume'])
    qtiles = dsw.quantile(probs=[0.1, 0.5, 0.9]).values
    bar['jma_wq10'] = qtiles[0]
    bar['jma_wq50'] = qtiles[1]
    bar['jma_wq90'] = qtiles[2]
    bar['jma_wq_range'] = bar['jma_wq90'] - bar['jma_wq10']
    bar['jma_wmean'] = dsw.mean
    bar['jma_wstd'] = dsw.std
    # tick/vol/dollar/imbalance
    bar['tick_count'] = len(trades)
    bar['volume'] = sum(trades['volume'])
    bar['dollars'] = bar['volume'] * bar['jma_wmean']
    bar['tick_imbalance'] = sum(trades['side'])
    bar['volume_imbalance'] = bar['tick_imbalance'] * bar['volume']
    bar['dollar_imbalance'] = bar['volume_imbalance'] * bar['jma_wmean']

    return bar
