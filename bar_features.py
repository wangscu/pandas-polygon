import pandas as pd
from statsmodels.stats.weightstats import DescrStatsW


def trades_to_bar(ticks: pd.DataFrame, bar_trigger: str='fixed') -> dict:
    
    if type(ticks) != pd.DataFrame:
        ticks = pd.DataFrame(ticks)
    
    bar = {'bar_trigger': bar_trigger}
    # time
    bar['open_at'] = ticks['utc_dt'].iloc[0]
    bar['close_at'] = ticks['utc_dt'].iloc[-1]
    bar['duration_td'] = bar['close_at'] - bar['open_at']
    # price
    bar['price_open'] = ticks.price.values[0]
    bar['price_close'] = ticks.price.values[-1]
    bar['price_low'] = ticks.price.min()
    bar['price_high'] = ticks.price.max()
    bar['price_range'] = bar['price_high'] - bar['price_low']
    bar['price_return'] = bar['price_close'] - bar['price_close']
    # volume weighted price
    dsw = DescrStatsW(data=ticks.price, weights=ticks.volume)
    qtiles = dsw.quantile(probs=[0.1, 0.5, 0.9]).values
    bar['price_wq10'] = qtiles[0]
    bar['price_wq50'] = qtiles[1]
    bar['price_wq90'] = qtiles[2]
    bar['price_wq_range'] = bar['price_wq90'] - bar['price_wq10']
    bar['price_wmean'] = dsw.mean
    bar['price_wstd'] = dsw.std
    # jma
    bar['jma_open'] = ticks.jma.values[0]
    bar['jma_close'] = ticks.jma.values[-1]
    bar['jma_low'] = ticks.jma.min()
    bar['jma_high'] = ticks.jma.max()
    bar['jma_range'] = bar['jma_high'] - bar['jma_low']
    bar['jma_return'] = bar['jma_close'] - bar['jma_open']
    # volume weighted jma
    dsw = DescrStatsW(data=ticks.jma, weights=ticks.volume)
    qtiles = dsw.quantile(probs=[0.1, 0.5, 0.9]).values
    bar['jma_wq10'] = qtiles[0]
    bar['jma_wq50'] = qtiles[1]
    bar['jma_wq90'] = qtiles[2]
    bar['jma_wq_range'] = bar['jma_wq90'] - bar['jma_wq10']
    bar['jma_wmean'] = dsw.mean
    bar['jma_wstd'] = dsw.std
    # tick/vol/dollar/imbalance
    bar['tick_count'] = ticks.shape[0]
    bar['volume'] = ticks.volume.sum()
    bar['dollars'] = (ticks.volume * ticks.price).sum()
    bar['tick_imbalance'] = ticks.side.sum()
    bar['volume_imbalance'] = (ticks.volume * ticks.side).sum()
    bar['dollar_imbalance'] = (ticks.volume * ticks.price * ticks.side).sum()

    return bar


def state_to_bar(state: dict) -> dict:
    
    new_bar = {}
    if state['stat']['tick_count'] < 11:
        return new_bar

    new_bar['bar_trigger'] = state['bar_trigger']
    # time
    new_bar['open_at'] = state['trades']['utc_dt'][0]
    new_bar['close_at'] = state['trades']['utc_dt'][-1]
    new_bar['duration_td'] = new_bar['close_at'] - new_bar['open_at']
    # price
    new_bar['price_open'] = state['trades']['price'][0]
    new_bar['price_close'] = state['trades']['price'][-1]
    new_bar['price_low'] = state['stat']['price_min']
    new_bar['price_high'] = state['stat']['price_max']
    new_bar['price_range'] = state['stat']['price_range']
    new_bar['price_return'] = state['stat']['price_return']
    # volume weighted price
    dsw = DescrStatsW(data=state['trades']['price'], weights=state['trades']['volume'])
    qtiles = dsw.quantile(probs=[0.1, 0.5, 0.9]).values
    new_bar['price_wq10'] = qtiles[0]
    new_bar['price_wq50'] = qtiles[1]
    new_bar['price_wq90'] = qtiles[2]
    new_bar['price_wq_range'] = new_bar['price_wq90'] - new_bar['price_wq10']
    new_bar['price_wmean'] = dsw.mean
    new_bar['price_wstd'] = dsw.std
    # jma
    new_bar['jma_open'] = state['trades']['jma'][0]
    new_bar['jma_close'] = state['trades']['jma'][-1]
    new_bar['jma_low'] = state['stat']['jma_min']
    new_bar['jma_high'] = state['stat']['jma_max']
    new_bar['jma_range'] = state['stat']['jma_range']
    new_bar['jma_return'] = state['stat']['jma_return']
    # volume weighted jma
    dsw = DescrStatsW(data=state['trades']['jma'], weights=state['trades']['volume'])
    qtiles = dsw.quantile(probs=[0.1, 0.5, 0.9]).values
    new_bar['jma_wq10'] = qtiles[0]
    new_bar['jma_wq50'] = qtiles[1]
    new_bar['jma_wq90'] = qtiles[2]
    new_bar['jma_wq_range'] = new_bar['jma_wq90'] - new_bar['jma_wq10']
    new_bar['jma_wmean'] = dsw.mean
    new_bar['jma_wstd'] = dsw.std
    # tick/vol/dollar/imbalance
    new_bar['tick_count'] = state['stat']['tick_count']
    new_bar['volume'] = state['stat']['volume']
    new_bar['dollars'] = state['stat']['dollars']
    new_bar['tick_imbalance'] = state['stat']['tick_imbalance']
    new_bar['volume_imbalance'] = state['stat']['volume_imbalance']
    new_bar['dollar_imbalance'] = state['stat']['dollar_imbalance']
    if False:
        new_bar['n_tick_count'] = len(state['trades']['price'])
        new_bar['n_volume'] = sum(state['trades']['volume'])
        new_bar['n_dollars'] = new_bar['price_wq50'] * new_bar['volume']
        new_bar['n_tick_imbalance'] = sum(state['trades']['side'])
        new_bar['n_open_at'] = state['trades']['utc_dt'][0]
        new_bar['n_close_at'] = state['trades']['utc_dt'][-1]
        # new_bar['n_volume_imbalance'] = 
        # new_bar['n_dollar_imbalance'] = 

    return new_bar
