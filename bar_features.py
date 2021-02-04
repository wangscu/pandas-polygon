import pandas as pd
from statsmodels.stats.weightstats import DescrStatsW



def trades_to_bar(ticks, bar_trigger: str='fixed') -> dict:
    
    if type(ticks) != pd.DataFrame:
        ticks = pd.DataFrame(ticks)
    
    bar = {'bar_trigger': bar_trigger}
    # time
    bar['open_at'] = ticks.nyc_time.iloc[0]
    bar['close_at'] = ticks.nyc_time.iloc[-1]
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
