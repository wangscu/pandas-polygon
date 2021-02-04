from datetime import timedelta
from dataclasses import dataclass


@dataclass
class BarStats():
    duration_td: timedelta
    price_min: int = 10 ** 5
    price_max: int = 0
    price_range: int = 0
    price_return: int = 0
    price_jma_min: int = 10 ** 5
    price_jma_max: int = 0
    price_jma_range: int = 0
    price_jma_return: int = 0
    tick_count: int = 0
    volume: int = 0
    dollars: int = 0
    tick_imbalance: int = 0
    volume_imbalance: int = 0
    dollar_imbalance: int = 0
    last_bar_return: float = 0


@dataclass
class Trades():
    date_time: list=[]
    price: list=[]
    jma: list=[]
    volume: list=[]
    side: list=[]

        
@dataclass
class Thresh():
    renko_size: float=0.11
    renko_bull: float
    renko_bear: float
    renko_return: float
    volume_imbalance: float
    min_duration_td: timedelta
    max_duration_td: timedelta
    min_tick_count: int
