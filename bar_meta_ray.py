from bar_meta import get_symbol_vol_filter, bar_workflow



def bar_dates_workflow(symbol: str, start_date: str, end_date: str, thresh: dict,
    add_label: bool=True, ray_on: bool=False) -> list:

    daily_stats_df = get_symbol_vol_filter(symbol, start_date, end_date)
    bar_dates = []
    if ray_on:
        import ray
        ray.init(dashboard_port=1111, ignore_reinit_error=True)
        bar_workflow_ray = ray.remote(bar_workflow)

    for row in daily_stats_df.itertuples():
        if 'range_jma_lag' in daily_stats_df.columns:
            rs = max(row.range_jma_lag / thresh['renko_range_frac'], row.vwap_jma_lag * 0.0005) # force min
            rs = min(rs, row.vwap_jma_lag * 0.005)  # enforce max
            thresh.update({'renko_size': rs})

        if ray_on:
            bar_date = bar_workflow_ray.remote(symbol, row.date, thresh, add_label)
        else:
            bar_date = bar_workflow(symbol, row.date, thresh, add_label)

        bar_dates.append(bar_date)

    if ray_on:
        bar_dates = ray.get(bar_dates)
    
    return bar_dates
