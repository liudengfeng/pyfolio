#
# Copyright 2019 Quantopian, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import division

import warnings
from time import time

# import matplotlib.gridspec as gridspec
# import matplotlib.pyplot as plt
import pandas as pd
import plotly
import plotly.figure_factory as ff
import plotly.graph_objects as go
from IPython.display import Markdown, display
from plotly.subplots import make_subplots

import empyrical as ep

# import os
# from . import _seaborn as sns
from . import (capacity, perf_attrib, plotting, pos, round_trips, timeseries,
               txn, utils)

FACTOR_PARTITIONS = {
    'style': ['momentum', 'size', 'value', 'reversal_short_term',
              'volatility'],
    'sector': ['basic_materials', 'consumer_cyclical', 'financial_services',
               'real_estate', 'consumer_defensive', 'health_care',
               'utilities', 'communication_services', 'energy', 'industrials',
               'technology']
}

# plotly.offline.init_notebook_mode(connected=True)


def timer(msg_body, previous_time):
    current_time = time()
    run_time = current_time - previous_time
    message = "\nFinished " + msg_body + " (required {:.2f} seconds)."
    print(message.format(run_time))

    return current_time


def create_full_tear_sheet(returns,
                           positions=None,
                           transactions=None,
                           market_data=None,
                           benchmark_rets=None,
                           slippage=None,
                           live_start_date=None,
                           sector_mappings=None,
                           round_trips=False,
                           estimate_intraday='infer',
                           hide_positions=False,
                           cone_std=(1.0, 1.5, 2.0),
                           bootstrap=False,
                           unadjusted_returns=None,
                           turnover_denom='AGB',
                           factor_returns=None,
                           factor_loadings=None,
                           pos_in_dollars=True,
                           header_rows=None,
                           factor_partitions=FACTOR_PARTITIONS):
    """
    Generate a number of tear sheets that are useful
    for analyzing a strategy's performance.

    - Fetches benchmarks if needed.
    - Creates tear sheets for returns, and significant events.
        If possible, also creates tear sheets for position analysis
        and transaction analysis.

    Parameters
    ----------
    returns : pd.Series
        Daily returns of the strategy, noncumulative.
         - Time series with decimal returns.
         - Example:
            2015-07-16    -0.012143
            2015-07-17    0.045350
            2015-07-20    0.030957
            2015-07-21    0.004902
    positions : pd.DataFrame, optional
        Daily net position values.
         - Time series of dollar amount invested in each position and cash.
         - Days where stocks are not held can be represented by 0 or NaN.
         - Non-working capital is labelled 'cash'
         - Example:
            index         'AAPL'         'MSFT'          cash
            2004-01-09    13939.3800     -14012.9930     711.5585
            2004-01-12    14492.6300     -14624.8700     27.1821
            2004-01-13    -13853.2800    13653.6400      -43.6375
    transactions : pd.DataFrame, optional
        Executed trade volumes and fill prices.
        - One row per trade.
        - Trades on different names that occur at the
          same time will have identical indicies.
        - Example:
            index                  amount   price    symbol
            2004-01-09 12:18:01    483      324.12   'AAPL'
            2004-01-09 12:18:01    122      83.10    'MSFT'
            2004-01-13 14:12:23    -75      340.43   'AAPL'
    market_data : pd.DataFrame, optional
        Daily market_data
        - DataFrame has a multi-index index, one level is dates and another is
        market_data contains volume & price, equities as columns
    slippage : int/float, optional
        Basis points of slippage to apply to returns before generating
        tearsheet stats and plots.
        If a value is provided, slippage parameter sweep
        plots will be generated from the unadjusted returns.
        Transactions and positions must also be passed.
        - See txn.adjust_returns_for_slippage for more details.
    live_start_date : datetime, optional
        The point in time when the strategy began live trading,
        after its backtest period. This datetime should be normalized.
    hide_positions : bool, optional
        If True, will not output any symbol names.
    round_trips: boolean, optional
        If True, causes the generation of a round trip tear sheet.
    sector_mappings : dict or pd.Series, optional
        Security identifier to sector mapping.
        Security ids as keys, sectors as values.
    estimate_intraday: boolean or str, optional
        Instead of using the end-of-day positions, use the point in the day
        where we have the most $ invested. This will adjust positions to
        better approximate and represent how an intraday strategy behaves.
        By default, this is 'infer', and an attempt will be made to detect
        an intraday strategy. Specifying this value will prevent detection.
    cone_std : float, or tuple, optional
        If float, The standard deviation to use for the cone plots.
        If tuple, Tuple of standard deviation values to use for the cone plots
         - The cone is a normal distribution with this standard deviation
             centered around a linear regression.
    bootstrap : boolean (optional)
        Whether to perform bootstrap analysis for the performance
        metrics. Takes a few minutes longer.
    turnover_denom : str
        Either AGB or portfolio_value, default AGB.
        - See full explanation in txn.get_turnover.
    factor_returns : pd.Dataframe, optional
        Returns by factor, with date as index and factors as columns
    factor_loadings : pd.Dataframe, optional
        Factor loadings for all days in the date range, with date and
        ticker as index, and factors as columns.
    pos_in_dollars : boolean, optional
        indicates whether positions is in dollars
    header_rows : dict or OrderedDict, optional
        Extra rows to display at the top of the perf stats table.
    factor_partitions : dict, optional
        dict specifying how factors should be separated in perf attrib
        factor returns and risk exposures plots
        - See create_perf_attrib_tear_sheet().
    """

    if (unadjusted_returns is None) and (slippage is not None) and\
       (transactions is not None):
        unadjusted_returns = returns.copy()
        returns = txn.adjust_returns_for_slippage(returns, positions,
                                                  transactions, slippage)

    positions = utils.check_intraday(estimate_intraday, returns,
                                     positions, transactions)

    create_returns_tear_sheet(
        returns,
        positions=positions,
        transactions=transactions,
        live_start_date=live_start_date,
        cone_std=cone_std,
        benchmark_rets=benchmark_rets,
        bootstrap=bootstrap,
        turnover_denom=turnover_denom,
        header_rows=header_rows)

    create_interesting_times_tear_sheet(returns,
                                        benchmark_rets=benchmark_rets)

    if positions is not None:
        create_position_tear_sheet(returns, positions,
                                   hide_positions=hide_positions,
                                   sector_mappings=sector_mappings,
                                   estimate_intraday=False)

        if transactions is not None:
            create_txn_tear_sheet(returns, positions, transactions,
                                  unadjusted_returns=unadjusted_returns,
                                  estimate_intraday=False)
            if round_trips:
                create_round_trip_tear_sheet(
                    returns=returns,
                    positions=positions,
                    transactions=transactions,
                    sector_mappings=sector_mappings,
                    estimate_intraday=False)

            if market_data is not None:
                create_capacity_tear_sheet(returns, positions, transactions,
                                           market_data,
                                           liquidation_daily_vol_limit=0.2,
                                           last_n_days=125,
                                           estimate_intraday=False)

        if factor_returns is not None and factor_loadings is not None:
            create_perf_attrib_tear_sheet(returns, positions, factor_returns,
                                          factor_loadings, transactions,
                                          pos_in_dollars=pos_in_dollars,
                                          factor_partitions=factor_partitions)


def create_simple_tear_sheet(returns,
                             positions=None,
                             transactions=None,
                             benchmark_rets=None,
                             slippage=None,
                             estimate_intraday='infer',
                             live_start_date=None,
                             turnover_denom='AGB',
                             header_rows=None):
    """
    Simpler version of create_full_tear_sheet; generates summary performance
    statistics and important plots as a single image.

    - Plots: cumulative returns, rolling beta, rolling Sharpe, underwater,
        exposure, top 10 holdings, total holdings, long/short holdings,
        daily turnover, transaction time distribution.
    - Never accept market_data input (market_data = None)
    - Never accept sector_mappings input (sector_mappings = None)
    - Never perform bootstrap analysis (bootstrap = False)
    - Never hide posistions on top 10 holdings plot (hide_positions = False)
    - Always use default cone_std (cone_std = (1.0, 1.5, 2.0))

    Parameters
    ----------
    returns : pd.Series
        Daily returns of the strategy, noncumulative.
         - Time series with decimal returns.
         - Example:
            2015-07-16    -0.012143
            2015-07-17    0.045350
            2015-07-20    0.030957
            2015-07-21    0.004902
    positions : pd.DataFrame, optional
        Daily net position values.
         - Time series of dollar amount invested in each position and cash.
         - Days where stocks are not held can be represented by 0 or NaN.
         - Non-working capital is labelled 'cash'
         - Example:
            index         'AAPL'         'MSFT'          cash
            2004-01-09    13939.3800     -14012.9930     711.5585
            2004-01-12    14492.6300     -14624.8700     27.1821
            2004-01-13    -13853.2800    13653.6400      -43.6375
    transactions : pd.DataFrame, optional
        Executed trade volumes and fill prices.
        - One row per trade.
        - Trades on different names that occur at the
          same time will have identical indicies.
        - Example:
            index                  amount   price    symbol
            2004-01-09 12:18:01    483      324.12   'AAPL'
            2004-01-09 12:18:01    122      83.10    'MSFT'
            2004-01-13 14:12:23    -75      340.43   'AAPL'
    benchmark_rets : pd.Series, optional
        Daily returns of the benchmark, noncumulative.
    slippage : int/float, optional
        Basis points of slippage to apply to returns before generating
        tearsheet stats and plots.
        If a value is provided, slippage parameter sweep
        plots will be generated from the unadjusted returns.
        Transactions and positions must also be passed.
        - See txn.adjust_returns_for_slippage for more details.
    live_start_date : datetime, optional
        The point in time when the strategy began live trading,
        after its backtest period. This datetime should be normalized.
    turnover_denom : str, optional
        Either AGB or portfolio_value, default AGB.
        - See full explanation in txn.get_turnover.
    header_rows : dict or OrderedDict, optional
        Extra rows to display at the top of the perf stats table.
    set_context : boolean, optional
        If True, set default plotting style context.
    """

    positions = utils.check_intraday(estimate_intraday, returns,
                                     positions, transactions)

    if (slippage is not None) and (transactions is not None):
        returns = txn.adjust_returns_for_slippage(returns, positions,
                                                  transactions, slippage)

    # always_sections = 4
    # positions_sections = 4 if positions is not None else 0
    # transactions_sections = 2 if transactions is not None else 0
    # live_sections = 1 if live_start_date is not None else 0
    # benchmark_sections = 1 if benchmark_rets is not None else 0

    # vertical_sections = sum([
    #     always_sections,
    #     positions_sections,
    #     transactions_sections,
    #     live_sections,
    #     benchmark_sections,
    # ])

    if live_start_date is not None:
        live_start_date = ep.utils.get_utc_timestamp(live_start_date)

    plotting.show_perf_stats(returns,
                             benchmark_rets,
                             positions=positions,
                             transactions=transactions,
                             turnover_denom=turnover_denom,
                             live_start_date=live_start_date,
                             header_rows=header_rows)
    figs = []

    ax_rolling_returns = make_subplots()

    if benchmark_rets is not None:
        ax_rolling_beta = make_subplots()
    ax_rolling_sharpe = make_subplots()

    ax_underwater = make_subplots()

    ax_rolling_returns = plotting.plot_rolling_returns(returns,
                                                       factor_returns=benchmark_rets,
                                                       live_start_date=live_start_date,
                                                       cone_std=(
                                                           1.0, 1.5, 2.0),
                                                       fig=ax_rolling_returns)

    # ax_rolling_returns.set_title('Cumulative returns')
    ax_rolling_returns.update_layout(title_text="累积收益率")
    figs.append(ax_rolling_returns)

    if benchmark_rets is not None:
        ax_rolling_beta = plotting.plot_rolling_beta(
            returns, benchmark_rets, fig=ax_rolling_beta)
        figs.append(ax_rolling_beta)
    else:
        ax_rolling_beta = None

    ax_rolling_sharpe = plotting.plot_rolling_sharpe(
        returns, fig=ax_rolling_sharpe)
    figs.append(ax_rolling_sharpe)

    ax_underwater = plotting.plot_drawdown_underwater(
        returns, fig=ax_underwater)
    figs.append(ax_underwater)

    # 处理共享x轴区域
    sharex_range = ax_rolling_returns.layout.xaxis.range
    for fig in [ax_rolling_beta, ax_rolling_sharpe, ax_underwater]:
        if fig:
            fig.update_xaxes(range=sharex_range)

    if positions is not None:
        # Plot simple positions tear sheet
        ax_exposures = make_subplots()

        ax_top_positions = make_subplots()

        ax_holdings = make_subplots()

        ax_long_short_holdings = make_subplots()

        positions_alloc = pos.get_percent_alloc(positions)

        ax_exposures = plotting.plot_exposures(
            returns, positions, fig=ax_exposures)
        figs.append(ax_exposures)

        ax_top_positions = plotting.show_and_plot_top_positions(returns,
                                                                positions_alloc,
                                                                show_and_plot=0,
                                                                hide_positions=False,
                                                                fig=ax_top_positions)
        figs.append(ax_top_positions)

        ax_holdings = plotting.plot_holdings(
            returns, positions_alloc, fig=ax_holdings)
        figs.append(ax_holdings)

        ax_long_short_holdings = plotting.plot_long_short_holdings(returns, positions_alloc,
                                                                   fig=ax_long_short_holdings)
        figs.append(ax_long_short_holdings)

        # 处理共享x轴区域
        sharex_range = ax_exposures.layout.xaxis.range
        for fig in [ax_top_positions, ax_holdings]:
            fig.update_xaxes(range=sharex_range)

        if transactions is not None:
            # Plot simple transactions tear sheet
            ax_turnover = make_subplots()

            ax_txn_timings = make_subplots()

            ax_turnover = plotting.plot_turnover(returns,
                                                 transactions,
                                                 positions,
                                                 turnover_denom=turnover_denom,
                                                 fig=ax_turnover)
            figs.append(ax_turnover)
            ax_txn_timings = plotting.plot_txn_time_hist(
                transactions, fig=ax_txn_timings)
            figs.append(ax_txn_timings)

    for fig in figs:
        fig.show()


def create_returns_tear_sheet(returns, positions=None,
                              transactions=None,
                              live_start_date=None,
                              cone_std=(1.0, 1.5, 2.0),
                              benchmark_rets=None,
                              bootstrap=False,
                              turnover_denom='AGB',
                              header_rows=None,
                              return_fig=False):
    """
    Generate a number of plots for analyzing a strategy's returns.

    - Fetches benchmarks, then creates the plots on a single figure.
    - Plots: rolling returns (with cone), rolling beta, rolling sharpe,
        rolling Fama-French risk factors, drawdowns, underwater plot, monthly
        and annual return plots, daily similarity plots,
        and return quantile box plot.
    - Will also print the start and end dates of the strategy,
        performance statistics, drawdown periods, and the return range.

    Parameters
    ----------
    returns : pd.Series
        Daily returns of the strategy, noncumulative.
         - See full explanation in create_full_tear_sheet.
    positions : pd.DataFrame, optional
        Daily net position values.
         - See full explanation in create_full_tear_sheet.
    transactions : pd.DataFrame, optional
        Executed trade volumes and fill prices.
        - See full explanation in create_full_tear_sheet.
    live_start_date : datetime, optional
        The point in time when the strategy began live trading,
        after its backtest period.
    cone_std : float, or tuple, optional
        If float, The standard deviation to use for the cone plots.
        If tuple, Tuple of standard deviation values to use for the cone plots
         - The cone is a normal distribution with this standard deviation
             centered around a linear regression.
    benchmark_rets : pd.Series, optional
        Daily noncumulative returns of the benchmark.
         - This is in the same style as returns.
    bootstrap : boolean, optional
        Whether to perform bootstrap analysis for the performance
        metrics. Takes a few minutes longer.
    turnover_denom : str, optional
        Either AGB or portfolio_value, default AGB.
        - See full explanation in txn.get_turnover.
    header_rows : dict or OrderedDict, optional
        Extra rows to display at the top of the perf stats table.
    return_fig : boolean, optional
        If True, returns the figure that was plotted on.
    """
    x_min, x_max = returns.index.min(), returns.index.max()

    if benchmark_rets is not None:
        returns = utils.clip_returns_to_benchmark(returns, benchmark_rets)

    plotting.show_perf_stats(returns, benchmark_rets,
                             positions=positions,
                             transactions=transactions,
                             turnover_denom=turnover_denom,
                             bootstrap=bootstrap,
                             live_start_date=live_start_date,
                             header_rows=header_rows)

    plotting.show_worst_drawdown_periods(returns)

    if live_start_date is not None:
        live_start_date = ep.utils.get_utc_timestamp(live_start_date)

    ax_rolling_returns = make_subplots(1, 1, subplot_titles=['累积收益率'])

    ax_rolling_returns_vol_match = make_subplots(
        1, 1, subplot_titles=['与基准匹配的累积收益波动率'])

    ax_rolling_returns_log = make_subplots(
        1, 1, subplot_titles=['对数刻度的累积收益率'])

    ax_returns = make_subplots(
        1, 1, subplot_titles=['收益率'])

    if benchmark_rets is not None:
        ax_rolling_beta = make_subplots(
            1, 1, subplot_titles=['滚动β'])
    else:
        ax_rolling_beta = None

    ax_rolling_volatility = make_subplots(1, 1)

    ax_rolling_sharpe = make_subplots(1, 1)

    ax_drawdown = make_subplots(1, 1)

    ax_underwater = make_subplots(1, 1)

    ax_annual_returns = make_subplots(
        1, 1, subplot_titles=['年收益率'])
    ax_monthly_dist = make_subplots(
        1, 1, subplot_titles=['月收益率分布'])

    ax_return_quantiles = make_subplots(
        1, 3, subplot_titles=['每日', '每周', '每月'], shared_yaxes=True)

    ax_rolling_returns = plotting.plot_rolling_returns(
        returns,
        factor_returns=benchmark_rets,
        live_start_date=live_start_date,
        cone_std=cone_std,
        fig=ax_rolling_returns)

    ax_rolling_returns_vol_match = plotting.plot_rolling_returns(
        returns,
        factor_returns=benchmark_rets,
        live_start_date=live_start_date,
        cone_std=None,
        volatility_match=(benchmark_rets is not None),
        legend_loc=None,
        fig=ax_rolling_returns_vol_match)

    ax_rolling_returns_log = plotting.plot_rolling_returns(
        returns,
        factor_returns=benchmark_rets,
        logy=True,
        live_start_date=live_start_date,
        cone_std=cone_std,
        fig=ax_rolling_returns_log)

    ax_returns = plotting.plot_returns(
        returns,
        live_start_date=live_start_date,
        fig=ax_returns,
    )

    if benchmark_rets is not None:
        ax_rolling_beta = plotting.plot_rolling_beta(
            returns, benchmark_rets, fig=ax_rolling_beta)

    ax_rolling_volatility = plotting.plot_rolling_volatility(
        returns, factor_returns=benchmark_rets, fig=ax_rolling_volatility)

    ax_rolling_sharpe = plotting.plot_rolling_sharpe(
        returns, fig=ax_rolling_sharpe)

    # Drawdowns
    ax_drawdown = plotting.plot_drawdown_periods(
        returns, top=5, fig=ax_drawdown)

    ax_underwater = plotting.plot_drawdown_underwater(
        returns, fig=ax_underwater)
    ax_monthly_heatmap = plotting.plot_monthly_returns_heatmap(returns)

    ax_annual_returns = plotting.plot_annual_returns(
        returns, fig=ax_annual_returns)

    ax_monthly_dist = plotting.plot_monthly_returns_dist(
        returns, fig=ax_monthly_dist)

    ax_return_quantiles = plotting.plot_return_quantiles(
        returns,
        live_start_date=live_start_date,
        fig=ax_return_quantiles)

    if bootstrap and (benchmark_rets is not None):
        ax_bootstrap = make_subplots(1, 1)
        ax_bootstrap = plotting.plot_perf_stats(returns, benchmark_rets,
                                                fig=ax_bootstrap)
    elif bootstrap:
        raise ValueError('bootstrap requires passing of benchmark_rets.')
    else:
        ax_bootstrap = None

    # for ax in fig.axes:
    #     plt.setp(ax.get_xticklabels(), visible=True)
    figs = [ax_rolling_returns, ax_rolling_returns_vol_match,
            ax_rolling_returns_log, ax_returns, ax_rolling_beta,
            ax_rolling_volatility, ax_rolling_sharpe,
            ax_drawdown, ax_underwater]
    figs = filter(None, figs)
    for fig in figs:
        fig.update_xaxes(showticklabels=True, range=[x_min, x_max])
        fig.show()
    # 无需共享x轴的图列表
    figs = [ax_monthly_heatmap, ax_annual_returns,
            ax_monthly_dist, ax_return_quantiles,
            ax_bootstrap]
    figs = filter(None, figs)
    for fig in figs:
        fig.show()


def create_position_tear_sheet(returns, positions,
                               show_and_plot_top_pos=2, hide_positions=False,
                               sector_mappings=None, transactions=None,
                               estimate_intraday='infer', return_fig=False):
    """
    Generate a number of plots for analyzing a
    strategy's positions and holdings.

    - Plots: gross leverage, exposures, top positions, and holdings.
    - Will also print the top positions held.

    Parameters
    ----------
    returns : pd.Series
        Daily returns of the strategy, noncumulative.
         - See full explanation in create_full_tear_sheet.
    positions : pd.DataFrame
        Daily net position values.
         - See full explanation in create_full_tear_sheet.
    show_and_plot_top_pos : int, optional
        By default, this is 2, and both prints and plots the
        top 10 positions.
        If this is 0, it will only plot; if 1, it will only print.
    hide_positions : bool, optional
        If True, will not output any symbol names.
        Overrides show_and_plot_top_pos to 0 to suppress text output.
    sector_mappings : dict or pd.Series, optional
        Security identifier to sector mapping.
        Security ids as keys, sectors as values.
    transactions : pd.DataFrame, optional
        Prices and amounts of executed trades. One row per trade.
         - See full explanation in create_full_tear_sheet.
    estimate_intraday: boolean or str, optional
        Approximate returns for intraday strategies.
        See description in create_full_tear_sheet.
    return_fig : boolean, optional
        If True, returns the figure that was plotted on.
    """
    figs = []
    positions = utils.check_intraday(estimate_intraday, returns,
                                     positions, transactions)

    if hide_positions:
        show_and_plot_top_pos = 0

    # gs = gridspec.GridSpec(vertical_sections, 3, wspace=0.5, hspace=0.5)
    ax_exposures = make_subplots()  # plt.subplot(gs[0, :])
    # TODO:sharex=ax_exposures
    # plt.subplot(gs[1, :], sharex=ax_exposures)
    ax_top_positions = make_subplots()
    # plt.subplot(gs[2, :], sharex=ax_exposures)
    ax_max_median_pos = make_subplots()
    # plt.subplot(gs[3, :], sharex=ax_exposures)
    ax_holdings = make_subplots()
    ax_long_short_holdings = make_subplots()  # plt.subplot(gs[4, :])
    # plt.subplot(gs[5, :], sharex=ax_exposures)
    ax_gross_leverage = make_subplots()

    figs.extend([ax_exposures, ax_top_positions, ax_max_median_pos,
                 ax_holdings, ax_long_short_holdings, ax_gross_leverage])

    positions_alloc = pos.get_percent_alloc(positions)

    plotting.plot_exposures(returns, positions, fig=ax_exposures)

    plotting.show_and_plot_top_positions(
        returns,
        positions_alloc,
        show_and_plot=show_and_plot_top_pos,
        hide_positions=hide_positions,
        fig=ax_top_positions)

    plotting.plot_max_median_position_concentration(positions,
                                                    fig=ax_max_median_pos)

    plotting.plot_holdings(returns, positions_alloc, fig=ax_holdings)

    plotting.plot_long_short_holdings(returns, positions_alloc,
                                      fig=ax_long_short_holdings)

    plotting.plot_gross_leverage(returns, positions,
                                 fig=ax_gross_leverage)

    if sector_mappings is not None:
        sector_exposures = pos.get_sector_exposures(positions,
                                                    sector_mappings)
        if len(sector_exposures.columns) > 1:
            sector_alloc = pos.get_percent_alloc(sector_exposures)
            sector_alloc = sector_alloc.drop('cash', axis='columns')
            # ax_sector_alloc = plt.subplot(gs[6, :], sharex=ax_exposures)
            ax_sector_alloc = make_subplots()
            plotting.plot_sector_allocations(returns, sector_alloc,
                                             fig=ax_sector_alloc)
            figs.append(ax_sector_alloc)

    if return_fig:
        return figs
    else:
        for fig in figs:
            fig.update_xaxes(range=[returns.index[0], returns.index[-1]])
            fig.show()


def create_txn_tear_sheet(returns, positions, transactions,
                          turnover_denom='AGB', unadjusted_returns=None,
                          estimate_intraday='infer', return_fig=False):
    """
    Generate a number of plots for analyzing a strategy's transactions.

    Plots: turnover, daily volume, and a histogram of daily volume.

    Parameters
    ----------
    returns : pd.Series
        Daily returns of the strategy, noncumulative.
         - See full explanation in create_full_tear_sheet.
    positions : pd.DataFrame
        Daily net position values.
         - See full explanation in create_full_tear_sheet.
    transactions : pd.DataFrame
        Prices and amounts of executed trades. One row per trade.
         - See full explanation in create_full_tear_sheet.
    turnover_denom : str, optional
        Either AGB or portfolio_value, default AGB.
        - See full explanation in txn.get_turnover.
    unadjusted_returns : pd.Series, optional
        Daily unadjusted returns of the strategy, noncumulative.
        Will plot additional swippage sweep analysis.
         - See pyfolio.plotting.plot_swippage_sleep and
           pyfolio.plotting.plot_slippage_sensitivity
    estimate_intraday: boolean or str, optional
        Approximate returns for intraday strategies.
        See description in create_full_tear_sheet.
    return_fig : boolean, optional
        If True, returns the figure that was plotted on.
    """
    figs = []
    positions = utils.check_intraday(estimate_intraday, returns,
                                     positions, transactions)

    ax_turnover = make_subplots()
    ax_daily_volume = make_subplots()
    ax_turnover_hist = make_subplots()
    ax_txn_timings = make_subplots()

    ax_turnover = plotting.plot_turnover(
        returns,
        transactions,
        positions,
        turnover_denom=turnover_denom,
        fig=ax_turnover)
    figs.append(ax_turnover)
    ax_daily_volume = plotting.plot_daily_volume(
        returns, transactions, fig=ax_daily_volume)
    figs.append(ax_daily_volume)
    try:
        ax_turnover_hist = plotting.plot_daily_turnover_hist(transactions,
                                                             positions,
                                                             turnover_denom=turnover_denom,
                                                             fig=ax_turnover_hist)
        figs.append(ax_turnover_hist)
    except ValueError:
        warnings.warn('Unable to generate turnover plot.', UserWarning)

    ax_txn_timings = plotting.plot_txn_time_hist(
        transactions, fig=ax_txn_timings)
    figs.append(ax_txn_timings)
    if unadjusted_returns is not None:
        ax_slippage_sweep = make_subplots()
        ax_slippage_sweep = plotting.plot_slippage_sweep(unadjusted_returns,
                                                         positions,
                                                         transactions,
                                                         fig=ax_slippage_sweep
                                                         )
        figs.append(ax_slippage_sweep)
        ax_slippage_sensitivity = make_subplots()
        ax_slippage_sensitivity = plotting.plot_slippage_sensitivity(unadjusted_returns,
                                                                     positions,
                                                                     transactions,
                                                                     fig=ax_slippage_sensitivity
                                                                     )
        figs.append(ax_slippage_sensitivity)
    else:
        ax_slippage_sweep = None

    if return_fig:
        return figs
    else:
        figs = filter(None, figs)
        shared_x_figs = [ax_turnover, ax_daily_volume]
        if ax_slippage_sweep:
            shared_x_figs.append(ax_slippage_sweep)
        for fig in shared_x_figs:
            fig.update_xaxes(range=[returns.index[0], returns.index[-1]])
        for fig in figs:
            fig.show()


def create_round_trip_tear_sheet(returns, positions, transactions,
                                 sector_mappings=None,
                                 estimate_intraday='infer', return_fig=False):
    """
    Generate a number of figures and plots describing the duration,
    frequency, and profitability of trade "round trips."
    A round trip is started when a new long or short position is
    opened and is only completed when the number of shares in that
    position returns to or crosses zero.

    Parameters
    ----------
    returns : pd.Series
        Daily returns of the strategy, noncumulative.
         - See full explanation in create_full_tear_sheet.
    positions : pd.DataFrame
        Daily net position values.
         - See full explanation in create_full_tear_sheet.
    transactions : pd.DataFrame
        Prices and amounts of executed trades. One row per trade.
         - See full explanation in create_full_tear_sheet.
    sector_mappings : dict or pd.Series, optional
        Security identifier to sector mapping.
        Security ids as keys, sectors as values.
    estimate_intraday: boolean or str, optional
        Approximate returns for intraday strategies.
        See description in create_full_tear_sheet.
    return_fig : boolean, optional
        If True, returns the figure that was plotted on.
    """
    figs = []
    positions = utils.check_intraday(estimate_intraday, returns,
                                     positions, transactions)

    transactions_closed = round_trips.add_closing_transactions(positions,
                                                               transactions)
    # extract_round_trips requires BoD portfolio_value
    trades = round_trips.extract_round_trips(
        transactions_closed,
        portfolio_value=positions.sum(axis='columns') / (1 + returns)
    )

    if len(trades) < 5:
        warnings.warn(
            """Fewer than 5 round-trip trades made.
               Skipping round trip tearsheet.""", UserWarning)
        return

    round_trips.print_round_trip_stats(trades)

    plotting.show_profit_attribution(trades)

    if sector_mappings is not None:
        sector_trades = round_trips.apply_sector_mappings_to_round_trips(
            trades, sector_mappings)
        plotting.show_profit_attribution(sector_trades)

    ax_trade_lifetimes = make_subplots()
    fig_1 = make_subplots(1, 2)
    fig_2 = make_subplots(1, 2)

    ax_trade_lifetimes = plotting.plot_round_trip_lifetimes(
        trades, fig=ax_trade_lifetimes)
    figs.append(ax_trade_lifetimes)

    fig_1 = plotting.plot_prob_profit_trade(trades, fig=fig_1, col=1)

    trade_holding_times = [x.days for x in trades['duration']]

    fig_1.add_trace(go.Histogram(x=trade_holding_times), row=1, col=2)
    fig_1.update_xaxes(
        title_text='持有时间(天)',
        row=1, col=2,
    )
    fig_1.update_yaxes(
        tickformat='.2f',
        row=1, col=2,
    )
    fig_1.update_layout(showlegend=False)
    figs.append(fig_1)

    fig_2.add_trace(go.Histogram(x=trades.pnl), row=1, col=1)
    fig_2.update_xaxes(
        title_text='每回交易盈亏额(RMB)',
        row=1, col=1,
    )

    fig_2.add_trace(go.Histogram(
        x=trades.returns.dropna()), row=1, col=2)
    fig_2.update_xaxes(
        title_text='每回交易盈亏率(%)',
        tickformat='%',
        row=1, col=2,
    )
    fig_2.update_layout(showlegend=False)

    figs.append(fig_2)

    if return_fig:
        return figs
    else:
        for fig in figs:
            fig.show()


def create_interesting_times_tear_sheet(returns, benchmark_rets=None,
                                        periods=None, legend_loc='best',
                                        return_fig=False):
    """
    Generate a number of returns plots around interesting points in time,
    like the flash crash and 9/11.

    Plots: returns around the dotcom bubble burst, Lehmann Brothers' failure,
    9/11, US downgrade and EU debt crisis, Fukushima meltdown, US housing
    bubble burst, EZB IR, Great Recession (August 2007, March and September
    of 2008, Q1 & Q2 2009), flash crash, April and October 2014.

    benchmark_rets must be passed, as it is meaningless to analyze performance
    during interesting times without some benchmark to refer to.

    Parameters
    ----------
    returns : pd.Series
        Daily returns of the strategy, noncumulative.
         - See full explanation in create_full_tear_sheet.
    benchmark_rets : pd.Series
        Daily noncumulative returns of the benchmark.
         - This is in the same style as returns.
    periods: dict or OrderedDict, optional
        historical event dates that may have had significant
        impact on markets
    legend_loc : plt.legend_loc, optional
         The legend's location.
    return_fig : boolean, optional
        If True, returns the figure that was plotted on.
    """

    rets_interesting = timeseries.extract_interesting_date_ranges(
        returns, periods)

    if not rets_interesting:
        warnings.warn('Passed returns do not overlap with any'
                      'interesting times.', UserWarning)
        return

    utils.print_table(pd.DataFrame(rets_interesting)
                      .describe().transpose()
                      .loc[:, ['mean', 'min', 'max']] * 100,
                      name='压力事件',
                      float_format='{0:.2f}%'.format)

    if benchmark_rets is not None:
        returns = utils.clip_returns_to_benchmark(returns, benchmark_rets)

        bmark_interesting = timeseries.extract_interesting_date_ranges(
            benchmark_rets, periods)

    names = list(rets_interesting.keys())
    figs = []
    for i in range(0, len(names), 2):
        subplot_titles = names[i:i + 2]
        fig = make_subplots(
            rows=1, cols=2, shared_yaxes=True, subplot_titles=subplot_titles)
        figs.append(fig)
        batch = names[i:i+2]
        for col, name in enumerate(batch, 1):
            rets_period = rets_interesting[name]
            cum_returns = ep.cum_returns(rets_period)
            fig.add_trace(
                go.Scatter(x=cum_returns.index,
                           y=cum_returns.values,
                           mode='lines',
                           showlegend=True if col == 1 else False,
                           name='策略',
                           opacity=0.70,
                           line=dict(color='forestgreen', width=2)),
                row=1, col=col,
            )
            if benchmark_rets is not None:
                bmark = ep.cum_returns(bmark_interesting[name])
                fig.add_trace(
                    go.Scatter(x=bmark.index, y=bmark.values,
                               mode='lines',
                               showlegend=True if col == 1 else False,
                               name='基准',
                               opacity=0.60,
                               line=dict(color='gray', width=2)),
                    row=1, col=col,
                )
    if return_fig:
        return figs
    else:
        for fig in figs:
            fig.update_yaxes(title_text='收益率')
            fig.update_layout(yaxis_tickformat='%')
            fig.update_layout(
                legend=dict(
                    x=0.01,
                    y=0.99,
                    yanchor="top",
                    xanchor="left",
                )
            )
            for col in [1, 2]:
                utils._date_tickformat(fig, col=col)
            fig.show()


def create_capacity_tear_sheet(returns, positions, transactions,
                               market_data,
                               liquidation_daily_vol_limit=0.2,
                               trade_daily_vol_limit=0.05,
                               last_n_days=utils.APPROX_BDAYS_PER_MONTH * 6,
                               days_to_liquidate_limit=1,
                               estimate_intraday='infer',
                               return_fig=False):
    """
    Generates a report detailing portfolio size constraints set by
    least liquid tickers. Plots a "capacity sweep," a curve describing
    projected sharpe ratio given the slippage penalties that are
    applied at various capital bases.

    Parameters
    ----------
    returns : pd.Series
        Daily returns of the strategy, noncumulative.
         - See full explanation in create_full_tear_sheet.
    positions : pd.DataFrame
        Daily net position values.
         - See full explanation in create_full_tear_sheet.
    transactions : pd.DataFrame
        Prices and amounts of executed trades. One row per trade.
         - See full explanation in create_full_tear_sheet.
    market_data : pd.DataFrame
        Daily market_data
        - DataFrame has a multi-index index, one level is dates and another is
        market_data contains volume & price, equities as columns
    liquidation_daily_vol_limit : float
        Max proportion of a daily bar that can be consumed in the
        process of liquidating a position in the
        "days to liquidation" analysis.
    trade_daily_vol_limit : float
        Flag daily transaction totals that exceed proportion of
        daily bar.
    last_n_days : integer
        Compute max position allocation and dollar volume for only
        the last N days of the backtest
    days_to_liquidate_limit : integer
        Display all tickers with greater max days to liquidation.
    estimate_intraday: boolean or str, optional
        Approximate returns for intraday strategies.
        See description in create_full_tear_sheet.
    return_fig : boolean, optional
        If True, returns the figure that was plotted on.
    """

    positions = utils.check_intraday(estimate_intraday, returns,
                                     positions, transactions)

    print("计算每个交易名称的最大清算天数\n\n"
          "假设：\n"
          "\t1. 限定20%的日线量\n"
          "\t2. 尾部5天平均成交量为可用量\n"
          "\t3. 清算大于1天的初始资本为1百万元")

    max_days_by_ticker = capacity.get_max_days_to_liquidate_by_ticker(
        positions, market_data,
        max_bar_consumption=liquidation_daily_vol_limit,
        capital_base=1e6,
        mean_volume_window=5)
    max_days_by_ticker.index = (
        max_days_by_ticker.index.map(utils.format_asset))
    print()
    print(f"{'='*30}整个回测{'='*30}")
    print()
    utils.print_table(
        max_days_by_ticker[max_days_by_ticker.days_to_liquidate >
                           days_to_liquidate_limit])

    max_days_by_ticker_lnd = capacity.get_max_days_to_liquidate_by_ticker(
        positions, market_data,
        max_bar_consumption=liquidation_daily_vol_limit,
        capital_base=1e6,
        mean_volume_window=5,
        last_n_days=last_n_days)
    max_days_by_ticker_lnd.index = (
        max_days_by_ticker_lnd.index.map(utils.format_asset))
    print()
    print(f"最近{last_n_days}个交易日")
    print()
    utils.print_table(
        max_days_by_ticker_lnd[max_days_by_ticker_lnd.days_to_liquidate > 1])

    llt = capacity.get_low_liquidity_transactions(transactions, market_data)
    llt.index = llt.index.map(utils.format_asset)

    factor = trade_daily_vol_limit * 100
    print()
    print(f'所有回测中，股票超过当日成交量的{factor}%清单')
    print()
    utils.print_table(
        llt[llt['max_pct_bar_consumed'] > trade_daily_vol_limit * 100])

    llt = capacity.get_low_liquidity_transactions(
        transactions, market_data, last_n_days=last_n_days)

    print()
    print(f"最近{last_n_days}个交易日")
    print()
    utils.print_table(
        llt[llt['max_pct_bar_consumed'] > trade_daily_vol_limit * 100])

    bt_starting_capital = positions.iloc[0].sum() / (1 + returns.iloc[0])
    fig = make_subplots()
    plotting.plot_capacity_sweep(returns, transactions, market_data,
                                 bt_starting_capital,
                                 min_pv=100000,
                                 max_pv=300000000,
                                 step_size=1000000,
                                 fig=fig)

    if return_fig:
        return fig
    else:
        fig.show()


def create_perf_attrib_tear_sheet(returns,
                                  positions,
                                  factor_returns,
                                  factor_loadings,
                                  transactions=None,
                                  pos_in_dollars=True,
                                  factor_partitions=FACTOR_PARTITIONS,
                                  return_fig=False):
    """
    Generate plots and tables for analyzing a strategy's performance.

    Parameters
    ----------
    returns : pd.Series
        Returns for each day in the date range.

    positions: pd.DataFrame
        Daily holdings (in dollars or percentages), indexed by date.
        Will be converted to percentages if positions are in dollars.
        Short positions show up as cash in the 'cash' column.

    factor_returns : pd.DataFrame
        Returns by factor, with date as index and factors as columns

    factor_loadings : pd.DataFrame
        Factor loadings for all days in the date range, with date
        and ticker as index, and factors as columns.

    transactions : pd.DataFrame, optional
        Prices and amounts of executed trades. One row per trade.
         - See full explanation in create_full_tear_sheet.
         - Default is None.

    pos_in_dollars : boolean, optional
        Flag indicating whether `positions` are in dollars or percentages
        If True, positions are in dollars.

    factor_partitions : dict
        dict specifying how factors should be separated in factor returns
        and risk exposures plots
        - Example:
          {'style': ['momentum', 'size', 'value', ...],
           'sector': ['technology', 'materials', ... ]}

    return_fig : boolean, optional
        If True, returns the figure that was plotted on.
    """
    portfolio_exposures, perf_attrib_data = perf_attrib.perf_attrib(
        returns, positions, factor_returns, factor_loadings, transactions,
        pos_in_dollars=pos_in_dollars
    )

    display(Markdown("## 相对于共同风险因子的绩效"))

    # aggregate perf attrib stats and show summary table
    perf_attrib.show_perf_attrib_stats(returns, positions, factor_returns,
                                       factor_loadings, transactions,
                                       pos_in_dollars)

    # one section for the returns plot, and for each factor grouping
    # one section for factor returns, and one for risk exposures
    if factor_partitions is not None:
        vertical_sections = 1 + 2 * max(len(factor_partitions), 1)
    else:
        vertical_sections = 1 + 2

    current_section = 0
    figs = [make_subplots() for _ in range(vertical_sections)]

    perf_attrib.plot_returns(perf_attrib_data,
                             fig=figs[current_section])
    current_section += 1
    if factor_partitions is not None:

        for factor_type, partitions in factor_partitions.items():

            columns_to_select = perf_attrib_data.columns.intersection(
                partitions
            )

            perf_attrib.plot_factor_contribution_to_perf(
                perf_attrib_data[columns_to_select],
                fig=figs[current_section],
                title=(
                    f'累积共同 {factor_type} 收益率归因'
                )
            )
            current_section += 1

        for factor_type, partitions in factor_partitions.items():

            columns_to_select = portfolio_exposures.columns.intersection(
                partitions
            )

            perf_attrib.plot_risk_exposures(
                portfolio_exposures[columns_to_select],
                fig=figs[current_section],
                title='每日 {} 因子敞口'.format(factor_type)
            )
            current_section += 1

    else:

        perf_attrib.plot_factor_contribution_to_perf(
            perf_attrib_data,
            fig=figs[current_section]
        )
        current_section += 1

        perf_attrib.plot_risk_exposures(
            portfolio_exposures,
            fig=figs[current_section]
        )

    # gs.tight_layout(fig)

    if return_fig:
        return figs
    else:
        for fig in figs:
            fig.show()
