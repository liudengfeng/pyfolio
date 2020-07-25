#
# Copyright 2018 Quantopian, Inc.
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

import datetime
from collections import OrderedDict
import plotly.express as px
import numpy as np
import pandas as pd
import plotly
import plotly.figure_factory as ff
import plotly.graph_objects as go
import pytz
import scipy as sp

import empyrical as ep

from . import capacity, pos, timeseries, txn, utils
from .utils import APPROX_BDAYS_PER_MONTH, MM_DISPLAY_UNIT, _date_tickformat


def plot_monthly_returns_heatmap(returns, **kwargs):
    """
    Plots a heatmap of returns by month.

    Parameters
    ----------
    returns : pd.Series
        Daily returns of the strategy, noncumulative.
         - See full explanation in tears.create_full_tear_sheet.
    **kwargs, optional
        Passed to seaborn plotting function.
    """
    monthly_ret_table = ep.aggregate_returns(returns, 'monthly')
    monthly_ret_table = monthly_ret_table.unstack().round(3)
    data = monthly_ret_table.fillna(0) * 100.0
    z = data.values
    z_text = np.around(z, decimals=3)
    fig = ff.create_annotated_heatmap(
        z=z,
        x=[f"{str(x).zfill(2)}月" for x in data.columns],
        y=[f"{y}年" for y in data.index],
        annotation_text=z_text,
        colorscale='Bluered',
        hoverinfo='z')

    # 自上而下升序排列
    fig.update_yaxes(title_text='年', autorange="reversed")
    fig.update_layout(title_text='月收益率(%)')
    return fig


def plot_annual_returns(returns, fig=None, **kwargs):
    """
    Plots a bar graph of returns by year.

    Parameters
    ----------
    returns : pd.Series
        Daily returns of the strategy, noncumulative.
         - See full explanation in tears.create_full_tear_sheet.
    fig : matplotlib.Axes, optional
        Axes upon which to plot.
    **kwargs, optional
        Passed to plotting function.

    Returns
    -------
    ax : matplotlib.Axes
        The axes that were plotted on.
    """
    if fig is None:
        fig = go.Figure()

    fig.update_xaxes(title_text='收益率')
    fig.update_yaxes(title_text='年')
    fig.update_layout(xaxis_tickformat='%')
    fig.update_layout(yaxis_tickformat='.0f')

    ann_ret_df = pd.DataFrame(
        ep.aggregate_returns(
            returns,
            'yearly'))

    m = ann_ret_df.values.mean()
    fig.add_trace(
        go.Scatter(x=[m]*2, y=[ann_ret_df.index.min(), ann_ret_df.index.max()],
                   mode='lines',
                   name='均值',
                   opacity=0.70,
                   line=dict(color='steelblue', width=4, dash='dash')),
    )

    a = ann_ret_df.sort_index(ascending=False)
    fig.add_trace(
        go.Bar(y=a.index, x=a.iloc[:, 0].values,
               name='年收益率',
               orientation='h'),
    )

    fig.add_trace(
        go.Scatter(x=[0]*2, y=[ann_ret_df.index.min(), ann_ret_df.index.max()],
                   mode='lines',
                   name='基线',
                   line=dict(color='black', width=3, dash='dash')),
    )
    fig.update_yaxes(
        ticktext=[str(x) for x in ann_ret_df.index],
        tickvals=ann_ret_df.index,
        autorange="reversed"
    )
    fig.update_layout(
        legend=dict(
            x=0.01,
            y=0.99,
            yanchor="top",
            xanchor="left",
        )
    )
    return fig


def plot_monthly_returns_dist(returns, fig=None, **kwargs):
    """
    Plots a distribution of monthly returns.

    Parameters
    ----------
    returns : pd.Series
        Daily returns of the strategy, noncumulative.
         - See full explanation in tears.create_full_tear_sheet.
    ax : matplotlib.Axes, optional
        Axes upon which to plot.
    **kwargs, optional
        Passed to plotting function.

    Returns
    -------
    ax : matplotlib.Axes
        The axes that were plotted on.
    """
    nbins = 20
    monthly_ret_table = ep.aggregate_returns(returns, 'monthly')
    m = monthly_ret_table.copy().mean()
    monthly_ret_table = monthly_ret_table.reset_index()
    monthly_ret_table.columns = ['年', '月', '收益率']

    fig = px.histogram(monthly_ret_table,
                       opacity=0.8,
                       color="月",
                       x='收益率', nbins=nbins)

    fig.add_shape(
        dict(
            type="line",
            opacity=1.0,
            yref="paper",
            name='均值',
            x0=m,
            y0=0,
            x1=m,
            y1=1,
            line=dict(color='gold', width=4, dash='dash')
        )
    )

    fig.add_shape(
        dict(
            type="line",
            opacity=0.75,
            yref="paper",
            name='基线',
            x0=0,
            y0=0,
            x1=0,
            y1=1,
            line=dict(color='black', width=3, dash='dash')
        )
    )
    fig.update_xaxes(title_text='收益率')
    fig.update_yaxes(title_text='次数')
    fig.update_layout(xaxis_tickformat='%', title_text="月度收益率分布")
    fig.update_layout(
        legend=dict(
            x=0.01,
            y=0.99,
            traceorder="normal",
            yanchor="top",
            xanchor="left",
            title_font_family="SimHei",
            font=dict(
                    family="SimHei",
                    size=12,
                    color="black",
            ),
            borderwidth=1
        )
    )
    return fig


def plot_holdings(returns, positions, legend_loc='best', fig=None, **kwargs):
    """
    Plots total amount of stocks with an active position, either short
    or long. Displays daily total, daily average per month, and
    all-time daily average.

    Parameters
    ----------
    returns : pd.Series
        Daily returns of the strategy, noncumulative.
         - See full explanation in tears.create_full_tear_sheet.
    positions : pd.DataFrame, optional
        Daily net position values.
         - See full explanation in tears.create_full_tear_sheet.
    legend_loc : matplotlib.loc, optional
        The location of the legend on the plot.
    fig : matplotlib.Axes, optional
        Axes upon which to plot.
    **kwargs, optional
        Passed to plotting function.

    Returns
    -------
    ax : matplotlib.Axes
        The axes that were plotted on.
    """

    if fig is None:
        fig = go.Figure()

    positions = positions.copy().drop('cash', axis='columns')
    df_holdings = positions.replace(0, np.nan).count(axis=1)
    df_holdings_by_month = df_holdings.resample('1M').mean()

    fig.add_trace(
        go.Scatter(x=df_holdings.index, y=df_holdings.values,
                   mode='lines',
                   name='每日持股',
                   opacity=0.60,
                   line=dict(color='steelblue', width=0.5)),
    )

    fig.add_trace(
        go.Scatter(x=df_holdings_by_month.index,
                   y=df_holdings_by_month.values,
                   mode='lines',
                   name='月度平均',
                   opacity=0.50,
                   line=dict(color='orangered', width=2)),
    )

    fig.add_trace(
        go.Scatter(x=[df_holdings.index.min(), df_holdings.index.max()],
                   y=[df_holdings.values.mean()]*2,
                   name='期间平均',
                   mode='lines',
                   line=dict(color='steelblue', width=3, dash='dash')),
    )

    range = [returns.index[0], returns.index[-1]]

    fig.update_xaxes(range=range)
    fig.update_yaxes(title_text='持股')
    fig.update_layout(title_text="总持股")
    fig.update_layout(
        legend=dict(
            x=0.01,
            y=0.50,
            yanchor="middle",
            xanchor="left",
        )
    )
    _date_tickformat(fig)
    return fig


def plot_long_short_holdings(returns, positions,
                             legend_loc='upper left', fig=None, **kwargs):
    """
    Plots total amount of stocks with an active position, breaking out
    short and long into transparent filled regions.

    Parameters
    ----------
    returns : pd.Series
        Daily returns of the strategy, noncumulative.
         - See full explanation in tears.create_full_tear_sheet.
    positions : pd.DataFrame, optional
        Daily net position values.
         - See full explanation in tears.create_full_tear_sheet.
    legend_loc : matplotlib.loc, optional
        The location of the legend on the plot.
    ax : matplotlib.Axes, optional
        Axes upon which to plot.
    **kwargs, optional
        Passed to plotting function.

    Returns
    -------
    ax : matplotlib.Axes
        The axes that were plotted on.

    """

    if fig is None:
        fig = go.Figure()

    positions = positions.drop('cash', axis='columns')
    positions = positions.replace(0, np.nan)
    df_longs = positions[positions > 0].count(axis=1)
    df_shorts = positions[positions < 0].count(axis=1)

    fig.add_trace(go.Scatter(
        x=df_longs.index,
        y=df_longs.values,
        line=dict(color='green', width=2.0),
        name='多头 (最多: %s只, 最少: %s只)' % (df_longs.max(),  df_longs.min()),
        opacity=0.50,
        fill='tozeroy')
    )

    fig.add_trace(go.Scatter(
        x=df_shorts.index,
        y=df_shorts.values,
        name='空头 (最多: %s只, 最少: %s只)' % (df_shorts.max(), df_shorts.min()),
        line=dict(color='red', width=2.0),
        opacity=0.50,
        fill='tozeroy')
    )

    fig.update_xaxes(range=[returns.index[0], returns.index[-1]])
    fig.update_yaxes(title_text='持有股票只数')
    fig.update_layout(title_text="多头与空头持股计数")
    fig.update_layout(
        legend=dict(
            x=0.01,
            y=0.50,
            yanchor="middle",
            xanchor="left",
        )
    )
    _date_tickformat(fig)
    return fig


def plot_drawdown_periods(returns, top=10, fig=None, **kwargs):
    """
    Plots cumulative returns highlighting top drawdown periods.

    Parameters
    ----------
    returns : pd.Series
        Daily returns of the strategy, noncumulative.
         - See full explanation in tears.create_full_tear_sheet.
    top : int, optional
        Amount of top drawdowns periods to plot (default 10).
    ax : matplotlib.Axes, optional
        Axes upon which to plot.
    **kwargs, optional
        Passed to plotting function.

    Returns
    -------
    ax : matplotlib.Axes
        The axes that were plotted on.
    """

    if fig is None:
        fig = go.Figure()

    title = f"最大{top}个回撤周期"
    # fig.update_xaxes(title_text='Beta')
    fig.update_yaxes(title_text='累积收益率')
    fig.update_layout(yaxis_tickformat='.2f',
                      title_text=title)

    df_cum_rets = ep.cum_returns(returns, starting_value=1.0)
    df_drawdowns = timeseries.gen_drawdown_table(returns, top=top)

    # df_cum_rets.plot(ax=ax, **kwargs)
    fig.add_trace(
        go.Scatter(x=df_cum_rets.index, y=df_cum_rets.values,
                   mode='lines',
                   #    showlegend=False,
                   name='累积收益率')
    )
    lim = [df_cum_rets.min(), df_cum_rets.max()]
    colors = plotly.colors.sequential.Rainbow
    for i, (peak, recovery) in df_drawdowns[
            ['波峰日期', '恢复日期']].iterrows():
        if pd.isnull(recovery):
            recovery = returns.index[-1]
        fig.add_trace(go.Scatter(
            x=[peak, peak, recovery, recovery],
            y=[lim[0], lim[1], lim[1], lim[0]],
            fill='toself',
            legendgroup="group",
            name=f"Portfolio {i+1}",
            showlegend=True if i == 0 else False,
            fillcolor=colors[i],
            # showlegend=False,
            opacity=0.40,
        ))
    fig.update_yaxes(range=lim)
    fig.update_layout(
        legend=dict(
            x=0.01,
            y=0.99,
            yanchor="top",
            xanchor="left",
            # borderwidth=2,
        )
    )
    _date_tickformat(fig)
    return fig


def plot_drawdown_underwater(returns, fig=None, **kwargs):
    """
    Plots how far underwaterr returns are over time, or plots current
    drawdown vs. date.

    Parameters
    ----------
    returns : pd.Series
        Daily returns of the strategy, noncumulative.
         - See full explanation in tears.create_full_tear_sheet.
    ax : matplotlib.Axes, optional
        Axes upon which to plot.
    **kwargs, optional
        Passed to plotting function.

    Returns
    -------
    ax : matplotlib.Axes
        The axes that were plotted on.
    """

    if fig is None:
        fig = go.Figure()

    title = "水下图"
    fig.update_yaxes(title_text='缩水')
    fig.update_layout(yaxis_tickformat='%',
                      title_text=title)

    df_cum_rets = ep.cum_returns(returns, starting_value=1.0)
    running_max = np.maximum.accumulate(df_cum_rets)
    underwater = -((running_max - df_cum_rets) / running_max)
    # (underwater).plot(ax=ax, kind='area', color='coral', alpha=0.7, **kwargs)
    fig.add_trace(go.Scatter(
        x=underwater.index,
        y=underwater.values,
        line=dict(color='coral'),
        opacity=0.70,
        fill='tozeroy')
    )
    _date_tickformat(fig)
    return fig


def plot_perf_stats(returns, factor_returns, fig=None):
    """
    Create box plot of some performance metrics of the strategy.
    The width of the box whiskers is determined by a bootstrap.

    Parameters
    ----------
    returns : pd.Series
        Daily returns of the strategy, noncumulative.
         - See full explanation in tears.create_full_tear_sheet.
    factor_returns : pd.Series
        Daily noncumulative returns of the benchmark factor to which betas are
        computed. Usually a benchmark such as market returns.
         - This is in the same style as returns.
    fig : matplotlib.Axes, optional
        Axes upon which to plot.

    Returns
    -------
    ax : matplotlib.Axes
        The axes that were plotted on.
    """

    if fig is None:
        fig = go.Figure()

    bootstrap_values = timeseries.perf_stats_bootstrap(returns,
                                                       factor_returns,
                                                       return_stats=False)

    bootstrap_values = bootstrap_values.drop('峰度', axis='columns')

    # sns.boxplot(data=bootstrap_values, orient='h', ax=ax)
    title = '绩效指标'
    fig = px.box(bootstrap_values, x=bootstrap_values.columns, title=title)
    fig.update_xaxes(title_text='')
    fig.update_yaxes(title_text='')
    return fig


STAT_FUNCS_PCT = [
    '年收益率',
    '累积收益率',
    '年波动',
    '最大回撤',
    '日在险价值',
    '日换手率',
]


def show_perf_stats(returns, factor_returns=None, positions=None,
                    transactions=None, turnover_denom='AGB',
                    live_start_date=None, bootstrap=False,
                    header_rows=None):
    """
    Prints some performance metrics of the strategy.

    - Shows amount of time the strategy has been run in backtest and
      out-of-sample (in live trading).

    - Shows Omega ratio, max drawdown, Calmar ratio, annual return,
      stability, Sharpe ratio, annual volatility, alpha, and beta.

    Parameters
    ----------
    returns : pd.Series
        Daily returns of the strategy, noncumulative.
         - See full explanation in tears.create_full_tear_sheet.
    factor_returns : pd.Series, optional
        Daily noncumulative returns of the benchmark factor to which betas are
        computed. Usually a benchmark such as market returns.
         - This is in the same style as returns.
    positions : pd.DataFrame, optional
        Daily net position values.
         - See full explanation in create_full_tear_sheet.
    transactions : pd.DataFrame, optional
        Prices and amounts of executed trades. One row per trade.
        - See full explanation in tears.create_full_tear_sheet
    turnover_denom : str, optional
        Either AGB or portfolio_value, default AGB.
        - See full explanation in txn.get_turnover.
    live_start_date : datetime, optional
        The point in time when the strategy began live trading, after
        its backtest period.
    bootstrap : boolean, optional
        Whether to perform bootstrap analysis for the performance
        metrics.
         - For more information, see timeseries.perf_stats_bootstrap
    header_rows : dict or OrderedDict, optional
        Extra rows to display at the top of the displayed table.
    """

    if bootstrap:
        perf_func = timeseries.perf_stats_bootstrap
    else:
        perf_func = timeseries.perf_stats

    perf_stats_all = perf_func(
        returns,
        factor_returns=factor_returns,
        positions=positions,
        transactions=transactions,
        turnover_denom=turnover_denom)

    date_rows = OrderedDict()
    if len(returns.index) > 0:
        date_rows['开始日期'] = returns.index[0].strftime(r'%Y-%m-%d')
        date_rows['结束日期'] = returns.index[-1].strftime(r'%Y-%m-%d')

    if live_start_date is not None:
        live_start_date = ep.utils.get_utc_timestamp(live_start_date)
        returns_is = returns[returns.index < live_start_date]
        returns_oos = returns[returns.index >= live_start_date]

        positions_is = None
        positions_oos = None
        transactions_is = None
        transactions_oos = None

        if positions is not None:
            positions_is = positions[positions.index < live_start_date]
            positions_oos = positions[positions.index >= live_start_date]
            if transactions is not None:
                transactions_is = transactions[(transactions.index <
                                                live_start_date)]
                transactions_oos = transactions[(transactions.index >
                                                 live_start_date)]

        perf_stats_is = perf_func(
            returns_is,
            factor_returns=factor_returns,
            positions=positions_is,
            transactions=transactions_is,
            turnover_denom=turnover_denom)

        perf_stats_oos = perf_func(
            returns_oos,
            factor_returns=factor_returns,
            positions=positions_oos,
            transactions=transactions_oos,
            turnover_denom=turnover_denom)
        if len(returns.index) > 0:
            date_rows['样本内月数'] = int(len(returns_is) /
                                     APPROX_BDAYS_PER_MONTH)
            date_rows['样本外月数'] = int(len(returns_oos) /
                                     APPROX_BDAYS_PER_MONTH)

        perf_stats = pd.concat(OrderedDict([
            ('样本内', perf_stats_is),
            ('样本外', perf_stats_oos),
            ('全部', perf_stats_all),
        ]), axis=1)
    else:
        if len(returns.index) > 0:
            date_rows['总月数'] = int(len(returns) /
                                   APPROX_BDAYS_PER_MONTH)
        perf_stats = pd.DataFrame(perf_stats_all, columns=['回测'])

    for column in perf_stats.columns:
        for stat, value in perf_stats[column].iteritems():
            if stat in STAT_FUNCS_PCT:
                perf_stats.loc[stat, column] = str(np.round(value * 100,
                                                            3)) + '%'
    if header_rows is None:
        header_rows = date_rows
    else:
        header_rows = OrderedDict(header_rows)
        header_rows.update(date_rows)

    utils.print_table(
        perf_stats,
        float_format='{0:.2f}'.format,
        header_rows=header_rows,
    )


def plot_returns(returns,
                 live_start_date=None,
                 fig=None):
    """
    Plots raw returns over time.

    Backtest returns are in green, and out-of-sample (live trading)
    returns are in red.

    Parameters
    ----------
    returns : pd.Series
        Daily returns of the strategy, noncumulative.
         - See full explanation in tears.create_full_tear_sheet.
    live_start_date : datetime, optional
        The date when the strategy began live trading, after
        its backtest period. This date should be normalized.
    ax : matplotlib.Axes, optional
        Axes upon which to plot.
    **kwargs, optional
        Passed to plotting function.

    Returns
    -------
    ax : matplotlib.Axes
        The axes that were plotted on.
    """

    if fig is None:
        fig = go.Figure()

    fig.update_yaxes(title_text='收益率')

    if live_start_date is not None:
        live_start_date = ep.utils.get_utc_timestamp(live_start_date)
        is_returns = returns.loc[returns.index < live_start_date]
        fig.add_trace(
            go.Scatter(x=is_returns.index, y=is_returns.values,
                       mode='lines',
                       showlegend=False,
                       line=dict(color='green'))
        )
        oos_returns = returns.loc[returns.index >= live_start_date]
        fig.add_trace(
            go.Scatter(x=oos_returns.index, y=oos_returns.values,
                       mode='lines',
                       showlegend=False,
                       line=dict(color='red'))
        )
    else:
        fig.add_trace(
            go.Scatter(x=returns.index, y=returns.values,
                       mode='lines',
                       showlegend=False,
                       line=dict(color='green'))
        )
    _date_tickformat(fig)
    return fig


def plot_rolling_returns(returns,
                         factor_returns=None,
                         live_start_date=None,
                         logy=False,
                         cone_std=None,
                         legend_loc='best',
                         volatility_match=False,
                         cone_function=timeseries.forecast_cone_bootstrap,
                         fig=None, **kwargs):
    """
    Plots cumulative rolling returns versus some benchmarks'.

    Backtest returns are in green, and out-of-sample (live trading)
    returns are in red.

    Additionally, a non-parametric cone plot may be added to the
    out-of-sample returns region.

    Parameters
    ----------
    returns : pd.Series
        Daily returns of the strategy, noncumulative.
         - See full explanation in tears.create_full_tear_sheet.
    factor_returns : pd.Series, optional
        Daily noncumulative returns of the benchmark factor to which betas are
        computed. Usually a benchmark such as market returns.
         - This is in the same style as returns.
    live_start_date : datetime, optional
        The date when the strategy began live trading, after
        its backtest period. This date should be normalized.
    logy : bool, optional
        Whether to log-scale the y-axis.
    cone_std : float, or tuple, optional
        If float, The standard deviation to use for the cone plots.
        If tuple, Tuple of standard deviation values to use for the cone plots
         - See timeseries.forecast_cone_bounds for more details.
    legend_loc : matplotlib.loc, optional
        The location of the legend on the plot.
    volatility_match : bool, optional
        Whether to normalize the volatility of the returns to those of the
        benchmark returns. This helps compare strategies with different
        volatilities. Requires passing of benchmark_rets.
    cone_function : function, optional
        Function to use when generating forecast probability cone.
        The function signiture must follow the form:
        def cone(in_sample_returns (pd.Series),
                 days_to_project_forward (int),
                 cone_std= (float, or tuple),
                 starting_value= (int, or float))
        See timeseries.forecast_cone_bootstrap for an example.
    fig :Figure, optional
        Figure upon which to plot.
    **kwargs, optional
        Passed to plotting function.

    Returns
    -------
    ax : matplotlib.Axes
        The axes that were plotted on.
    """

    if fig is None:
        fig = go.Figure()

    fig.update_yaxes(title_text='累积收益率')
    if logy:
        fig.update_yaxes(type="log")

    if volatility_match and factor_returns is None:
        raise ValueError('volatility_match requires passing of '
                         'factor_returns.')
    elif volatility_match and factor_returns is not None:
        bmark_vol = factor_returns.loc[returns.index].std()
        returns = (returns / returns.std()) * bmark_vol

    cum_rets = ep.cum_returns(returns, 1.0)

    fig.update_layout(yaxis_tickformat='.2f')

    if factor_returns is not None:
        cum_factor_returns = ep.cum_returns(
            factor_returns[cum_rets.index], 1.0)
        fig.add_trace(
            go.Scatter(x=cum_factor_returns.index, y=cum_factor_returns.values,
                       mode='lines',
                       opacity=0.60,
                       showlegend=True if legend_loc else False,
                       line=dict(color='gray', width=2),
                       name=factor_returns.name)
        )

    if live_start_date is not None:
        live_start_date = ep.utils.get_utc_timestamp(live_start_date)
        is_cum_returns = cum_rets.loc[cum_rets.index < live_start_date]
        oos_cum_returns = cum_rets.loc[cum_rets.index >= live_start_date]
    else:
        is_cum_returns = cum_rets
        oos_cum_returns = pd.Series([])

    fig.add_trace(
        go.Scatter(x=is_cum_returns.index, y=is_cum_returns.values,
                   mode='lines',
                   opacity=0.60,
                   showlegend=True if legend_loc else False,
                   line=dict(color='forestgreen', width=3),
                   name='回测')
    )
    if len(oos_cum_returns) > 0:
        fig.add_trace(
            go.Scatter(x=oos_cum_returns.index, y=oos_cum_returns.values,
                       mode='lines',
                       opacity=0.60,
                       showlegend=True if legend_loc else False,
                       line=dict(color='red', width=4),
                       name='直播')
        )

        if cone_std is not None:
            if isinstance(cone_std, (float, int)):
                cone_std = [cone_std]

            is_returns = returns.loc[returns.index < live_start_date]
            cone_bounds = cone_function(
                is_returns,
                len(oos_cum_returns),
                cone_std=cone_std,
                starting_value=is_cum_returns[-1])

            cone_bounds = cone_bounds.set_index(oos_cum_returns.index)
            for std in cone_std:
                # fill 重复序列 2 次
                x = cone_bounds.index.to_list()
                x += x[::-1]
                y = cone_bounds[float(std)].tolist()
                y += cone_bounds[float(-std)].tolist()[::-1]
                fig.add_trace(go.Scatter(
                    x=x,
                    y=y,
                    fill='toself',
                    fillcolor='steelblue',
                    # line_color='rgba(255,255,255,0)',
                    showlegend=False,
                    # name='Ideal',
                    opacity=0.50,
                ))

    fig.add_trace(
        go.Scatter(x=[cum_rets.index.min(), cum_rets.index.max()], y=[1.0]*2,
                   mode='lines',
                   opacity=0.60,
                   showlegend=True if legend_loc else False,
                   line=dict(color='black', width=2, dash='dash'),
                   name='基线')
    )

    if legend_loc:
        fig.update_layout(
            legend=dict(
                x=0.01,
                y=0.99,
                yanchor="top",
                xanchor="left",
                title_font_family="SimHei",
                font=dict(
                    family="SimHei",
                    size=12,
                    color="black",
                ),
                borderwidth=1
            )
        )
    _date_tickformat(fig)
    return fig


def plot_rolling_beta(returns, factor_returns, legend_loc='best',
                      fig=None, **kwargs):
    """
    Plots the rolling 6-month and 12-month beta versus date.

    Parameters
    ----------
    returns : pd.Series
        Daily returns of the strategy, noncumulative.
         - See full explanation in tears.create_full_tear_sheet.
    factor_returns : pd.Series
        Daily noncumulative returns of the benchmark factor to which betas are
        computed. Usually a benchmark such as market returns.
         - This is in the same style as returns.
    legend_loc : matplotlib.loc, optional
        The location of the legend on the plot.
    ax : matplotlib.Axes, optional
        Axes upon which to plot.
    **kwargs, optional
        Passed to plotting function.

    Returns
    -------
    ax : matplotlib.Axes
        The axes that were plotted on.
    """

    if fig is None:
        fig = go.Figure()

    title = f"相对于 {str(factor_returns.name)} 投资组合滚动β"
    fig.update_yaxes(title_text='Beta')
    fig.update_layout(yaxis_tickformat='.2f',
                      title_text=title)

    rb_1 = timeseries.rolling_beta(
        returns, factor_returns, rolling_window=APPROX_BDAYS_PER_MONTH * 6)
    fig.add_trace(
        go.Scatter(x=rb_1.index, y=rb_1.values,
                   mode='lines',
                   opacity=0.60,
                   legendgroup="group",
                   name="6个月滚动β",
                   showlegend=True if legend_loc else False,
                   line=dict(color='steelblue', width=3))
    )

    rb_2 = timeseries.rolling_beta(
        returns, factor_returns, rolling_window=APPROX_BDAYS_PER_MONTH * 12)

    fig.add_trace(
        go.Scatter(x=rb_2.index, y=rb_2.values,
                   mode='lines',
                   opacity=0.40,
                   legendgroup="group",
                   name="12个月滚动β",
                   showlegend=True if legend_loc else False,
                   line=dict(color='grey', width=3))
    )

    fig.add_trace(
        go.Scatter(x=[rb_1.index[0], rb_1.index[-1]], y=[rb_1.mean()]*2,
                   mode='lines',
                   name='6个月平均值',
                   showlegend=True if legend_loc else False,
                   line=dict(color='steelblue', width=3, dash='dash'))
    )

    # ax.axhline(0.0, color='black', linestyle='-', lw=2)
    fig.add_trace(
        go.Scatter(x=rb_1.index, y=[0.0]*len(rb_1),
                   mode='lines',
                   name='基线',
                   showlegend=True if legend_loc else False,
                   line=dict(color='black', width=2, dash='dash'))
    )

    if legend_loc:
        fig.update_layout(
            legend=dict(
                x=0.01,
                y=0.99,
                yanchor="top",
                xanchor="left",
            )
        )
    fig.update_yaxes(range=[-1.0, 1.0])
    _date_tickformat(fig)
    return fig


def plot_rolling_volatility(returns, factor_returns=None,
                            rolling_window=APPROX_BDAYS_PER_MONTH * 6,
                            legend_loc='best', fig=None, **kwargs):
    """
    Plots the rolling volatility versus date.

    Parameters
    ----------
    returns : pd.Series
        Daily returns of the strategy, noncumulative.
         - See full explanation in tears.create_full_tear_sheet.
    factor_returns : pd.Series, optional
        Daily noncumulative returns of the benchmark factor for which the
        benchmark rolling volatility is computed. Usually a benchmark such
        as market returns.
         - This is in the same style as returns.
    rolling_window : int, optional
        The days window over which to compute the volatility.
    legend_loc : matplotlib.loc, optional
        The location of the legend on the plot.
    ax : matplotlib.Axes, optional
        Axes upon which to plot.
    **kwargs, optional
        Passed to plotting function.

    Returns
    -------
    ax : matplotlib.Axes
        The axes that were plotted on.
    """

    if fig is None:
        fig = go.Figure()

    title = '滚动波动(6-month)'
    fig.update_layout(yaxis_tickformat='.2f',
                      title_text=title)

    rolling_vol_ts = timeseries.rolling_volatility(
        returns, rolling_window)
    # rolling_vol_ts.plot(alpha=.7, lw=3, color='orangered', ax=ax,
    #                     **kwargs)
    fig.add_trace(
        go.Scatter(x=rolling_vol_ts.index, y=rolling_vol_ts.values,
                   mode='lines',
                   opacity=0.70,
                   #    legendgroup="group",
                   name="6个月波动率",
                   showlegend=True if legend_loc else False,
                   line=dict(color='orangered', width=3))
    )

    if factor_returns is not None:
        rolling_vol_ts_factor = timeseries.rolling_volatility(
            factor_returns, rolling_window)
        # rolling_vol_ts_factor.plot(alpha=.7, lw=3, color='grey', ax=ax,
        #                            **kwargs)
        fig.add_trace(
            go.Scatter(x=rolling_vol_ts_factor.index, y=rolling_vol_ts_factor.values,
                       mode='lines',
                       opacity=0.70,
                       # legendgroup="group",
                       name="指数波动率",
                       showlegend=True if legend_loc else False,
                       line=dict(color='grey', width=3))
        )
    # ax.set_title('Rolling volatility (6-month)')
    # ax.axhline(
    #     rolling_vol_ts.mean(),
    #     color='steelblue',
    #     linestyle='--',
    #     lw=3)
    fig.add_trace(
        go.Scatter(x=rolling_vol_ts.index, y=[rolling_vol_ts.mean()]*len(rolling_vol_ts),
                   mode='lines',
                   name='平均波动率',
                   showlegend=True if legend_loc else False,
                   line=dict(color='steelblue', width=3, dash='dash'))
    )
    # ax.axhline(0.0, color='black', linestyle='-', lw=2)
    fig.add_trace(
        go.Scatter(x=[rolling_vol_ts.index.min(), rolling_vol_ts.index.max()], y=[0.0]*2,
                   mode='lines',
                   name='基线',
                   showlegend=True if legend_loc else False,
                   line=dict(color='black', width=2, dash='dash'))
    )
    # ax.set_ylabel('Volatility')
    fig.update_yaxes(title_text='波动')
    # ax.set_xlabel('')
    # if factor_returns is None:
    #     ax.legend(['Volatility', 'Average volatility'],
    #               loc=legend_loc, frameon=True, framealpha=0.5)
    # else:
    #     ax.legend(['Volatility', 'Benchmark volatility', 'Average volatility'],
    #               loc=legend_loc, frameon=True, framealpha=0.5)
    if legend_loc:
        fig.update_layout(
            legend=dict(
                x=0.01,
                y=0.99,
                yanchor="top",
                xanchor="left",
            )
        )
    _date_tickformat(fig)
    return fig


def plot_rolling_sharpe(returns, factor_returns=None,
                        rolling_window=APPROX_BDAYS_PER_MONTH * 6,
                        legend_loc='best', fig=None, **kwargs):
    """
    Plots the rolling Sharpe ratio versus date.

    Parameters
    ----------
    returns : pd.Series
        Daily returns of the strategy, noncumulative.
         - See full explanation in tears.create_full_tear_sheet.
    factor_returns : pd.Series, optional
        Daily noncumulative returns of the benchmark factor for
        which the benchmark rolling Sharpe is computed. Usually
        a benchmark such as market returns.
         - This is in the same style as returns.
    rolling_window : int, optional
        The days window over which to compute the sharpe ratio.
    legend_loc : matplotlib.loc, optional
        The location of the legend on the plot.
    ax : matplotlib.Axes, optional
        Axes upon which to plot.
    **kwargs, optional
        Passed to plotting function.

    Returns
    -------
    ax : matplotlib.Axes
        The axes that were plotted on.
    """

    if fig is None:
        fig = go.Figure()

    title = '滚动Sharpe比率(6个月)'
    fig.update_layout(yaxis_tickformat='.2f',
                      title_text=title)

    rolling_sharpe_ts = timeseries.rolling_sharpe(
        returns, rolling_window)
    # rolling_sharpe_ts.plot(alpha=.7, lw=3, color='orangered', ax=ax,
    #                        **kwargs)
    fig.add_trace(
        go.Scatter(x=rolling_sharpe_ts.index, y=rolling_sharpe_ts.values,
                   mode='lines',
                   opacity=0.70,
                   name="6个月Sharpe比率",
                   showlegend=True if legend_loc else False,
                   line=dict(color='orangered', width=3))
    )

    if factor_returns is not None:
        rolling_sharpe_ts_factor = timeseries.rolling_sharpe(
            factor_returns, rolling_window)
        # rolling_sharpe_ts_factor.plot(alpha=.7, lw=3, color='grey', ax=ax,
        #                               **kwargs)
        fig.add_trace(
            go.Scatter(x=rolling_sharpe_ts_factor.index, y=rolling_sharpe_ts_factor.values,
                       mode='lines',
                       opacity=0.70,
                       name="指数sharpe比率",
                       showlegend=True if legend_loc else False,
                       line=dict(color='grey', width=3))
        )
    # ax.set_title('Rolling Sharpe ratio (6-month)')
    # ax.axhline(
    #     rolling_sharpe_ts.mean(),
    #     color='steelblue',
    #     linestyle='--',
    #     lw=3)
    fig.add_trace(
        go.Scatter(x=rolling_sharpe_ts.index, y=[rolling_sharpe_ts.mean()]*len(rolling_sharpe_ts),
                   mode='lines',
                   name='平均',
                   showlegend=True if legend_loc else False,
                   line=dict(color='steelblue', width=3, dash='dash'))
    )
    # ax.axhline(0.0, color='black', linestyle='-', lw=3)
    fig.add_trace(
        go.Scatter(x=[rolling_sharpe_ts.index.min(), rolling_sharpe_ts.index.max()], y=[0.0]*2,
                   mode='lines',
                   name='基线',
                   showlegend=True if legend_loc else False,
                   line=dict(color='black', width=2, dash='dash'))
    )
    # ax.set_ylabel('Sharpe ratio')
    fig.update_yaxes(title_text='Sharpe比率')
    # ax.set_xlabel('')
    # if factor_returns is None:
    #     ax.legend(['Sharpe', 'Average'],
    #               loc=legend_loc, frameon=True, framealpha=0.5)
    # else:
    #     ax.legend(['Sharpe', 'Benchmark Sharpe', 'Average'],
    #               loc=legend_loc, frameon=True, framealpha=0.5)
    if legend_loc:
        fig.update_layout(
            legend=dict(
                x=0.01,
                y=0.99,
                yanchor="top",
                xanchor="left",
            )
        )
    _date_tickformat(fig)
    return fig


def plot_gross_leverage(returns, positions, fig=None, **kwargs):
    """
    Plots gross leverage versus date.

    Gross leverage is the sum of long and short exposure per share
    divided by net asset value.

    Parameters
    ----------
    returns : pd.Series
        Daily returns of the strategy, noncumulative.
         - See full explanation in tears.create_full_tear_sheet.
    positions : pd.DataFrame
        Daily net position values.
         - See full explanation in create_full_tear_sheet.
    fig : matplotlib.Axes, optional
        Axes upon which to plot.
    **kwargs, optional
        Passed to plotting function.

    Returns
    -------
    ax : matplotlib.Axes
        The axes that were plotted on.
    """

    if fig is None:
        fig = go.Figure()
    gl = timeseries.gross_lev(positions)
    fig.add_trace(
        go.Scatter(x=gl.index,
                   y=gl.values,
                   mode='lines',
                   name='杠杆系数',
                   line=dict(color='limegreen', width=0.5))
    )
    fig.add_trace(
        go.Scatter(x=[gl.index.min(), gl.index.max()], y=[gl.mean()]*2,
                   mode='lines',
                   name='均值',
                   line=dict(color='green', width=3, dash='dash'))
    )

    fig.update_yaxes(title_text='总杠杆', tickformat='.2f')
    fig.update_layout(title_text='总杠杆', showlegend=False)
    _date_tickformat(fig)
    return fig


def plot_exposures(returns, positions, fig=None, **kwargs):
    """
    Plots a cake chart of the long and short exposure.

    Parameters
    ----------
    returns : pd.Series
        Daily returns of the strategy, noncumulative.
         - See full explanation in tears.create_full_tear_sheet.
    positions_alloc : pd.DataFrame
        Portfolio allocation of positions. See
        pos.get_percent_alloc.
    fig : matplotlib.Axes, optional
        Axes upon which to plot.
    **kwargs, optional
        Passed to plotting function.

    Returns
    -------
    fig : matplotlib.Axes
        The axes that were plotted on.
    """

    if fig is None:
        fig = go.Figure()

    pos_no_cash = positions.drop('cash', axis=1)
    l_exp = pos_no_cash[pos_no_cash > 0].sum(axis=1) / positions.sum(axis=1)
    s_exp = pos_no_cash[pos_no_cash < 0].sum(axis=1) / positions.sum(axis=1)
    net_exp = pos_no_cash.sum(axis=1) / positions.sum(axis=1)

    fig.add_trace(go.Scatter(
        x=l_exp.index,
        y=l_exp.values,
        name='多头',
        line=dict(color='green'),
        opacity=0.50,
        fill='tozeroy')
    )

    fig.add_trace(go.Scatter(
        x=s_exp.index,
        y=s_exp.values,
        name='空头',
        line=dict(color='red'),
        opacity=0.50,
        fill='tozeroy')
    )

    fig.add_trace(go.Scatter(
        x=net_exp.index,
        y=net_exp.values,
        name='净头寸',
        line=dict(color='black', dash='dot'),
        fill='tozeroy')
    )

    fig.update_xaxes(title_text='日期', range=[
                     returns.index[0], returns.index[-1]])
    fig.update_yaxes(title_text='风险敞口')
    fig.update_layout(
        legend=dict(
            x=0.01,
            y=0.99,
            yanchor="top",
            xanchor="left",
        )
    )
    _date_tickformat(fig)
    return fig


def show_and_plot_top_positions(returns, positions_alloc,
                                show_and_plot=2, hide_positions=False,
                                legend_loc='real_best', fig=None,
                                **kwargs):
    """
    Prints and/or plots the exposures of the top 10 held positions of
    all time.

    Parameters
    ----------
    returns : pd.Series
        Daily returns of the strategy, noncumulative.
         - See full explanation in tears.create_full_tear_sheet.
    positions_alloc : pd.DataFrame
        Portfolio allocation of positions. See pos.get_percent_alloc.
    show_and_plot : int, optional
        By default, this is 2, and both prints and plots.
        If this is 0, it will only plot; if 1, it will only print.
    hide_positions : bool, optional
        If True, will not output any symbol names.
    legend_loc : matplotlib.loc, optional
        The location of the legend on the plot.
        By default, the legend will display below the plot.
    ax : matplotlib.Axes, optional
        Axes upon which to plot.
    **kwargs, optional
        Passed to plotting function.

    Returns
    -------
    ax : matplotlib.Axes, conditional
        The axes that were plotted on.

    """
    positions_alloc = positions_alloc.copy()
    positions_alloc.columns = positions_alloc.columns.map(utils.format_asset)

    df_top_long, df_top_short, df_top_abs = pos.get_top_long_short_abs(
        positions_alloc)

    if show_and_plot == 1 or show_and_plot == 2:
        utils.print_table(pd.DataFrame(df_top_long * 100, columns=['max']),
                          float_format='{0:.2f}%'.format,
                          name='期间排名前10位的【多头】头寸')

        utils.print_table(pd.DataFrame(df_top_short * 100, columns=['max']),
                          float_format='{0:.2f}%'.format,
                          name='期间排名前10位的【空头】头寸')

        utils.print_table(pd.DataFrame(df_top_abs * 100, columns=['max']),
                          float_format='{0:.2f}%'.format,
                          name='期间排名前10位总头寸')

    if show_and_plot == 0 or show_and_plot == 2:
        if fig is None:
            fig = go.Figure()
        title = "排名前10投资组合分配"
        df = positions_alloc[df_top_abs.index]
        for col in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df[col].values,
                    name=col,
                    opacity=0.5,
                )
            )
        if hide_positions:
            fig.update_layout(showlegend=False)
        fig.update_layout(title_text=title)
        fig.update_yaxes(title_text='持有风险敞口', tickformat='%')
        _date_tickformat(fig)
        return fig


def plot_max_median_position_concentration(positions, fig=None, **kwargs):
    """
    Plots the max and median of long and short position concentrations
    over the time.

    Parameters
    ----------
    positions : pd.DataFrame
        The positions that the strategy takes over time.
    fig : matplotlib.Axes, optional
        Axes upon which to plot.

    Returns
    -------
    fig : matplotlib.Axes
        The axes that were plotted on.
    """

    if fig is None:
        fig = go.Figure()

    alloc_summary = pos.get_max_median_position_concentration(positions)
    colors = ['mediumblue', 'steelblue', 'tomato', 'firebrick']
    for i, col in enumerate(alloc_summary.columns):
        fig.add_trace(
            go.Scatter(
                x=alloc_summary.index,
                y=alloc_summary[col].values,
                name=col,
                opacity=0.6,
                line=dict(color=colors[i], width=1)
            )
        )

    fig.update_yaxes(title_text='风险敞口', tickformat='%')
    fig.update_layout(
        title_text='多头/空头 最大值与中位数头寸',
        legend=dict(
            x=0.01,
            yanchor="auto",
            xanchor="left",
        )
    )
    _date_tickformat(fig)
    return fig


def plot_sector_allocations(returns, sector_alloc, fig=None, **kwargs):
    """
    Plots the sector exposures of the portfolio over time.

    Parameters
    ----------
    returns : pd.Series
        Daily returns of the strategy, noncumulative.
         - See full explanation in tears.create_full_tear_sheet.
    sector_alloc : pd.DataFrame
        Portfolio allocation of positions. See pos.get_sector_alloc.
    fig : matplotlib.Axes, optional
        Axes upon which to plot.
    **kwargs, optional
        Passed to plotting function.

    Returns
    -------
    ax : matplotlib.Axes
        The axes that were plotted on.
    """

    if fig is None:
        fig = go.Figure()

    for col in sector_alloc.columns:
        fig.add_trace(
            go.Scatter(
                x=sector_alloc.index,
                y=sector_alloc[col].values,
                name=col,
                opacity=0.5,
            )
        )

    fig.update_xaxes(range=[sector_alloc.index[0], sector_alloc.index[-1]])
    fig.update_yaxes(title_text='行业风险敞口', tickformat='%')
    fig.update_layout(
        title_text='行业分配',
        legend=dict(
            x=0.01,
            yanchor="auto",
            xanchor="left",
        )
    )
    return fig


def plot_return_quantiles(returns, live_start_date=None, fig=None, **kwargs):
    """
    Creates a box plot of daily, weekly, and monthly return
    distributions.

    Parameters
    ----------
    returns : pd.Series
        Daily returns of the strategy, noncumulative.
         - See full explanation in tears.create_full_tear_sheet.
    live_start_date : datetime, optional
        The point in time when the strategy began live trading, after
        its backtest period.
    ax : matplotlib.Axes, optional
        Axes upon which to plot.
    **kwargs, optional
        Passed to seaborn plotting function.

    Returns
    -------
    ax : matplotlib.Axes
        The axes that were plotted on.
    """

    if fig is None:
        fig = go.Figure()

    is_returns = returns if live_start_date is None \
        else returns.loc[returns.index < live_start_date]
    is_weekly = ep.aggregate_returns(is_returns, 'weekly')
    is_monthly = ep.aggregate_returns(is_returns, 'monthly')

    colors = ["#4c72B0", "#55A868", "#CCB974"]
    fig.add_trace(
        go.Box(y=is_returns.values,
               legendgroup="样本内",
               name='样本内',
               #    showlegend=False,
               marker_color=colors[0],
               boxpoints='all',
               jitter=0.3,  # add some jitter for a better separation between points
               marker_size=2,
               line_width=1,
               ),
        row=1, col=1
    )
    fig.add_trace(
        go.Box(y=is_weekly.values,
               legendgroup="样本内",
               name='样本内',
               marker_color=colors[1],
               showlegend=False,
               boxpoints='all',
               jitter=0.3,  # add some jitter for a better separation between points
               marker_size=2,
               line_width=1,
               ),
        row=1, col=2
    )
    fig.add_trace(
        go.Box(y=is_monthly.values,
               legendgroup="样本内",
               name='样本内',
               marker_color=colors[2],
               showlegend=False,
               boxpoints='all',
               jitter=0.3,  # add some jitter for a better separation between points
               marker_size=2,
               line_width=1,
               ),
        row=1, col=3
    )

    if live_start_date is not None:
        oos_returns = returns.loc[returns.index >= live_start_date]
        oos_weekly = ep.aggregate_returns(oos_returns, 'weekly')
        oos_monthly = ep.aggregate_returns(oos_returns, 'monthly')

        fig.add_trace(
            go.Box(y=oos_returns.values,
                   legendgroup="样本外",
                   name='样本外',
                   showlegend=True,
                   marker_color="red",
                   boxpoints='all',
                   marker_size=2,
                   line_width=1,
                   jitter=0.3,
                   ),
            row=1, col=1
        )
        fig.add_trace(
            go.Box(y=oos_weekly.values,
                   legendgroup="样本外",
                   name='样本外',
                   marker_color="red",
                   showlegend=False,
                   boxpoints='all',  # can also be outliers, or suspectedoutliers, or False
                   jitter=0.3,  # add some jitter for a better separation between points
                   marker_size=2,
                   line_width=1,
                   ),
            row=1, col=2
        )
        fig.add_trace(
            go.Box(y=oos_monthly.values,
                   legendgroup="样本外",
                   name='样本外',
                   marker_color="red",
                   showlegend=False,
                   boxpoints='all',  # can also be outliers, or suspectedoutliers, or False
                   jitter=0.3,  # add some jitter for a better separation between points
                   marker_size=2,
                   line_width=1,
                   ),
            row=1, col=3
        )
    fig.update_layout(yaxis_tickformat='%',
                      title_text='分位数分组收益率')

    fig.update_layout(
        # boxmode='group',  # group together boxes of the different traces for each value of x
        legend=dict(
            x=0.01,
            y=0.99,
            yanchor="top",
            xanchor="left",
        )
    )
    return fig


def plot_turnover(returns, transactions, positions, turnover_denom='AGB',
                  legend_loc='best', fig=None, **kwargs):
    """
    Plots turnover vs. date.

    Turnover is the number of shares traded for a period as a fraction
    of total shares.

    Displays daily total, daily average per month, and all-time daily
    average.

    Parameters
    ----------
    returns : pd.Series
        Daily returns of the strategy, noncumulative.
         - See full explanation in tears.create_full_tear_sheet.
    transactions : pd.DataFrame
        Prices and amounts of executed trades. One row per trade.
         - See full explanation in tears.create_full_tear_sheet.
    positions : pd.DataFrame
        Daily net position values.
         - See full explanation in tears.create_full_tear_sheet.
    turnover_denom : str, optional
        Either AGB or portfolio_value, default AGB.
        - See full explanation in txn.get_turnover.
    legend_loc : matplotlib.loc, optional
        The location of the legend on the plot.
    fig : matplotlib.Axes, optional
        Axes upon which to plot.
    **kwargs, optional
        Passed to plotting function.

    Returns
    -------
    ax : matplotlib.Axes
        The axes that were plotted on.
    """

    if fig is None:
        fig = go.Figure()

    fig.update_layout(yaxis_tickformat='.2f')

    df_turnover = txn.get_turnover(positions, transactions, turnover_denom)
    df_turnover_by_month = df_turnover.resample("M").mean()
    # df_turnover.plot(color='steelblue', alpha=1.0, lw=0.5, ax=ax, **kwargs)
    fig.add_trace(
        go.Scatter(
            x=df_turnover.index,
            y=df_turnover.values,
            name='日换手率',
            opacity=1.0,
            line=dict(color='steelblue', width=0.5)
        )
    )
    # df_turnover_by_month.plot(
    #     color='orangered',
    #     alpha=0.5,
    #     lw=2,
    #     ax=ax,
    #     **kwargs)
    fig.add_trace(
        go.Scatter(
            x=df_turnover_by_month.index,
            y=df_turnover_by_month.values,
            name='月均换手率',
            opacity=0.5,
            line=dict(color='orangered', width=2)
        )
    )

    # ax.axhline(
    #     df_turnover.mean(), color='steelblue', linestyle='--', lw=3, alpha=1.0)
    fig.add_trace(
        go.Scatter(
            x=[df_turnover.index.min(), df_turnover.index.max()],
            y=[df_turnover.mean()]*2,
            name='日换手率均值',
            opacity=1.0,
            line=dict(color='steelblue', width=3, dash='dash')
        )
    )

    fig.update_xaxes(range=[returns.index[0], returns.index[-1]])
    fig.update_yaxes(title_text='换手率', range=[0, 2])
    fig.update_layout(title_text='每日换手率')
    fig.update_layout(
        legend=dict(
            x=0.01,
            y=0.99,
            yanchor="top",
            xanchor="left",
        )
    )
    _date_tickformat(fig)
    return fig


def plot_slippage_sweep(returns, positions, transactions,
                        slippage_params=(3, 8, 10, 12, 15, 20, 50),
                        fig=None, **kwargs):
    """
    Plots equity curves at different per-dollar slippage assumptions.

    Parameters
    ----------
    returns : pd.Series
        Timeseries of portfolio returns to be adjusted for various
        degrees of slippage.
    positions : pd.DataFrame
        Daily net position values.
         - See full explanation in tears.create_full_tear_sheet.
    transactions : pd.DataFrame
        Prices and amounts of executed trades. One row per trade.
         - See full explanation in tears.create_full_tear_sheet.
    slippage_params: tuple
        Slippage pameters to apply to the return time series (in
        basis points).
    ax : matplotlib.Axes, optional
        Axes upon which to plot.
    **kwargs, optional
        Passed to seaborn plotting function.

    Returns
    -------
    ax : matplotlib.Axes
        The axes that were plotted on.
    """

    if fig is None:
        fig = go.Figure()

    slippage_sweep = pd.DataFrame()
    for bps in slippage_params:
        adj_returns = txn.adjust_returns_for_slippage(returns, positions,
                                                      transactions, bps)
        label = str(bps) + " bps"
        slippage_sweep[label] = ep.cum_returns(adj_returns, 1)

    # slippage_sweep.plot(alpha=1.0, lw=0.5, ax=ax)
    for col in slippage_sweep.columns:
        fig.add_trace(
            go.Scatter(
                x=slippage_sweep.index,
                y=slippage_sweep[col].values,
                name=col,
                opacity=1.0,
                line=dict(width=0.5),
            )
        )

    # ax.set_title('Cumulative returns given additional per-dollar slippage')
    # ax.set_ylabel('')

    # ax.legend(loc='center left', frameon=True, framealpha=0.5)

    fig.update_layout(title_text='考虑额外每元滑点后的累积收益率')
    fig.update_layout(
        legend=dict(
            x=0.01,
            # y=0.99,
            yanchor="auto",
            xanchor="left",
        )
    )
    _date_tickformat(fig)
    return fig


def plot_slippage_sensitivity(returns, positions, transactions,
                              fig=None, **kwargs):
    """
    Plots curve relating per-dollar slippage to average annual returns.

    Parameters
    ----------
    returns : pd.Series
        Timeseries of portfolio returns to be adjusted for various
        degrees of slippage.
    positions : pd.DataFrame
        Daily net position values.
         - See full explanation in tears.create_full_tear_sheet.
    transactions : pd.DataFrame
        Prices and amounts of executed trades. One row per trade.
         - See full explanation in tears.create_full_tear_sheet.
    ax : matplotlib.Axes, optional
        Axes upon which to plot.
    **kwargs, optional
        Passed to seaborn plotting function.

    Returns
    -------
    ax : matplotlib.Axes
        The axes that were plotted on.
    """

    if fig is None:
        fig = go.Figure()

    avg_returns_given_slippage = pd.Series()
    for bps in range(1, 100):
        adj_returns = txn.adjust_returns_for_slippage(returns, positions,
                                                      transactions, bps)
        avg_returns = ep.annual_return(adj_returns)
        avg_returns_given_slippage.loc[bps] = avg_returns

    fig.add_trace(
        go.Scatter(
            x=avg_returns_given_slippage.index,
            y=avg_returns_given_slippage.values,
            name='年收益率',
            opacity=1.0,
            line=dict(width=2),
        )
    )

    fig.update_layout(title_text='考虑到额外每元滑点后的平均年收益率')
    fig.update_xaxes(
        ticktext=np.arange(0, 100, 10),
        tickvals=np.arange(0, 100, 10),
    )
    fig.update_xaxes(title_text='每元滑点(bps)')
    fig.update_yaxes(title_text='平均年收益率', tickformat='%')
    return fig


def plot_capacity_sweep(returns, transactions, market_data,
                        bt_starting_capital,
                        min_pv=100000,
                        max_pv=300000000,
                        step_size=1000000,
                        fig=None):
    txn_daily_w_bar = capacity.daily_txns_with_bar_data(transactions,
                                                        market_data)

    captial_base_sweep = pd.Series()
    for start_pv in range(min_pv, max_pv, step_size):
        adj_ret = capacity.apply_slippage_penalty(returns,
                                                  txn_daily_w_bar,
                                                  start_pv,
                                                  bt_starting_capital)
        sharpe = ep.sharpe_ratio(adj_ret)
        if sharpe < -1:
            break
        captial_base_sweep.loc[start_pv] = sharpe
    captial_base_sweep.index = captial_base_sweep.index / MM_DISPLAY_UNIT

    if fig is None:
        fig = go.Figure()
    # captial_base_sweep.plot(ax=ax)
    fig.add_trace(
        go.Scatter(
            x=captial_base_sweep.index,
            y=captial_base_sweep.values,
            showlegend=False
        )
    )

    fig.update_xaxes(title_text='基础资本(百万)')
    fig.update_yaxes(title_text='Sharpe比率')
    fig.update_layout(title_text='基础资本绩效范围')
    return fig


def plot_daily_turnover_hist(transactions, positions, turnover_denom='AGB',
                             fig=None, **kwargs):
    """
    Plots a histogram of daily turnover rates.

    Parameters
    ----------
    transactions : pd.DataFrame
        Prices and amounts of executed trades. One row per trade.
         - See full explanation in tears.create_full_tear_sheet.
    positions : pd.DataFrame
        Daily net position values.
         - See full explanation in tears.create_full_tear_sheet.
    turnover_denom : str, optional
        Either AGB or portfolio_value, default AGB.
        - See full explanation in txn.get_turnover.
    fig : matplotlib.Axes, optional
        Axes upon which to plot.
    **kwargs, optional
        Passed to seaborn plotting function.

    Returns
    -------
    ax : matplotlib.Axes
        The axes that were plotted on.
    """

    if fig is None:
        fig = go.Figure()
    turnover = txn.get_turnover(positions, transactions, turnover_denom)
    # fig = ff.create_distplot([turnover.values], ['换手率'],
    #                          show_rug=False)
    fig = go.Figure(data=[go.Histogram(x=turnover.values)])
    fig.update_yaxes(title_text='换手率')
    fig.update_layout(title_text='每日换手率分布', showlegend=False)
    return fig


def plot_daily_volume(returns, transactions, fig=None, **kwargs):
    """
    Plots trading volume per day vs. date.

    Also displays all-time daily average.

    Parameters
    ----------
    returns : pd.Series
        Daily returns of the strategy, noncumulative.
         - See full explanation in tears.create_full_tear_sheet.
    transactions : pd.DataFrame
        Prices and amounts of executed trades. One row per trade.
         - See full explanation in tears.create_full_tear_sheet.
    fig : matplotlib.Axes, optional
        Axes upon which to plot.
    **kwargs, optional
        Passed to plotting function.

    Returns
    -------
    ax : matplotlib.Axes
        The axes that were plotted on.
    """

    if fig is None:
        fig = go.Figure()
    daily_txn = txn.get_txn_vol(transactions)
    # daily_txn.txn_shares.plot(alpha=1.0, lw=0.5, ax=ax, **kwargs)
    fig.add_trace(
        go.Scatter(
            x=daily_txn.txn_shares.index,
            y=daily_txn.txn_shares.values,
            name='日成交量',
            opacity=1.0,
            line=dict(width=0.5)
        )
    )
    # ax.axhline(daily_txn.txn_shares.mean(), color='steelblue',
    #            linestyle='--', lw=3, alpha=1.0)
    fig.add_trace(
        go.Scatter(
            x=[daily_txn.txn_shares.index.min(), daily_txn.txn_shares.index.max()],
            y=[daily_txn.txn_shares.mean()]*2,
            name='均值',
            opacity=1.0,
            line=dict(width=3, color='steelblue', dash='dash')
        )
    )

    fig.update_xaxes(range=[returns.index[0], returns.index[-1]])
    fig.update_yaxes(title_text='股票交易量')
    fig.update_layout(title_text='每日成交', showlegend=False)
    _date_tickformat(fig)
    return fig


def plot_txn_time_hist(transactions, bin_minutes=5, tz='Asia/Shanghai',
                       fig=None, **kwargs):
    """
    Plots a histogram of transaction times, binning the times into
    buckets of a given duration.

    Parameters
    ----------
    transactions : pd.DataFrame
        Prices and amounts of executed trades. One row per trade.
         - See full explanation in tears.create_full_tear_sheet.
    bin_minutes : float, optional
        Sizes of the bins in minutes, defaults to 5 minutes.
    tz : str, optional
        Time zone to plot against. Note that if the specified
        zone does not apply daylight savings, the distribution
        may be partially offset.
    ax : matplotlib.Axes, optional
        Axes upon which to plot.
    **kwargs, optional
        Passed to plotting function.

    Returns
    -------
    ax : matplotlib.Axes
        The axes that were plotted on.
    """

    if fig is None:
        fig = go.Figure()

    txn_time = transactions.copy()

    txn_time.index = txn_time.index.tz_convert(pytz.timezone(tz))
    txn_time.index = txn_time.index.map(lambda x: x.hour * 60 + x.minute)
    txn_time['trade_value'] = (txn_time.amount * txn_time.price).abs()
    txn_time = txn_time.groupby(level=0).sum().reindex(index=range(570, 961))
    txn_time.index = (txn_time.index / bin_minutes).astype(int) * bin_minutes
    txn_time = txn_time.groupby(level=0).sum()

    txn_time['time_str'] = txn_time.index.map(lambda x:
                                              str(datetime.time(int(x / 60),
                                                                x % 60))[:-3])

    trade_value_sum = txn_time.trade_value.sum()
    txn_time.trade_value = txn_time.trade_value.fillna(0) / trade_value_sum
    # ax.bar(txn_time.index, txn_time.trade_value, width=bin_minutes, **kwargs)
    fig.add_trace(
        go.Bar(
            x=txn_time.index,
            y=txn_time.trade_value.values,
            width=bin_minutes,
        )
    )

    fig.update_xaxes(
        ticktext=txn_time.time_str[::int(30 / bin_minutes)],
        tickvals=txn_time.index[::int(30 / bin_minutes)],
    )
    fig.update_xaxes(range=[570, 960])
    fig.update_yaxes(title_text='比例')
    fig.update_layout(title_text='交易时间分布', yaxis_tickformat='%')
    return fig


def show_worst_drawdown_periods(returns, top=5):
    """
    Prints information about the worst drawdown periods.

    Prints peak dates, valley dates, recovery dates, and net
    drawdowns.

    Parameters
    ----------
    returns : pd.Series
        Daily returns of the strategy, noncumulative.
         - See full explanation in tears.create_full_tear_sheet.
    top : int, optional
        Amount of top drawdowns periods to plot (default 5).
    """

    drawdown_df = timeseries.gen_drawdown_table(returns, top=top)
    utils.print_table(
        drawdown_df.sort_values('净回撤百分比%', ascending=False),
        name='最大回撤',
        float_format='{0:.2f}'.format,
    )


def plot_monthly_returns_timeseries(returns, fig=None, **kwargs):
    """
    Plots monthly returns as a timeseries.

    Parameters
    ----------
    returns : pd.Series
        Daily returns of the strategy, noncumulative.
         - See full explanation in tears.create_full_tear_sheet.
    ax : matplotlib.Axes, optional
        Axes upon which to plot.
    **kwargs, optional
        Passed to seaborn plotting function.

    Returns
    -------
    ax : matplotlib.Axes
        The axes that were plotted on.
    """

    def cumulate_returns(x):
        return ep.cum_returns(x)[-1]

    if fig is None:
        fig = go.Figure()

    monthly_rets = returns.resample('M').apply(lambda x: cumulate_returns(x))
    monthly_rets = monthly_rets.to_period()

    fig.add_trace(
        go.Bar(
            y=monthly_rets.values,
            marker_color='steelblue',
        )
    )

    fig.update_layout(xaxis_tickangle=90)

    # only show x-labels on year boundary
    xticks_coord = []
    xticks_label = []
    count = 0
    for i in monthly_rets.index:
        if i.month == 1:
            xticks_label.append(str(i))
            xticks_coord.append(count)
            # plot yearly boundary line
            fig.add_shape(
                # Line Vertical
                dict(
                    type="line",
                    yref="paper",
                    opacity=0.3,
                    x0=count,
                    y0=0,
                    x1=count,
                    y1=0.01,
                    line=dict(
                        color='gray',
                        dash='dash',
                    )
                )
            )

        count += 1

    fig.add_shape(
        # 水平线
        dict(
            type="line",
            x0=0,
            y0=0,
            x1=len(monthly_rets),
            y1=0,
            line=dict(
                color='darkgray',
                dash='dash',
            )
        )
    )

    fig.update_xaxes(
        ticktext=xticks_label,
        tickvals=xticks_coord,
    )
    fig.update_yaxes(
        tickformat='%',
    )
    fig.update_layout(title_text='月度收益率时间序列')
    return fig


def plot_round_trip_lifetimes(round_trips, disp_amount=16, lsize=18, fig=None):
    """
    Plots timespans and directions of a sample of round trip trades.

    Parameters
    ----------
    round_trips : pd.DataFrame
        DataFrame with one row per round trip trade.
        - See full explanation in round_trips.extract_round_trips
    ax : matplotlib.Axes, optional
        Axes upon which to plot.

    Returns
    -------
    ax : matplotlib.Axes
        The axes that were plotted on.
    """

    if fig is None:
        fig = go.Figure()

    symbols_sample = round_trips.symbol.unique()
    np.random.seed(1)
    sample = np.random.choice(round_trips.symbol.unique(), replace=False,
                              size=min(disp_amount, len(symbols_sample)))
    sample_round_trips = round_trips[round_trips.symbol.isin(sample)]

    symbol_idx = pd.Series(np.arange(len(sample)), index=sample)

    only_show_first = []

    for symbol, sym_round_trips in sample_round_trips.groupby('symbol'):
        for _, row in sym_round_trips.iterrows():
            c = 'blue' if row.long else 'red'
            y_ix = symbol_idx[symbol] + 0.05
            fig.add_trace(
                go.Scatter(x=[row['open_dt'], row['close_dt']],
                           y=[y_ix, y_ix],
                           mode='lines',
                           showlegend=False if c in only_show_first else True,
                           name="多头" if row.long else '空头',
                           #    legendgroup="多头" if row.long else '空头',
                           line=dict(color=c, width=lsize)),
            )
            if c not in only_show_first:
                only_show_first.append(c)

    fig.update_xaxes(
        showgrid=False,
    )
    fig.update_yaxes(
        ticktext=[utils.format_asset(s) for s in sample],
        tickvals=list(range(disp_amount)),
        range=[-0.5, min(len(sample), disp_amount) - 0.5],
        showgrid=False,
    )

    fig.update_layout(
        legend=dict(
            x=0.01,
            y=0.05,
            yanchor="bottom",
            xanchor="left",
        )
    )
    _date_tickformat(fig)
    return fig


def show_profit_attribution(round_trips):
    """
    Prints the share of total PnL contributed by each
    traded name.

    Parameters
    ----------
    round_trips : pd.DataFrame
        DataFrame with one row per round trip trade.
        - See full explanation in round_trips.extract_round_trips
    ax : matplotlib.Axes, optional
        Axes upon which to plot.

    Returns
    -------
    ax : matplotlib.Axes
        The axes that were plotted on.
    """
    total_pnl = round_trips['pnl'].sum()
    pnl_attribution = round_trips.groupby('symbol')['pnl'].sum() / total_pnl
    pnl_attribution.name = ''

    pnl_attribution.index = pnl_attribution.index.map(utils.format_asset)
    utils.print_table(
        pnl_attribution.sort_values(
            inplace=False,
            ascending=False,
        ),
        name='分项获利能力(项目盈亏 / 盈亏总计)',
        float_format='{:.2%}'.format,
    )


def plot_prob_profit_trade(round_trips, fig=None, col=None):
    """
    Plots a probability distribution for the event of making
    a profitable trade.

    Parameters
    ----------
    round_trips : pd.DataFrame
        DataFrame with one row per round trip trade.
        - See full explanation in round_trips.extract_round_trips
    ax : matplotlib.Axes, optional
        Axes upon which to plot.

    Returns
    -------
    ax : matplotlib.Axes
        The axes that were plotted on.
    """

    x = np.linspace(0, 1., 500)

    round_trips['profitable'] = round_trips.pnl > 0

    dist = sp.stats.beta(round_trips.profitable.sum(),
                         (~round_trips.profitable).sum())
    y = dist.pdf(x)
    lower_perc = dist.ppf(.025)
    upper_perc = dist.ppf(.975)

    lower_plot = dist.ppf(.001)
    upper_plot = dist.ppf(.999)

    if fig is None:
        fig = go.Figure()

    # ax.plot(x, y)
    fig.add_trace(
        go.Scatter(x=x,
                   y=y,
                   showlegend=False,
                   mode='lines'),
        row=1 if col else None, col=col,
    )

    fig.add_shape(
        # Line Vertical
        dict(
            type="line",
            x0=lower_perc,
            y0=0,
            x1=lower_perc,
            y1=y.max(),
            line=dict(
                color='gray',
            )
        ),
        row=1 if col else None, col=col
    )
    m = (lower_perc+upper_perc) / 2
    fig.add_shape(
        # Line Vertical
        dict(
            type="line",
            x0=m,
            y0=0,
            x1=m,
            y1=y.max(),
            line=dict(
                color='blue',
                dash='dash',
            )
        ),
        row=1 if col else None, col=col
    )
    fig.add_shape(
        # Line Vertical
        dict(
            type="line",
            x0=upper_perc,
            y0=0,
            x1=upper_perc,
            y1=y.max(),
            line=dict(
                color='gray',
            )
        ),
        row=1 if col else None, col=col,
    )

    fig.update_xaxes(
        title_text='盈利决策的可能性',
        range=[lower_plot, upper_plot],
        row=1 if col else None, col=col,
    )
    fig.update_yaxes(
        title_text='想法',
        range=[0, y.max() + 1.0],
        row=1 if col else None, col=col,
    )
    return fig


def plot_cones(name, bounds, oos_returns, num_samples=1000, fig=None,
               cone_std=(1., 1.5, 2.), random_seed=None, num_strikes=3):
    """
    Plots the upper and lower bounds of an n standard deviation
    cone of forecasted cumulative returns. Redraws a new cone when
    cumulative returns fall outside of last cone drawn.

    Parameters
    ----------
    name : str
        Account name to be used as figure title.
    bounds : pandas.core.frame.DataFrame
        Contains upper and lower cone boundaries. Column names are
        strings corresponding to the number of standard devations
        above (positive) or below (negative) the projected mean
        cumulative returns.
    oos_returns : pandas.core.frame.DataFrame
        Non-cumulative out-of-sample returns.
    num_samples : int
        Number of samples to draw from the in-sample daily returns.
        Each sample will be an array with length num_days.
        A higher number of samples will generate a more accurate
        bootstrap cone.
    ax : matplotlib.Axes, optional
        Axes upon which to plot.
    cone_std : list of int/float
        Number of standard devations to use in the boundaries of
        the cone. If multiple values are passed, cone bounds will
        be generated for each value.
    random_seed : int
        Seed for the pseudorandom number generator used by the pandas
        sample method.
    num_strikes : int
        Upper limit for number of cones drawn. Can be anything from 0 to 3.

    Returns
    -------
    Returns are either an ax or fig option, but not both. If a
    matplotlib.Axes instance is passed in as ax, then it will be modified
    and returned. This allows for users to plot interactively in jupyter
    notebook. When no ax object is passed in, a matplotlib.figure instance
    is generated and returned. This figure can then be used to save
    the plot as an image without viewing it.

    ax : matplotlib.Axes
        The axes that were plotted on.
    fig : matplotlib.figure
        The figure instance which contains all the plot elements.
    """
    if fig is None:
        fig = go.Figure()

    returns = ep.cum_returns(oos_returns, starting_value=1.)
    bounds_tmp = bounds.copy()
    returns_tmp = returns.copy()
    cone_start = returns.index[0]
    colors = ["green", "orange", "orangered", "darkred"]

    for c in range(num_strikes + 1):
        if c > 0:
            tmp = returns.loc[cone_start:]
            bounds_tmp = bounds_tmp.iloc[0:len(tmp)]
            bounds_tmp = bounds_tmp.set_index(tmp.index)
            crossing = (tmp < bounds_tmp[float(-2.)].iloc[:len(tmp)])
            if crossing.sum() <= 0:
                break
            cone_start = crossing.loc[crossing].index[0]
            returns_tmp = returns.loc[cone_start:]
            bounds_tmp = (bounds - (1 - returns.loc[cone_start]))
        for std in cone_std:
            x = returns_tmp.index
            y1 = bounds_tmp[float(std)].iloc[:len(returns_tmp)]
            y2 = bounds_tmp[float(-std)].iloc[:len(returns_tmp)]
            # axes.fill_between(x, y1, y2, color=colors[c], alpha=0.5)
            fig.add_shape(
                # filled Rectangle
                type="rect",
                opacity=0.5,
                x0=x,
                y0=y1,
                x1=x,
                y1=y2,
                fillcolor=colors[c],
            )

    # Plot returns line graph
    label = '累积收益率 = {:.2f}%'.format((returns.iloc[-1] - 1) * 100)
    # axes.plot(returns.index, returns.values, color='black', lw=3.,
    #           label=label)
    fig.add_trace(
        go.Scatter(x=returns.index,
                   y=returns.values,
                   name=label,
                   line=dict(color='black', width=3),
                   mode='lines'),
    )
    if name is not None:
        # axes.set_title(name)
        fig.update_layout(title_text=name)

    # axes.axhline(1, color='black', alpha=0.2)
    fig.add_shape(
        dict(
            type="line",
            opacity=0.2,
            x0=returns.index.min(),
            y0=1,
            x1=returns.index.max(),
            y1=1,
            line=dict(
                color='black',
            )
        )
    )
    # axes.legend(frameon=True, framealpha=0.5)
    fig.update_layout(
        legend=dict(
            x=0.01,
            y=0.50,
            yanchor="middle",
            xanchor="left",
        )
    )
    return fig
