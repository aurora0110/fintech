from __future__ import annotations

import json

import pandas as pd
import streamlit as st

try:
    from .pricing import enrich_positions, latest_snapshot, next_open_after_signal, trading_days_between
    from .storage import (
        clear_demo_records,
        get_signal_row,
        init_db,
        insert_position,
        list_basket_daily_snapshot,
        list_basket_members,
        list_operation_logs,
        list_positions,
        list_signals,
        list_trades,
        close_position,
    )
    from .sync_signals import sync_all_signals
except ImportError:  # pragma: no cover - for direct script execution
    from pricing import enrich_positions, latest_snapshot, next_open_after_signal, trading_days_between
    from storage import (
        clear_demo_records,
        get_signal_row,
        init_db,
        insert_position,
        list_basket_daily_snapshot,
        list_basket_members,
        list_operation_logs,
        list_positions,
        list_signals,
        list_trades,
        close_position,
    )
    from sync_signals import sync_all_signals


st.set_page_config(page_title="BRICK 独立交易看板", layout="wide", initial_sidebar_state="expanded")
st.title("BRICK 独立交易看板")
st.caption("仅覆盖 brick_list 与 brick_case_rank_lgbm_top20_list；独立账本，不回写 holding.yaml。")


@st.cache_data(ttl=120)
def _load_all_data(sync_token: int):
    positions = list_positions()
    trades = list_trades()
    signals = list_signals()
    baskets = list_basket_daily_snapshot()
    return positions, trades, signals, baskets


def _format_pct(value):
    if pd.isna(value):
        return "—"
    return f"{value * 100:.2f}%"


def _format_price(value):
    if pd.isna(value):
        return "—"
    return f"{value:.2f}"


def _format_dt(value):
    if value in (None, "", pd.NaT) or pd.isna(value):
        return "—"
    return str(value)


def _position_status(row: pd.Series) -> str:
    if pd.isna(row.get("latest_close")):
        return "价格缺失"
    if bool(row.get("warning_over_3d")):
        return "超期提醒"
    if pd.notna(row.get("current_return")) and float(row["current_return"]) < 0:
        return "浮亏"
    return "正常"


def _build_buy_reason_snapshot(row: dict) -> str:
    return json.dumps(
        {
            "signal_date": row.get("signal_date"),
            "strategy": row.get("strategy"),
            "source_list": row.get("source_list", ""),
            "raw_score": row.get("raw_score"),
            "raw_reason": row.get("raw_reason", ""),
            "signal_close": row.get("signal_close"),
            "signal_low": row.get("signal_low"),
            "stop_loss_price": row.get("stop_loss_price"),
        },
        ensure_ascii=False,
    )


def _filter_text_mask(df: pd.DataFrame, query: str, cols: list[str]) -> pd.Series:
    if not query:
        return pd.Series([True] * len(df), index=df.index)
    query = query.strip().lower()
    return df.apply(
        lambda r: any(query in str(r[col]).lower() for col in cols if col in df.columns),
        axis=1,
    )


def _apply_global_filters(positions_df: pd.DataFrame, trades_df: pd.DataFrame, signals_df: pd.DataFrame):
    st.sidebar.header("全局筛选")
    query = st.sidebar.text_input("按代码或名称搜索", value="")
    selected_strategies = st.sidebar.multiselect(
        "策略过滤",
        options=["brick", "brick_case_rank"],
        default=["brick", "brick_case_rank"],
    )
    signal_start = pd.to_datetime(signals_df["signal_date"]).min() if not signals_df.empty else None
    signal_end = pd.to_datetime(signals_df["signal_date"]).max() if not signals_df.empty else None
    if signal_start is not None and signal_end is not None:
        date_range = st.sidebar.date_input("信号日期范围", value=(signal_start.date(), signal_end.date()))
    else:
        date_range = None

    def _filter(df: pd.DataFrame, name_cols: list[str], date_col: str | None = None) -> pd.DataFrame:
        if df.empty:
            return df.copy()
        out = df[df["strategy"].isin(selected_strategies)].copy() if "strategy" in df.columns else df.copy()
        out = out[_filter_text_mask(out, query, name_cols)].copy()
        if date_range and date_col and len(date_range) == 2 and date_col in out.columns:
            start = pd.Timestamp(date_range[0]).strftime("%Y-%m-%d")
            end = pd.Timestamp(date_range[1]).strftime("%Y-%m-%d")
            out = out[(out[date_col] >= start) & (out[date_col] <= end)].copy()
        return out

    return (
        _filter(positions_df, ["code", "display_name", "name"], "signal_date"),
        _filter(trades_df, ["code", "name"], "signal_date"),
        _filter(signals_df, ["code", "name"], "signal_date"),
    )


def _sorted_positions(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    sort_choice = st.selectbox(
        "当前持仓排序",
        options=[
            "持有天数(高到低)",
            "个股收益率(高到低)",
            "个股收益率(低到高)",
            "买入日期(新到旧)",
            "买入日期(旧到新)",
        ],
        index=0,
        key="position_sort_choice",
    )
    display = df.copy()
    if sort_choice == "持有天数(高到低)":
        display = display.sort_values(["holding_days", "current_return"], ascending=[False, False])
    elif sort_choice == "个股收益率(高到低)":
        display = display.sort_values(["current_return", "holding_days"], ascending=[False, False], na_position="last")
    elif sort_choice == "个股收益率(低到高)":
        display = display.sort_values(["current_return", "holding_days"], ascending=[True, False], na_position="last")
    elif sort_choice == "买入日期(新到旧)":
        display = display.sort_values(["entry_date", "code"], ascending=[False, True])
    else:
        display = display.sort_values(["entry_date", "code"], ascending=[True, True])
    return display


def _render_metrics(positions_df: pd.DataFrame, trades_df: pd.DataFrame) -> None:
    realized_pnl = float(trades_df["pnl_amount"].sum()) if not trades_df.empty else 0.0
    unrealized_pnl = float(positions_df["unrealized_pnl_amount"].sum()) if not positions_df.empty else 0.0
    total_cost = (
        float((trades_df["entry_price"] * trades_df["quantity"]).sum()) if not trades_df.empty else 0.0
    ) + (
        float((positions_df["entry_price"] * positions_df["quantity"]).sum()) if not positions_df.empty else 0.0
    )
    total_return_pct = (realized_pnl + unrealized_pnl) / total_cost if total_cost > 0 else 0.0
    win_rate = float((trades_df["return_pct"] > 0).mean()) if not trades_df.empty else 0.0
    avg_trade_return = float(trades_df["return_pct"].mean()) if not trades_df.empty else 0.0

    cols = st.columns(6)
    cols[0].metric("当前持仓数", f"{len(positions_df)}")
    cols[1].metric("历史交易数", f"{len(trades_df)}")
    cols[2].metric("策略累计收益额", f"{realized_pnl + unrealized_pnl:,.2f}")
    cols[3].metric("策略累计收益率", _format_pct(total_return_pct))
    cols[4].metric("胜率", _format_pct(win_rate))
    cols[5].metric("平均单笔收益", _format_pct(avg_trade_return))


def _render_reminders(positions_df: pd.DataFrame, signals_df: pd.DataFrame) -> None:
    st.subheader("今日提醒")
    if positions_df.empty and signals_df.empty:
        st.info("暂无提醒。")
        return

    reminders = []
    if not positions_df.empty:
        for _, row in positions_df.iterrows():
            name = row["display_name"]
            if bool(row["warning_over_3d"]):
                reminders.append({"类型": "超期提醒", "股票": f'{row["code"]} {name}', "说明": f'已持有 {row["holding_days"]} 天，超过 3 天'})
            latest_close = row.get("latest_close")
            stop_price = row.get("stop_loss_price")
            atr_tp_price = row.get("atr_tp_price")
            fixed_tp_3_price = row.get("fixed_tp_3_price")
            fixed_tp_8_price = row.get("fixed_tp_8_price")
            if pd.notna(latest_close) and pd.notna(stop_price) and latest_close <= stop_price * 1.02:
                reminders.append({"类型": "接近止损", "股票": f'{row["code"]} {name}', "说明": f'最新价 {latest_close:.2f} 接近止损价 {stop_price:.2f}'})
            if pd.notna(latest_close) and pd.notna(atr_tp_price) and latest_close >= atr_tp_price * 0.98:
                reminders.append({"类型": "接近ATR止盈", "股票": f'{row["code"]} {name}', "说明": f'最新价 {latest_close:.2f} 接近 ATR 止盈价 {atr_tp_price:.2f}'})
            if pd.notna(latest_close) and pd.notna(fixed_tp_3_price) and latest_close >= fixed_tp_3_price * 0.98:
                reminders.append({"类型": "接近固定3%止盈", "股票": f'{row["code"]} {name}', "说明": f'最新价 {latest_close:.2f} 接近 3% 止盈价 {fixed_tp_3_price:.2f}'})
            if pd.notna(latest_close) and pd.notna(fixed_tp_8_price) and latest_close >= fixed_tp_8_price * 0.98:
                reminders.append({"类型": "接近固定8%止盈", "股票": f'{row["code"]} {name}', "说明": f'最新价 {latest_close:.2f} 接近 8% 止盈价 {fixed_tp_8_price:.2f}'})

    if not signals_df.empty:
        open_keys = set(zip(positions_df["signal_date"], positions_df["code"], positions_df["strategy"])) if not positions_df.empty else set()
        latest_signal_date = signals_df["signal_date"].max()
        pending = signals_df[
            (signals_df["signal_date"] == latest_signal_date)
            & ~signals_df.apply(lambda r: (r["signal_date"], r["code"], r["strategy"]) in open_keys, axis=1)
        ]
        if not pending.empty:
            reminders.append(
                {"类型": "今日新信号未处理", "股票": "多只", "说明": f'{latest_signal_date} 还有 {len(pending)} 条新信号未建仓'}
            )

    if not reminders:
        st.success("当前没有需要处理的提醒。")
        return
    st.dataframe(pd.DataFrame(reminders), use_container_width=True, hide_index=True)


def _styled_positions(df: pd.DataFrame):
    display = _sorted_positions(df)
    display["个股累计收益率"] = display["current_return"]
    display["止损价"] = display["stop_loss_price"]
    display["ATR止盈价"] = display["atr_tp_price"]
    display["固定3%止盈价"] = display["fixed_tp_3_price"]
    display["固定8%止盈价"] = display["fixed_tp_8_price"]
    display["最新价"] = display["latest_close"]
    display["股票名称"] = display["display_name"]
    display["状态"] = display.apply(_position_status, axis=1)
    show_cols = [
        "code",
        "股票名称",
        "strategy",
        "状态",
        "entry_date",
        "entry_price",
        "quantity",
        "最新价",
        "止损价",
        "ATR止盈价",
        "固定3%止盈价",
        "固定8%止盈价",
        "个股累计收益率",
        "holding_days",
        "manual_note",
    ]
    rename_map = {
        "code": "股票代码",
        "strategy": "策略",
        "状态": "状态",
        "entry_date": "买入日",
        "entry_price": "买入价",
        "quantity": "股数",
        "holding_days": "持有天数",
        "manual_note": "备注",
    }
    styled = (
        display[show_cols]
        .rename(columns=rename_map)
        .style.format(
            {
                "买入价": _format_price,
                "最新价": _format_price,
                "止损价": _format_price,
                "ATR止盈价": _format_price,
                "固定3%止盈价": _format_price,
                "固定8%止盈价": _format_price,
                "个股累计收益率": _format_pct,
            }
        )
        .map(
            lambda value: "color: #b91c1c; font-weight: 700;"
            if value == "超期提醒"
            else ("color: #b45309; font-weight: 700;" if value == "浮亏" else ""),
            subset=["状态"],
        )
        .map(
            lambda value: "color: #15803d; font-weight: 700;"
            if pd.notna(value) and value > 0
            else ("color: #b91c1c; font-weight: 700;" if pd.notna(value) and value < 0 else ""),
            subset=["个股累计收益率"],
        )
        .apply(
            lambda row: [
                "background-color: #ffe5e5; color: #9b1c1c;" if row["持有天数"] > 3 else ""
                for _ in row
            ],
            axis=1,
        )
    )
    st.dataframe(styled, use_container_width=True, hide_index=True)


def _render_equity_curve(positions_df: pd.DataFrame, trades_df: pd.DataFrame) -> None:
    st.subheader("账户净值/盈亏曲线")
    if trades_df.empty and positions_df.empty:
        st.info("暂无可绘制的权益数据。")
        return
    series = []
    if not trades_df.empty:
        trade_part = trades_df.copy()
        trade_part["curve_date"] = pd.to_datetime(trade_part["exit_date"])
        daily = trade_part.groupby("curve_date", as_index=False)["pnl_amount"].sum().sort_values("curve_date")
        daily["cum_pnl"] = daily["pnl_amount"].cumsum()
        series.append(daily[["curve_date", "cum_pnl"]])
    if not positions_df.empty:
        latest_date = positions_df["latest_price_date"].dropna().max()
        if latest_date:
            current_total = float(positions_df["unrealized_pnl_amount"].sum())
            realized_total = float(trades_df["pnl_amount"].sum()) if not trades_df.empty else 0.0
            series.append(pd.DataFrame({"curve_date": [pd.to_datetime(latest_date)], "cum_pnl": [realized_total + current_total]}))
    if not series:
        st.info("暂无可绘制的权益数据。")
        return
    curve = pd.concat(series, ignore_index=True).sort_values("curve_date").drop_duplicates(subset=["curve_date"], keep="last")
    curve["净收益额"] = curve["cum_pnl"]
    st.line_chart(curve.set_index("curve_date")["净收益额"], use_container_width=True)
    curve_table = curve[["curve_date", "净收益额"]].rename(columns={"curve_date": "日期"})
    st.dataframe(curve_table, use_container_width=True, hide_index=True)


def _render_recent_stats(trades_df: pd.DataFrame) -> None:
    st.subheader("最近阶段统计")
    if trades_df.empty:
        st.info("暂无历史交易统计。")
        return
    trades = trades_df.copy()
    trades["exit_date"] = pd.to_datetime(trades["exit_date"])
    latest_exit = trades["exit_date"].max()
    recent_trade_windows = [10, 20]
    recent_day_windows = [30, 60, 120]
    rows = []
    for n in recent_trade_windows:
        subset = trades.head(n)
        if subset.empty:
            continue
        rows.append(
            {
                "统计类型": f"最近{n}笔",
                "交易数": len(subset),
                "平均收益率": subset["return_pct"].mean(),
                "胜率": (subset["return_pct"] > 0).mean(),
                "总盈亏": subset["pnl_amount"].sum(),
            }
        )
    for n in recent_day_windows:
        start = latest_exit - pd.Timedelta(days=n)
        subset = trades[trades["exit_date"] >= start]
        if subset.empty:
            continue
        rows.append(
            {
                "统计类型": f"最近{n}天",
                "交易数": len(subset),
                "平均收益率": subset["return_pct"].mean(),
                "胜率": (subset["return_pct"] > 0).mean(),
                "总盈亏": subset["pnl_amount"].sum(),
            }
        )
    stats_df = pd.DataFrame(rows)
    st.dataframe(
        stats_df.style.format({"平均收益率": _format_pct, "胜率": _format_pct, "总盈亏": "{:,.2f}"}),
        use_container_width=True,
        hide_index=True,
    )


def _render_period_summary(trades_df: pd.DataFrame) -> None:
    st.subheader("月度 / 季度汇总")
    if trades_df.empty:
        st.info("暂无历史交易汇总。")
        return
    trades = trades_df.copy()
    trades["exit_date"] = pd.to_datetime(trades["exit_date"])
    trades["month"] = trades["exit_date"].dt.to_period("M").astype(str)
    trades["quarter"] = trades["exit_date"].dt.to_period("Q").astype(str)
    month_df = trades.groupby("month", as_index=False).agg(
        交易数=("id", "count"),
        胜率=("return_pct", lambda s: (s > 0).mean()),
        平均收益率=("return_pct", "mean"),
        总盈亏=("pnl_amount", "sum"),
    ).sort_values("month", ascending=False)
    quarter_df = trades.groupby("quarter", as_index=False).agg(
        交易数=("id", "count"),
        胜率=("return_pct", lambda s: (s > 0).mean()),
        平均收益率=("return_pct", "mean"),
        总盈亏=("pnl_amount", "sum"),
    ).sort_values("quarter", ascending=False)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**月度汇总**")
        st.dataframe(month_df.style.format({"胜率": _format_pct, "平均收益率": _format_pct, "总盈亏": "{:,.2f}"}), use_container_width=True, hide_index=True)
    with c2:
        st.markdown("**季度汇总**")
        st.dataframe(quarter_df.style.format({"胜率": _format_pct, "平均收益率": _format_pct, "总盈亏": "{:,.2f}"}), use_container_width=True, hide_index=True)


def _render_trades(trades_df: pd.DataFrame) -> None:
    if trades_df.empty:
        st.info("暂无历史交易。")
        return
    display = trades_df.copy()
    display["股票名称"] = display.apply(lambda r: r["name"] or r["code"], axis=1)
    display["收益率"] = display["return_pct"]
    show_cols = [
        "code",
        "股票名称",
        "strategy",
        "entry_date",
        "entry_price",
        "exit_date",
        "exit_price",
        "quantity",
        "exit_reason_category",
        "收益率",
        "pnl_amount",
        "holding_days",
        "exit_reason",
        "manual_note",
    ]
    rename_map = {
        "code": "股票代码",
        "strategy": "策略",
        "entry_date": "买入日期",
        "entry_price": "买入价",
        "exit_date": "卖出日期",
        "exit_price": "卖出价",
        "quantity": "股数",
        "exit_reason_category": "卖出分类",
        "pnl_amount": "盈亏金额",
        "holding_days": "持有天数",
        "exit_reason": "卖出原因",
        "manual_note": "备注",
    }
    st.dataframe(
        display[show_cols].rename(columns=rename_map).style.format(
            {
                "买入价": _format_price,
                "卖出价": _format_price,
                "收益率": _format_pct,
                "盈亏金额": "{:,.2f}",
            }
        ).map(
            lambda value: "color: #15803d; font-weight: 700;"
            if pd.notna(value) and value > 0
            else ("color: #b91c1c; font-weight: 700;" if pd.notna(value) and value < 0 else ""),
            subset=["收益率", "盈亏金额"],
        ),
        use_container_width=True,
        hide_index=True,
    )


def _render_lifecycle(signals_df: pd.DataFrame, positions_df: pd.DataFrame, trades_df: pd.DataFrame) -> None:
    st.subheader("持仓生命周期 / 操作日志")
    all_codes = sorted(set(signals_df["code"].astype(str)).union(set(positions_df["code"].astype(str))).union(set(trades_df["code"].astype(str))))
    if not all_codes:
        st.info("暂无可展示的生命周期。")
        return
    selected_code = st.selectbox("选择股票查看生命周期", all_codes, key="lifecycle_code")
    logs_df = list_operation_logs(selected_code)
    signal_rows = signals_df[signals_df["code"] == selected_code].copy()
    position_rows = positions_df[positions_df["code"] == selected_code].copy()
    trade_rows = trades_df[trades_df["code"] == selected_code].copy()
    events = []
    for _, row in signal_rows.iterrows():
        events.append(
            {
                "时间": row["signal_date"],
                "事件": "signal",
                "说明": f'{row["strategy"]} | 分数 {row["raw_score"] if pd.notna(row["raw_score"]) else "—"} | {row["raw_reason"]}',
            }
        )
    for _, row in position_rows.iterrows():
        events.append(
            {
                "时间": row["entry_date"],
                "事件": "open_position",
                "说明": f'建仓 {row["strategy"]} | 买入价 {row["entry_price"]:.2f} | 股数 {int(row["quantity"])}',
            }
        )
    for _, row in trade_rows.iterrows():
        events.append(
            {
                "时间": row["exit_date"],
                "事件": "close_position",
                "说明": f'平仓 {row["strategy"]} | 卖出价 {row["exit_price"]:.2f} | 收益 {_format_pct(row["return_pct"])} | {row["exit_reason_category"]}',
            }
        )
    for _, row in logs_df.iterrows():
        payload = row["payload_json"]
        events.append(
            {
                "时间": row["event_date"] or row["created_at"],
                "事件": row["event_type"],
                "说明": payload,
            }
        )
    event_df = pd.DataFrame(events)
    if event_df.empty:
        st.info("该股票暂无生命周期记录。")
        return
    event_df = event_df.sort_values("时间", ascending=False)
    st.dataframe(event_df, use_container_width=True, hide_index=True)


def _render_baskets(baskets_df: pd.DataFrame) -> None:
    st.subheader("最近 5 个交易日选股篮子表现")
    if baskets_df.empty:
        st.info("暂无篮子快照。请先同步信号。")
        return
    strategy_choice = st.selectbox(
        "篮子策略范围",
        options=["All", "brick", "brick_case_rank"],
        index=0,
    )
    if strategy_choice == "All":
        filtered = baskets_df.copy()
        grouped = (
            filtered.groupby("signal_date", as_index=False)
            .agg(
                stock_count=("stock_count", "sum"),
                avg_return_to_latest_close=("avg_return_to_latest_close", "mean"),
                latest_price_date=("latest_price_date", "max"),
            )
            .sort_values("signal_date", ascending=False)
            .head(5)
        )
        display_rows = grouped.to_dict("records")
        member_strategy_lookup = None
    else:
        grouped = (
            baskets_df[baskets_df["strategy"] == strategy_choice]
            .sort_values("signal_date", ascending=False)
            .head(5)
        )
        display_rows = grouped.to_dict("records")
        member_strategy_lookup = strategy_choice

    for row in display_rows:
        signal_date = row["signal_date"]
        stock_count = int(row["stock_count"])
        avg_ret = row["avg_return_to_latest_close"]
        latest_date = row["latest_price_date"]
        with st.expander(
            f"{signal_date} | 股票数 {stock_count} | 等权累计收益 {_format_pct(avg_ret)} | 最新价日期 {latest_date or '—'}"
        ):
            if strategy_choice == "All":
                parts = []
                for part_strategy in ("brick", "brick_case_rank"):
                    members = list_basket_members(signal_date, part_strategy)
                    if not members.empty:
                        parts.append(members)
                members_df = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
            else:
                members_df = list_basket_members(signal_date, member_strategy_lookup)
            if members_df.empty:
                st.info("该日没有可展开的单票明细。")
                continue
            st.dataframe(
                members_df.rename(
                    columns={
                        "code": "股票代码",
                        "name": "股票名称",
                        "entry_date": "买入日期",
                        "entry_price": "买入价",
                        "latest_close": "最新收盘价",
                        "latest_price_date": "最新价格日期",
                        "return_to_latest_close": "累计收益率",
                        "strategy": "策略",
                    }
                ).style.format(
                    {"买入价": _format_price, "最新收盘价": _format_price, "累计收益率": _format_pct}
                ),
                use_container_width=True,
                hide_index=True,
            )


def _render_signals(signals_df: pd.DataFrame) -> None:
    st.subheader("最新策略信号")
    if signals_df.empty:
        st.info("暂无可展示的 BRICK 信号。")
        return
    strategy_choice = st.selectbox(
        "信号策略范围",
        options=["All", "brick", "brick_case_rank"],
        index=0,
        key="signal_strategy_choice",
    )
    display = signals_df.copy()
    if strategy_choice != "All":
        display = display[display["strategy"] == strategy_choice].copy()
    display["股票名称"] = display.apply(lambda r: r["name"] or r["code"], axis=1)
    display["raw_reason"] = display["raw_reason"].fillna("")
    display = display.sort_values(["signal_date", "raw_score", "code"], ascending=[False, False, True])
    show_cols = [
        "signal_date",
        "strategy",
        "code",
        "股票名称",
        "signal_close",
        "stop_loss_price",
        "raw_score",
        "raw_reason",
    ]
    rename_map = {
        "signal_date": "信号日期",
        "strategy": "策略",
        "code": "股票代码",
        "signal_close": "信号收盘价",
        "stop_loss_price": "止损价",
        "raw_score": "原始分数",
        "raw_reason": "信号原因",
    }
    st.dataframe(
        display[show_cols]
        .rename(columns=rename_map)
        .head(200)
        .style.format(
            {
                "信号收盘价": _format_price,
                "止损价": _format_price,
                "原始分数": _format_price,
            }
        ),
        use_container_width=True,
        hide_index=True,
    )


def _render_position_forms(signals_df: pd.DataFrame, positions_df: pd.DataFrame) -> None:
    st.subheader("手动记账")
    open_keys = set(
        zip(
            positions_df["signal_date"].astype(str),
            positions_df["code"].astype(str),
            positions_df["strategy"].astype(str),
        )
    ) if not positions_df.empty else set()
    signal_options_df = signals_df[
        ~signals_df.apply(lambda r: (str(r["signal_date"]), str(r["code"]), str(r["strategy"])) in open_keys, axis=1)
    ].head(300)

    with st.expander("从信号创建持仓", expanded=False):
        if signal_options_df.empty:
            st.info("没有可用于建仓的信号。")
        else:
            option_map = {
                f'{row["signal_date"]} | {row["strategy"]} | {row["code"]} | {row["name"] or row["code"]}': (
                    row["signal_date"],
                    row["code"],
                    row["strategy"],
                )
                for _, row in signal_options_df.iterrows()
            }
            with st.form("create_from_signal"):
                selected_label = st.selectbox("选择信号", list(option_map.keys()))
                entry_date = st.date_input("买入日期")
                entry_price = st.number_input("买入价", min_value=0.0, step=0.01, format="%.2f")
                quantity = st.number_input("股数", min_value=1, step=100, value=100)
                tags = st.text_input("标签")
                manual_note = st.text_input("备注")
                submitted = st.form_submit_button("创建持仓")
            if submitted:
                signal_date, code, strategy = option_map[selected_label]
                row = get_signal_row(signal_date, code, strategy)
                if row is None:
                    st.error("未找到对应信号。")
                else:
                    insert_position(
                        code=code,
                        name=row["name"] or code,
                        strategy=strategy,
                        signal_date=signal_date,
                        entry_date=pd.Timestamp(entry_date).strftime("%Y-%m-%d"),
                        entry_price=float(entry_price),
                        entry_signal_low=float(row["signal_low"] or row["stop_loss_price"]),
                        quantity=int(quantity),
                        source_list=row["source_list"] or "",
                        raw_score=row["raw_score"],
                        raw_reason=row["raw_reason"] or "",
                        buy_reason_snapshot=_build_buy_reason_snapshot(dict(row)),
                        tags=tags,
                        manual_note=manual_note,
                    )
                    st.success("持仓已创建。")
                    st.rerun()

    with st.expander("批量从信号建仓", expanded=False):
        if signal_options_df.empty:
            st.info("没有可批量建仓的信号。")
        else:
            option_map = {
                f'{row["signal_date"]} | {row["strategy"]} | {row["code"]} | {row["name"] or row["code"]}': (
                    row["signal_date"],
                    row["code"],
                    row["strategy"],
                )
                for _, row in signal_options_df.head(100).iterrows()
            }
            with st.form("batch_create_from_signal"):
                selected_labels = st.multiselect("选择多条信号", list(option_map.keys()))
                quantity = st.number_input("统一股数", min_value=1, step=100, value=100, key="batch_quantity")
                tags = st.text_input("统一标签", key="batch_tags")
                manual_note = st.text_input("统一备注", key="batch_manual_note")
                submitted = st.form_submit_button("按建议次日开盘价批量建仓")
            if submitted:
                created = 0
                for label in selected_labels:
                    signal_date, code, strategy = option_map[label]
                    row = get_signal_row(signal_date, code, strategy)
                    if row is None:
                        continue
                    next_date, next_open = next_open_after_signal(code, signal_date)
                    if next_date is None or next_open is None:
                        continue
                    insert_position(
                        code=code,
                        name=row["name"] or code,
                        strategy=strategy,
                        signal_date=signal_date,
                        entry_date=next_date,
                        entry_price=float(next_open),
                        entry_signal_low=float(row["signal_low"] or row["stop_loss_price"]),
                        quantity=int(quantity),
                        source_list=row["source_list"] or "",
                        raw_score=row["raw_score"],
                        raw_reason=row["raw_reason"] or "",
                        buy_reason_snapshot=_build_buy_reason_snapshot(dict(row)),
                        tags=tags,
                        manual_note=manual_note or "批量建仓",
                    )
                    created += 1
                st.success(f"批量建仓完成：{created} 笔。")
                st.rerun()

    with st.expander("手动新增持仓", expanded=False):
        with st.form("manual_create_position"):
            code = st.text_input("股票代码")
            name = st.text_input("股票名称")
            strategy = st.selectbox("策略", ["brick", "brick_case_rank"])
            signal_date = st.date_input("信号日期")
            entry_date = st.date_input("买入日期", key="manual_entry_date")
            entry_price = st.number_input("买入价", min_value=0.0, step=0.01, format="%.2f", key="manual_entry_price")
            quantity = st.number_input("股数", min_value=1, step=100, value=100, key="manual_quantity")
            entry_signal_low = st.number_input("买入K线最低价", min_value=0.0, step=0.01, format="%.2f")
            tags = st.text_input("标签", key="manual_tags")
            buy_reason_snapshot = st.text_area("买入原因快照", key="manual_reason_snapshot")
            manual_note = st.text_input("备注", key="manual_position_note")
            submitted = st.form_submit_button("手动新增")
        if submitted:
            insert_position(
                code=str(code).zfill(6),
                name=name or latest_snapshot(str(code).zfill(6))["name"] or str(code).zfill(6),
                strategy=strategy,
                signal_date=pd.Timestamp(signal_date).strftime("%Y-%m-%d"),
                entry_date=pd.Timestamp(entry_date).strftime("%Y-%m-%d"),
                entry_price=float(entry_price),
                entry_signal_low=float(entry_signal_low),
                quantity=int(quantity),
                buy_reason_snapshot=buy_reason_snapshot,
                tags=tags,
                manual_note=manual_note,
            )
            st.success("持仓已新增。")
            st.rerun()

    with st.expander("平仓并写入历史交易", expanded=False):
        if positions_df.empty:
            st.info("当前没有持仓。")
        else:
            position_map = {
                f'{row["id"]} | {row["strategy"]} | {row["code"]} | {row["display_name"]}': row
                for _, row in positions_df.iterrows()
            }
            with st.form("close_position_form"):
                selected = st.selectbox("选择持仓", list(position_map.keys()))
                exit_date = st.date_input("卖出日期")
                exit_price = st.number_input("卖出价", min_value=0.0, step=0.01, format="%.2f")
                exit_reason_category = st.selectbox(
                    "卖出分类",
                    ["止损", "ATR止盈", "固定止盈3%", "固定止盈8%", "超3天卖出", "手动卖出", "其他"],
                )
                exit_reason = st.text_input("卖出原因说明")
                manual_note = st.text_input("备注", key="close_manual_note")
                submitted = st.form_submit_button("确认平仓")
            if submitted:
                row = position_map[selected]
                exit_date_str = pd.Timestamp(exit_date).strftime("%Y-%m-%d")
                holding_days = trading_days_between(row["code"], row["entry_date"], exit_date_str)
                return_pct = (float(exit_price) - float(row["entry_price"])) / float(row["entry_price"])
                pnl_amount = (float(exit_price) - float(row["entry_price"])) * float(row["quantity"])
                close_position(
                    position_id=int(row["id"]),
                    exit_date=exit_date_str,
                    exit_price=float(exit_price),
                    exit_reason=exit_reason or exit_reason_category,
                    exit_reason_category=exit_reason_category,
                    holding_days=holding_days,
                    return_pct=return_pct,
                    pnl_amount=pnl_amount,
                    manual_note=manual_note,
                )
                st.success("已平仓并写入历史交易。")
                st.rerun()

    with st.expander("批量按最新价平仓", expanded=False):
        if positions_df.empty:
            st.info("当前没有持仓。")
        else:
            batch_map = {
                f'{row["id"]} | {row["strategy"]} | {row["code"]} | {row["display_name"]} | 最新价 {_format_price(row["latest_close"])}': row
                for _, row in positions_df.iterrows()
                if pd.notna(row["latest_close"])
            }
            with st.form("batch_close_position_form"):
                selected_labels = st.multiselect("选择持仓", list(batch_map.keys()))
                exit_date = st.date_input("统一卖出日期", key="batch_exit_date")
                exit_reason_category = st.selectbox(
                    "统一卖出分类",
                    ["止损", "ATR止盈", "固定止盈3%", "固定止盈8%", "超3天卖出", "手动卖出", "其他"],
                    key="batch_exit_reason_category",
                )
                exit_reason = st.text_input("统一卖出说明", key="batch_exit_reason")
                manual_note = st.text_input("统一备注", key="batch_close_note")
                submitted = st.form_submit_button("按最新价批量平仓")
            if submitted:
                closed = 0
                exit_date_str = pd.Timestamp(exit_date).strftime("%Y-%m-%d")
                for label in selected_labels:
                    row = batch_map[label]
                    exit_price = float(row["latest_close"])
                    holding_days = trading_days_between(row["code"], row["entry_date"], exit_date_str)
                    return_pct = (exit_price - float(row["entry_price"])) / float(row["entry_price"])
                    pnl_amount = (exit_price - float(row["entry_price"])) * float(row["quantity"])
                    close_position(
                        position_id=int(row["id"]),
                        exit_date=exit_date_str,
                        exit_price=exit_price,
                        exit_reason=exit_reason or exit_reason_category,
                        exit_reason_category=exit_reason_category,
                        holding_days=holding_days,
                        return_pct=return_pct,
                        pnl_amount=pnl_amount,
                        manual_note=manual_note or "批量平仓",
                    )
                    closed += 1
                st.success(f"批量平仓完成：{closed} 笔。")
                st.rerun()


def _render_sample_management() -> None:
    st.subheader("示例数据管理")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("清空示例数据"):
            clear_demo_records()
            _load_all_data.clear()
            st.success("示例持仓 / 示例历史交易已清空。")
            st.rerun()
    with c2:
        st.info("示例数据备注前缀为“示例”或“演示”。")


def main():
    init_db()
    if "brick_panel_synced" not in st.session_state:
        with st.spinner("同步 BRICK 信号中..."):
            st.session_state["brick_panel_sync_count"] = sync_all_signals()
        st.session_state["brick_panel_synced"] = True

    col_a, col_b = st.columns([1, 1])
    with col_a:
        st.info(f'本次会话已同步信号 {st.session_state.get("brick_panel_sync_count", 0)} 条。')
    with col_b:
        if st.button("重新同步信号"):
            with st.spinner("同步 BRICK 信号中..."):
                st.session_state["brick_panel_sync_count"] = sync_all_signals()
            _load_all_data.clear()
            st.success("同步完成。")
            st.rerun()

    sync_token = int(st.session_state.get("brick_panel_sync_count", 0))
    raw_positions_df, trades_df, signals_df, baskets_df = _load_all_data(sync_token)
    positions_df = enrich_positions(raw_positions_df)
    positions_df, trades_df, signals_df = _apply_global_filters(positions_df, trades_df, signals_df)

    _render_metrics(positions_df, trades_df)
    _render_reminders(positions_df, signals_df)
    _render_signals(signals_df)
    _render_equity_curve(positions_df, trades_df)
    _render_recent_stats(trades_df)
    _render_period_summary(trades_df)
    st.subheader("当前持仓")
    if positions_df.empty:
        st.info("暂无当前持仓。")
    else:
        _styled_positions(positions_df)

    _render_baskets(baskets_df)
    st.subheader("历史交易")
    _render_trades(trades_df)
    _render_lifecycle(signals_df, positions_df, trades_df)
    _render_position_forms(signals_df, positions_df)
    _render_sample_management()


if __name__ == "__main__":
    main()
