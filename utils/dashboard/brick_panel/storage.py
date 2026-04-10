from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from typing import Iterable

import pandas as pd

try:
    from .paths import DB_PATH
except ImportError:  # pragma: no cover - for direct script execution
    from paths import DB_PATH


def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


@contextmanager
def connection_ctx():
    conn = get_connection()
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def _ensure_column(conn: sqlite3.Connection, table: str, column: str, ddl: str) -> None:
    existing = {row["name"] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}
    if column not in existing:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {ddl}")


def init_db() -> None:
    with connection_ctx() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS signals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                signal_date TEXT NOT NULL,
                code TEXT NOT NULL,
                name TEXT,
                strategy TEXT NOT NULL,
                source_list TEXT NOT NULL,
                signal_close REAL,
                signal_low REAL,
                stop_loss_price REAL,
                raw_score REAL,
                raw_reason TEXT,
                import_batch TEXT,
                UNIQUE(signal_date, code, strategy)
            );

            CREATE TABLE IF NOT EXISTS positions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                code TEXT NOT NULL,
                name TEXT,
                strategy TEXT NOT NULL,
                signal_date TEXT NOT NULL,
                entry_date TEXT NOT NULL,
                entry_price REAL NOT NULL,
                entry_signal_low REAL NOT NULL,
                quantity INTEGER NOT NULL,
                source_list TEXT,
                raw_score REAL,
                raw_reason TEXT,
                buy_reason_snapshot TEXT,
                tags TEXT,
                strategy_version TEXT,
                manual_note TEXT,
                status TEXT NOT NULL DEFAULT 'open',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                code TEXT NOT NULL,
                name TEXT,
                strategy TEXT NOT NULL,
                signal_date TEXT NOT NULL,
                entry_date TEXT NOT NULL,
                entry_price REAL NOT NULL,
                entry_signal_low REAL NOT NULL,
                quantity INTEGER NOT NULL,
                source_list TEXT,
                raw_score REAL,
                raw_reason TEXT,
                buy_reason_snapshot TEXT,
                tags TEXT,
                strategy_version TEXT,
                exit_date TEXT NOT NULL,
                exit_price REAL NOT NULL,
                exit_reason TEXT NOT NULL,
                exit_reason_category TEXT,
                holding_days INTEGER NOT NULL,
                return_pct REAL NOT NULL,
                pnl_amount REAL NOT NULL,
                manual_note TEXT,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS basket_daily_snapshot (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                signal_date TEXT NOT NULL,
                strategy TEXT NOT NULL,
                stock_count INTEGER NOT NULL,
                avg_return_to_latest_close REAL,
                latest_price_date TEXT,
                updated_at TEXT NOT NULL,
                UNIQUE(signal_date, strategy)
            );

            CREATE TABLE IF NOT EXISTS basket_members_snapshot (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                signal_date TEXT NOT NULL,
                strategy TEXT NOT NULL,
                code TEXT NOT NULL,
                name TEXT,
                entry_date TEXT,
                entry_price REAL,
                latest_close REAL,
                latest_price_date TEXT,
                return_to_latest_close REAL,
                updated_at TEXT NOT NULL,
                UNIQUE(signal_date, strategy, code)
            );

            CREATE TABLE IF NOT EXISTS operation_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                code TEXT,
                name TEXT,
                strategy TEXT,
                event_type TEXT NOT NULL,
                event_date TEXT,
                ref_type TEXT,
                ref_id INTEGER,
                payload_json TEXT,
                created_at TEXT NOT NULL
            );
            """
        )
        _ensure_column(conn, "positions", "source_list", "TEXT")
        _ensure_column(conn, "positions", "raw_score", "REAL")
        _ensure_column(conn, "positions", "raw_reason", "TEXT")
        _ensure_column(conn, "positions", "buy_reason_snapshot", "TEXT")
        _ensure_column(conn, "positions", "tags", "TEXT")
        _ensure_column(conn, "positions", "strategy_version", "TEXT")
        _ensure_column(conn, "trades", "source_list", "TEXT")
        _ensure_column(conn, "trades", "raw_score", "REAL")
        _ensure_column(conn, "trades", "raw_reason", "TEXT")
        _ensure_column(conn, "trades", "buy_reason_snapshot", "TEXT")
        _ensure_column(conn, "trades", "tags", "TEXT")
        _ensure_column(conn, "trades", "strategy_version", "TEXT")
        _ensure_column(conn, "trades", "exit_reason_category", "TEXT")


def read_sql(query: str, params: Iterable | None = None) -> pd.DataFrame:
    with get_connection() as conn:
        return pd.read_sql_query(query, conn, params=params or ())


def upsert_signals(records: list[dict]) -> int:
    if not records:
        return 0
    with connection_ctx() as conn:
        conn.executemany(
            """
            INSERT INTO signals (
                signal_date, code, name, strategy, source_list, signal_close,
                signal_low, stop_loss_price, raw_score, raw_reason, import_batch
            )
            VALUES (
                :signal_date, :code, :name, :strategy, :source_list, :signal_close,
                :signal_low, :stop_loss_price, :raw_score, :raw_reason, :import_batch
            )
            ON CONFLICT(signal_date, code, strategy) DO UPDATE SET
                name=excluded.name,
                source_list=excluded.source_list,
                signal_close=excluded.signal_close,
                signal_low=excluded.signal_low,
                stop_loss_price=excluded.stop_loss_price,
                raw_score=excluded.raw_score,
                raw_reason=excluded.raw_reason,
                import_batch=excluded.import_batch
            """,
            records,
        )
    return len(records)


def list_signals(limit: int | None = None) -> pd.DataFrame:
    sql = """
        SELECT signal_date, code, COALESCE(name, '') AS name, strategy, source_list,
               signal_close, signal_low, stop_loss_price, raw_score, raw_reason
        FROM signals
        ORDER BY signal_date DESC, strategy, code
    """
    if limit is not None:
        sql += f" LIMIT {int(limit)}"
    return read_sql(sql)


def list_positions() -> pd.DataFrame:
    return read_sql(
        """
        SELECT id, code, COALESCE(name, '') AS name, strategy, signal_date, entry_date,
               entry_price, entry_signal_low, quantity,
               COALESCE(source_list, '') AS source_list,
               raw_score, COALESCE(raw_reason, '') AS raw_reason,
               COALESCE(buy_reason_snapshot, '') AS buy_reason_snapshot,
               COALESCE(tags, '') AS tags,
               COALESCE(strategy_version, '') AS strategy_version,
               COALESCE(manual_note, '') AS manual_note,
               status, created_at, updated_at
        FROM positions
        WHERE status = 'open'
        ORDER BY entry_date DESC, code
        """
    )


def list_trades() -> pd.DataFrame:
    return read_sql(
        """
        SELECT id, code, COALESCE(name, '') AS name, strategy, signal_date, entry_date,
               entry_price, entry_signal_low, quantity,
               COALESCE(source_list, '') AS source_list,
               raw_score, COALESCE(raw_reason, '') AS raw_reason,
               COALESCE(buy_reason_snapshot, '') AS buy_reason_snapshot,
               COALESCE(tags, '') AS tags,
               COALESCE(strategy_version, '') AS strategy_version,
               exit_date, exit_price, exit_reason, COALESCE(exit_reason_category, '') AS exit_reason_category,
               holding_days, return_pct, pnl_amount, COALESCE(manual_note, '') AS manual_note,
               created_at
        FROM trades
        ORDER BY exit_date DESC, code
        """
    )


def list_operation_logs(code: str | None = None) -> pd.DataFrame:
    if code:
        return read_sql(
            """
            SELECT id, code, COALESCE(name, '') AS name, strategy, event_type, event_date,
                   COALESCE(ref_type, '') AS ref_type, ref_id, COALESCE(payload_json, '') AS payload_json, created_at
            FROM operation_logs
            WHERE code = ?
            ORDER BY created_at DESC, id DESC
            """,
            (code,),
        )
    return read_sql(
        """
        SELECT id, code, COALESCE(name, '') AS name, strategy, event_type, event_date,
               COALESCE(ref_type, '') AS ref_type, ref_id, COALESCE(payload_json, '') AS payload_json, created_at
        FROM operation_logs
        ORDER BY created_at DESC, id DESC
        """
    )


def _log_event(
    conn: sqlite3.Connection,
    *,
    code: str,
    name: str,
    strategy: str,
    event_type: str,
    event_date: str,
    ref_type: str,
    ref_id: int | None,
    payload: dict,
) -> None:
    conn.execute(
        """
        INSERT INTO operation_logs (
            code, name, strategy, event_type, event_date, ref_type, ref_id, payload_json, created_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            code,
            name,
            strategy,
            event_type,
            event_date,
            ref_type,
            ref_id,
            json.dumps(payload, ensure_ascii=False),
            datetime.now().isoformat(timespec="seconds"),
        ),
    )


def get_signal_row(signal_date: str, code: str, strategy: str) -> sqlite3.Row | None:
    with get_connection() as conn:
        row = conn.execute(
            """
            SELECT *
            FROM signals
            WHERE signal_date = ? AND code = ? AND strategy = ?
            """,
            (signal_date, code, strategy),
        ).fetchone()
    return row


def insert_position(
    *,
    code: str,
    name: str,
    strategy: str,
    signal_date: str,
    entry_date: str,
    entry_price: float,
    entry_signal_low: float,
    quantity: int,
    source_list: str = "",
    raw_score: float | None = None,
    raw_reason: str = "",
    buy_reason_snapshot: str = "",
    tags: str = "",
    strategy_version: str = "brick_panel_v1",
    manual_note: str,
) -> None:
    now = datetime.now().isoformat(timespec="seconds")
    with connection_ctx() as conn:
        cursor = conn.execute(
            """
            INSERT INTO positions (
                code, name, strategy, signal_date, entry_date, entry_price,
                entry_signal_low, quantity, source_list, raw_score, raw_reason, buy_reason_snapshot,
                tags, strategy_version, manual_note, status, created_at, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'open', ?, ?)
            """,
            (
                code,
                name,
                strategy,
                signal_date,
                entry_date,
                entry_price,
                entry_signal_low,
                quantity,
                source_list,
                raw_score,
                raw_reason,
                buy_reason_snapshot,
                tags,
                strategy_version,
                manual_note,
                now,
                now,
            ),
        )
        _log_event(
            conn,
            code=code,
            name=name,
            strategy=strategy,
            event_type="open_position",
            event_date=entry_date,
            ref_type="position",
            ref_id=cursor.lastrowid,
            payload={
                "signal_date": signal_date,
                "entry_price": entry_price,
                "entry_signal_low": entry_signal_low,
                "quantity": quantity,
                "source_list": source_list,
                "raw_score": raw_score,
                "raw_reason": raw_reason,
                "buy_reason_snapshot": buy_reason_snapshot,
                "tags": tags,
                "strategy_version": strategy_version,
                "manual_note": manual_note,
            },
        )


def close_position(
    *,
    position_id: int,
    exit_date: str,
    exit_price: float,
    exit_reason: str,
    exit_reason_category: str,
    holding_days: int,
    return_pct: float,
    pnl_amount: float,
    manual_note: str,
) -> None:
    now = datetime.now().isoformat(timespec="seconds")
    with connection_ctx() as conn:
        row = conn.execute("SELECT * FROM positions WHERE id = ?", (position_id,)).fetchone()
        if row is None:
            raise ValueError(f"未找到持仓: {position_id}")
        cursor = conn.execute(
            """
            INSERT INTO trades (
                code, name, strategy, signal_date, entry_date, entry_price, entry_signal_low,
                quantity, source_list, raw_score, raw_reason, buy_reason_snapshot, tags, strategy_version,
                exit_date, exit_price, exit_reason, exit_reason_category, holding_days, return_pct,
                pnl_amount, manual_note, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                row["code"],
                row["name"],
                row["strategy"],
                row["signal_date"],
                row["entry_date"],
                row["entry_price"],
                row["entry_signal_low"],
                row["quantity"],
                row["source_list"],
                row["raw_score"],
                row["raw_reason"],
                row["buy_reason_snapshot"],
                row["tags"],
                row["strategy_version"],
                exit_date,
                exit_price,
                exit_reason,
                exit_reason_category,
                holding_days,
                return_pct,
                pnl_amount,
                manual_note,
                now,
            ),
        )
        conn.execute(
            "UPDATE positions SET status = 'closed', updated_at = ? WHERE id = ?",
            (now, position_id),
        )
        _log_event(
            conn,
            code=row["code"],
            name=row["name"],
            strategy=row["strategy"],
            event_type="close_position",
            event_date=exit_date,
            ref_type="trade",
            ref_id=cursor.lastrowid,
            payload={
                "entry_date": row["entry_date"],
                "entry_price": row["entry_price"],
                "exit_price": exit_price,
                "exit_reason": exit_reason,
                "exit_reason_category": exit_reason_category,
                "holding_days": holding_days,
                "return_pct": return_pct,
                "pnl_amount": pnl_amount,
                "manual_note": manual_note,
            },
        )


def clear_demo_records() -> None:
    with connection_ctx() as conn:
        conn.execute("DELETE FROM trades WHERE manual_note LIKE '示例%' OR manual_note LIKE '演示%'")
        conn.execute("DELETE FROM positions WHERE manual_note LIKE '示例%' OR manual_note LIKE '演示%'")
        conn.execute(
            """
            DELETE FROM operation_logs
            WHERE payload_json LIKE '%示例%' OR payload_json LIKE '%演示%'
            """
        )


def replace_basket_snapshot(
    signal_date: str,
    strategy: str,
    stock_count: int,
    avg_return_to_latest_close: float | None,
    latest_price_date: str | None,
    members: list[dict],
) -> None:
    now = datetime.now().isoformat(timespec="seconds")
    with connection_ctx() as conn:
        conn.execute(
            """
            INSERT INTO basket_daily_snapshot (
                signal_date, strategy, stock_count, avg_return_to_latest_close, latest_price_date, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(signal_date, strategy) DO UPDATE SET
                stock_count = excluded.stock_count,
                avg_return_to_latest_close = excluded.avg_return_to_latest_close,
                latest_price_date = excluded.latest_price_date,
                updated_at = excluded.updated_at
            """,
            (
                signal_date,
                strategy,
                stock_count,
                avg_return_to_latest_close,
                latest_price_date,
                now,
            ),
        )
        conn.execute(
            "DELETE FROM basket_members_snapshot WHERE signal_date = ? AND strategy = ?",
            (signal_date, strategy),
        )
        if members:
            payload = [{**member, "updated_at": now} for member in members]
            conn.executemany(
                """
                INSERT INTO basket_members_snapshot (
                    signal_date, strategy, code, name, entry_date, entry_price,
                    latest_close, latest_price_date, return_to_latest_close, updated_at
                )
                VALUES (
                    :signal_date, :strategy, :code, :name, :entry_date, :entry_price,
                    :latest_close, :latest_price_date, :return_to_latest_close, :updated_at
                )
                """,
                payload,
            )


def list_basket_daily_snapshot() -> pd.DataFrame:
    return read_sql(
        """
        SELECT signal_date, strategy, stock_count, avg_return_to_latest_close,
               latest_price_date, updated_at
        FROM basket_daily_snapshot
        ORDER BY signal_date DESC, strategy
        """
    )


def list_basket_members(signal_date: str, strategy: str) -> pd.DataFrame:
    return read_sql(
        """
        SELECT signal_date, strategy, code, COALESCE(name, '') AS name, entry_date,
               entry_price, latest_close, latest_price_date, return_to_latest_close
        FROM basket_members_snapshot
        WHERE signal_date = ? AND strategy = ?
        ORDER BY code
        """,
        (signal_date, strategy),
    )
