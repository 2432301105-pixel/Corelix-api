from __future__ import annotations

import math
from typing import Iterable

import numpy as np
import pandas as pd

from app.models.schemas import (
    AnalysisRequest, AnalysisResponse, AssetItem, AssetSearchResponse,
    DisclaimerBlock, GARCHOutput, InsightBlock, MetaBlock, NormalizedPoint,
    RiskBatchProfileRequest, RiskBatchProfilesResponse, RiskCommentary,
    RiskForecast, RiskHistoryBundle, RiskMeta, RiskModelSpec, RiskPrice,
    RiskProfileRequest, RiskProfileResponse, RiskScores, RiskSeriesPoint,
    RollingPoint, ScatterPoint, SummaryBlock,
)
from app.services import mock_engine as engine

def search_assets(query: str, limit: int = 10) -> AssetSearchResponse:
    raw = engine.search_assets(query, limit=limit)
    items = [AssetItem(**row, currency="USD") for row in raw]
    return AssetSearchResponse(query=query, results=items, total=len(items))

def run_analysis(req: AnalysisRequest) -> AnalysisResponse:
    sym_a     = req.asset_a.strip().upper()
    sym_b     = req.asset_b.strip().upper()
    period    = req.period.value
    frequency = req.frequency.value
    rtype     = req.return_type.value

    if engine.get_asset(sym_a) is None:
        raise ValueError(f"Unknown symbol: '{sym_a}'. Check /v1/assets/search for valid symbols.")
    if engine.get_asset(sym_b) is None:
        raise ValueError(f"Unknown symbol: '{sym_b}'. Check /v1/assets/search for valid symbols.")

    prices_a  = engine.generate_price_series(sym_a, period, frequency)
    prices_b  = engine.generate_price_series(sym_b, period, frequency)
    returns_a = engine.to_returns(prices_a, rtype)
    returns_b = engine.to_returns(prices_b, rtype)

    stat_dict, rolling_raw = engine.compute_statistics(returns_a, returns_b, frequency)

    aligned   = pd.DataFrame({"a": prices_a, "b": prices_b}).dropna()
    date_from = aligned.index[0].date().isoformat()
    date_to   = aligned.index[-1].date().isoformat()

    return AnalysisResponse(
        meta=MetaBlock(
            asset_a=sym_a, asset_b=sym_b,
            period=period, frequency=frequency, return_type=rtype,
            date_from=date_from, date_to=date_to,
        ),
        summary=SummaryBlock(**stat_dict),
        normalized_series=[NormalizedPoint(**p) for p in engine.build_normalized_series(prices_a, prices_b)],
        scatter_points=[ScatterPoint(**p) for p in engine.build_scatter_points(returns_a, returns_b)],
        rolling_series=[RollingPoint(**p) for p in engine.build_rolling_series(rolling_raw)],
        insight=InsightBlock(**engine.build_insight(stat_dict)),
        disclaimer=DisclaimerBlock(text=engine.DISCLAIMER_TEXT),
    )


def run_risk_profile(req: RiskProfileRequest) -> RiskProfileResponse:
    symbol = req.symbol.strip().upper()
    price_series = _generate_risk_price_series(
        symbol=symbol,
        lookback=req.lookback,
        interval=req.interval,
    )

    returns = np.log(price_series / price_series.shift(1)).dropna()
    if len(returns) < 60:
        raise ValueError(f"Insufficient data for '{symbol}'. Need at least 60 observations.")

    metadata = _resolve_asset_metadata(symbol)
    long_run_vol = max(_safe_std(returns), 0.0001)
    latest_conditional_vol = max(_rolling_volatility(returns, window=min(20, len(returns))), 0.0001)

    persistence_raw = max(_lag_one_autocorrelation((returns ** 2).tolist()), 0.0)
    shock_ratio = min(max(abs(float(returns.iloc[-1])) / max(latest_conditional_vol * 2.0, 0.0001), 0.0), 1.0)
    alpha1, beta1 = _garch_like_parameters(
        symbol=symbol,
        persistence_raw=persistence_raw,
        shock_ratio=shock_ratio,
    )
    alpha_plus_beta = round(alpha1 + beta1, 4)
    omega = round((long_run_vol ** 2) * max(1.0 - alpha_plus_beta, 0.001), 8)
    mu = round(float(returns.mean()), 6)

    risk_score = _clamp_score(int(round(((latest_conditional_vol / long_run_vol) / 2.0) * 100.0)))
    persistence_score = _clamp_score(int(round(alpha_plus_beta * 100.0)))
    shock_score = _clamp_score(int(round(min(alpha1 / 0.10, 1.0) * 100.0)))
    regime = _risk_regime(risk_score)
    forecast = _forecast_scores(risk_score, persistence_score)

    rolling_window = min(20, len(returns))
    rolling_sigma = returns.rolling(rolling_window).std(ddof=1).dropna()
    conditional_history = [
        RiskSeriesPoint(date=index.date().isoformat(), value=round(float(value), 6))
        for index, value in rolling_sigma.tail(60).items()
    ]
    returns_history = [
        RiskSeriesPoint(date=index.date().isoformat(), value=round(float(value), 6))
        for index, value in returns.tail(60).items()
    ]

    last_price = float(price_series.iloc[-1])
    previous_close = float(price_series.iloc[-2])
    change_percent = ((last_price / previous_close) - 1.0) * 100.0
    as_of = price_series.index[-1].date().isoformat()

    summary = _commentary_summary(symbol, risk_score, persistence_score, shock_score, alpha_plus_beta)
    watchlist_note = _watchlist_note(regime)

    return RiskProfileResponse(
        meta=RiskMeta(
            symbol=symbol,
            name=metadata["name"],
            exchange=metadata["exchange"],
            currency=metadata["currency"],
            interval=req.interval,
            lookback=req.lookback,
            as_of=as_of,
        ),
        price=RiskPrice(
            last=round(last_price, 4),
            previous_close=round(previous_close, 4),
            change_percent=round(change_percent, 4),
        ),
        model=RiskModelSpec(distribution=req.distribution.value),
        garch=GARCHOutput(
            mu=mu,
            omega=omega,
            alpha1=round(alpha1, 4),
            beta1=round(beta1, 4),
            alpha_plus_beta=alpha_plus_beta,
            latest_conditional_vol=round(latest_conditional_vol, 6),
            long_run_vol=round(long_run_vol, 6),
        ),
        scores=RiskScores(
            risk=risk_score,
            persistence=persistence_score,
            shock=shock_score,
            regime=regime,
        ),
        forecast=RiskForecast(
            day_1=forecast[0],
            day_5=forecast[1],
            day_20=forecast[2],
        ),
        history=RiskHistoryBundle(
            conditional_volatility=conditional_history,
            returns=returns_history,
        ),
        commentary=RiskCommentary(
            summary=summary,
            watchlist_note=watchlist_note,
        ),
        disclaimer=engine.DISCLAIMER_TEXT,
    )


def run_risk_profiles(req: RiskBatchProfileRequest) -> RiskBatchProfilesResponse:
    profiles = [
        run_risk_profile(
            RiskProfileRequest(
                symbol=symbol,
                lookback=req.lookback,
                interval=req.interval,
                distribution=req.distribution,
                refresh=req.refresh,
            )
        )
        for symbol in req.symbols
    ]
    return RiskBatchProfilesResponse(profiles=profiles)


def _resolve_asset_metadata(symbol: str) -> dict[str, str]:
    asset = engine.get_asset(symbol)
    if asset is not None:
        return {
            "name": asset["name"],
            "exchange": asset["exchange"],
            "currency": "USD" if asset["exchange"] != "BIST" else "TRY",
        }

    if symbol.endswith(".IS"):
        base = symbol.removesuffix(".IS")
        return {
            "name": f"{base} BIST 100",
            "exchange": "BIST",
            "currency": "TRY",
        }

    return {
        "name": symbol,
        "exchange": "SYNTH",
        "currency": "USD",
    }


def _generate_risk_price_series(symbol: str, lookback: int, interval: str) -> pd.Series:
    period = _period_from_lookback(lookback)
    frequency = _frequency_from_interval(interval)
    series = engine.generate_price_series(symbol, period, frequency)
    if len(series) > lookback + 5:
        series = series.tail(lookback + 5)
    return series


def _period_from_lookback(lookback: int) -> str:
    if lookback <= 31:
        return "1M"
    if lookback <= 190:
        return "6M"
    if lookback <= 380:
        return "1Y"
    if lookback <= 820:
        return "3Y"
    return "5Y"


def _frequency_from_interval(interval: str) -> str:
    normalized = interval.strip().lower()
    if normalized in {"1wk", "1w", "wk", "weekly"}:
        return "weekly"
    if normalized in {"1mo", "1m", "monthly"}:
        return "monthly"
    return "daily"


def _safe_std(values: pd.Series | Iterable[float]) -> float:
    if isinstance(values, pd.Series):
        count = len(values)
        return float(values.std(ddof=1)) if count >= 2 else 0.0
    values_list = list(values)
    return float(np.std(values_list, ddof=1)) if len(values_list) >= 2 else 0.0


def _rolling_volatility(returns: pd.Series, window: int) -> float:
    if len(returns) < window:
        return _safe_std(returns)
    return _safe_std(returns.tail(window))


def _lag_one_autocorrelation(values: list[float]) -> float:
    if len(values) < 3:
        return 0.0
    x = np.asarray(values[:-1])
    y = np.asarray(values[1:])
    x_centered = x - x.mean()
    y_centered = y - y.mean()
    denominator = math.sqrt(float((x_centered ** 2).sum() * (y_centered ** 2).sum()))
    if denominator <= 0:
        return 0.0
    correlation = float((x_centered * y_centered).sum() / denominator)
    return min(max(correlation, -1.0), 1.0)


def _garch_like_parameters(symbol: str, persistence_raw: float, shock_ratio: float) -> tuple[float, float]:
    if symbol.endswith(".IS"):
        base_alpha = 0.045
        persistence_floor = 0.88
    elif symbol.endswith("-USD"):
        base_alpha = 0.060
        persistence_floor = 0.80
    else:
        base_alpha = 0.038
        persistence_floor = 0.84

    alpha1 = min(max(base_alpha + shock_ratio * 0.045, 0.02), 0.14)
    target_persistence = min(max(persistence_floor + persistence_raw * 0.10, 0.78), 0.985)
    beta1 = min(max(target_persistence - alpha1, 0.55), 0.96)
    if alpha1 + beta1 >= 0.995:
        beta1 = 0.994 - alpha1
    return alpha1, beta1


def _risk_regime(score: int) -> str:
    if score < 35:
        return "low"
    if score < 56:
        return "balanced"
    if score < 76:
        return "elevated"
    return "high"


def _forecast_scores(risk_score: int, persistence_score: int) -> tuple[int, int, int]:
    persistence_weight = max(persistence_score, 20) / 100.0
    day1 = _clamp_score(int(round(risk_score * (0.94 * persistence_weight + 0.74 * (1 - persistence_weight)))))
    day5 = _clamp_score(int(round(risk_score * (0.86 * persistence_weight + 0.70 * (1 - persistence_weight)))))
    day20 = _clamp_score(int(round(risk_score * (0.74 * persistence_weight + 0.66 * (1 - persistence_weight)))))
    return day1, day5, day20


def _clamp_score(value: int) -> int:
    return min(max(value, 0), 100)


def _commentary_summary(symbol: str, risk_score: int, persistence_score: int, shock_score: int, alpha_plus_beta: float) -> str:
    base = f"{symbol} icin risk skoru {risk_score}, kalicilik {persistence_score}, sok hassasiyeti {shock_score} olarak hesaplandi."
    if alpha_plus_beta >= 0.94:
        return base + " Oynaklik rejimi yuksek derecede kalici ve sarsintilar yavas sonuyor."
    if alpha_plus_beta >= 0.88:
        return base + " Oynaklik rejimi belirgin kalici; yeni soklar bir sure daha tasiniyor."
    return base + " Oynaklik rejimi gorece daha hizli normallesme egiliminde."


def _watchlist_note(regime: str) -> str:
    if regime == "high":
        return "Watchlist icinde kirmizi bayrak olarak tutulmali; gunluk risk yenilenmesi izlenmeli."
    if regime == "elevated":
        return "Watchlist icinde aktif izleme gerektirir; alarm esikleri acik tutulabilir."
    if regime == "balanced":
        return "Watchlist icinde temel izleme seviyesi icin uygun."
    return "Watchlist icinde referans veya dengeleyici varlik gibi izlenebilir."
