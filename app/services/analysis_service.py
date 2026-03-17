from __future__ import annotations
import pandas as pd
from app.models.schemas import (
    AnalysisRequest, AnalysisResponse, AssetItem, AssetSearchResponse,
    DisclaimerBlock, InsightBlock, MetaBlock, NormalizedPoint,
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
