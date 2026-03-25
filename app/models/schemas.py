from __future__ import annotations
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field, model_validator

class Period(str, Enum):
    one_month   = "1M"
    six_months  = "6M"
    one_year    = "1Y"
    three_years = "3Y"
    five_years  = "5Y"

class Frequency(str, Enum):
    daily   = "daily"
    weekly  = "weekly"
    monthly = "monthly"

class ReturnType(str, Enum):
    price      = "price"
    pct_return = "pct"
    log_return = "log"

class HealthResponse(BaseModel):
    status: str
    version: str
    environment: str

class AssetType(str, Enum):
    stock     = "stock"
    etf       = "etf"
    crypto    = "crypto"
    commodity = "commodity"

class AssetItem(BaseModel):
    symbol:     str
    name:       str
    asset_type: AssetType
    exchange:   str
    currency:   str = "USD"

class AssetSearchResponse(BaseModel):
    query:   str
    results: list[AssetItem]
    total:   int

class AnalysisRequest(BaseModel):
    asset_a:     str = Field(..., min_length=1, max_length=20)
    asset_b:     str = Field(..., min_length=1, max_length=20)
    period:      Period     = Period.one_year
    frequency:   Frequency  = Frequency.daily
    return_type: ReturnType = ReturnType.pct_return

    @model_validator(mode="after")
    def assets_must_differ(self) -> AnalysisRequest:
        if self.asset_a.strip().upper() == self.asset_b.strip().upper():
            raise ValueError("asset_a and asset_b must be different symbols.")
        return self

class MetaBlock(BaseModel):
    asset_a: str
    asset_b: str
    period: str
    frequency: str
    return_type: str
    date_from: str
    date_to: str

class SummaryBlock(BaseModel):
    pearson_r: float
    pearson_p_value: float
    spearman_rho: float
    spearman_p_value: float
    r_squared: float
    observations: int
    is_significant: bool
    pearson_label: str
    rolling_window: int

class NormalizedPoint(BaseModel):
    date: str
    value_a: float
    value_b: float

class ScatterPoint(BaseModel):
    x: float
    y: float

class RollingPoint(BaseModel):
    date: str
    correlation: Optional[float]

class InsightBlock(BaseModel):
    headline: str
    body: str
    label: str
    is_significant: bool
    r_squared_pct: float

class DisclaimerBlock(BaseModel):
    text: str
    version: str = "1.0"

class AnalysisResponse(BaseModel):
    meta:              MetaBlock
    summary:           SummaryBlock
    normalized_series: list[NormalizedPoint]
    scatter_points:    list[ScatterPoint]
    rolling_series:    list[RollingPoint]
    insight:           InsightBlock
    disclaimer:        DisclaimerBlock


class RiskDistribution(str, Enum):
    normal = "normal"
    student_t = "student_t"


class RiskProfileRequest(BaseModel):
    symbol: str = Field(..., min_length=1, max_length=24)
    lookback: int = Field(default=252, ge=60, le=1260)
    interval: str = Field(default="1d", min_length=2, max_length=8)
    distribution: RiskDistribution = RiskDistribution.student_t
    refresh: str = Field(default="if_stale", min_length=2, max_length=20)


class RiskBatchProfileRequest(BaseModel):
    symbols: list[str] = Field(..., min_length=1, max_length=120)
    lookback: int = Field(default=252, ge=60, le=1260)
    interval: str = Field(default="1d", min_length=2, max_length=8)
    distribution: RiskDistribution = RiskDistribution.student_t
    refresh: str = Field(default="if_stale", min_length=2, max_length=20)

    @model_validator(mode="after")
    def normalize_symbols(self) -> "RiskBatchProfileRequest":
        normalized: list[str] = []
        seen: set[str] = set()
        for raw_symbol in self.symbols:
            symbol = raw_symbol.strip().upper()
            if not symbol or symbol in seen:
                continue
            seen.add(symbol)
            normalized.append(symbol)
        if not normalized:
            raise ValueError("At least one valid symbol is required.")
        self.symbols = normalized
        return self


class RiskMeta(BaseModel):
    symbol: str
    name: str
    exchange: str
    currency: str
    interval: str
    lookback: int
    as_of: str


class RiskPrice(BaseModel):
    last: float
    previous_close: float
    change_percent: float


class RiskModelSpec(BaseModel):
    family: str = "GARCH"
    mean: str = "Constant"
    p: int = 1
    q: int = 1
    distribution: str


class GARCHOutput(BaseModel):
    mu: float
    omega: float
    alpha1: float
    beta1: float
    alpha_plus_beta: float
    latest_conditional_vol: float
    long_run_vol: float


class RiskScores(BaseModel):
    risk: int
    persistence: int
    shock: int
    regime: str


class RiskForecast(BaseModel):
    day_1: int
    day_5: int
    day_20: int


class RiskSeriesPoint(BaseModel):
    date: str
    value: float


class RiskHistoryBundle(BaseModel):
    conditional_volatility: list[RiskSeriesPoint]
    returns: list[RiskSeriesPoint]


class RiskCommentary(BaseModel):
    summary: str
    watchlist_note: str


class RiskProfileResponse(BaseModel):
    meta: RiskMeta
    price: RiskPrice
    model: RiskModelSpec
    garch: GARCHOutput
    scores: RiskScores
    forecast: RiskForecast
    history: RiskHistoryBundle
    commentary: RiskCommentary
    disclaimer: str


class RiskBatchProfilesResponse(BaseModel):
    profiles: list[RiskProfileResponse]
