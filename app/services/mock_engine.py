from __future__ import annotations

from datetime import date, timedelta
import numpy as np
import pandas as pd
from scipy import stats

CATALOGUE = [
    {"symbol": "AAPL",    "name": "Apple Inc.",                   "asset_type": "stock",     "exchange": "NASDAQ"},
    {"symbol": "MSFT",    "name": "Microsoft Corporation",        "asset_type": "stock",     "exchange": "NASDAQ"},
    {"symbol": "GOOGL",   "name": "Alphabet Inc.",                "asset_type": "stock",     "exchange": "NASDAQ"},
    {"symbol": "AMZN",    "name": "Amazon.com Inc.",              "asset_type": "stock",     "exchange": "NASDAQ"},
    {"symbol": "TSLA",    "name": "Tesla Inc.",                   "asset_type": "stock",     "exchange": "NASDAQ"},
    {"symbol": "NVDA",    "name": "NVIDIA Corporation",           "asset_type": "stock",     "exchange": "NASDAQ"},
    {"symbol": "META",    "name": "Meta Platforms Inc.",          "asset_type": "stock",     "exchange": "NASDAQ"},
    {"symbol": "JPM",     "name": "JPMorgan Chase & Co.",         "asset_type": "stock",     "exchange": "NYSE"},
    {"symbol": "SPY",     "name": "SPDR S&P 500 ETF",             "asset_type": "etf",       "exchange": "NYSE"},
    {"symbol": "QQQ",     "name": "Invesco QQQ Trust",            "asset_type": "etf",       "exchange": "NASDAQ"},
    {"symbol": "GLD",     "name": "SPDR Gold Shares",             "asset_type": "etf",       "exchange": "NYSE"},
    {"symbol": "TLT",     "name": "iShares 20+ Year Treasury ETF","asset_type": "etf",       "exchange": "NASDAQ"},
    {"symbol": "BTC-USD", "name": "Bitcoin",                      "asset_type": "crypto",    "exchange": "CRYPTO"},
    {"symbol": "ETH-USD", "name": "Ethereum",                     "asset_type": "crypto",    "exchange": "CRYPTO"},
    {"symbol": "SOL-USD", "name": "Solana",                       "asset_type": "crypto",    "exchange": "CRYPTO"},
    {"symbol": "XAU",     "name": "Gold Spot",                    "asset_type": "commodity", "exchange": "OTC"},
]

_MAP = {r["symbol"].upper(): r for r in CATALOGUE}

DISCLAIMER_TEXT = (
    "Corelix is an analytical tool for informational purposes only. "
    "Correlation data does not constitute financial advice, investment "
    "recommendations, or an offer to buy or sell any security. "
    "Past statistical relationships do not guarantee future results."
)

_PERIOD_DAYS = {"1M": 30, "6M": 182, "1Y": 365, "3Y": 365 * 3, "5Y": 365 * 5}
_FREQ_ALIAS = {"daily": "B", "weekly": "W-FRI", "monthly": "BME"}
_ROLLING_WINDOW = {"daily": 30, "weekly": 8, "monthly": 3}

_SYMBOL_PARAMS = {
    "AAPL":    {"drift": 0.00045, "idio": 0.0070, "mkt": 1.15, "tech": 1.10, "rates": -0.10, "gold": 0.00, "crypto": 0.00},
    "MSFT":    {"drift": 0.00042, "idio": 0.0065, "mkt": 1.05, "tech": 1.00, "rates": -0.05, "gold": 0.00, "crypto": 0.00},
    "GOOGL":   {"drift": 0.00040, "idio": 0.0070, "mkt": 1.05, "tech": 1.00, "rates": -0.05, "gold": 0.00, "crypto": 0.00},
    "AMZN":    {"drift": 0.00038, "idio": 0.0090, "mkt": 1.10, "tech": 0.95, "rates": -0.05, "gold": 0.00, "crypto": 0.00},
    "TSLA":    {"drift": 0.00035, "idio": 0.0180, "mkt": 1.35, "tech": 1.20, "rates": -0.10, "gold": 0.00, "crypto": 0.10},
    "NVDA":    {"drift": 0.00060, "idio": 0.0120, "mkt": 1.30, "tech": 1.35, "rates": -0.05, "gold": 0.00, "crypto": 0.05},
    "META":    {"drift": 0.00040, "idio": 0.0090, "mkt": 1.10, "tech": 1.00, "rates": -0.05, "gold": 0.00, "crypto": 0.00},
    "JPM":     {"drift": 0.00028, "idio": 0.0060, "mkt": 0.95, "tech": 0.00, "rates": 0.60,  "gold": 0.00, "crypto": 0.00},
    "SPY":     {"drift": 0.00030, "idio": 0.0025, "mkt": 1.00, "tech": 0.20, "rates": -0.05, "gold": 0.00, "crypto": 0.00},
    "QQQ":     {"drift": 0.00036, "idio": 0.0030, "mkt": 1.08, "tech": 1.25, "rates": -0.08, "gold": 0.00, "crypto": 0.00},
    "GLD":     {"drift": 0.00012, "idio": 0.0035, "mkt": 0.05, "tech": 0.00, "rates": -0.35, "gold": 1.00, "crypto": 0.00},
    "TLT":     {"drift": -0.00002, "idio": 0.0030, "mkt": -0.10, "tech": 0.00, "rates": -1.10, "gold": 0.10, "crypto": 0.00},
    "BTC-USD": {"drift": 0.00065, "idio": 0.0200, "mkt": 0.35, "tech": 0.20, "rates": 0.00, "gold": 0.00, "crypto": 1.00},
    "ETH-USD": {"drift": 0.00060, "idio": 0.0220, "mkt": 0.30, "tech": 0.25, "rates": 0.00, "gold": 0.00, "crypto": 1.15},
    "SOL-USD": {"drift": 0.00070, "idio": 0.0280, "mkt": 0.25, "tech": 0.20, "rates": 0.00, "gold": 0.00, "crypto": 1.30},
    "XAU":     {"drift": 0.00010, "idio": 0.0030, "mkt": 0.00, "tech": 0.00, "rates": -0.30, "gold": 1.00, "crypto": 0.00},
}
_DEFAULT_PARAMS = {"drift": 0.00020, "idio": 0.0080, "mkt": 0.80, "tech": 0.00, "rates": 0.00, "gold": 0.00, "crypto": 0.00}

def search_assets(query: str, limit: int = 10):
    q = query.strip().upper()
    return [r for r in CATALOGUE if q in r["symbol"].upper() or q in r["name"].upper()][:limit]

def get_asset(symbol: str):
    return _MAP.get(symbol.strip().upper())

def generate_price_series(symbol: str, period: str, frequency: str) -> pd.Series:
    sym = symbol.upper()
    params = _SYMBOL_PARAMS.get(sym, _DEFAULT_PARAMS)

    start = date.today() - timedelta(days=_PERIOD_DAYS[period])
    bdays = pd.bdate_range(start=start, end=date.today())

    common_rng = np.random.default_rng(20260316)
    market = common_rng.normal(0.00025, 0.0065, size=len(bdays))
    tech = common_rng.normal(0.00010, 0.0050, size=len(bdays))
    rates = common_rng.normal(0.00000, 0.0040, size=len(bdays))
    gold = common_rng.normal(0.00005, 0.0035, size=len(bdays))
    crypto = common_rng.normal(0.00040, 0.0180, size=len(bdays))

    idio_seed = sum(ord(c) * (i + 1) for i, c in enumerate(sym))
    idio_rng = np.random.default_rng(idio_seed)
    idio = idio_rng.normal(0.0, params["idio"], size=len(bdays))

    daily_returns = (
        params["drift"]
        + params["mkt"] * market
        + params["tech"] * tech
        + params["rates"] * rates
        + params["gold"] * gold
        + params["crypto"] * crypto
        + idio
    )

    prices = 100.0 * np.exp(np.cumsum(daily_returns))
    daily = pd.Series(prices, index=bdays, name=sym)

    if frequency == "daily":
        return daily
    return daily.resample(_FREQ_ALIAS[frequency]).last().dropna()

def to_returns(prices: pd.Series, return_type: str) -> pd.Series:
    if return_type == "price":
        return prices
    if return_type == "log":
        return np.log(prices / prices.shift(1)).dropna()
    return prices.pct_change().dropna()

def compute_statistics(returns_a: pd.Series, returns_b: pd.Series, frequency: str):
    df = pd.DataFrame({"a": returns_a, "b": returns_b}).dropna()
    window = _ROLLING_WINDOW[frequency]
    if len(df) < window + 2:
        raise ValueError(f"Only {len(df)} data points. Need at least {window + 2} for {frequency}. Try a longer period.")

    a, b = df["a"], df["b"]
    pr, pp = stats.pearsonr(a, b)
    sr, sp = stats.spearmanr(a, b)
    rolling = a.rolling(window).corr(b)

    stat = {
        "pearson_r": round(float(pr), 4),
        "pearson_p_value": round(float(pp), 6),
        "spearman_rho": round(float(sr), 4),
        "spearman_p_value": round(float(sp), 6),
        "r_squared": round(float(pr ** 2), 4),
        "observations": len(a),
        "is_significant": bool(pp < 0.05),
        "pearson_label": _label(pr),
        "rolling_window": window,
    }
    return stat, rolling

def build_normalized_series(prices_a, prices_b):
    df = pd.DataFrame({"a": prices_a, "b": prices_b}).dropna()
    normed = df.div(df.iloc[0]) * 100.0
    return [{"date": idx.date().isoformat(), "value_a": round(float(r["a"]), 4), "value_b": round(float(r["b"]), 4)} for idx, r in normed.iterrows()]

def build_scatter_points(returns_a, returns_b, max_points=300):
    df = pd.DataFrame({"x": returns_a, "y": returns_b}).dropna()
    if len(df) > max_points:
        df = df.sample(max_points, random_state=42)
    return [{"x": round(float(r["x"]), 6), "y": round(float(r["y"]), 6)} for _, r in df.iterrows()]

def build_rolling_series(rolling):
    return [{"date": idx.date().isoformat(), "correlation": round(float(v), 4) if not np.isnan(v) else None} for idx, v in rolling.items()]

def build_insight(stat):
    r = stat["pearson_r"]
    r2 = round(stat["r_squared"] * 100, 1)
    sig = stat["is_significant"]
    sig_note = "Statistically significant (p < 0.05)." if sig else "Not statistically significant (p >= 0.05)."

    if abs(r) >= 0.70:
        headline = f"{'Strong positive' if r >= 0 else 'Strong negative'} relationship"
        body = f"These assets move {'together' if r >= 0 else 'in opposite directions'} very consistently. {r2}% of variance explained. {sig_note}"
    elif abs(r) >= 0.40:
        headline = f"{'Moderate positive' if r >= 0 else 'Moderate negative'} relationship"
        body = f"A meaningful relationship (r={r:+.2f}). {r2}% shared variance. {sig_note}"
    elif abs(r) >= 0.20:
        headline = f"{'Low positive' if r >= 0 else 'Low negative'} relationship"
        body = f"A weak relationship (r={r:+.2f}). Only {r2}% shared variance. {sig_note}"
    else:
        headline = "No meaningful relationship"
        body = f"With r={r:+.2f}, essentially no linear relationship. {sig_note}"

    return {"headline": headline, "body": body, "label": stat["pearson_label"], "is_significant": sig, "r_squared_pct": r2}

def _label(r):
    a = abs(r)
    d = "positive" if r >= 0 else "negative"
    if a >= 0.70:
        return f"strong {d}"
    elif a >= 0.40:
        return f"moderate {d}"
    elif a >= 0.20:
        return f"low {d}"
    return "very weak"
