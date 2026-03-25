from app.models.schemas import RiskBatchProfileRequest, RiskDistribution, RiskProfileRequest
from app.services.analysis_service import run_risk_profile, run_risk_profiles


def test_run_risk_profile_accepts_bist_symbol():
    response = run_risk_profile(
        RiskProfileRequest(
            symbol="THYAO.IS",
            lookback=252,
            interval="1d",
            distribution=RiskDistribution.student_t,
            refresh="if_stale",
        )
    )

    assert response.meta.symbol == "THYAO.IS"
    assert response.meta.exchange == "BIST"
    assert response.scores.risk >= 0
    assert response.garch.alpha_plus_beta < 1
    assert len(response.history.conditional_volatility) > 0


def test_run_risk_profiles_returns_requested_symbols():
    response = run_risk_profiles(
        RiskBatchProfileRequest(
            symbols=["TSLA", "AKBNK.IS", "TSLA"],
            lookback=252,
            interval="1d",
            distribution=RiskDistribution.student_t,
            refresh="if_stale",
        )
    )

    symbols = [profile.meta.symbol for profile in response.profiles]
    assert symbols == ["TSLA", "AKBNK.IS"]
    assert all(profile.garch.alpha_plus_beta < 1 for profile in response.profiles)
