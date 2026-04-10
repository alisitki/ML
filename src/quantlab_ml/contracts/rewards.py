from __future__ import annotations

from datetime import datetime

from pydantic import Field, model_validator

from quantlab_ml.contracts.common import QuantBaseModel


class RewardEventSpec(QuantBaseModel):
    horizon_steps: int = 1
    fee_bps: float
    slippage_bps: float
    risk_aversion: float
    turnover_penalty: float
    funding_weight: float
    timestamping: str
    realized_event: str
    infeasible_action_penalty: float = -0.001


class ActionReward(QuantBaseModel):
    action_key: str
    gross_return: float
    fee: float
    funding: float
    slippage: float
    risk_penalty: float
    turnover_penalty: float
    net_reward: float
    applicable: bool = True
    # V2: hangi venue üzerinden hesaplandığı; None ise abstain veya legacy
    venue: str | None = None


# ---------------------------------------------------------------------------
# V2: Venue-Aware Reward Context
# ---------------------------------------------------------------------------


class VenueExecutionRef(QuantBaseModel):
    """Decision-time'da bilinen, tek bir exchange için execution referansı.

    Bu yapı **yalnızca geçmiş ve karar-anı bilgisi** içerir; future fiyat
    veya sonraki reward timestamp buraya girmez.
    """

    exchange: str
    reference_price: float
    fee_regime_bps: float
    slippage_proxy_bps: float
    funding_rate: float
    funding_freshness_seconds: float  # funding verisinin yaşı (saniye)


class RewardContext(QuantBaseModel):
    """Her TrajectoryStep için venue-aware ekonomik durum.

    Yalnızca decision-time'da bilinen verilerden türetilir.
    """

    venues: dict[str, VenueExecutionRef] = Field(default_factory=dict)  # exchange → ref
    hold_horizon_steps: int = 1
    turnover_state: float = 0.0  # birikimli turnover skorü
    previous_position_state: str = "flat"  # flat | long | short
    selected_venue: str | None = None  # policy'nin seçtiği venue (uygulanmamışsa None)

    @model_validator(mode="after")
    def validate_selected_venue(self) -> "RewardContext":
        if self.selected_venue is not None and self.selected_venue not in self.venues:
            raise ValueError(
                f"selected_venue '{self.selected_venue}' is not in venues dict"
            )
        return self


class RewardTimeline(QuantBaseModel):
    """Horizon boyunca reward replay için yeterli venue-aware referans serisi.

    Yalnızca `horizon_steps` adım taşır; tam backtest replay motoru bu
    milestone kapsamında değildir.
    """

    horizon_steps: int
    # exchange → fiyat listesi; her liste horizon_steps uzunluğunda
    venue_reference_series: dict[str, list[float]] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_series_length(self) -> "RewardTimeline":
        for exchange, series in self.venue_reference_series.items():
            if len(series) != self.horizon_steps:
                raise ValueError(
                    f"venue_reference_series[{exchange}] length {len(series)} "
                    f"!= horizon_steps {self.horizon_steps}"
                )
        return self


# ---------------------------------------------------------------------------
# RewardSnapshot — scalar legacy alanlar korunuyor, V2 context + timeline eklendi
# ---------------------------------------------------------------------------


class RewardSnapshot(QuantBaseModel):
    event_time: datetime
    # Legacy-compat scalar alanlar; yeni kod RewardContext'i kullanır.
    # Bu alanlar compatibility adapter dışında kullanılmamalıdır.
    reference_price: float = 0.0
    next_price: float = 0.0
    action_rewards: list[ActionReward] = Field(default_factory=list)
    # V2 alanları
    context: RewardContext | None = None
    timeline: RewardTimeline | None = None

    def for_action(self, action_key: str) -> ActionReward:
        """action_key ile ilk eşleşen ActionReward'u döner.

        Guarantee: 'abstain' için her zaman çalışır (abstain venue=None ile oluşturulur).
        Directional action'lar için venue=None fallback kaydı artık üretilmiyor;
        bunun yerine `RewardEngine._select_best_reward()` veya
        `for_action_venue(action_key, venue)` kullanın.
        """
        for reward in self.action_rewards:
            if reward.action_key == action_key:
                return reward
        raise KeyError(f"unknown action reward: {action_key}")

    def for_action_venue(self, action_key: str, venue: str) -> ActionReward:
        """Belirli bir venue için action reward'unu döner.

        Bulunamazsa `for_action(action_key)` ile fallback yapar (abstain için çalışır).
        Directional action'lar için venue yoksa KeyError olabilir.
        """
        for reward in self.action_rewards:
            if reward.action_key == action_key and reward.venue == venue:
                return reward
        # Abstain için her zaman fallback çalışır; directional için burada exception olabilir.
        return self.for_action(action_key)
