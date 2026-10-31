"""Transaction-cost model for simulated positions (analysis only; nothing is executed).

All rates are in basis points of traded notional, charged on |change in position|
(position is a fraction of equity, so a trade from 0 to 1 is 1.0 of notional).

    cost_t = |dpos_t| * (commission + half_spread + slippage + impact_k * sigma_t) / 1e4
           + |pos_t| * 1{pos_t < 0} * borrow_bps_per_year / 1e4 / periods_per_year
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CostModel:
    commission_bps: float = 1.0      # broker commission, per side
    half_spread_bps: float = 1.0     # cross half the bid/ask spread, per side
    slippage_bps: float = 2.0        # fixed extra slippage, per side
    vol_slippage_k: float = 0.0      # extra slippage in bps per unit of daily vol (in bps), e.g. 0.05
    borrow_bps_per_year: float = 0.0 # cost of holding short positions
    periods_per_year: int = 252

    @property
    def per_side_bps(self) -> float:
        return self.commission_bps + self.half_spread_bps + self.slippage_bps

    def trade_cost(self, dpos: pd.Series, vol: pd.Series | None = None) -> pd.Series:
        bps = pd.Series(self.per_side_bps, index=dpos.index, dtype=float)
        if self.vol_slippage_k and vol is not None:
            bps = bps + self.vol_slippage_k * vol.reindex(dpos.index).fillna(0.0) * 1e4
        return dpos.abs() * bps / 1e4

    def holding_cost(self, pos: pd.Series) -> pd.Series:
        if not self.borrow_bps_per_year:
            return pd.Series(0.0, index=pos.index)
        return pos.clip(upper=0).abs() * self.borrow_bps_per_year / 1e4 / self.periods_per_year

    def to_dict(self): return asdict(self)


ZERO = CostModel(0.0, 0.0, 0.0)
