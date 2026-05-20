"""
Ornstein-Uhlenbeck process — continuous mean-reverting dynamics.

Used to model how intimacy and conflict drift over time:
- Without events, values decay toward a baseline
- Events can push values away from baseline
- The "spring" force (theta) pulls values back

Math:
    dX = theta * (mu - X) * dt + sigma * dW

where:
    X = current value (intimacy or conflict)
    mu = long-term mean (baseline)
    theta = speed of mean reversion (how fast it returns to baseline)
    sigma = volatility (how much randomness)
    dW = Wiener process (Brownian motion)
"""

from __future__ import annotations

import math
import random


class OUProcess:
    """
    Ornstein-Uhlenbeck process for a single emotional dimension.

    Example:
        >>> ou = OUProcess(baseline=0.5, reversion_speed=0.1, volatility=0.05)
        >>> ou.value  # starts at baseline
        0.5
        >>> ou.bump(0.3)  # event pushes it up
        >>> ou.step(dt=1.0)  # one time step
        >>> ou.value  # drifted back toward 0.5
    """

    def __init__(
        self,
        baseline: float = 0.5,
        reversion_speed: float = 0.1,
        volatility: float = 0.05,
        seed: int | None = None,
    ):
        """
        Args:
            baseline: Long-term mean (mu). Value drifts toward this.
            reversion_speed: How fast it returns to baseline (theta). Higher = faster.
            volatility: Random noise magnitude (sigma).
            seed: Random seed for reproducibility.
        """
        self.baseline = baseline
        self.reversion_speed = reversion_speed
        self.volatility = volatility
        self.value = baseline

        self._rng = random.Random(seed)

    def bump(self, amount: float) -> None:
        """Push the value away from baseline (event-driven)."""
        self.value = max(0.0, min(1.0, self.value + amount))

    def step(self, dt: float = 1.0) -> float:
        """
        Advance one time step.

        Uses Euler-Maruyama discretization:
            X(t+dt) = X(t) + theta*(mu - X(t))*dt + sigma*sqrt(dt)*N(0,1)

        Args:
            dt: Time step size.

        Returns:
            New value after the step.
        """
        drift = self.reversion_speed * (self.baseline - self.value) * dt
        noise = self.volatility * math.sqrt(dt) * self._rng.gauss(0, 1)
        self.value += drift + noise

        # Clamp to [0, 1] for emotional dimensions
        self.value = max(0.0, min(1.0, self.value))
        return self.value

    def reset(self, value: float | None = None) -> None:
        """Reset to baseline or a specific value."""
        self.value = value if value is not None else self.baseline

    def __repr__(self) -> str:
        return (
            f"OUProcess(value={self.value:.3f}, "
            f"baseline={self.baseline:.3f}, "
            f"theta={self.reversion_speed:.3f})"
        )
