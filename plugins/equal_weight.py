"""Plugin `plugins/equal_weight.py:EqualWeight`.

1/N do universo elegível (o teto de peso nunca morde quando N >= 1/max_weight)."""

import numpy as np

from newsmanager.backtester.plugin import Context, Model


class EqualWeight(Model):
    def weights(self, ctx: Context) -> np.ndarray:
        return np.full(ctx.p, 1 / ctx.p)
