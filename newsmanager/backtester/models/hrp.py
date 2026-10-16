"""Hierarchical Risk Parity (López de Prado, 2016): ligação simples na distância
√(½(1-ρ)) da correlação de Σ̂, quasi-diagonalização e bissecção recursiva com
variância inversa. Não inverte Σ̂. O teto de peso é imposto por projeção
(registrada em info_hrp_projection_l1).

    params = { cov = "LW fator único" }
"""

import numpy as np

from ...portfolio.optimize import hrp, project_box_simplex
from .base import CovModel


class HRP(CovModel):
    def optimize(self, S, ctx):
        w = hrp(S)
        v = project_box_simplex(w, ctx.min_weight, ctx.max_weight)
        ctx.info["hrp_projection_l1"] = float(np.abs(v - w).sum())
        return v
