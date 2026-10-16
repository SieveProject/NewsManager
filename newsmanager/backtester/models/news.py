"""Compatibilidade: NewsFactorResidual(features, placebo, ...) dos TOMLs da
seção 12 (commit 54c5095) = MinVariance com cov "fator + resíduo <features>"."""

from . import min_variance

_COV = {"setor": "fator + resíduo setor", "noticias": "fator + resíduo notícias",
        "setor+noticias": "fator + resíduo setor + notícias"}


class NewsFactorResidual(min_variance.MinVariance):
    def __init__(self, features: str = "setor+noticias", **cov_params):
        if features not in _COV:
            raise ValueError(f"features {features!r}: use um de {list(_COV)}")
        super().__init__(cov=_COV[features], cov_params=cov_params)
