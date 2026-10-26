"""G4 (seção 18.9, exploratório): a exposição de notícias ao Fed prediz a
sensibilidade das ações nos dias de FOMC?

    python -m newsmanager.grafo fomc        # monta data/reference/fomc_dates.csv (calendário do Fed)
    python -m newsmanager.grafo g4

Dia de evento = dia do comunicado (último dia da reunião, ou o 1º pregão
depois dele), reuniões agendadas e extraordinárias de 2011 a 2023 (sem
"notation vote", conference call e reunião cancelada). Surpresa s = ΔDGS2 do
dia (FRED), em pontos percentuais.

Em cada dia τ, transversal nas ações de U_τ:
    r_iτ - β̂_i r_mτ = α_τ + γ_τ ã_i,Fed + δ_τ β̂_i,Fed + e
com β̂_i (mercado) e β̂_i,Fed (beta a -ΔDGS2) dos 252 pregões anteriores e
ã_i,Fed = exposição com sinal ao nó FEDERAL_RESERVE (padronizada). Segundo
passo: γ_τ = c0 + c·s_τ + u (c ≠ 0: a exposição das notícias muda a
resposta à surpresa); e o mesmo para δ_τ (o controle histórico).
Carteira: spread Q5 - Q1 de ã_i,Fed no dia contra o sinal da surpresa.
"""

from __future__ import annotations

import json
import re
from datetime import date

import numpy as np

from .. import query
from ..config import Config
from ..portfolio import data
from . import dados, inferencia as inf, rede

HIST = "https://www.federalreserve.gov/monetarypolicy/fomchistorical{y}.htm"
CURRENT = "https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm"
MONTHS = {m: k for k, m in enumerate(["January", "February", "March", "April", "May", "June", "July", "August",
                                      "September", "October", "November", "December"], start=1)}
ABBR = {m[:3]: k for m, k in MONTHS.items()}


def _last_day(year: int, month_txt: str, days_txt: str) -> date:
    """'Apr/May' + '30-1' -> 1º de maio; 'January' + '27-28' -> 28 de janeiro."""
    months = month_txt.split("/")
    last_m = months[-1]
    m = MONTHS.get(last_m) or ABBR[last_m[:3]]
    d = int(re.findall(r"\d+", days_txt)[-1])
    return date(year, m, d)


def build_fomc(cfg: Config, first: int = 2011, last: int = 2023) -> list[date]:
    import httpx
    out: set[date] = set()
    cur = httpx.get(CURRENT, timeout=60).text
    for y in range(first, last + 1):
        r = httpx.get(HIST.format(y=y), timeout=60)
        if r.status_code == 200:
            for txt in re.findall(r"<h5[^>]*>([^<]*)</h5>", r.text):
                m = re.match(r"\s*([A-Za-z/]+)\s+([\d\-]+)\s*(\((\w+)\))?\s*Meeting\s*-\s*(\d{4})", txt)
                if not m or (m.group(4) or "").lower() == "cancelled":
                    continue
                out.add(_last_day(int(m.group(5)), m.group(1), m.group(2)))
            continue
        # anos recentes: página de calendários (blocos "<ano> FOMC Meetings")
        blocks = re.split(r"(\d{4}) FOMC Meetings", cur)
        for k in range(1, len(blocks), 2):
            if int(blocks[k]) != y:
                continue
            pairs = re.findall(r"<strong>([A-Za-z/]+)</strong>.*?fomc-meeting__date[^>]*>([^<]*)", blocks[k + 1], re.S)
            for mon, dd in pairs:
                low = dd.lower()
                if "notation" in low or "cancel" in low or "call" in low:
                    continue
                out.add(_last_day(y, mon, dd))
    dates = sorted(out)
    path = cfg.root / "reference" / "fomc_dates.csv"
    path.write_text("date\n" + "\n".join(str(d) for d in dates) + "\n")
    print(f"[fomc] {len(dates)} reuniões ({dates[0]} a {dates[-1]}): {path}")
    return dates


def g4(cfg: Config, tau: float = 90.0) -> dict:
    con = query.connect(cfg)
    pn = dados.Panel(cfg, con)
    ev = rede.Events(con)
    con.close()
    ds = dados.fomc_dates(cfg)
    dgs2 = dados.fred_changes(cfg, "DGS2", pn.days)
    days = np.array(pn.days, dtype="datetime64[D]")
    gam, dlt, surp, spread = [], [], [], []
    for d in ds:
        i = int(np.searchsorted(days, np.datetime64(d, "D")))
        if i >= len(pn.days) or i < 300 or not np.isfinite(dgs2[i]):
            continue
        U = pn.universe(i)
        X = pn.window(i, U)
        m = data.market_factor(X)
        _, b = data.market_betas(X, m)
        f = -np.nan_to_num(dgs2[i - 252:i])
        bf = np.array([np.cov(np.nan_to_num(X[:, j]), f)[0, 1] / f.var() for j in range(len(U))])
        r = pn.R[i, [pn.col[x] for x in U]]
        rm = np.nanmean(r)
        snap = ev.snapshot(pn.days[i], U, tau=tau)
        ix = {ev.names[a]: k for k, a in enumerate(snap.agents)}
        if "FEDERAL_RESERVE" not in ix:
            continue
        col = np.asarray(snap.A[:, ix["FEDERAL_RESERVE"]].todense()).ravel()
        a = np.where(snap.st >= 0, col[np.maximum(snap.st, 0)], 0.0)
        ok = np.isfinite(r)
        if a[ok].std() == 0:
            continue
        y = r[ok] - b[ok] * rm
        za = (a[ok] - a[ok].mean()) / a[ok].std()
        zb = (bf[ok] - bf[ok].mean()) / bf[ok].std()
        X2 = np.column_stack([np.ones(ok.sum()), za, zb])
        c, *_ = np.linalg.lstsq(X2, y, rcond=None)
        gam.append(c[1])
        dlt.append(c[2])
        surp.append(dgs2[i])
        q = np.quantile(za, [0.2, 0.8])
        spread.append(y[za >= q[1]].mean() - y[za <= q[0]].mean())
    gam, dlt, surp, spread = map(np.array, (gam, dlt, surp, spread))

    def second(v):
        X = np.column_stack([np.ones(len(surp)), surp])
        bb, *_ = np.linalg.lstsq(X, v, rcond=None)
        e = v - X @ bb
        V = inf.nw_cov(X * e[:, None], 1) * len(v)
        Ai = np.linalg.inv(X.T @ X / len(v))
        se = np.sqrt(np.diag(Ai @ V @ Ai) / len(v))
        return {"c": float(bb[1]), "t": float(bb[1] / se[1])}

    res = {"eventos": int(len(gam)), "gamma_em_surpresa": second(gam), "delta_hist_em_surpresa": second(dlt),
           "spread_Q5_Q1": {"corr_com_surpresa": float(np.corrcoef(spread, surp)[0, 1]),
                            "acerto_sinal": float(np.mean(np.sign(spread) == np.sign(-surp)))}}
    out = cfg.root.parent / "logs" / "secao18_g4.json"
    out.write_text(json.dumps(res, indent=2, ensure_ascii=False))
    print(json.dumps(res, indent=2, ensure_ascii=False))
    return res
