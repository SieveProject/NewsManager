"""Peças da seção 18 (newsmanager/grafo) contra casos conhecidos. Sem dados reais.

    python tests/test_grafo.py
"""

from __future__ import annotations

import sys
import tempfile
from datetime import date, timedelta
from pathlib import Path

import duckdb
import numpy as np
from scipy import sparse

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from newsmanager.grafo import estimadores as ge, inferencia as inf, rede  # noqa: E402
from newsmanager.grafo.programa import FastShrink  # noqa: E402
from newsmanager.portfolio import estimators as est  # noqa: E402


def make_graph_db(path: Path) -> None:
    """Grafo mínimo: 3 ações (AAA, BBB, CCC), 2 agentes (FED, OIL), eventos datados."""
    con = duckdb.connect(str(path))
    con.execute("CREATE SCHEMA news")
    con.execute("""CREATE TABLE news.graph_nodes (node_id VARCHAR, ticker VARCHAR, tickers VARCHAR[], kind VARCHAR)""")
    con.executemany("INSERT INTO news.graph_nodes VALUES (?, ?, ?, ?)",
                    [("AAA", "AAA", ["AAA", "AAA2"], "company"), ("BBB", "BBB", ["BBB"], "company"),
                     ("CCC", "CCC", ["CCC"], "company"), ("FED", None, None, "central_bank"),
                     ("OIL", None, None, "commodity"), ("ZZZ", "ZZZ", ["ZZZ"], "company")])
    con.execute("""CREATE TABLE news.graph_events (dt DATE, doc_id VARCHAR, src VARCHAR, dst VARCHAR,
                   rel_class VARCHAR, sign TINYINT, strength DOUBLE)""")
    ev = [
        (date(2020, 1, 10), "d1", "AAA", "BBB", "supply_chain", 1, 0.8),
        (date(2020, 1, 20), "d2", "BBB", "AAA", "competition", -1, 0.5),
        (date(2020, 1, 15), "d3", "FED", "AAA", "monetary", -1, 1.0),
        (date(2020, 1, 15), "d3", "FED", "BBB", "monetary", -1, 1.0),
        (date(2020, 1, 16), "d4", "CCC", "OIL", "impact", 1, 0.6),
        (date(2020, 1, 16), "d5", "ZZZ", "OIL", "impact", 1, 0.6),
        (date(2020, 2, 1), "d6", "AAA", "CCC", "partnership", 1, 1.0),     # = t: não pode entrar
        (date(2018, 1, 1), "d7", "AAA", "CCC", "partnership", 1, 1.0),     # fora da janela de 365 dias
    ]
    con.executemany("INSERT INTO news.graph_events VALUES (?, ?, ?, ?, ?, ?, ?)", ev)
    con.close()


def main() -> int:
    rng = np.random.default_rng(0)

    # 1. snapshot: janela [t - 365, t), simetria, decaimento, IDF, classes de ação
    with tempfile.TemporaryDirectory() as tmp:
        db = Path(tmp) / "g.duckdb"
        make_graph_db(db)
        con = duckdb.connect(str(db), read_only=True)
        ev = rede.Events(con)
        con.close()
        t = date(2020, 2, 1)
        s = ev.snapshot(t, ["AAA", "BBB", "CCC", "AAA2"], tau=1e12)
        W = s.direct("raw")
        assert np.allclose(W, W.T)
        assert np.isclose(W[0, 1], 0.8 - 0.5) and W[0, 2] == 0           # evento em t e de 2018 fora
        assert np.isclose(W[0, 3], 0) and np.allclose(W[3, :3], W[0, :3]) and W[3, 1] != 0  # AAA2 = nó AAA
        assert np.isclose(s.direct("pos")[0, 1], np.log1p(0.8)) and np.isclose(s.direct("neg")[0, 1], np.log1p(0.5))
        assert np.isclose(s.direct("cadeia")[0, 1], np.log1p(0.8))
        G = s.projection("bruta")
        assert np.isclose(G[0, 1], 1.0) and np.isclose(G[0, 2], 0.0) and np.allclose(np.diag(G), 1)
        assert np.isclose(G[0, 3], 1.0)
        # IDF: FED liga 2 nós com ticker (AAA, BBB), OIL liga 2 (CCC, ZZZ); 4 nós com ticker ativos
        ix = {ev.names[a]: k for k, a in enumerate(s.agents)}
        assert np.isclose(s.idf[ix["FED"]], np.log(4 / 2)) and np.isclose(s.idf[ix["OIL"]], np.log(4 / 2))
        s2 = ev.snapshot(t, ["AAA", "BBB"], tau=10.0)
        age = (t - date(2020, 1, 10)).days
        assert np.isclose(s2.direct("raw")[0, 1], 0.8 * 0.5 ** (age / 10) - 0.5 * 0.5 ** (12 / 10))
        assert s.cooccur()[0, 1] == 3 and s.cooccur()[0, 2] == 0            # d1, d2, d3 têm AAA e BBB
        print("ok  snapshot sem look-ahead, simétrico, com decaimento e IDF")

    # 2. curveball preserva graus de linhas e colunas e o valor viaja com a aresta
    B = sparse.random(30, 50, density=0.2, random_state=1, format="csr")
    B.data = rng.standard_normal(len(B.data))
    P = rede.curveball(B, rng, trades=2000)
    deg = lambda M, ax: np.asarray((M != 0).sum(axis=ax)).ravel()  # noqa: E731
    assert (deg(P, 0) == deg(B, 0)).all() and (deg(P, 1) == deg(B, 1)).all()
    for c in range(50):
        assert np.allclose(np.sort(B[:, c].data), np.sort(P[:, c].data))
    assert (P != B).nnz > 0
    sec = np.array([0, 0, 1, 1, 1, 2])
    C = np.arange(36.0).reshape(6, 6)
    C3 = rede.placebo_p3(C, sec, np.random.default_rng(3))
    # cada linha de C3 é a linha de uma ação do mesmo setor
    for i in range(6):
        src = [k for k in range(6) if sec[k] == sec[i] and C3[i, i] == C[k, k]]
        assert src
    print("ok  placebos P2 (curveball) e P3 (dentro do setor)")

    # 3. kernels e FastShrink
    Wm = rng.standard_normal((8, 8))
    Wm = (Wm + Wm.T) / 2
    np.fill_diagonal(Wm, 0)
    K = rede.diffusion(Wm, 1.0)
    assert np.linalg.eigvalsh(K)[0] > 0 and np.allclose(np.diag(K), 1)
    X = rng.standard_normal((120, 10)) @ rng.standard_normal((10, 10)) * 0.01
    R = np.eye(10)
    R[:8, :8] = rede.higham(Wm, 0.4)
    fs = FastShrink(X)
    assert np.allclose(fs(R), est.shrink_to_corr(X, R), atol=1e-14)
    print("ok  kernel de difusão PD e FastShrink = shrink_to_corr")

    # 4. glasso ponderado: condições de KKT
    p = 12
    A = rng.standard_normal((300, p))
    Sc = np.corrcoef(A.T)
    Lam = np.full((p, p), 0.1)
    Lam[0, 1] = Lam[1, 0] = 0.01
    Th, Z, info = ge.glasso(Sc, Lam, tol=1e-7, max_iter=20000)
    G_ = np.linalg.inv(Z) - Sc if np.linalg.eigvalsh(Z)[0] > 0 else np.linalg.inv(Th) - Sc
    off = ~np.eye(p, dtype=bool)
    nz, zz = (Z != 0) & off, (Z == 0) & off
    assert np.allclose(np.diag(np.linalg.inv(Th) - Sc), 0, atol=1e-4)
    assert np.allclose(G_[nz], (Lam * np.sign(Z))[nz], atol=1e-3)
    assert (np.abs(G_[zz]) <= Lam[zz] + 1e-3).all()
    print(f"ok  glasso ponderado satisfaz KKT ({info['iter']} iterações)")

    # 5. POET: limiar enorme = resíduo diagonal = suporte vazio (M3 sem pares)
    X = rng.standard_normal((400, 15))
    Z_ = est.demeaned(X)
    S_big, i_big = ge.poet(X, C=1e6, k=2)
    lam, V = np.linalg.eigh(Z_.T @ Z_ / 400)
    low = (V[:, -2:] * lam[-2:]) @ V[:, -2:].T
    assert np.allclose(S_big - low, np.diag(np.diag(S_big - low)), atol=1e-12) and i_big["density"] == 0
    S_emp, _ = ge.poet(X, support=np.zeros((15, 15), bool), k=2)
    assert np.allclose(S_emp, S_big, atol=1e-12)
    S_0, i_0 = ge.poet(X, C=0.0, k=2)                      # sem limiar: a amostral (se PD)
    assert np.allclose(S_0, Z_.T @ Z_ / 400, atol=1e-12) and i_0["density"] == 1.0
    print("ok  POET nos dois limites")

    # 6. modelos de fatores, alvo por blocos, ERC
    X = rng.standard_normal((250, 20)) * 0.01
    Fm = rng.standard_normal((250, 3)) * 0.01
    S1, _ = ge.ridge_factor(X, Fm, lam=0.0)
    assert np.allclose(S1, est.multi_factor(X, Fm, "diag"), atol=1e-14)
    assert np.allclose(ge.block_target(X, np.zeros(20, int)), est.lw_const_corr(X), atol=1e-14)
    S = np.cov((rng.standard_normal((500, 6)) @ rng.standard_normal((6, 6))).T)
    w = ge.erc(S)
    rc = w * (S @ w)
    assert np.allclose(rc, rc.mean(), rtol=1e-6) and np.isclose(w.sum(), 1)
    print("ok  ridge = multi_factor com λ = 0; blocos de 1 grupo = LW corr. constante; ERC")

    # 7. Louvain, ARI, NMI
    A = np.zeros((20, 20))
    A[:10, :10] = 1
    A[10:, 10:] = 1
    A[0, 10] = A[10, 0] = 0.1
    np.fill_diagonal(A, 0)
    lab = ge.louvain(A)
    truth = np.r_[np.zeros(10), np.ones(10)]
    assert ge.ari(lab, truth) == 1.0 and np.isclose(ge.nmi(lab, truth), 1.0)
    assert abs(ge.ari(rng.integers(0, 3, 2000), rng.integers(0, 3, 2000))) < 0.01
    print("ok  Louvain separa dois cliques; ARI/NMI")

    # 8. inferência
    assert np.allclose(inf.holm(np.array([0.01, 0.04, 0.03, 0.2])), [0.04, 0.09, 0.09, 0.2])
    assert np.allclose(inf.bh(np.array([0.01, 0.04, 0.03, 0.2])), [0.04, 0.16 / 3, 0.16 / 3, 0.2])
    e = rng.standard_normal(3000)
    ar = np.zeros(3000)
    for k in range(1, 3000):
        ar[k] = 0.9 * ar[k - 1] + e[k]
    assert inf.pw_block(ar) > 5 * inf.pw_block(e)
    dm = inf.dm_hln(rng.standard_normal(200) + 0.5)
    assert dm["stat"] > 4 and dm["p_b_better"] < 1e-4
    L = rng.standard_normal((300, 4)) * 0.5
    L[:, 3] += 1.0                                            # o 4º modelo é claramente pior
    pv = inf.mcs(L + 5, ["a", "b", "c", "ruim"], B=500, rng=np.random.default_rng(1))
    assert pv["ruim"] < 0.05 and max(pv.values()) == 1.0
    L0 = rng.standard_normal(500)
    Lm = np.column_stack([L0 + rng.standard_normal(500) * 0.1 + 0.3, L0 - 0.5 + rng.standard_normal(500) * 0.1])
    sp = inf.spa(L0, Lm, B=500, rng=np.random.default_rng(2))
    assert sp["spa_p"] < 0.01 and sp["best"] == 1
    sp = inf.spa(L0, Lm[:, :1], B=500, rng=np.random.default_rng(2))
    assert sp["spa_p"] > 0.2
    tabs = [np.array([[30, 10], [10, 30]]), np.array([[20, 10], [10, 25]])]
    assert inf.cmh(tabs)["z"] > 3
    stat = np.array([3.0, 0.1])
    boot = np.random.default_rng(3).standard_normal((2000, 2))
    adj = inf.romano_wolf(stat, boot)
    assert adj[0] < 0.01 and adj[1] > 0.3
    print("ok  Holm, BH, Politis-White, DM-HLN, MCS, SPA, CMH, Romano-Wolf")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
