"""O grafo de notícias em t como matrizes (seção 18.3 do método).

`Events` carrega os eventos limpos (news.graph_events, src <> dst) uma vez;
`Events.snapshot(t, tickers)` agrega a janela [t - window, t) com decaimento
0,5^(idade/τ) e devolve um `Snapshot` com:

  W      k x k, soma com sinal dos eventos entre dois nós do universo
  Wpos   módulo dos eventos positivos; Wneg dos negativos
  Wg     módulo por grupo de classe (GROUPS)
  Wn     nº de eventos entre os dois nós
  A      k x M (esparsa), soma com sinal dos eventos entre o nó i do universo
         e o agente m fora dele; Aabs módulo; An contagem
  idf    log(N/df_m): df = nós com ticker (no grafo todo) ligados a m na janela
  m      eventos de cada nó do universo (com qualquer outro nó)
  docs   incidência documento x nó (co-ocorrência no artigo extraído)

k = nós distintos do universo (GOOG e GOOGL viram o mesmo nó); `expand`
leva uma matriz k x k para as p ações. Ação sem nó no grafo fica com linha 0.

Nada usa eventos com data >= t (o grafo não tem hora: dt < t).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

import numpy as np
from scipy import sparse

from ..portfolio.estimators import nearest_corr

GROUPS = {
    "cadeia": ("supply_chain",),
    "parceria": ("partnership",),
    "concorrencia": ("competition",),
    "legal": ("legal_regulatory",),
    "controle": ("ma", "ownership"),
    "impacto": ("impact",),
}
OUTROS = "outros"
GROUP_NAMES = (*GROUPS, OUTROS)
NORMS = ("idf", "bruta", "grau", "pmi", "z")
EPOCH = date(1970, 1, 1)
WRAP_MIN = 5          # artigo com >= 5 tickers distintos = "market wrap" (robustez da seção 18.8)


def _day(d) -> int:
    return (d - EPOCH).days if isinstance(d, date) else int(np.datetime64(d, "D").astype(int))


class Events:
    def __init__(self, con, mentions_glob: str | None = None):
        ev = con.execute("""
            SELECT CAST(dt AS DATE) AS dt, doc_id, src, dst, rel_class, sign, strength
            FROM news.graph_events WHERE src <> dst ORDER BY dt""").fetchnumpy()
        nodes = con.execute("SELECT node_id, ticker, tickers, kind FROM news.graph_nodes").fetchall()
        names = np.array([n for n, *_ in nodes], dtype=object)
        self.node = {n: k for k, n in enumerate(names)}
        self.names = names
        self.is_ticker = np.array([t is not None for _, t, _, _ in nodes])
        self.kind = np.array([k or "" for *_, k in nodes], dtype=object)
        # ticker de preço -> nó (o próprio, ou o nó cujas classes de ação o incluem)
        self.ticker_node: dict[str, int] = {}
        for n, t, ts, _ in nodes:
            for x in [t, *(ts or [])]:
                if x:
                    self.ticker_node.setdefault(x, self.node[n])
        for n in names:                       # o nó com o próprio nome tem precedência
            if n in self.ticker_node and self.is_ticker[self.node[n]]:
                self.ticker_node[n] = self.node[n]
        self.dt = ev["dt"].astype("datetime64[D]").astype(np.int64)
        self.src = np.array([self.node[x] for x in ev["src"]], np.int64)
        self.dst = np.array([self.node[x] for x in ev["dst"]], np.int64)
        docs, self.doc = np.unique(ev["doc_id"].astype(str), return_inverse=True)
        self.n_docs = len(docs)
        self.sign = ev["sign"].astype(float)
        self.strength = ev["strength"].astype(float)
        cls = ev["rel_class"].astype(str)
        gmap = {c: g for g, cs in GROUPS.items() for c in cs}
        self.group = np.array([GROUP_NAMES.index(gmap.get(c, OUTROS)) for c in cls], np.int64)
        self.rel_class = cls
        # nº de nós com ticker distintos por artigo (market wrap: >= WRAP_MIN)
        tk = np.concatenate([np.stack([self.doc[self.is_ticker[self.src]], self.src[self.is_ticker[self.src]]], 1),
                             np.stack([self.doc[self.is_ticker[self.dst]], self.dst[self.is_ticker[self.dst]]], 1)])
        tk = np.unique(tk, axis=0)
        self.doc_ntk = np.bincount(tk[:, 0], minlength=self.n_docs)
        self._mentions = None
        self._mentions_glob = mentions_glob

    def window(self, t, window_days: int) -> slice:
        hi = np.searchsorted(self.dt, _day(t), side="left")
        lo = np.searchsorted(self.dt, _day(t) - window_days, side="left")
        return slice(int(lo), int(hi))

    def nodes_of(self, tickers: list[str]) -> np.ndarray:
        return np.array([self.ticker_node.get(x, -1) for x in tickers], np.int64)

    def snapshot(self, t, tickers: list[str], tau: float = 90.0, window_days: int = 365,
                 agents: str = "todos", exclude_wrap: bool = False) -> "Snapshot":
        """Grafo em t para as ações `tickers`. agents: "todos" (qualquer nó fora
        do universo) ou "sem_ticker" (só nós sem ticker: Fed, petróleo...).
        exclude_wrap: ignora artigos com >= WRAP_MIN tickers distintos."""
        w = self.window(t, window_days)
        sl = np.arange(w.start, w.stop)
        if exclude_wrap:
            sl = sl[self.doc_ntk[self.doc[sl]] < WRAP_MIN]
        s, d = self.src[sl], self.dst[sl]
        age = _day(t) - self.dt[sl]
        dec = 0.5 ** (age / tau)
        mag = self.strength[sl] * dec
        val = self.sign[sl] * mag
        grp = self.group[sl]

        snode = self.nodes_of(tickers)
        un = np.unique(snode[snode >= 0])
        k = len(un)
        pos = np.full(len(self.names), -1, np.int64)
        pos[un] = np.arange(k)
        st = np.where(snode >= 0, pos[np.maximum(snode, 0)], -1)
        ps, pd_ = pos[s], pos[d]
        ins, ind = ps >= 0, pd_ >= 0

        # 1. arestas diretas (os dois sentidos somados, simétricas)
        dm = ins & ind
        a, b = ps[dm], pd_[dm]

        def sym(v):
            M = np.zeros((k, k))
            np.add.at(M, (a, b), v)
            M = M + M.T
            np.fill_diagonal(M, 0.0)
            return M

        W, Wn = sym(val[dm]), sym(np.ones(dm.sum()))
        Wpos, Wneg = sym(np.where(val[dm] > 0, mag[dm], 0.0)), sym(np.where(val[dm] < 0, mag[dm], 0.0))
        Wg = {g: sym(np.where(grp[dm] == gi, mag[dm], 0.0)) for gi, g in enumerate(GROUP_NAMES)}

        # 2. exposição: exatamente uma ponta no universo; o agente é a outra
        em = ins ^ ind
        row = np.where(ins[em], ps[em], pd_[em])
        other = np.where(ins[em], d[em], s[em])
        if agents == "sem_ticker":
            keep = ~self.is_ticker[other]
            em_idx = np.flatnonzero(em)[keep]
            row, other = row[keep], other[keep]
        else:
            em_idx = np.flatnonzero(em)
        ag, col = np.unique(other, return_inverse=True)
        M = len(ag)
        shp = (k, M)

        def spm(v):
            return sparse.csr_matrix((v, (row, col)), shape=shp)

        A, Aabs, An = spm(val[em_idx]), spm(mag[em_idx]), spm(np.ones(len(em_idx)))
        for X_ in (A, Aabs, An):
            X_.sum_duplicates()
            X_.sort_indices()
        assert (A.indptr == An.indptr).all() and (A.indices == An.indices).all()
        # IDF: nós com ticker (no grafo todo) ligados a cada agente na janela
        tk_s, tk_d = self.is_ticker[s], self.is_ticker[d]
        pairs = np.concatenate([np.stack([s[tk_s], d[tk_s]], 1), np.stack([d[tk_d], s[tk_d]], 1)])
        pairs = np.unique(pairs, axis=0) if len(pairs) else pairs.reshape(0, 2)
        n_tk = len(np.unique(pairs[:, 0])) if len(pairs) else 0
        dfa = np.zeros(len(self.names))
        if len(pairs):
            np.add.at(dfa, pairs[:, 1], 1)
        df = dfa[ag]
        idf = np.log(np.maximum(n_tk, 1) / np.maximum(df, 1))

        # 3. menções (eventos de cada nó do universo) e documentos
        m = np.bincount(ps[ins], minlength=k) + np.bincount(pd_[ind], minlength=k)
        dd = np.concatenate([self.doc[sl][ins], self.doc[sl][ind]])
        dn = np.concatenate([ps[ins], pd_[ind]])
        D = sparse.csr_matrix((np.ones(len(dd)), (dd, dn)), shape=(self.n_docs, k))
        D.data[:] = 1.0                       # incidência 0/1 (vários eventos no mesmo artigo)
        D.sum_duplicates()
        D.data[:] = 1.0
        return Snapshot(t=t, tickers=list(tickers), st=st, un=un, W=W, Wn=Wn, Wpos=Wpos, Wneg=Wneg, Wg=Wg,
                        A=A, Aabs=Aabs, An=An, agents=ag, idf=idf, m=m.astype(float), docs=D,
                        agent_kind=self.kind[ag])

    # -------------------------------------------------- co-menção do FNSPID (P6)
    def prepare_mentions(self, con, tickers: list[str]) -> None:
        """Carrega (uma vez) as marcações do FNSPID cujos símbolos caem nos nós
        de `tickers` (todas as ações que podem entrar no universo)."""
        nodes = set(self.nodes_of(tickers).tolist()) - {-1}
        syms = sorted(x for x, n in self.ticker_node.items() if n in nodes)
        lit = ", ".join("'" + x.replace("'", "''") + "'" for x in syms)
        r = con.execute(f"""SELECT CAST(published_at AS DATE) AS d, doc_id, symbol
                            FROM read_parquet('{self._mentions_glob}')
                            WHERE published_at >= TIMESTAMP '2009-01-01' AND symbol IN ({lit})""").fetchnumpy()
        _, doc = np.unique(r["doc_id"].astype(str), return_inverse=True)
        node = np.array([self.ticker_node[x] for x in r["symbol"].astype(str)], np.int64)
        order = np.argsort(r["d"].astype("datetime64[D]").astype(np.int64), kind="stable")
        self._mentions = (r["d"].astype("datetime64[D]").astype(np.int64)[order], doc[order], node[order])

    def comention(self, t, tickers: list[str], window_days: int = 365) -> np.ndarray:
        """p x p: artigos do FNSPID marcados com os dois tickers (símbolo -> nó
        do grafo); a diagonal guarda os artigos de cada ação. Exige prepare_mentions."""
        dts, doc, node = self._mentions
        lo = np.searchsorted(dts, _day(t) - window_days, side="left")
        hi = np.searchsorted(dts, _day(t), side="left")
        snode = self.nodes_of(tickers)
        un = np.unique(snode[snode >= 0])
        pos = np.full(len(self.names), -1, np.int64)
        pos[un] = np.arange(len(un))
        col = pos[node[lo:hi]]
        ok = col >= 0
        st = np.where(snode >= 0, pos[np.maximum(snode, 0)], -1)
        if not ok.any():
            return np.zeros((len(tickers), len(tickers)))
        ud, di = np.unique(doc[lo:hi][ok], return_inverse=True)
        D = sparse.csr_matrix((np.ones(ok.sum()), (di, col[ok])), shape=(len(ud), len(un)))
        D.sum_duplicates()
        D.data[:] = 1.0
        return _expand((D.T @ D).toarray(), st, diag=None)


def _expand(M: np.ndarray, st: np.ndarray, diag: float | None = 0.0) -> np.ndarray:
    """k x k (nós) -> p x p (ações); ação sem nó: linha e coluna 0."""
    p = len(st)
    out = np.zeros((p, p))
    ok = st >= 0
    ix = np.flatnonzero(ok)
    out[np.ix_(ix, ix)] = M[np.ix_(st[ok], st[ok])]
    if diag is not None:
        np.fill_diagonal(out, diag)
    return out


def sgnlog(x: np.ndarray) -> np.ndarray:
    return np.sign(x) * np.log1p(np.abs(x))


def cosine_rows(B) -> np.ndarray:
    """B B' com linhas normalizadas (esparsa ou densa) -> k x k denso."""
    if sparse.issparse(B):
        nrm = np.sqrt(np.asarray(B.multiply(B).sum(axis=1)).ravel())
        inv = np.divide(1.0, nrm, out=np.zeros_like(nrm), where=nrm > 0)
        Bn = sparse.diags(inv) @ B
        G = (Bn @ Bn.T).toarray()
    else:
        nrm = np.linalg.norm(B, axis=1)
        Bn = B / np.where(nrm > 0, nrm, 1)[:, None]
        G = Bn @ Bn.T
    return G


@dataclass
class Snapshot:
    t: object
    tickers: list[str]
    st: np.ndarray
    un: np.ndarray
    W: np.ndarray
    Wn: np.ndarray
    Wpos: np.ndarray
    Wneg: np.ndarray
    Wg: dict
    A: sparse.csr_matrix
    Aabs: sparse.csr_matrix
    An: sparse.csr_matrix
    agents: np.ndarray
    idf: np.ndarray
    m: np.ndarray
    docs: sparse.csr_matrix
    agent_kind: np.ndarray
    cache: dict = field(default_factory=dict)

    @property
    def p(self) -> int:
        return len(self.tickers)

    def expand(self, M: np.ndarray, diag: float | None = 0.0) -> np.ndarray:
        return _expand(M, self.st, diag)

    # ---------------------------------------------- pares (p x p, diagonal 0)
    def direct(self, what: str = "sgnlog") -> np.ndarray:
        """sgnlog (sinal x log(1+|w|)), pos, neg (log(1+·) dos módulos),
        n (contagem), raw, ou um grupo de GROUP_NAMES (log(1+módulo))."""
        if what == "sgnlog":
            return self.expand(sgnlog(self.W))
        if what == "raw":
            return self.expand(self.W)
        if what == "pos":
            return self.expand(np.log1p(self.Wpos))
        if what == "neg":
            return self.expand(np.log1p(self.Wneg))
        if what == "n":
            return self.expand(self.Wn)
        return self.expand(np.log1p(self.Wg[what]))

    def mentions(self) -> np.ndarray:
        """Eventos de cada ação (0 sem nó)."""
        return np.where(self.st >= 0, self.m[np.maximum(self.st, 0)], 0.0)

    def cooccur(self, diag: bool = False) -> np.ndarray:
        """Artigos extraídos em comum; diag=True mantém o nº de artigos de cada ação."""
        C = (self.docs.T @ self.docs).toarray()
        return self.expand(C, diag=None if diag else 0.0)

    def exposure(self, norm: str = "idf", kind: str = "sinal") -> sparse.csr_matrix:
        """Ã (k x M) antes da normalização das linhas. kind: "sinal" (padrão),
        "abs" (módulos) ou "bin" (1 onde há evento). Normalizações (18.3):
          idf    peso log(N/df_m) por agente (padrão)
          bruta  sem peso
          grau   1/√grau do agente (soma dos módulos no universo)
          pmi    sinal médio x max(PMI, 0) da contagem ação-agente
          z      sinal médio x max(z, 0) contra o modelo de configuração
        """
        base = {"sinal": self.A, "abs": self.Aabs, "bin": (self.An > 0).astype(float)}[kind]
        if norm == "idf":
            return (base @ sparse.diags(self.idf)).tocsr()
        if norm == "bruta":
            return base.tocsr()
        if norm == "grau":
            deg = np.asarray(self.Aabs.sum(axis=0)).ravel()
            return (base @ sparse.diags(np.divide(1, np.sqrt(deg), out=np.zeros_like(deg), where=deg > 0))).tocsr()
        if norm in ("pmi", "z"):
            # A, Aabs e An têm a mesma estrutura (mesmas coordenadas, csr canônica)
            An = self.An
            row = np.repeat(np.arange(An.shape[0]), np.diff(An.indptr))
            col = An.indices
            ci = np.asarray(An.sum(axis=1)).ravel()
            cm = np.asarray(An.sum(axis=0)).ravel()
            e0 = ci[row] * cm[col] / max(ci.sum(), 1)
            if norm == "pmi":
                sc = np.maximum(np.log(An.data / np.maximum(e0, 1e-12)), 0)
            else:
                sc = np.maximum((An.data - e0) / np.sqrt(np.maximum(e0, 1e-12)), 0)
            sg = {"sinal": self.A.data / np.maximum(self.Aabs.data, 1e-12),
                  "abs": np.ones(len(sc)), "bin": np.ones(len(sc))}[kind]
            return sparse.csr_matrix((sg * sc, (row, col)), shape=An.shape)
        raise ValueError(f"normalização {norm!r}: use uma de {NORMS}")

    def projection(self, norm: str = "idf", kind: str = "sinal", B=None) -> np.ndarray:
        """C^G = cosseno das linhas de Ã (p x p, PSD, diagonal 1). Ação sem
        exposição: linha 0 fora da diagonal. `B` (k x M) substitui Ã (placebos)."""
        key = ("proj", norm, kind)
        if B is None and key in self.cache:
            return self.cache[key]
        G = cosine_rows(self.exposure(norm, kind) if B is None else B)
        C = self.expand(G, diag=1.0)
        if B is None:
            self.cache[key] = C
        return C


# ------------------------------------------------------------------ kernels PSD

def diffusion(W: np.ndarray, beta: float = 1.0) -> np.ndarray:
    """Kernel de difusão no Laplaciano assinado L = D̄ - W, D̄ = diag(Σ|w|)
    (Kunegis et al. 2010), normalizado para correlação. Sempre PD."""
    L = np.diag(np.abs(W).sum(axis=1)) - W
    lam, U = np.linalg.eigh((L + L.T) / 2)
    K = (U * np.exp(-beta * lam)) @ U.T
    d = np.sqrt(np.diag(K))
    return K / np.outer(d, d)


def higham(W: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """Correlação válida mais próxima de I + α W/max|W| (Higham 2002, aproximado)."""
    mx = np.abs(W).max()
    R = np.eye(len(W)) + (alpha * W / mx if mx > 0 else 0)
    return nearest_corr(R)


# ------------------------------------------------------------------ placebos

def _weighted(snap: Snapshot, norm: str, kind: str = "sinal") -> sparse.csr_matrix:
    return snap.exposure(norm, kind).tocsr()


def placebo_p1(snap: Snapshot, norm: str, rng) -> sparse.csr_matrix:
    """P1: os valores não nulos de Ã em posições sorteadas (sem repetição)."""
    B = _weighted(snap, norm).tocoo()
    k, M = B.shape
    n = len(B.data)
    cells = rng.choice(k * M, size=n, replace=False)
    return sparse.csr_matrix((B.data, (cells // M, cells % M)), shape=(k, M))


def curveball(B: sparse.csr_matrix, rng, trades: int | None = None) -> sparse.csr_matrix:
    """P2: rewiring do bipartido com grau de cada linha e de cada coluna
    preservados (curveball, Strona et al. 2014). O valor viaja com a aresta."""
    B = B.tocsr()
    k = B.shape[0]
    rows = [dict(zip(B.indices[B.indptr[i]:B.indptr[i + 1]].tolist(), B.data[B.indptr[i]:B.indptr[i + 1]].tolist()))
            for i in range(k)]
    trades = trades or 5 * k
    pairs = rng.integers(0, k, size=(trades, 2))
    for a, b in pairs:
        if a == b:
            continue
        ra, rb = rows[a], rows[b]
        ea = [c for c in ra if c not in rb]
        eb = [c for c in rb if c not in ra]
        if not ea and not eb:
            continue
        pool = ea + eb
        vals = [ra[c] for c in ea] + [rb[c] for c in eb]
        order = rng.permutation(len(pool))
        for c in ea:
            del ra[c]
        for c in eb:
            del rb[c]
        na = len(ea)
        for j, o in enumerate(order):
            (ra if j < na else rb)[pool[o]] = vals[o]
    r = np.concatenate([np.full(len(x), i) for i, x in enumerate(rows)]) if k else np.array([])
    c = np.concatenate([np.fromiter(x.keys(), int, len(x)) for x in rows]) if k else np.array([])
    v = np.concatenate([np.fromiter(x.values(), float, len(x)) for x in rows]) if k else np.array([])
    return sparse.csr_matrix((v, (r.astype(int), c.astype(int))), shape=B.shape)


def placebo_p2(snap: Snapshot, norm: str, rng) -> sparse.csr_matrix:
    return curveball(_weighted(snap, norm), rng)


def placebo_p3(C: np.ndarray, sector: np.ndarray, rng) -> np.ndarray:
    """P3: permuta as ações dentro do setor (FF12; sem setor = um grupo)."""
    perm = np.arange(len(C))
    for s in np.unique(sector):
        ix = np.flatnonzero(sector == s)
        perm[ix] = rng.permutation(ix)
    return C[np.ix_(perm, perm)]


def placebo_p4(snap: Snapshot, norm: str, rng) -> sparse.csr_matrix:
    """P4: sinais aleatórios nas entradas de Ã (topologia e módulos preservados)."""
    B = _weighted(snap, norm).tocoo()
    return sparse.csr_matrix((B.data * rng.choice([-1.0, 1.0], len(B.data)), (B.row, B.col)), shape=B.shape)


def count_cosine(C: np.ndarray) -> np.ndarray:
    """P6: cosseno de vetores de incidência a partir da matriz de contagens
    em comum (diagonal = nº de documentos de cada ação)."""
    d = np.sqrt(np.diag(C))
    out = np.divide(C, np.outer(d, d), out=np.zeros_like(C, dtype=float), where=np.outer(d, d) > 0)
    np.fill_diagonal(out, 1.0)
    return out
