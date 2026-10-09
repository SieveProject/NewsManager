"""Marca tuplas ruidosas da extração -- marca, não apaga.

Cada relação recebe uma coluna `noise` com o primeiro motivo que se aplica, ou
NULL se passou limpa. A análise escolhe o que excluir; os dados brutos ficam
intactos e uma regra pode ser revista sem nova extração.

Motivos, em ordem de prioridade (o primeiro que casa vence):

  self_loop   agent_a e agent_b são o mesmo agente ("DuPont expands DuPont")
  source      um lado é a fonte ou provedor de dados, não um agente econômico
              (Zacks, ETF Channel, Capital IQ...)
  metric      um lado é um item contábil da própria empresa (revenue, gross
              margin, EPS, dividend, stock price) -- desempenho, não relação
  generic     um lado é um grupo sem nome (investors, analysts, customers...)
  orgchart    nomeação/contratação de pessoa ("Boeing appoints John Wojick")
  comparison  desempenho relativo, não causal ("X outperforms S&P 500")
  membership  composição de índice/ETF ("Nasdaq 100 includes Tesla"), inclusive
              com verbo genérico ("Technology SPDR affects ADP")

Calibrado em amostras de 10-15 tuplas por motivo sobre as 882.914 relações da
run v=a05d0c0dc189 (qwen2.5:14b-instruct). Precisão estimada: self_loop,
source, membership, comparison ~10/10; metric e generic ~14/15; orgchart após
restringir a alvos com cara de nome de pessoa. Duas armadilhas evitadas de
propósito -- não as reintroduza:

  * Indicadores macro NÃO são métricas: "retail sales -> pound", "Fed ->
    dividend stocks", "CDS traders -> European government debt" são relações
    válidas. Por isso `metric` exige que o agente *termine* num item contábil
    de empresa e não aceita sales/debt/income/yield soltos.
  * Setores e agregados NÃO são genéricos: "banks", "industry", "economy",
    "government" são agentes que o prompt aceita.
  * "hires" com banco assessor é relação real ("Georgian Railways hires JP
    Morgan"); `orgchart` só pega hires/employs quando o alvo parece pessoa.

Não resolve sinônimos ("Fed", "the Fed", "Federal Reserve"): isso é resolução
de entidades, outra etapa.
"""

from __future__ import annotations

import duckdb

from ..config import Config

REASONS: dict[str, str] = {
    "self_loop": "agent_a e agent_b são o mesmo agente",
    "source": "fonte ou provedor de dados como agente",
    "metric": "item contábil da empresa como agente",
    "generic": "grupo genérico sem nome como agente",
    "orgchart": "nomeação ou contratação de pessoa",
    "comparison": "desempenho relativo, não causal",
    "membership": "composição de índice ou ETF",
}

# Excluídos por padrão em news.relations_clean. `comparison` e `membership` são
# informação de mercado legítima para algumas perguntas; ficam marcados e a
# análise decide -- basta filtrar news.relations_flagged diretamente.
DEFAULT_EXCLUDE = tuple(REASONS)

GENERIC = (
    "investors", "investor", "analysts", "analyst", "shareholders", "shareholder",
    "stockholders", "market", "markets", "stock market", "shares", "stock", "stocks",
    "u.s. stocks", "us stocks", "equities", "wall street", "consumers", "consumer",
    "customers", "customer", "clients", "employees", "workers", "traders", "buyers",
    "sellers", "users", "people", "investors and analysts", "bondholders", "creditors",
    "lenders", "borrowers", "homeowners", "savers", "retail investors",
    "institutional investors", "advertisers", "partners", "the company", "management",
    "board", "board of directors",
)

SOURCE = (
    r"(?i)(zacks|capital iq|etf channel|stock options channel|bnk invest|motley fool"
    r"|benzinga|seeking alpha|marketwatch|thestreet|investorplace|tipranks|nasdaq\.com"
    r"|validea|wall street journal|^bloomberg( news)?$|^reuters$|^cnbc$|^the street$)"
)

# Item contábil no FIM do nome, com até 3 qualificadores antes ("adjusted
# operating income", "non-GAAP diluted EPS", "advertising revenues").
METRIC = (
    r"(?i)^([^ ]+ ){0,3}(revenue|revenues|net revenue|sales growth|same-store sales"
    r"|comparable sales|net sales|ebitda|ebit|eps|earnings per share|net income"
    r"|operating income|fee income|profit|profits|operating profit|gross profit|margin"
    r"|margins|gross margin|operating margin|dividend|dividends|dividend payout|guidance"
    r"|cash flow|free cash flow|expenses|operating expenses|net loss|loss per share"
    r"|stock price|share price|backlog|bookings|book value|market cap"
    r"|market capitalization)$"
)

# Marcadores de empresa: protegem "Realty Income Corp.", "Fixed Income ETF".
COMPANY = (
    r"(?i)(\binc\b|\bcorp|\bco\.|\bltd\b|\bplc\b|\bllc\b|\bgroup\b|\bholdings\b|\betf\b"
    r"|\bfund\b|\btrust\b|\(|\bbank\b|\bn\.?v\.?\b|\bs\.?a\.?\b|\bag\b|\breit\b)"
)

# Nomeação de pessoa: sempre organograma.
ORG_ALWAYS = (
    r"(?i)^(appoints|names|named|elects|is led by|steps down from|resigns from|ceo of"
    r"|chairs|nominates)$"
)
# Contratação: só quando o alvo parece pessoa -- banco assessor é relação real.
ORG_HIRES = r"(?i)^(hires|hired|employs)$"
PERSON = r"^[A-Z][a-zA-Z'.-]+( [A-Z][a-zA-Z'.-]+){1,2}$"
NOT_PERSON = (
    r"(?i)(morgan|goldman|citi|bank|securities|capital|partners|advisors|advisers"
    r"|associates|management|lazard|barclays|ubs|credit suisse|consult|group|holdings)"
)

COMPARE = (
    r"(?i)^(outperforms|underperforms|outperformed|underperformed|lags|lagged|trails"
    r"|trailed|beats the market|is outperformed by|outpaces)$"
)
MEMBER = (
    r"(?i)^(includes|is part of|is included in|is a component of|ranks within"
    r"|belongs to|is a member of|is constituent of)$"
)

# ETF/fundo "afetando" as próprias posições é composição de carteira com outro
# verbo: "Technology Select Sector SPDR affects ADP", "ProShares UltraPro Dow30
# affects American Express" passavam limpos com o verbo genérico.
FUND = r"(?i)(\betf\b|spdr|proshares|ishares|direxion|invesco qqq|\bindex fund\b)"
FUND_VERBS = r"(?i)^(affects|has exposure to|holds|weighs on|lifts|drags|drives)$"


def _q(s: str) -> str:
    return s.replace("'", "''")


def classify_sql(a: str = "agent_a", b: str = "agent_b", rt: str = "relation_type") -> str:
    """SQL CASE com o motivo de ruído (ou NULL) para uma linha de relações."""
    norm = lambda x: f"regexp_replace(lower(trim({x})), '^the ', '')"  # noqa: E731
    generic = ", ".join(f"'{_q(g)}'" for g in GENERIC)
    metric = lambda x: (  # noqa: E731
        f"(regexp_matches(trim({x}), '{_q(METRIC)}') AND NOT regexp_matches({x}, '{_q(COMPANY)}'))"
    )
    return f"""CASE
        WHEN {norm(a)} = {norm(b)} THEN 'self_loop'
        WHEN regexp_matches({a}, '{_q(SOURCE)}') OR regexp_matches({b}, '{_q(SOURCE)}') THEN 'source'
        WHEN {metric(a)} OR {metric(b)} THEN 'metric'
        WHEN {norm(a)} IN ({generic}) OR {norm(b)} IN ({generic}) THEN 'generic'
        WHEN regexp_matches(trim({rt}), '{_q(ORG_ALWAYS)}') THEN 'orgchart'
        WHEN regexp_matches(trim({rt}), '{_q(ORG_HIRES)}')
             AND regexp_matches(trim({b}), '{_q(PERSON)}')
             AND NOT regexp_matches({b}, '{_q(NOT_PERSON)}') THEN 'orgchart'
        WHEN regexp_matches(trim({rt}), '{_q(COMPARE)}') THEN 'comparison'
        WHEN regexp_matches(trim({rt}), '{_q(MEMBER)}') THEN 'membership'
        WHEN regexp_matches(trim({rt}), '{_q(FUND_VERBS)}')
             AND (regexp_matches({a}, '{_q(FUND)}') OR regexp_matches({b}, '{_q(FUND)}')) THEN 'membership'
    END"""


def build(cfg: Config, prompt_version: str, exclude: tuple[str, ...] = DEFAULT_EXCLUDE) -> dict:
    """Grava relations_flagged.parquet e as views news.relations_flagged/_clean."""
    unknown = set(exclude) - set(REASONS)
    if unknown:
        raise ValueError(f"motivos desconhecidos: {sorted(unknown)}; válidos: {list(REASONS)}")
    out_dir = cfg.curated / "relations" / f"v={prompt_version}"
    src = out_dir / "relations.parquet"
    if not src.exists():
        raise FileNotFoundError(f"{src} não existe; rode `collect` antes")
    dst = out_dir / "relations_flagged.parquet"

    con = duckdb.connect()
    con.execute(
        f"COPY (SELECT *, {classify_sql()} AS noise FROM read_parquet('{src}')) "
        f"TO '{dst}' (FORMAT parquet, COMPRESSION zstd)"
    )
    total = con.execute(f"SELECT count(*) FROM read_parquet('{dst}')").fetchone()[0]
    counts = dict(con.execute(
        f"SELECT coalesce(noise, 'clean'), count(*) FROM read_parquet('{dst}') GROUP BY 1"
    ).fetchall())
    con.close()

    db = duckdb.connect(str(cfg.db))
    db.execute("CREATE SCHEMA IF NOT EXISTS news")
    db.execute(f"CREATE OR REPLACE VIEW news.relations_flagged AS SELECT * FROM read_parquet('{dst}')")
    excl = ", ".join(f"'{r}'" for r in exclude)
    db.execute(
        "CREATE OR REPLACE VIEW news.relations_clean AS SELECT * EXCLUDE (noise) "
        f"FROM news.relations_flagged WHERE noise IS NULL"
        + (f" OR noise NOT IN ({excl})" if excl else "")
    )
    db.close()

    kept = counts.get("clean", 0) + sum(n for r, n in counts.items() if r != "clean" and r not in exclude)
    print(f"[filter] {total:,} relações em v={prompt_version}")
    for r in ("clean", *REASONS):
        n = counts.get(r, 0)
        tag = "" if r == "clean" else ("  (excluída)" if r in exclude else "  (mantida)")
        print(f"  {r:11} {n:9,}  {100 * n / total:5.1f}%{tag}")
    print(f"[filter] news.relations_clean: {kept:,} relações ({100 * kept / total:.1f}%)")
    print(f"[filter] {dst}")
    return {"total": total, "kept": kept, **counts}
