# Seção 18: resultados

Os resultados do programa pré-registrado na seção 18 de
`docs/metodo_covariancia.md` (commit 200bdd5) estão aqui, avaliados contra os
critérios fixados lá. As hipóteses G1–G13 correspondem às H1–H13 do roteiro
`grafo_noticias_covariancia.md`. Os números exploratórios, vistos depois dos
testes, estão marcados como tal e não alteram nenhum veredito.

- **Fase 1** (2026-10-26):
  - auditoria;
  - amostras;
  - descritivas;
  - testes diádicos;
  - G4;
  - grades do dev.
  
  Log: `logs/secao18_fase1.log`.
- **Fase 2**:
  - holdout 2019–2023, rodado uma vez a partir de 2026-10-28 13:36 UTC;
  - placebos;
  - robustez;
  - avaliação.
  
  Log: `logs/secao18_fase2.log`. As escolhas do dev estavam congeladas nesse
  momento: o sha256 de `logs/secao18_escolhas.json` era `ee9e01ad…ef91b82`.

## 1. Resumo

| hipótese | pergunta | resultado | veredito |
|---|---|---|---|
| G1 | a aresta direta com sinal prevê correlação residual futura? | β = −0,0016 (t = −9,1); p = 1 | **rejeitada** |
| G2 | o efeito das arestas positivas é maior que o das negativas? | diferença t = −7,3 (negativas > positivas) | **rejeitada** |
| G3 | as classes de relação têm efeitos diferentes? | Wald = 77,9; p QAP = 0,005 | **confirmada** |
| G4 | a exposição a agentes prevê a co-reação ao FOMC? | γ = 0,0029 (t = 1,8) | fraca |
| G6 | o efeito depende da cobertura e do tamanho? | interação t = 4,7 | descritiva |
| G8 | a notícia antecede a correlação? | à frente t = −2,2; reverso t = 0,1 | **rejeitada** |
| G9 | a centralidade prevê o beta de baixa futuro? | t = 2,9; p = 0,002 | **confirmada** |
| G5, G7, G10–G13 | carteiras e placebos no holdout | fase 2 | pendente |

Testes de permutação (QAP) com 200 permutações. Neles, o menor p possível é
1/201 ≈ 0,005. A correção de Holm na família primária sai da avaliação
(fase 2).

## 2. Extração (etapa 1)

### 2.1 Auditoria de presença no texto

Entre as 1.063.603 tuplas, 740.467 (70%) têm texto disponível. O texto das
unidades no Drive é o da amostra de 25%.

| | literal | por tokens |
|---|---|---|
| total | 72,8% | 98,2% |
| maior (partnership) | 83,5% | 99,4% |
| menor (capital_return) | 67,4% | 92,1% |

Por classe:

| classe | tuplas | literal | tokens |
|---|---|---|---|
| impact | 341.934 | 69,5% | 97,6% |
| other | 149.433 | 71,0% | 98,5% |
| competition | 97.837 | 81,0% | 98,6% |
| ownership | 78.072 | 70,5% | 98,7% |
| ma | 74.396 | 79,9% | 98,9% |
| supply_chain | 64.610 | 72,9% | 98,3% |
| partnership | 47.246 | 83,5% | 99,4% |
| legal_regulatory | 43.448 | 70,4% | 98,7% |
| stance | 34.988 | 69,0% | 98,2% |
| analyst | 32.980 | 75,0% | 97,4% |
| business | 30.685 | 73,8% | 98,1% |
| financing | 24.590 | 67,6% | 98,3% |
| monetary | 21.151 | 68,9% | 95,9% |
| governance | 9.756 | 77,3% | 99,4% |
| trade_policy | 9.706 | 74,6% | 97,2% |
| capital_return | 2.771 | 67,4% | 92,1% |

Quase todo agente aparece no texto quando se compara por tokens. A diferença
para a comparação literal vem de nomes normalizados pelo LLM ("Apple Inc."
contra "Apple").

### 2.2 Amostras para anotação (pendentes)

Arquivos em `data/anotacao/` (ver `LEIA-ME.txt`):

- 800 tuplas para os anotadores A e B;
- recall em 63 unidades, das 100 previstas;
- anonimização em 218 unidades, das 300 previstas.

Faltam:

- a anotação manual (tarefa do usuário);
- a reextração anonimizada, que exige GPU e autorização.

Precisão, recall, κ e a taxa de vazamento de memória só saem depois disso.

## 3. Descritivas do grafo (etapa 2)

- **Portão:** o R² médio da projeção g sobre o setor é 0,017 em 156 meses.
  O critério era < 0,5, então passa: g não é um proxy de setor.
- **Sobreposição com o setor:** cerca de metade das arestas entre ações liga
  empresas do mesmo FF12 (47–54% por ano), contra cerca de 10% dos pares em
  geral. No FF49 são 31–39% das arestas, contra cerca de 4% dos pares.
- **Fração de arestas negativas por ano:** de 41% a 53%.
- **Estabilidade:** o Jaccard ano a ano das arestas entre tickers fica entre
  0,12 e 0,17.
- **Cobertura:** cresce com o tamanho. No quintil menor, 73–86% das ações têm
  algum evento, contra cerca de 97% no maior.
- **Grau:** mediana 1 em todos os anos (cauda longa, com p90 = 6).

Detalhes em `logs/secao18_descritivas.json`.

## 4. Testes diádicos

Painel: S&P 500 point-in-time, 155 meses (2011–2023), cerca de 350–460 ações
por mês. y é a correlação residual futura em z (h = 21 pregões).

O método é Fama-MacBeth com regressores padronizados por mês, erro-padrão
Newey-West e QAP. Controles:

- correlação histórica (z_hist);
- mesmo FF49;
- mesmo SIC2;
- distância de tamanho;
- soma e diferença de atenção.

### 4.1 G1: aresta direta com sinal

| regressor | coef. | t |
|---|---|---|
| **sgnlog(w)** | **−0,0016** | **−9,1** |
| g (projeção bipartida) | +0,0038 | +11,1 |
| z_hist | +0,105 | +16,7 |
| mesmo FF49 | +0,013 | +14,4 |
| mesmo SIC2 | +0,011 | +15,3 |
| distância de tamanho | −0,0018 | −4,2 |
| atenção (soma) | 0,0000 | +0,1 |
| atenção (diferença) | −0,0002 | −0,6 |

- O critério de sucesso falha nas três condições:
  - o sinal é negativo nos três subperíodos (2011–14: −0,0017; 2015–18:
    −0,0014; 2019–23: −0,0016);
  - intra-setor: t = −3,2;
  - inter-setor: t = −8,4.
- Com h = 63 e h = 126, o coeficiente também é negativo (t = −8,5 e −7,3).
- **Rejeitada.**

**Diagnóstico (exploratório, visto depois do teste).**

| especificação | coef. | t |
|---|---|---|
| só sgnlog(w) | −0,0111 | −30,4 |
| \|w\| + controles | +0,0027 | +12,2 |
| log(1+w⁺) + controles | +0,0009 | +6,7 |
| log(1+w⁻) + controles | +0,0025 | +11,8 |

- Só 1,0% dos pares tem aresta direta: 0,41% positivas e 0,60% negativas.
- Os dois tipos de aresta aumentam a correlação futura, e as negativas
  aumentam mais. As negativas são principalmente `competition`, com sinal
  médio −0,87: rivais do mesmo ramo, que andam juntos.
- Por isso o sinal entra com a direção trocada: o sinal extraído pelo LLM
  não indica a direção da correlação.
- A existência e a intensidade do vínculo preveem correlação (|w| com
  t = +12; g com t = +11), além de setor e histórico.

### 4.2 G2: assimetria positiva × negativa

| | coef. | t |
|---|---|---|
| positivas | +0,0010 | +6,7 |
| negativas | +0,0025 | +11,7 |
| diferença (pos − neg) | −0,0015 | −7,3 |

A hipótese era β⁺ > β⁻ e saiu o contrário. A hipótese auxiliar β⁻ < 0
também falha (t = +11,7). **Rejeitada.**

### 4.3 G3: classes de relação

| grupo | coef. | t | p BH |
|---|---|---|---|
| concorrência | +0,0022 | +9,9 | < 10⁻²¹ |
| parceria | +0,0004 | +4,7 | < 10⁻⁴ |
| controle (participação, M&A) | +0,0007 | +4,1 | 0,0001 |
| cadeia de suprimentos | +0,0006 | +3,1 | 0,003 |
| outros | +0,0004 | +2,9 | — |
| impacto | +0,0003 | +1,9 | 0,065 |
| legal | −0,0000 | −0,2 | 0,81 |

- O Wald de igualdade dá 77,9 (p χ² < 10⁻¹⁴; p QAP = 0,005). **Confirmada.**
- As expectativas registradas se confirmam: cadeia > 0, parceria pequena e
  legal ≤ 0.

### 4.4 G6: cobertura e tamanho (descritivo)

- A interação sgnlog × cobertura dá +0,0021 (t = 4,7).
- Coeficiente de sgnlog por tercil de tamanho médio do par:

| tercil | coef. | t |
|---|---|---|
| pequenas | −0,0012 | −5,7 |
| médias | −0,0017 | −7,7 |
| grandes | −0,0021 | −8,3 |

Com o diagnóstico da 4.1: quanto maior a empresa, mais o termo com sinal
reflete a concorrência entre grandes do mesmo ramo.

### 4.5 G8: direção temporal

| direção | coef. | t |
|---|---|---|
| Δw_t → Δz futura (à frente) | −0,0005 | −2,2 |
| Δz_t → Δw futura (reverso) | +0,00004 | +0,1 |

Mudanças na rede de notícias não antecedem mudanças na correlação. O
coeficiente à frente tem sinal contrário ao esperado, e o reverso é nulo.
**Rejeitada.**

### 4.6 G9: centralidade e risco de baixa

Regressão em corte transversal mensal, com tamanho, beta passado e dummies
FF12.

| variável dependente | coef. | t |
|---|---|---|
| **beta de baixa futuro** | **+0,011** | **+2,9** (p = 0,002) |
| beta futuro | +0,007 | +2,4 |
| co-assimetria futura | −0,0004 | −0,2 |
| beta de baixa, centralidade PageRank* | +0,015 | +3,1 |
| beta de baixa, centralidade grau* | +0,017 | +3,3 |

\* Nas centralidades alternativas, os meses sem beta futuro (fim da
amostra) não são descartados e entram com coeficiente 0. Isso viesa esses
dois números levemente para zero. O p registrado não é afetado. O defeito
foi visto depois do teste e não foi corrigido.

**Confirmada** na especificação registrada.

### 4.7 Robustez dos testes diádicos (sem QAP)

| painel | G1 sgnlog (t) | g (t) | G2 diferença (t) | G3 Wald | G6 interação (t) | G8 à frente (t) | G9 t (p) |
|---|---|---|---|---|---|---|---|
| **τ = 90, IDF (registrado)** | −9,1 | 11,1 | −7,3 | 77,9 | 4,7 | −2,2 | 2,9 (0,002) |
| τ = 30 | −8,9 | 12,1 | −7,2 | 72,4 | 3,1 | −1,5 | 2,2 (0,015) |
| τ = 180 | −9,3 | 10,2 | −7,6 | 90,8 | 5,2 | −2,3 | 3,1 (0,001) |
| τ = 365 | −9,4 | 9,7 | −8,0 | 104,2 | 5,3 | −2,1 | 2,9 (0,002) |
| norma bruta | −9,6 | 10,8 | −7,7 | 87,0 | 5,0 | −2,2 | 3,0 (0,001) |
| norma grau | −9,2 | 11,4 | −7,5 | 81,8 | 4,7 | −2,2 | 0,7 (0,25) |
| norma PMI | −9,5 | 11,9 | −7,8 | 88,3 | 4,6 | −2,2 | 2,3 (0,010) |
| norma z | −9,0 | 11,6 | −7,3 | 79,4 | 4,5 | −2,2 | 0,5 (0,31) |

- Os vereditos de G1, G2, G3 e G8 não mudam em nenhum painel.
- O G9 depende da normalização da exposição: some com a norma grau e com a
  norma z, e se mantém nas demais.
- O G8 à frente não muda entre normalizações: a aresta direta não depende
  da norma.

## 5. G4: dias de FOMC

Foram 104 reuniões, de 2011-01 a 2023-12. A surpresa é a variação do DGS2
no dia do comunicado.

| | coef. | t |
|---|---|---|
| exposição comum ao Fed (γ) × surpresa | +0,0029 | +1,8 |
| correlação histórica (δ) × surpresa | −0,023 | −3,0 |

- Spread Q5 − Q1 de exposição:
  - correlação com a surpresa: 0,21;
  - acerto de sinal: 41%.
- O efeito vai na direção esperada, mas não chega a t = 2. **Não confirmada**
  pelo critério registrado.

## 6. Dev (2013–2018): escolhas congeladas

| estimador | critério | escolha | observação |
|---|---|---|---|
| M2 | variância GMV | τ = 180, norma z | 20 tentativas; var. GMV 4,09·10⁻⁵ |
| B8 / M3 (POET) | variância GMV | C = 0,5 | var. GMV 3,15·10⁻⁵ |
| M5 | QLIKE | q = 10 | |
| M1 | QLIKE | κ = 1, c = 0,1 | κ muda a QLIKE em só 10⁻⁵ |
| M2+W | variância GMV | α = 0,25, β = 2 | |
| M4 / B9 | QLIKE | λ₀ = 0,2, γ = 2 | borda da grade |
| M8b | HRP | λ = 0,25 | única configuração |

- No dev, o POET teve variância GMV cerca de 23% menor que o melhor M2.
- O prior de notícias do M1 quase não muda a perda.
- Como o dev não é o teste, esses números não decidem nada: o veredito das
  carteiras é o do holdout.

## 7. Holdout, placebos e avaliação (fase 2)

*Em execução. Preencher com `logs/secao18_avaliacao.json` (G5, G7,
G10–G13, Holm).*

## 8. Desvios e correções declarados

- **Correções depois do commit 200bdd5**, só na leitura de dados:
  - `amostra`: KeyError em `unit_id`;
  - `anonimizacao`: passou a ler de `news.relations_linked`.
- **Cobertura da etapa 1:** texto para 70% das tuplas; 63 unidades de
  recall; 218 de anonimização.
- **G9:** no resultado secundário das centralidades alternativas, os meses
  vazios entram com coeficiente 0 (ver 4.6).
- **Commit:** a nota do dev em `metodo_covariancia.md` ainda não estava
  commitada quando o holdout começou. O hash das escolhas no início do
  holdout está registrado acima.
