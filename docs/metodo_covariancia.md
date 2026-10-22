# Grafo de notícias → matriz de covariância: desenho do método

Objetivo da tese: usar as relações extraídas das notícias (`news.graph_events`)
para melhorar a estimativa da matriz de covariância no período *in-sample* de
cada rebalanceamento e medir o efeito em carteiras que dependem dela.

## 1. Hipóteses

- **H1 (estatística).** Os vínculos das notícias até a data t preveem a
  correlação futura entre duas ações *além* do que a correlação histórica já
  prevê.
- **H2 (econômica).** Usar esse sinal como alvo de *shrinkage* reduz a
  volatilidade fora da amostra da carteira de variância mínima (GMV) e melhora
  carteiras que dependem de Σ, em relação aos estimadores de referência.
- **H3 (regime).** O ganho é maior quando a história de preços é pouco
  informativa: janela curta (N/p pequeno) e mudança de regime (2020, 2022).
  Becquin e Esmeir (2023) formulam a mesma hipótese para alvos semânticos.

H1 é condição para H2: se as notícias não preveem correlação, um ganho de
carteira seria só efeito de regularização genérica. Os placebos da seção 5.4
existem para separar uma coisa da outra.

## 2. Base teórica

### 2.1 Estimar Σ: o problema do erro de estimação

- Markowitz (1952): a carteira depende de Σ. Com p ativos e N observações, a
  covariância amostral S tem p(p+1)/2 parâmetros. É singular se N < p e
  mal-condicionada se N não for muito maior que p: o otimizador explora
  justamente os erros (*error maximization*).
- Kan e Zhou (2007): com muitos ativos, o erro em Σ pesa tanto quanto o erro na
  média. Por isso a GMV, que não usa a média, é o laboratório padrão para
  comparar estimadores de Σ.
- **Estrutura ajuda.** Elton e Gruber (1973) mostram que impor correlação
  constante, inclusive por blocos de setor, prevê melhor que a amostral. Chan,
  Karceski e Lakonishok (1999) chegam a conclusão semelhante comparando modelos
  de risco.
- **Shrinkage linear.** Σ̂ = δ·T + (1−δ)·S, com δ ótimo analítico (Ledoit e
  Wolf). Alvos possíveis:
  - identidade escalada (2004, *JMVA*);
  - correlação constante (2004, "Honey, I Shrunk the Sample Covariance
    Matrix", *JPM*);
  - fator único (2003, *JEF*).

  O shrinkage pode ser lido como um Bayes empírico: T é o *prior*. **Nossa
  contribuição entra exatamente aqui, um prior vindo das notícias.**
- **Shrinkage não linear** (Ledoit e Wolf, 2020, *Annals of Statistics*, forma
  analítica) e **limpeza por matrizes aleatórias** (Laloux et al., 1999;
  Bun, Bouchaud e Potters, 2017): referências fortes, que não usam informação
  externa.
- **Graphical lasso** (Friedman, Hastie e Tibshirani, 2008) estima a matriz de
  precisão Σ⁻¹ esparsa, que é o objeto que a GMV usa (w ∝ Σ⁻¹1). A variante
  *weighted glasso* penaliza menos as arestas apoiadas por conhecimento prévio;
  na genômica, o prior são redes biológicas. Há ainda o *tailored glasso*, que
  estima quanto confiar no prior.

### 2.2 Redes em finanças

- Redes de correlação filtradas: MST (Mantegna, 1999) e PMFG (Tumminello et
  al., 2005).
- Ativos periféricos diversificam melhor (Pozzi, Di Matteo e Aste, 2013,
  *Sci. Rep.*). Os pesos de Markowitz são negativamente ligados à centralidade
  na rede (Peralta e Zareei, 2016).
- HRP (López de Prado, 2016) agrupa ativos hierarquicamente pela correlação
  antes de alocar. É um consumidor natural de uma matriz de correlação melhor.
- Lu, Reinert e Cucuringu (2023) estimam a covariância com uma rede externa
  aos retornos (co-negociação): clusters espectrais da rede definem quais
  blocos da covariância idiossincrática ficam. Supera blocos GICS em
  volatilidade e Sharpe. **É o precedente metodológico mais próximo de "rede
  externa → estrutura de Σ"**, embora não use texto.

### 2.3 Vínculos econômicos, texto e notícias

- **Vínculos reais propagam informação**:
  - clientes e fornecedores (Cohen e Frazzini, 2008, *JF*);
  - indústrias ligadas na cadeia (Menzly e Ozbas, 2010, *JF*);
  - similaridade textual de produtos nos 10-K, TNIC (Hoberg e Phillips, 2016,
    *JPE*).
- **Notícias revelam vínculos**:
  - firmas citadas juntas preveem os retornos umas das outras, mesmo sem
    relação cliente-fornecedor (Scherbina e Schlusche, 2016, AEI WP; versão
    "Follow the Leader", *Review of Finance*, 2020);
  - a cobertura conjunta gera transbordamento de atenção e co-movimento
    (Guo, Peng, Tao e Tu, "Joint News, Attention Spillover, and Stock
    Returns").
- **Os três trabalhos que ligam texto a Σ** (o núcleo da revisão):
  1. **Tang, Zhou e Hong (2019, *JRFM* 12(1):45)**, "News Co-Occurrences,
     Stock Return Correlations, and Portfolio Construction Implications". A
     co-ocorrência em notícias prevê a correlação futura mesmo controlando
     pela persistência da correlação. Incorporada à covariância, melhora a
     GMV. É a evidência direta de H1, mas usa só co-menção, sem tipo, sinal ou
     entidades não negociadas.
  2. **Becquin e Esmeir (2023, Findings of EMNLP)**, "Semantic Similarity
     Covariance Matrix Shrinkage" (Bloomberg). Similaridade semântica
     (embeddings de descrições ou de grafo de conhecimento) vira **alvo de
     shrinkage**, não estimador direto, porque a similaridade é descalibrada
     e depende da dimensão do embedding. Construção:
     - Sim = EEᵀ, lida como correlação;
     - SimCov = Sim ⊙ √(diag S · diag Sᵀ);
     - correção de autovalores para ficar PSD;
     - δ ótimo à Ledoit-Wolf.

     Testado em S&P 500, Nasdaq-100 e Nikkei, de 2007 a 2023, com 3 meses de
     estimação e 1 de avaliação. **É o molde técnico que vamos seguir.**
  3. **Lu, Ndiaye e Simaan (2024, *IRFA* 96:103572)**. Encolhem a matriz de
     correlação com aprendizado por reforço usando as redes TNIC de
     Hoberg-Phillips. Em GMV, mais de 400 ações e 20 anos de teste, superam
     os shrinkages padrão em volatilidade, Sharpe e risco de cauda, líquidos
     de custos.
- Linha "LLM → grafo de empresas": grafos extraídos por LLM alimentam GNNs para
  prever retorno (Zhou et al., FinNLP 2025, entre outros). **Não encontrei
  trabalho que use relações extraídas por LLM, tipadas, com sinal e com
  entidades macro, para estimar Σ.** Esta é a lacuna que a tese ocupa.

### 2.4 Avaliação

- **Referência ingênua**: 1/N (DeMiguel, Garlappi e Uppal, 2009, *RFS*). Muitos
  modelos não a batem.
- **Avaliação econômica de previsões de correlação** pela variância da
  carteira: Engle e Colacito (2006, *JBES*).
- **Testes robustos** a caudas pesadas e autocorrelação: Ledoit e Wolf (2008,
  *JEF*) para Sharpe; Ledoit e Wolf (2011, *Wilmott*) para variância, com
  bootstrap studentizado em blocos.
- **Perdas robustas** para comparar previsões de covariância com uma proxy
  ruidosa: Patton (2011).

## 3. O que os nossos dados permitem (medido em 2026-10-14)

- Data só ao dia (`published_at` sempre à meia-noite): o grafo em t usa
  `dt < t` (`newsmanager.graph.snapshot`).
- Densidade ao longo do tempo: arestas ticker-ticker por ano de 1,6 mil (2010)
  a 12 mil (2023), com 2006–2009 quase vazios. **Período útil: 2011–2023.**
- Amostra de 25% dos artigos, a caminho de ~34% (run de ampliação). A
  densidade sobe quase linearmente com a amostra.
- **Arestas diretas são raras.** Em 2018, entre os 100 tickers com mais
  eventos, só 9,7% dos pares têm aresta direta. Mas **57,9% dos pares
  compartilham ao menos uma entidade não negociada**: Fed, petróleo, China,
  "coronavirus", "banks". Com 300 tickers: 2,5% e 24,3%.

**Consequência para o desenho:** o sinal relevante para Σ é a **exposição comum
a entidades**, uma estrutura de fatores semântica, mais do que a aresta direta.
É isso que diferencia o nosso grafo da co-menção de Tang et al. (2019), que só
enxerga pares de tickers.

## 4. O modelo: shrinkage relacional semântico

Em cada data de rebalanceamento t, com universo de p ações e janela in-sample
de N dias:

### 4.1 Atributos de par a partir do grafo (sem look-ahead)

Snapshot com janela W e meia-vida h (`graph.snapshot(con, t, window_days=W,
half_life_days=h)`):

1. **Vínculo direto por classe**: d_ij^c = Σ |w|·decaimento sobre os eventos
   i→j e j→i da classe c. As 16 classes são agrupadas em poucos grupos, por
   exemplo:
   - cadeia (`supply_chain`, `partnership`);
   - rivalidade (`competition`);
   - controle (`ma`, `ownership`);
   - impacto (`impact`);
   - outros.

   Feature: log(1 + d_ij^c).
2. **Exposição comum (fatores semânticos).** Matriz B (ações × entidades não
   negociadas), com B_ik = Σ |w|·decaimento dos eventos entre a ação i e a
   entidade k, nos dois sentidos.
   - Pondera cada entidade por um **IDF**: log(p / nº de ações ligadas a k).
     Assim "S&P 500", que liga tudo, pesa pouco, e "crude oil" pesa para as
     petroleiras.
   - Normaliza as linhas e usa s_ij = cos(B_i, B_j).
   - Variante: um s_ij^k por tipo de entidade (`central_bank`, `commodity`,
     `country`, `sector`, `macro`).
3. **Sinal**: proporção de eventos positivos no vínculo. Hipótese testável: a
   rivalidade pode reduzir a correlação.
4. **Controles**:
   - co-menção bruta no FNSPID (coluna `symbols`): é a variável de Tang et al.;
   - 1[mesmo setor], se obtivermos a classificação setorial.

### 4.2 Ponte semântica → numérica: regressão preditiva de correlação

Com os rebalanceamentos **anteriores** a t (painel empilhado e janela
expansiva), estima-se:

    z(ρ_ij^{fut}) = a + b·z(ρ_ij^{hist}) + Σ_c β_c·log(1+d_ij^c) + γ·s_ij + θ'controles + ε

- z = transformação de Fisher (atanh);
- ρ^{hist} = correlação na janela in-sample;
- ρ^{fut} = correlação realizada no período seguinte de avaliação.

Os β_c e γ são **a tradução, estimada nos dados, de cada tipo de relação semântica
em correlação**. É essa a "ponte" que a tese defende, e ela é interpretável:
dá para mostrar, por exemplo, que `supply_chain` aumenta a correlação e que
`competition` aumenta menos.

**Atribuição limpa.** O mesmo modelo com β = γ = 0 é um alvo de "persistência
de correlação" sem notícias. A diferença entre os dois resultados é o valor das
notícias.

### 4.3 Alvo e shrinkage

1. Correlação-alvo R_T = tanh(ẑ), com diagonal 1, projetada para a matriz de
   correlação mais próxima (Higham, 2002) se não for PSD.
2. Covariância-alvo T = D^½ R_T D^½, com D = diag(S): as variâncias vêm dos
   preços, como em Becquin e Esmeir.
3. Σ̂ = δ·T + (1−δ)·S. O δ ótimo sai da fórmula de Ledoit e Wolf (2004,
   "Honey") com a correlação constante r̄ trocada por r_ij^T **fixo**: o alvo
   não usa os mesmos retornos, exceto as variâncias. A generalização é direta:

       π = Σ_ij AsyVar(√N s_ij)
       ρ = Σ_i π_ii + Σ_{i≠j} (r_ij^T/2)·(√(s_jj/s_ii)·ϑ_ii,ij + √(s_ii/s_jj)·ϑ_jj,ij)
       γ = ‖T − S‖²_F
       δ* = max(0, min(1, (π − ρ)/(N·γ)))

   Alternativa: escolher δ por validação temporal dentro do in-sample.

### 4.4 Variantes (escada de complexidade, para as ablações)

| | alvo | parâmetros | papel |
|---|---|---|---|
| V0 | correlação constante (Ledoit e Wolf) | 1 | referência |
| V1 | **correlação constante por bloco do grafo**: ρ̄ para pares ligados (direta ou exposição s_ij > τ) e ρ̄ para os não ligados, estimados no in-sample | 2–3 | prova de conceito simples, generaliza Elton e Gruber por setor |
| V2 | regressão preditiva da 4.2 | ~10 | **modelo principal** |
| V3 | *weighted graphical lasso*: penalidade λ_ij = λ·exp(−κ·a_ij), com a_ij o vínculo das notícias | 2 | ataca Σ⁻¹ diretamente; robustez |
| V4 | V2 com o alvo encolhido de forma não linear (NLS sobre o resíduo), ou HRP com distância da correlação V2 | — | extensões, se der tempo |

## 5. Desenho experimental

### 5.1 Universo e período

- **Universo point-in-time**: em cada t, as p ações com preço completo na
  janela e mais eventos limpos nos 12 meses anteriores a t. Testar p = 50, 100
  e 200. A seleção usa só o passado. Viés de sobrevivência: o FNSPID só tem os
  tickers que sobreviveram até a coleta; limitação a declarar.
- **Período**:
  - 2011–2013: calibração de W, h, τ e κ;
  - **2014–2023: teste**, com os hiperparâmetros congelados. Isso evita
    *data snooping*.
- **Rebalanceamento** mensal. Janela in-sample de N = 63, 126 e 252 dias úteis.
  N/p variando é o eixo de H3.

### 5.2 Estimadores comparados

Ficam todos no mesmo *pipeline*:

- S (amostral);
- Ledoit e Wolf identidade, correlação constante e fator único;
- Ledoit e Wolf não linear analítico (2020);
- limpeza por matrizes aleatórias (*eigenvalue clipping*);
- **alvo de co-menção** (a ideia de Tang et al.: substituir o grafo LLM pela
  co-ocorrência de `symbols`);
- alvo de persistência (V2 sem notícias);
- **V1, V2 e V3**;
- 1/N.

### 5.3 Carteiras

GMV sem restrição, GMV long-only, paridade de risco (ERC) e máxima
diversificação; HRP como extensão. Nenhuma usa retorno esperado, para isolar o
efeito de Σ.

### 5.4 Métricas e testes

- **Principal**: volatilidade anualizada fora da amostra da GMV, com o teste de
  razão de variâncias de Ledoit e Wolf (2011), bootstrap em blocos.
- **Secundárias**:
  - Sharpe, com o teste de Ledoit e Wolf (2008);
  - drawdown máximo;
  - turnover;
  - retorno líquido de custos (10 bps por lado);
  - número de condição de Σ̂.
- **Estatísticas**:
  - erro de previsão da correlação dos pares (e só dos pares ligados);
  - perdas de Patton (2011) contra a covariância realizada do mês seguinte;
  - Diebold-Mariano entre estimadores.
- **Placebos**, decisivos para a tese:
  1. grafo **embaralhado** com reconexão que preserva os graus: mesma
     esparsidade, arestas erradas;
  2. grafo **deslocado no tempo**: o do ano anterior;
  3. só co-menção.

  Se V2 superar os três, o ganho vem do *conteúdo* das relações, e não da
  estrutura nem da regularização.
- **Subperíodos**: 2014–2019, 2020 (COVID) e 2022 (aperto monetário), para H3.

## 6. Limitações a declarar

- Amostra de artigos (25–34%) e cobertura desigual entre os anos.
- Ruído do LLM: ~70% das tuplas limpas são relações econômicas claras.
- Ligação a tickers com precisão estimada de ~94%.
- Tickers ausentes do painel (UNH, VZ, BRK…) e reuso de ticker (FB).
- Data sem hora (mitigado com `dt < t`).
- Viés de sobrevivência do FNSPID.
- Múltiplas comparações: hiperparâmetros congelados e conjunto de modelos
  pequeno e decidido antes do teste.

## 7. Implementação (próximos passos no código)

1. **Painel de preços**: ler `Stock_price/full_history.zip` do FNSPID por HTTP
   range, só os membros dos tickers do universo; o diretório central já é lido
   no link. Gravar retornos diários em `data/curated/prices/`.
2. `newsmanager/portfolio/features.py`: atributos de par em t (4.1), a partir
   de `news.graph_events`.
3. `newsmanager/portfolio/estimators.py`: S, Ledoit e Wolf (identidade,
   correlação constante, fator único, não linear), RMT, V1, V2, V3, com a
   fórmula de δ da 4.3. Testes contra o `sklearn.covariance.LedoitWolf` onde
   houver equivalente.
4. `newsmanager/portfolio/backtest.py`: loop de rebalanceamento, carteiras e
   custos.
5. `newsmanager/portfolio/evaluate.py`: métricas, testes e placebos.
6. **Primeiro marco**: H1 sozinha, a regressão da 4.2 com p = 100 e
   2011–2023. Se as notícias não preverem a correlação além da persistência,
   revemos as features antes de construir o resto.

## 8. Resultado do primeiro marco: H1 (2026-10-14)

`python -m newsmanager h1 --n 100` (`newsmanager/portfolio/h1.py`).

**Configuração.**
- 156 rebalanceamentos (2011-01 a 2023-12), universo de 100 ações e 772.200
  pares-data.
- y = z(ρ) no mês de carteira seguinte; ρ_hist em 252 pregões antes de t.
- Notícias dos 365 dias antes de t, com meia-vida de 90 dias. Esses valores
  NÃO foram calibrados; ficam para a calibração em 2011–2013.
- Amostra de artigos de 25%: a ampliação para ~34% ainda não entrou.

**Cobertura dos pares:**

| atributo | pares-data com valor > 0 |
|---|---|
| exposição comum | 64,3% |
| co-menção FNSPID | 34,7% |
| co-ocorrência no artigo extraído | 23,7% |
| vínculo direto, qualquer classe | 1–4% por grupo |

**Fora da amostra** (janela expansiva, 132 meses de teste, 2013–2023; R² da
cross-section de z(ρ_fut)):

| modelo | R²oos | ganho sobre M0 | t (NW) | meses melhores |
|---|---|---|---|---|
| M0 persistência | 0,1858 | — | — | — |
| M1 + co-menção FNSPID (Tang et al.) | 0,1873 | +0,0015 | 2,5 | 67% |
| M2 + co-ocorrência no artigo extraído | 0,1900 | +0,0042 | 5,7 | 74% |
| M3 + relações tipadas + exposição | 0,1891 | +0,0033 | 5,4 | 76% |
| M4 tudo | **0,1911** | **+0,0053** | 4,8 | 75% |

O ganho é positivo nos três subperíodos (2013–16, 2017–19, 2020–23) para
todos os modelos.

**Leitura.**

1. **H1 se confirma.** As notícias preveem a correlação futura além da
   persistência, de forma estável. O ganho absoluto é pequeno (+0,5 ponto de
   R² sobre 18,6%, ~3% relativo), mas y é uma correlação de 21 dias,
   dominada por ruído. O teste econômico que interessa é o da carteira
   (seção 5).
2. **O grafo do LLM domina a co-menção do FNSPID.** A co-ocorrência no artigo
   extraído rende quase 3× o ganho da co-menção marcada (0,0042 contra
   0,0015), e as duas somam.
3. **A exposição comum às entidades é o sinal semântico que sobrevive.** No
   Fama-MacBeth com tudo (M4), ela fica com t = 7,4 mesmo controlando
   co-ocorrência e co-menção.
4. **As relações diretas tipadas quase não acrescentam.** Isoladas, rivalidade
   e "outras" são significantes (t de 3,8 e 4,0), mas somem quando a
   co-ocorrência entra. A cadeia (`supply_chain`, `partnership`) chega a
   ficar negativa (t = −3,2). Com 1–4% de cobertura e alvo mensal ruidoso,
   não dá para separar o tipo da relação do fato de as empresas aparecerem
   juntas.

**Controle de setor** (SIC da SEC → Fama-French 49 e SIC de 2 dígitos,
`python -m newsmanager sectors`; cobre 99,7% das ações-mês do universo):

| comparação fora da amostra | ganho | t (NW) | meses melhores |
|---|---|---|---|
| setor sobre persistência (Elton e Gruber) | +0,0030 | 4,5 | 67% |
| notícias sobre persistência | +0,0053 | 4,8 | 75% |
| co-menção FNSPID sobre persistência + setor | +0,0008 | 1,8 | 63% |
| **notícias sobre persistência + setor** | **+0,0031** | **4,0** | **74%** |
| **notícias sobre persistência + setor + co-menção** | **+0,0023** | **5,2** | **73%** |

5. **As notícias acrescentam além do setor, e sozinhas valem mais que ele.**
   A co-menção do FNSPID quase some depois do setor (t = 1,8); o grafo do LLM,
   não. No Fama-MacBeth completo (S4), co-ocorrência (t = 10,3) e exposição
   (t = 6,3) continuam fortes ao lado do setor, cujos coeficientes caem à
   metade: parte do que o setor explica, as notícias também explicam, mas não
   o contrário.

**Consequências para o modelo (seção 4).**
- O alvo V2 passa a usar exposição + co-ocorrência (+ co-menção) como núcleo,
  **com setor como controle**: o alvo de referência a bater é "persistência +
  setor", não só Ledoit e Wolf.
- As classes de relação entram como teste secundário.

Antes do backtest, faltam duas verificações:
- (a) sensibilidade a horizonte (63 dias), janela e meia-vida, calibrando só
  em 2011–2013;
- (b) refazer com a amostra ampliada quando a run terminar.

## 9. Primeiro backtest (2026-10-14)

`python -m newsmanager backtest --n 100` (`newsmanager/portfolio/backtest.py`).

**Configuração.**
- 132 meses (2013-01 a 2023-12); 2011–2012 servem só para treinar os alvos.
- Janela in-sample de 252 pregões; carteira comprada e mantida no mês.
- Custo de 10 bps por unidade de giro.
- Notícias com janela de 365 dias e meia-vida de 90, sem calibrar.

**Caminhos tentados (todos declarados).**

1. **Alvo com a correlação histórica dentro** (regressão completa da H1).
   - δ ≈ 1: o alvo depende da própria amostra, e a fórmula de δ supõe alvo
     fixo. Σ̂ vira a previsão da regressão, mal condicionada.
   - GMV sem restrição a 35,6% de volatilidade (placebo a 20,4%).
   - **Descartado** por construção, não pelo resultado.
2. **Alvos-prior** (Becquin e Esmeir): R_ij = tanh(z̄ + β'f_ij), sem z_hist;
   β = 0 é o LW de correlação constante. δ médio ≈ 0,2. Resultados abaixo.

**Volatilidade realizada da GMV** (razão de variâncias contra LW correlação
constante, bootstrap em blocos):

| estimador | sem restrição | p | long-only | p |
|---|---|---|---|---|
| LW fator único | **0,918** | 0,000 | 0,999 | 0,91 |
| LW não linear | 0,949 | 0,005 | 1,006 | 0,79 |
| LW identidade | 0,959 | 0,035 | 1,017 | 0,36 |
| alvo setor | 1,008 | 0,26 | 1,006 | 0,002 |
| alvo notícias | 1,042 | 0,000 | 1,006 | 0,20 |
| alvo setor + notícias | 1,035 | 0,002 | 1,006 | 0,16 |
| placebo (notícias embaralhadas) | 1,032 | 0,015 | 1,002 | 0,39 |
| amostral | 1,254 | 0,000 | 1,029 | 0,09 |

Vol. anual da GMV sem restrição: LW fator único 12,9%, alvo setor + notícias
13,7%, 1/N 20,1%. A versão long-only empata tudo (13,4–13,7%).

**Erro quadrático da correlação estimada contra a realizada no mês:**

| estimador | erro vs LW corr. constante | t |
|---|---|---|
| alvo setor + notícias | −0,09% (o menor de todos) | 0,7 |
| placebo | +0,33% | −3,7 |
| LW fator único | +0,97% | −3,8 |
| amostral | +2,48% | −9,6 |

**Leitura.**

1. **H2 não se confirma nesta formulação.** O alvo de notícias não reduz a
   variância da GMV: fica um pouco pior que a correlação constante e empata
   com o placebo.
2. **A informação existe, mas é fraca demais neste canal.** Em correlação, o
   alvo verdadeiro é melhor que o placebo com significância (t = 3,7 de
   diferença). Mas o ganho sobre a correlação constante é de 0,09%: a
   previsibilidade da H1 é pequena (+0,3 a +0,5 ponto de R²), e o shrinkage
   ainda a dilui (δ ≈ 0,2).
3. **A GMV premia a estrutura de fatores, não a precisão par a par.** O fator
   único é o pior em erro de correlação e o melhor em variância. Um alvo
   construído sobre correlação constante parte da base errada.

**Próximo desenho (proposto, ainda não testado).** As notícias devem atuar na
correlação **residual** ao fator de mercado:

    Σ = β β' σ²_m + D^½ R_res D^½

com R_res encolhido para a estrutura prevista pelas notícias e pelo setor.
Esse é o desenho de Lu, Reinert e Cucuringu (2023), com a rede de notícias no
lugar da co-negociação. Pré-requisito: refazer a H1 na correlação residual.

**Aviso de *data snooping*.** Esta revisão foi motivada pelo resultado do
período de teste. Para não contaminar a conclusão, a tese deve:
- reportar todas as variantes, inclusive as que falharam;
- fixar o novo desenho antes de rodá-lo;
- usar como confirmação independente os universos n = 50 e n = 200 e a
  amostra ampliada de artigos.

## 10. Desenho fixado antes de rodar: notícias na correlação residual (2026-10-14)

Escrito ANTES de qualquer resultado deste desenho. Mudanças depois disto
precisam ser declaradas como tal.

**Modelo.** Mercado m_t = média igual dos retornos do universo no dia (o mesmo
fator do LW fator único). Na janela in-sample, para cada ação:
x_it = a_i + b_i·m_t + e_it (MQO).

    Σ̂ = σ²_m b b' + Σ̂_e,   Σ̂_e = shrink_to_corr(E, R_prior)

com E = resíduos in-sample e R_prior a correlação residual a priori:

| estimador | R_prior | papel |
|---|---|---|
| fator + resíduo diagonal | identidade | modelo de fator único puro |
| fator + resíduo corr. constante | r̄_e | referência estatística |
| **fator + resíduo setor** | tanh(z̄_e + β'setor) | **referência a bater** |
| fator + resíduo notícias | tanh(z̄_e + β'notícias) | — |
| **fator + resíduo setor + notícias** | tanh(z̄_e + β'[setor, notícias]) | **modelo principal** |
| placebo | o anterior com as notícias embaralhadas entre ações | controle |

β é estimado como nos alvos-prior da seção 9 (MQO empilhado nos meses < t, y e
x centrados no mês, sem z_hist). Aqui, porém, y = z da correlação **residual**
realizada no mês de carteira: resíduos fora da amostra com os b_i da janela
in-sample e o m_t do próprio mês.

**H1-res (pré-requisito).** A mesma análise da seção 8 com a correlação
residual. Comparação principal: "S4 + notícias" contra "S0 persistência +
setor".

**Parâmetros congelados.**
- Universo n = 100; janela in-sample de 252 pregões.
- Notícias de 365 dias, meia-vida de 90 dias, sem calibrar.
- Atributos e grupos iguais aos da seção 8.
- Custo de 10 bps; 2013-01 a 2023-12.

**Critério de sucesso** (decidido agora):

1. H1-res: ganho fora da amostra de "S4" sobre "S0" > 0 com t > 2.
2. GMV sem restrição, n = 100: razão de variâncias "fator + resíduo setor +
   notícias" / "fator + resíduo setor" < 1 com p < 0,05, **e** menor que a do
   placebo.
3. Confirmação: a mesma razão < 1 em n = 50 e em n = 200 (sem exigir
   significância), e o resultado se mantém com a amostra ampliada de artigos.

Se (2) falhar, o resultado negativo da tese fica registrado com os dois
desenhos.

*Adendo, depois do commit 5c01017 e antes de rodar:* acrescentado o
diagnóstico de estabilidade de López de Prado (2016, 2020): número de condição
λ_max/λ_min da correlação estimada de cada estimador, mês a mês. É descritivo,
**não** é critério de sucesso.

## 11. Resultado do desenho residual (2026-10-14, avaliado contra a seção 10)

`python -m newsmanager h1 --residual` e `python -m newsmanager backtest --n {50,100,200}`.

**Critério 1 (H1-res): atendido.** "S4 + notícias" sobre "S0 persistência +
setor" na correlação residual:
- ganho de +0,0026 com t = 9,7, melhor em 86% dos meses, positivo nos três
  subperíodos;
- na correlação total eram +0,0031 com t = 4,0 e 74%: o sinal fica bem mais
  estável depois de tirar o mercado.

**Critério 2 (n = 100, GMV sem restrição): NÃO atendido.** Razão de variâncias
"fator + resíduo setor + notícias" / "fator + resíduo setor" = 1,001 (IC95
0,986–1,013, p = 0,92).

**Critério 3 (confirmação):**

| n | fator + resíduo setor | + notícias | razão | p | placebo (razão, p) | LW fator único | LW não linear | 1/N |
|---|---|---|---|---|---|---|---|---|
| 50 | 13,85% | 13,83% | 0,997 | 0,57 | 1,028; 0,08 | 13,79% | 14,01% | 21,6% |
| 100 | 12,88% | 12,89% | 1,001 | 0,92 | 1,028; 0,009 | 12,90% | 13,12% | 20,1% |
| 200 | 11,92% | 12,03% | 1,018 | 0,10 | 1,058; 0,000 | 11,96% | 12,45% | 19,9% |

(volatilidade anual da GMV sem restrição; na versão long-only tudo fica a
menos de 0,1 ponto de diferença.)

**Estabilidade (López de Prado, descritivo):** mediana de λmax/λmin da
correlação estimada.

| n | fator + resíduo setor | + notícias | placebo | LW fator único | amostral |
|---|---|---|---|---|---|
| 50 | 102 | 109 | 118 | 96 | 241 |
| 100 | 183 | 210 | 218 | 163 | 765 |
| 200 | 375 | 533 | 482 | 331 | 13.410 |

**Conclusão.**

1. **H2 é rejeitada nos dois desenhos.** As notícias melhoram a previsão da
   correlação (H1, inclusive residual, com t até 9,7), mas não reduzem a
   variância realizada da GMV além do setor. A razão vai de 0,997 (n = 50) a
   1,018 (n = 200), nunca significante.
2. **O conteúdo das notícias não é ruído.** Em todos os universos o grafo
   verdadeiro bate o placebo (variância 2,8–5,8% menor; p de 0,000 a 0,08).
   O que falta é ele superar o setor.
3. **As notícias pioram a estabilidade da matriz.** O número de condição sobe
   com as notícias em todos os universos (de +7% em n = 50 a +42% em n = 200).
   A estrutura extra, esparsa e pequena, abre o leque de autovalores, e a GMV
   inverte a matriz: o ganho estatístico se perde na inversão. É o mecanismo
   de instabilidade de Markowitz descrito por López de Prado (2016, 2020).
4. **Achado metodológico positivo.** O modelo fator de mercado + correlação
   residual encolhida para o setor (seção 10) empata com o LW fator único e
   bate o LW não linear em todos os universos (p ≤ 0,05 em n = 100 e 200).
5. **O ganho das notícias encolhe com n**: em universos maiores a cobertura
   dos pares cai e as notícias ficam mais esparsas.

**Pendente (já previsto na seção 10):** refazer com a amostra ampliada de
artigos (~34%) quando a run na Vast terminar.

## 12. Desenho fixado antes de rodar: carteira long-only com teto de peso (2026-10-15)

Escrito ANTES de rodar este desenho com o grafo da amostra ampliada. Mudanças
depois do commit desta seção precisam ser declaradas como tal.

**Por que um segundo desenho.** A seção 10 testou a GMV sem restrição, que
inverte Σ̂ diretamente e por isso sofre com o número de condição (seção 11). Na
prática a variância mínima é usada comprada e com teto por ativo, e as
restrições funcionam elas mesmas como shrinkage (Jagannathan e Ma, 2003). A
pergunta aqui é se as notícias reduzem a variância da carteira que um gestor
de fato montaria.

**Transparência: o que já foi visto.** As configurações abaixo foram escritas
como exemplos do motor novo (`backtests/*.toml`) antes de qualquer resultado.
Depois rodaram uma vez com o grafo antigo (amostra de 25%), num teste do motor.
Aquele resultado é **exploratório** e não conta para os critérios:

| (grafo 25%, n = 100) | vol. | razão de variâncias contra setor | p |
|---|---|---|---|
| fator + resíduo setor | 13,41% | — | — |
| setor + notícias | 13,37% | 0,994 | 0,017 |
| placebo | 13,42% | 1,001 | 0,79 |
| LW fator único | 13,37% | 0,993 | 0,19 |

O teste confirmatório usa o grafo da amostra ampliada. São ~171 mil artigos
novos (+43% de unidades extraídas), sobre o mesmo período de preços: os dados
de notícias são novos, mas o período de carteira não é.

**Motor e configuração** (`python -m newsmanager.backtester`, arquivos
`backtests/*.toml` deste commit):

- 2013-01-01 a 2023-12-31, 252 pregões in-sample, rebalanceamento mensal (1º
  pregão do mês);
- long-only (`min_weight = 0`), teto `max_weight = 0.10`, custo de 10 bps
  sobre o giro, benchmark SPY;
- universo `news.universe` n = 100, cobertura mínima de 95% na janela;
- otimização: `min_variance` (FISTA com projeção no simplex com teto).

| run | plugin | papel |
|---|---|---|
| `minvar_setor` | `NewsFactorResidual(features="setor")` | **referência a bater** |
| `minvar_setor_noticias` | `NewsFactorResidual(features="setor+noticias")` | **modelo principal** |
| `minvar_setor_noticias_placebo` | idem, `placebo = true` (seed 7) | controle |
| `minvar_lw_fator` | `MinVariance(estimator="LW fator único")` | referência estatística |
| `equal_weight` | `EqualWeight` | referência ingênua |

O modelo e a regressão são os da seção 10:
- notícias de 365 dias, meia-vida de 90 dias, `min_train = 24`;
- β treinado só nos meses cujo período de carteira terminou antes de t.

**Teste.** `analysis.compare`: razão de variâncias dos retornos diários
líquidos, com p bicaudal por bootstrap circular em blocos de 10 dias, 2.000
reamostragens e seed 7.

**Critério de sucesso** (decidido agora):

1. **Principal.** n = 100: a razão `minvar_setor_noticias` / `minvar_setor`
   é < 1 com **p < 0,025**, **e** é menor que a razão `placebo` / `minvar_setor`.
   O nível é 0,025, e não 0,05, porque este é o segundo desenho testado para a
   mesma hipótese H2 (Bonferroni para dois desenhos).
2. **Confirmação (sem exigir significância).** Razão < 1 com a mesma
   configuração em n = 50 e n = 200 (`--set universe.n=50 --name <run>_n50`;
   idem n200).
3. **Robustez (descritiva, não é critério).** n = 100 com teto de 5% e de 20%
   (`--set max_weight=0.05` / `0.20`); diferença de Sharpe; número de condição.

O resultado é registrado junto com o da seção 11, seja qual for. Se (1)
falhar, H2 fica rejeitada nos três desenhos. Se (1) passar, a tese afirma o
ganho para a carteira restrita, e não para a GMV, com o mecanismo da seção 11
(instabilidade na inversão) como explicação da diferença.

O critério 3 da seção 10 (amostra ampliada, GMV sem restrição, `python -m
newsmanager backtest`) continua valendo e é avaliado à parte.

## 13. Resultado com a amostra ampliada (2026-10-16, avaliado contra as seções 10 e 12)

**Dados.** Extração parada em 572.762 unidades (~34% do corpus; 171.153 novas, 0
falhas), 1.270.582 relações. Link dos 70.024 nomes novos completo (0 falhas).
Grafo: 1.064.801 eventos limpos (antes 741.409, +44%), 255.384 nós. Universos
refeitos: 93–95% das ações em comum por mês com os de antes; n = 100 passa de
334 para 315 tickers distintos. Setores: 20 de 651 sem setor.

*Desvio declarado:* os 434 tickers novos aceitos só por evidência (875
menções, 0,11% das 773 mil com ticker) não passaram pelo `link-verify`: a VM
ficou sem crédito depois do link. Na verificação anterior ~25% desse grupo
foi rejeitado, então são ~200 menções com ticker errado.

O estado anterior (25%) está em `data/archive/*_25pct/`.

**H1 com o grafo ampliado** (`h1 --n 100`):

| | antes (25%) | agora (34%) |
|---|---|---|
| correlação total, S4 contra S0 | +0,0031 (t = 4,0; 74%) | +0,0033 (t = 4,3; 73%) |
| correlação residual, S4 contra S0 | +0,0026 (t = 9,7; 86%) | +0,0027 (t = 8,8; 84%) |

O sinal preditivo das notícias se mantém e fica um pouco maior.

**Seção 10, critério 3 (GMV sem restrição, `backtest --n N`): não se mantém.**

| n | razão notícias/setor | p | placebo/setor | p | λmax/λmin (setor → notícias) |
|---|---|---|---|---|---|
| 50 | 1,003 | 0,77 | 1,011 | 0,30 | 104 → 110 |
| 100 | 1,001 | 0,89 | 1,022 | 0,006 | 183 → 217 |
| 200 | 1,025 | 0,046 | 1,021 | 0,27 | 374 → 544 |

Com mais notícias, a GMV sem restrição fica igual (n = 50, 100) ou **pior**
(n = 200, p = 0,046). O número de condição sobe mais que antes: em n = 200
passa de +42% para +45%.

**Seção 12 (long-only, teto de 10%, `python -m newsmanager.backtester`).**

Critério 1 (principal), n = 100: **NÃO atendido.**

| run | vol. | Sharpe | razão contra setor | p |
|---|---|---|---|---|
| fator + resíduo setor | 13,43% | 0,778 | — | — |
| setor + notícias | 13,42% | 0,789 | 0,998 | 0,46 |
| placebo | 13,39% | 0,783 | 0,994 | 0,12 |
| LW fator único | 13,39% | 0,788 | 0,993 | 0,15 |
| 1/N | 19,94% | 0,663 | 2,205 | 0,000 |

Exigia p < 0,025 e notícias abaixo do placebo; não houve nenhum dos dois. O
resultado exploratório com o grafo de 25% (0,994, p = 0,017) **não se
replicou**.

Critério 2 (confirmação) e robustez:

| variante | razão notícias/setor | p | placebo/setor | p | Δ Sharpe |
|---|---|---|---|---|---|
| n = 50 | 0,996 | 0,06 | 0,999 | 0,68 | +0,013 |
| n = 100 | 0,998 | 0,46 | 0,994 | 0,12 | +0,011 |
| n = 200 | 0,996 | 0,09 | 1,001 | 0,85 | −0,019 |
| n = 100, teto 5% | 0,995 | 0,20 | 0,995 | 0,07 | +0,022 |
| n = 100, teto 20% | 0,998 | 0,51 | 0,996 | 0,29 | +0,003 |

O critério 2 seria atendido sozinho, com razão < 1 em n = 50 e n = 200, mas
ele só valia como confirmação do critério 1. As diferenças são de 0,2% a 0,5%
da variância e menores que o ruído. Com teto, o placebo empata com as notícias
de verdade.

**Conclusão (as três avaliações juntas).**

1. **H1 confirmada e estável:** as notícias preveem a correlação futura além
   do histórico e do setor. Com 44% mais eventos o ganho fica igual ou
   maior, e o t vai a 8,8 na correlação residual.
2. **H2 rejeitada nos três desenhos:**
   - GMV com alvo-prior (seção 9);
   - GMV residual (seções 10 e 11, agora também com a amostra ampliada);
   - variância mínima long-only com teto (seção 12).

   A informação preditiva não vira redução de variância realizada da carteira.
3. **Mecanismo:**
   - **Sem restrição:** as notícias pioram o condicionamento de Σ̂ (λmax/λmin
     +6% a +45%), e a inversão consome o ganho.
   - **Com restrição:** o teto e a proibição de venda a descoberto já
     regularizam tanto (Jagannathan e Ma, 2003) que todos os estimadores
     baseados em fator empatam: setor, notícias, placebo e LW ficam a menos de
     0,05 ponto de volatilidade entre si.
4. **Mais dados não mudam a conclusão:** de 25% para 34% do corpus, o ganho
   preditivo cresce um pouco, e o efeito na carteira continua nulo.

## 14. Desenho fixado antes de rodar: outros otimizadores (2026-10-16)

Escrito ANTES de rodar qualquer um destes otimizadores com dados reais. Só
foram rodados testes sintéticos (`tests/test_backtester.py`), que conferem
cada otimizador contra a solução fechada quando ela existe. Mudanças depois
do commit desta seção precisam ser declaradas como tal.

**Pergunta.** As seções 9–13 só testaram variância mínima. Aqui a pergunta é
se a Σ̂ com notícias melhora a carteira quando o otimizador:
- usa retorno esperado (Markowitz, máximo Sharpe);
- busca diversificação (máxima diversificação);
- não inverte Σ̂ (HRP), que é onde as notícias perdiam o ganho (seção 11);
- persegue um benchmark (tracking error mínimo).

**Otimizadores** (um plugin cada, `newsmanager/backtester/models/`):

| plugin | problema |
|---|---|
| `mean_variance:MeanVariance` | max w'μ − (γ/2) w'Σ̂w, γ = 5 |
| `max_sharpe:MaxSharpe` | max w'μ / √(w'Σ̂w), rf = 0 |
| `max_diversification:MaxDiversification` | max w'σ / √(w'Σ̂w), σ = √diag Σ̂ (Choueifaty e Coignard, 2008) |
| `hrp:HRP` | López de Prado (2016): ligação simples em √(½(1−ρ)), bissecção recursiva com variância inversa; teto por projeção |
| `min_tracking_error:MinTrackingError` | min w'Σ̂w − 2w'c, c = Cov(R, SPY) amostral na janela |

**Retorno esperado:** μ de Bayes-Stein (Jorion, 1986), calculado nos 252
pregões in-sample e encolhido para a média da GMV de Σ̂. A média amostral
pura tem erro grande demais (DeMiguel, Garlappi e Uppal, 2009).

**Restrições:** Σw = 1 e 0 ≤ w ≤ 10%, as mesmas da seção 12. O máximo Sharpe
e a máxima diversificação percorrem a fronteira restrita
min w'Σ̂w − λa'w, com uma grade em λ seguida de seção áurea.

**Estimadores de Σ̂ (parâmetro `cov`), iguais para os cinco otimizadores:**

| `cov` | papel |
|---|---|
| fator + resíduo setor | **referência a bater** |
| fator + resíduo setor + notícias | **modelo principal** |
| idem, `placebo = true` (seed 7) | controle |
| LW fator único | referência estatística |

**Configuração:** a da seção 12. São 2013-01 a 2023-12, 252 pregões, mensal,
n = 100, 10 bps, SPY, notícias de 365 dias, meia-vida de 90 e
`min_train = 24`. Arquivos: `backtests/secao14/*.toml` (20 runs),
`rodar.sh` e `avaliar.py`, deste commit.

**Estatística principal por otimizador** (notícias contra setor, pregões em
comum, bootstrap circular em blocos de 10, 2.000 reamostragens, seed 7, p
bicaudal):

| otimizador | estatística | melhora |
|---|---|---|
| Markowitz, máximo Sharpe | diferença de Sharpe líquido | > 0 |
| máxima diversificação, HRP | razão de variâncias dos retornos líquidos | < 1 |
| tracking error mínimo | razão de variâncias do retorno ativo (r − SPY) | < 1 |

**Critério de sucesso** (decidido agora), avaliado por otimizador:

1. **Principal.** Melhora com **p < 0,01** (Bonferroni para cinco
   otimizadores: 0,05/5), **e** melhor que o placebo na mesma estatística.
2. **Confirmação (sem exigir significância).** Mesma direção em n = 50 e
   n = 200.
3. **Robustez (descritiva).** Markowitz com γ = 2 e 10; volatilidade,
   Sharpe, TE e número de condição de cada run.

Para declarar H2 aceita num otimizador, (1) e (2) precisam ser atendidos
nele. Os resultados dos cinco são registrados juntos, seja qual for o
desfecho.

*Nota de implementação, antes de rodar:* os modelos passaram a ser um plugin
por arquivo. `basic.py` e `news.py` continuam aceitando os nomes antigos das
seções 12–13. Rodar de novo os TOMLs da seção 12 reproduz os retornos diários
a menos de 1e-15, a mesma diferença entre duas execuções seguidas, que vem da
ordem de soma do DuckDB.

*Adendo, depois do commit a09119a e antes de avaliar (2026-10-16): troca do
solver do máximo Sharpe e da máxima diversificação.* A busca na fronteira
(grade em λ + seção áurea, cada ponto resolvido por FISTA) ficou lenta demais
com Σ̂ mal condicionada. Com n = 200, os meses de 2020 em diante levavam ~5 min
cada, e depois de 8 h as 6 runs desses dois otimizadores com n = 200 estavam
no mês 84–90 de 132. Elas foram interrompidas.

- **Novo solver:** o mesmo problema pela reformulação convexa homogênea
  (Cornuéjols e Tütüncü, 2007), resolvida por pontos interiores (Clarabel):

      min y'Σ̂y  s.t.  a'y = 1,  Σy = κ,  lo·κ ≤ y ≤ hi·κ,  κ ≥ 0,  w = y/κ

  O fallback para a GMV quando nenhum a'w > 0 é atingível continua igual.
- **Conferência:** 24 casos reais (n = 200; seis datas de 2013 a 2023; Σ̂ de
  setor e de setor + notícias; a = μ e a = σ). Maior Σ|Δw| entre as duas
  implementações = 2,3e-7, razão igual até 1e-9, tempo de ~0,1 s contra
  15–60 s. A versão registrada continua no código como `max_ratio_frontier`,
  e `tests/test_backtester.py` confere a equivalência.
- **O que nada disso muda:** o problema, as configurações, a estatística e o
  critério.
- **O que foi visto antes do adendo:** as linhas de progresso das runs
  interrompidas mostravam o patrimônio acumulado até 2020 (de 1,86 a 1,89 na
  máxima diversificação, de 1,93 a 2,72 no máximo Sharpe). Os outros três
  otimizadores (HRP, Markowitz, TE mínimo), que não mudaram, já rodaram
  completos com o código do commit a09119a, mas ainda não foram avaliados.

## 15. Resultado dos outros otimizadores (2026-10-16, avaliado contra a seção 14)

`backtests/secao14/rodar.sh` (os mesmos comandos, 6 em paralelo) e
`python backtests/secao14/avaliar.py` (`logs/secao14_avaliacao.log`). São 56
runs. Máximo Sharpe e máxima diversificação usam o solver do adendo (commit
7c6bfcd).

**Critério 1 (principal): NÃO atendido em nenhum dos cinco otimizadores.**

| otimizador | estatística | notícias contra setor | p | placebo contra setor | p |
|---|---|---|---|---|---|
| Markowitz (γ = 5) | Δ Sharpe | −0,015 | 0,16 | +0,004 | 0,62 |
| máximo Sharpe | Δ Sharpe | −0,010 | 0,43 | +0,004 | 0,76 |
| máxima diversificação | razão de variâncias | 0,992 | 0,46 | 0,921 | 0,17 |
| HRP | razão de variâncias | 1,007 | 0,14 | 1,002 | 0,59 |
| tracking error mínimo | razão de variâncias ativas | **1,214** | **0,000** | 1,008 | 0,63 |

**Critério 2 (confirmação, n = 50 / n = 200):** a direção não se repete em
nenhum otimizador.

| otimizador | n = 50 | n = 200 |
|---|---|---|
| Markowitz | −0,003 | +0,006 |
| máximo Sharpe | −0,008 | +0,015 |
| máxima diversificação | 1,003 | 0,964 (p = 0,09) |
| HRP | 0,992 | 1,001 |
| tracking error mínimo | **1,104** (p = 0,000) | **1,432** (p = 0,000) |

**Descritivo, n = 100** (vol. anual, Sharpe líquido, tracking error contra
SPY):

| otimizador | setor | setor + notícias | placebo | LW fator único |
|---|---|---|---|---|
| Markowitz | 24,6%; 0,79; 17,9% | 24,4%; 0,77; 17,8% | 24,7%; 0,79; 18,0% | 24,7%; 0,78; 18,0% |
| máximo Sharpe | 20,4%; 0,88; 14,8% | 20,4%; 0,87; 14,8% | 20,3%; 0,89; 14,6% | 20,2%; 0,88; 14,7% |
| máxima diversificação | 20,0%; 0,79; 15,1% | 20,0%; 0,81; 14,9% | 19,2%; 0,80; 13,9% | 19,7%; 0,79; 14,7% |
| HRP | 15,2%; 0,81; 5,0% | 15,3%; 0,80; 5,0% | 15,2%; 0,80; 5,0% | 15,2%; 0,82; 5,1% |
| TE mínimo | 17,2%; 0,80; 2,48% | 17,1%; 0,81; **2,73%** | 17,2%; 0,79; 2,49% | 17,2%; 0,80; 2,47% |

Para comparação, com os mesmos dados e restrições: variância mínima com setor
(seção 13) 13,4%; 0,78. 1/N 19,9%; 0,66. SPY 17,0%; 0,82.

**Robustez do Markowitz:** Δ Sharpe das notícias de −0,004 (γ = 2) e −0,018
(γ = 10); o placebo fica positivo nos dois.

**Conclusão.**

1. **H2 rejeitada também com outros otimizadores.** Em quatro dos cinco
   (Markowitz, máximo Sharpe, máxima diversificação, HRP), Σ̂ com notícias
   empata com Σ̂ com setor. As diferenças de Sharpe ficam abaixo de 0,02 e
   as de variância abaixo de 1%, sem direção estável entre n = 50, 100 e
   200. Nesses quatro o placebo fica igual ou melhor.
2. **No tracking error mínimo as notícias pioram, de forma significativa e
   crescente com n:** o TE realizado sobe 10% (n = 50), 21% (n = 100) e 43%
   (n = 200), com p < 0,001. O placebo não piora (1,008), então o efeito vem
   da estrutura real das notícias, não de ruído genérico no alvo. Replicar um
   índice depende de acertar a covariância conjunta de muitos ativos. As
   correlações residuais que as notícias acrescentam acertam a média dos
   pares (H1), mas distorcem o hedge, e o giro sobe de 3,3 para 4,0 por ano.
   É o mesmo mecanismo da GMV sem restrição (seções 11 e 13): a estrutura
   esparsa extra piora a matriz que o otimizador explora.
3. **O otimizador importa muito mais que o estimador.** A volatilidade vai de
   13% (variância mínima) a 25% (Markowitz) conforme o otimizador. Para um
   otimizador fixo, trocar o estimador (setor, notícias, placebo, LW) muda a
   volatilidade em menos de 1 ponto. Nenhum otimizador com retorno esperado
   bateu o SPY em Sharpe líquido (0,82). O máximo Sharpe (0,88) chega
   perto, mas sem significância.
4. **Panorama da tese (seções 8–15):** as notícias preveem a correlação (H1,
   t até 9,7), mas em nenhum dos 8 desenhos de carteira testados isso virou
   melhora significativa. São eles: GMV com alvo-prior, GMV residual,
   variância mínima com teto, Markowitz, máximo Sharpe, máxima
   diversificação, HRP e TE mínimo. Em dois deles (GMV sem restrição com
   n = 200 e TE mínimo) as notícias pioram a carteira.

Notebooks: `notebooks/secao14_{mv,maxsharpe,maxdiv,hrp,minte}.ipynb`.

## 16. Desenho fixado antes de rodar: Black-Litterman na covariância (2026-10-22)

Escrito ANTES de implementar ou rodar este desenho. Mudanças depois do commit
desta seção precisam ser declaradas como tal.

**Transparência: o que já foi visto.** Os resultados das seções 8–15. Também
os dois testes exploratórios de 2026-10-22, sem critério pré-registrado:

- **Estabilidade diária de Σ̂** (`python -m newsmanager.backtester stability`):
  - notícias/setor = 1,12 no κ mediano, contra 1,10 do placebo;
  - mercado + sentimento sobre mercado = 1,02, contra 1,02 do placebo.
- **Fatores de sentimento por agente** (`python -m newsmanager sentiment`): R²
  mediano de 4,2%, o nível esperado do ruído com 10 regressores.

Nenhuma configuração deste desenho foi rodada.

**Por que um quarto desenho.** Nos desenhos anteriores a notícia entra por um
único alvo, com um único δ de shrinkage para todos os pares. Um par com
centenas de eventos e um par ligado só por exposição comum recebem o mesmo
peso da notícia. A H1 mostra que a notícia acerta a média dos pares. A
pergunta aqui é se ela melhora Σ̂ quando cada par recebe a confiança que a sua
própria evidência justifica. É a estrutura do Black-Litterman (Black e
Litterman, 1992), com visões sobre a covariância e não sobre a média (Qian e
Gorman, 2001).

**Modelo.** As definições de mercado, betas e resíduos são as da seção 10:
m_t, b_i e E. Só a covariância residual Σ_e é atualizada:

    Σ̂ = σ²_m b b' + Σ̂_e^BL

1. **Prior ("equilíbrio").** Σ₀ = Σ̂_e do "fator + resíduo setor" da seção 10,
   a referência a bater, com sᵢ = (Σ₀)ᵢᵢ. A incerteza do prior sobre os
   elementos fora da diagonal é a covariância de Wishart da covariância
   amostral (Anderson, 2003), avaliada em Σ₀, com T = 252 pregões:

       V_(ij),(kl) = (σ₀,ik σ₀,jl + σ₀,il σ₀,jk) / T

   É o análogo, na covariância, da convenção τ = 1/T de He e Litterman
   (1999). Em escala de Fisher z, ela dá Var ≈ 1/T por par. Uma visão sobre
   o par (i, j) também move os pares (i, l) e (j, l), na proporção das
   covariâncias do prior.
2. **Visões.** Há uma visão para cada par com algum atributo de notícia não
   nulo na janela (`comention`, `cooccur`, `direct_*` ou `exposure`). Pares
   sem notícia ficam sem visão.
   - O valor da visão é a correlação prevista pelo modelo "setor + notícias"
     da seção 10, ẑ_ij = z̄_e + β'x_ij, na escala da covariância:

         q_ij = tanh(ẑ_ij) · √(sᵢ sⱼ)

   - P seleciona o elemento (i, j) de vec(Σ_e). A visão é, portanto, linear
     em Σ_e, como no BL original.
   - A diagonal (as variâncias) não recebe visão nem é atualizada.
3. **Confiança Ω (diagonal) por classe de evidência.** Cada par com visão cai
   em uma classe c:
   - c = 0: há atributo de notícia, mas `cooccur` = 0 (só exposição comum
     e/ou co-menção);
   - c = 1–4: quartis de `cooccur` > 0, com os cortes calculados nos meses de
     treino.

   Para cada classe, o erro da visão em escala z é estimado no painel da
   seção 10, só nos meses já disponíveis para β:

       ω_c = média(ê² | c) − média_m 1/(d_m − 3),   ω_c ≥ 1/(T − 3)

   - ê = z da correlação residual realizada no mês menos a previsão
     (resíduo dentro da amostra da regressão que dá β).
   - O termo subtraído é o ruído amostral de uma correlação de d_m pregões.
     Sem ele, ω mediria o erro contra a correlação realizada no mês, e não
     contra a verdadeira.
   - O piso impede que uma visão seja mais precisa que a correlação amostral
     de T pregões.
   - Na escala da covariância, pelo método delta:

         Ω_ij = (1 − tanh²ẑ_ij)² · sᵢ sⱼ · ω_c(ij)

4. **Posterior.** É a média do BL na forma de Kalman, sobre os elementos fora
   da diagonal:

       vec Σ_e^BL = vec Σ₀ + V P' (P V P' + Ω)⁻¹ (q − P vec Σ₀)

   O resultado é convertido em correlação e projetado para PSD com
   `nearest_corr` (Higham, 2002), com as variâncias sᵢ de volta. Não se soma
   a incerteza do posterior (o M do BL), porque o objeto estimado é Σ, não um
   retorno futuro.
5. **Antes de 24 meses de treino** (`min_train`) não há β nem ω, e o BL é
   igual ao prior. É a mesma regra dos desenhos anteriores.

**Estimadores comparados** (parâmetro `cov`):

| `cov` | papel |
|---|---|
| fator + resíduo setor | **referência a bater** (o prior) |
| **fator + resíduo setor + BL notícias** | **modelo principal** |
| idem, `placebo = true` (seed 7) | controle: atributos de notícia permutados entre ações, como na seção 10 (q, classe e Ω vão juntos) |
| idem, Ω homogêneo | ablação descritiva: ω igual à média ponderada das classes |
| fator + resíduo setor + notícias (seção 10) | descritivo: o shrinkage com δ único |
| LW fator único | referência estatística |

**Carteira principal: GMV sem restrição** (w = Σ̂⁻¹1 / 1'Σ̂⁻¹1;
`min_weight = -inf`, `max_weight = inf` no backtester, que usa a forma
fechada). É a carteira em que o erro de Σ̂ passa inteiro para os pesos, e onde
as notícias pioraram nas seções 11 e 13. O resto da configuração é o da seção
12:
- 2013-01 a 2023-12, 252 pregões in-sample, rebalanceamento mensal;
- custo de 10 bps, benchmark SPY;
- `news.universe` n = 100, cobertura mínima de 95%;
- notícias de 365 dias, meia-vida de 90 dias, `min_train = 24`.

Arquivos: `backtests/secao16/*.toml`, `rodar.sh` e `avaliar.py`, no commit
da implementação.

**Teste.** É o das seções 12 e 14:
- razão de variâncias dos retornos diários líquidos nos pregões em comum;
- p bicaudal por bootstrap circular em blocos de 10 dias, 2.000
  reamostragens, seed 7.

**Critério de sucesso** (decidido agora):

1. **Principal.** n = 100: a razão "setor + BL notícias" / "setor" é < 1 com
   **p < 0,0125**, **e** é menor que a razão placebo / setor. O nível é
   0,05/4: este é o quarto desenho testado para H2 (seções 10, 12, 14 e 16).
2. **Confirmação (sem exigir significância).** Razão < 1 em n = 50 e
   n = 200.
3. **Robustez e diagnóstico (descritivos, não são critério).**
   - variância mínima long-only com teto de 10% (configuração da seção 12);
   - tracking error mínimo (configuração da seção 14), em que as notícias
     pioraram;
   - Ω homogêneo contra Ω por classe (isola o ganho da confiança por par);
   - os ω_c estimados por ano e o número de visões por mês;
   - o número de condição mensal de cada Σ̂.

Para declarar H2 aceita neste desenho, (1) e (2) precisam ser atendidos. O
resultado é registrado junto com o das seções 11, 13 e 15, seja qual for.

**Limitações declaradas agora.**
- O prior e as visões compartilham a informação de setor e a mesma janela de
  retornos (o nível z̄_e). Os erros dos dois não são independentes, como o BL
  supõe, e o posterior tende a confiar demais nas visões.
- O ω_c vem de resíduos dentro da amostra, com o mês centrado. Ele não inclui
  o erro do nível z̄_e.
- A incerteza de Wishart é a da covariância amostral. Σ₀ já é encolhida e
  tem menos variância que isso, então o prior pode estar fraco demais.
- A atualização linear não garante PSD. A projeção para PSD muda o posterior
  nos meses em que ela age; a sua frequência é registrada em `ctx.info`.

**Implementação** (depois do commit desta seção, antes de rodar):
- função de estimação em `newsmanager/portfolio/estimators.py`;
- estimador no `cov` de `newsmanager/backtester/models/covariance.py`;
- ω_c treinado no mesmo `_Trainer` que dá β.

Teste sintético em `tests/test_backtester.py`:
- Ω → ∞ devolve o prior;
- Ω → 0 com uma visão leva o elemento visto a q e move os outros por
  V P'(P V P')⁻¹;
- conferência contra o BL escrito com matrizes densas num caso pequeno.

## Referências

- Anderson, T. W. (2003). *An Introduction to Multivariate Statistical Analysis*, 3ª ed. Wiley.
- Becquin, G.; Esmeir, S. (2023). Semantic Similarity Covariance Matrix Shrinkage. *Findings of EMNLP 2023*, 9977–9992. https://aclanthology.org/2023.findings-emnlp.668/
- Black, F.; Litterman, R. (1992). Global portfolio optimization. *Financial Analysts Journal* 48(5), 28–43.
- Bun, J.; Bouchaud, J.-P.; Potters, M. (2017). Cleaning large correlation matrices: tools from random matrix theory. *Physics Reports* 666.
- Chan, L.; Karceski, J.; Lakonishok, J. (1999). On portfolio optimization: forecasting covariances and choosing the risk model. *RFS* 12, 937–974.
- Cohen, L.; Frazzini, A. (2008). Economic links and predictable returns. *Journal of Finance* 63(4).
- Choueifaty, Y.; Coignard, Y. (2008). Toward maximum diversification. *J. Portfolio Management* 35(1).
- Cornuéjols, G.; Tütüncü, R. (2007). *Optimization Methods in Finance*. Cambridge University Press.
- DeMiguel, V.; Garlappi, L.; Uppal, R. (2009). Optimal versus naive diversification. *RFS* 22(5).
- Elton, E.; Gruber, M. (1973). Estimating the dependence structure of share prices. *Journal of Finance* 28.
- Engle, R.; Colacito, R. (2006). Testing and valuing dynamic correlations for asset allocation. *JBES* 24(2).
- Friedman, J.; Hastie, T.; Tibshirani, R. (2008). Sparse inverse covariance estimation with the graphical lasso. *Biostatistics* 9(3).
- Guo, L.; Peng, L.; Tao, Y.; Tu, J. Joint News, Attention Spillover, and Stock Returns. arXiv:1703.02715.
- He, G.; Litterman, R. (1999). The intuition behind Black-Litterman model portfolios. Goldman Sachs Investment Management Research.
- Higham, N. (2002). Computing the nearest correlation matrix. *IMA J. Numer. Anal.* 22.
- Hoberg, G.; Phillips, G. (2016). Text-based network industries and endogenous product differentiation. *JPE* 124(5).
- Jagannathan, R.; Ma, T. (2003). Risk reduction in large portfolios: why imposing the wrong constraints helps. *J. Finance* 58(4).
- Jorion, P. (1986). Bayes-Stein estimation for portfolio analysis. *JFQA* 21(3).
- Kan, R.; Zhou, G. (2007). Optimal portfolio choice with parameter uncertainty. *JFQA* 42(3).
- Laloux, L.; Cizeau, P.; Bouchaud, J.-P.; Potters, M. (1999). Noise dressing of financial correlation matrices. *PRL* 83.
- Ledoit, O.; Wolf, M. (2003). Improved estimation of the covariance matrix of stock returns with an application to portfolio selection. *J. Empirical Finance* 10.
- Ledoit, O.; Wolf, M. (2004a). Honey, I shrunk the sample covariance matrix. *J. Portfolio Management* 30(4).
- Ledoit, O.; Wolf, M. (2004b). A well-conditioned estimator for large-dimensional covariance matrices. *JMVA* 88.
- Ledoit, O.; Wolf, M. (2008). Robust performance hypothesis testing with the Sharpe ratio. *J. Empirical Finance* 15.
- Ledoit, O.; Wolf, M. (2011). Robust performance hypothesis testing with the variance. *Wilmott*, set. 2011, 86–89.
- Ledoit, O.; Wolf, M. (2020). Analytical nonlinear shrinkage of large-dimensional covariance matrices. *Annals of Statistics* 48(5).
- López de Prado, M. (2016). Building diversified portfolios that outperform out of sample. *J. Portfolio Management* 42(4).
- Lu, C.; Ndiaye, P. M.; Simaan, M. (2024). Improved estimation of the correlation matrix using reinforcement learning and text-based networks. *IRFA* 96, 103572.
- Lu, Y.; Reinert, G.; Cucuringu, M. (2023). Co-trading networks for modeling dynamic interdependency structures and estimating high-dimensional covariances in US equity markets. arXiv:2302.09382.
- Mantegna, R. (1999). Hierarchical structure in financial markets. *EPJ B* 11.
- Markowitz, H. (1952). Portfolio selection. *Journal of Finance* 7(1).
- Menzly, L.; Ozbas, O. (2010). Market segmentation and cross-predictability of returns. *Journal of Finance* 65(4).
- Patton, A. (2011). Volatility forecast comparison using imperfect volatility proxies. *J. Econometrics* 160(1).
- Peralta, G.; Zareei, A. (2016). A network approach to portfolio selection. *J. Empirical Finance* 38.
- Pozzi, F.; Di Matteo, T.; Aste, T. (2013). Spread of risk across financial markets: better to invest in the peripheries. *Scientific Reports* 3, 1665.
- Qian, E.; Gorman, S. (2001). Conditional distribution in portfolio theory. *Financial Analysts Journal* 57(2), 44–51.
- Scherbina, A.; Schlusche, B. (2016). Economic linkages inferred from news stories and the predictability of stock returns. AEI WP; (2020) Follow the leader, *Review of Finance* 24(1).
- Tang, Y.; Zhou, Y.; Hong, M. (2019). News co-occurrences, stock return correlations, and portfolio construction implications. *JRFM* 12(1), 45.
- Tumminello, M.; Aste, T.; Di Matteo, T.; Mantegna, R. (2005). A tool for filtering information in complex systems. *PNAS* 102(30).
