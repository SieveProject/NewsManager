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

*Nota de implementação, depois do commit ce5f630 e antes de rodar
(2026-10-22):*
- **Código:**
  - `estimators.black_litterman_cov` resolve o sistema por gradiente
    conjugado: V P'α = vec(Σ₀ A Σ₀)/T, sem formar V;
  - `covariance.NewsBL` e `_Omega` montam as visões e a confiança;
  - `backtests/secao16/` tem 14 TOMLs, `rodar.sh` e `avaliar.py`.
- **Ruído amostral:** o termo 1/(d_m − 3) usa, para cada par, o número de
  pregões em comum no mês (`n_fut` do painel), e não o número de pregões do
  mês. Os dois só diferem quando falta retorno.
- **Conferências:**
  - o BL bate com o BL denso a 1e-12, e Ω → ∞ e Ω → 0 se comportam como
    previsto (`tests/test_backtester.py`);
  - ω_c bate com um cálculo independente em pandas a 4e-15 (mês 60).
- **Visto antes de rodar:** num teste de fumaça de 4 meses (2015-01 a
  2015-04), os ω_c crescem com a evidência, de 0,010 (classe 0) a 0,038
  (classe 4).

## 17. Resultado do Black-Litterman na covariância (2026-10-22, avaliado contra a seção 16)

`backtests/secao16/rodar.sh` e `python backtests/secao16/avaliar.py`
(`logs/secao16_avaliacao.log`).

**Critério 1 (principal, GMV sem restrição, n = 100): NÃO atendido.**

| Σ̂ | vol. | Sharpe | razão contra setor | p |
|---|---|---|---|---|
| fator + resíduo setor | 12,83% | 0,632 | — | — |
| setor + BL notícias | 13,71% | 0,609 | 1,143 | 0,015 |
| BL placebo | 14,62% | 0,556 | 1,300 | 0,012 |
| BL Ω homogêneo | 13,61% | 0,596 | 1,126 | 0,021 |
| setor + notícias (δ único, seção 10) | 12,83% | 0,641 | 1,001 | 0,87 |
| LW fator único | 12,85% | 0,634 | 1,003 | 0,52 |

O BL **aumenta** a variância da GMV em 14%. Ele fica melhor que o placebo
(1,14 contra 1,30), mas pior que o prior.

**Critério 2 (confirmação): NÃO atendido.**
- n = 50: razão 1,10 (p = 0,19); placebo 1,16.
- n = 200: razão 1,20 (p = 0,001); placebo 1,36.

**Robustez (descritiva).**
- Variância mínima com teto de 10%: 0,998 (p = 0,88), um empate.
- TE mínimo (variância ativa): 2,02 (p = 0,000); o placebo dá 2,27.

**Diagnóstico.**
- Os ω_c crescem com a evidência: as medianas vão de 0,013 (classe 0) a
  0,058 (classe 4).
- Visões por mês, medianas: ~2.700 na classe 0 e 190–660 nas classes 1–4.
- A projeção PSD não agiu em nenhum mês.
- O número de condição mediano **cai** de 183 (setor) para 137. Mesmo assim, a
  variância realizada sobe.

**Conclusão.**
1. H2 é rejeitada também no quarto desenho.
2. Confiança por par não resolve o problema. As visões movem milhares de
   covariâncias residuais em direções que a GMV explora, e o erro de cada
   visão (ω ≈ 0,01–0,06 em z) é da mesma ordem do sinal que ela carrega.
3. O placebo piora mais que o grafo verdadeiro. A informação existe, como na
   H1, mas não vira carteira melhor.

## 18. Desenho fixado antes de rodar: o roteiro `grafo_noticias_covariancia.md` (2026-10-26)

Escrito ANTES de implementar ou rodar qualquer parte deste desenho com dados
reais. Mudanças depois do commit desta seção precisam ser declaradas como tal.

O arquivo `grafo_noticias_covariancia.md` ("o roteiro", na raiz do
repositório) propõe 9 modelos (M1–M9), 13 hipóteses e um protocolo de
avaliação. Para não confundir com as H1–H3 e os M0–M4 das seções 1 e 8, as
hipóteses do roteiro aparecem aqui como **G1–G13**, e os modelos como **M1–M9
do roteiro**.

### 18.1 O que já foi visto (transparência)

- **Seções 8–17.** H1 confirmada, com t até 9,7 na correlação residual. H2
  rejeitada em quatro desenhos de carteira. O universo era `news.universe`
  (top-n por eventos).
- **Pertinentes ao roteiro:**
  - A "exposição comum" da seção 4.1 é o cosseno entre vetores de exposição
    com IDF, sem sinal. É quase a projeção bipartida do M2 com Ω = I.
    - Ela sobreviveu na H1 (t = 7,4).
    - Seção 8: as relações diretas tipadas quase não acrescentam, e a cadeia
      (`supply_chain`, `partnership`) fica negativa (t = −3,2) com controle
      de co-ocorrência.
- **Notebook `notebooks/tuplas_e_precos.ipynb` (exploratório, sem critério).**
  - Usou o S&P 500 point-in-time de 2019 a 2023, janela de 252 pregões.
  - A regra "exposição PSD", (1−λ)r̄ + λG com G = cosseno das exposições
    incluindo ligações indiretas, empatou com o LW de correlação constante
    no erro de correlação do mês seguinte (λ ≈ 0,006–0,019).
  - Shrinkage para o alvo de notícias: κ pior que o do LW correlação
    constante e igual ao placebo.
  - Médias simples com a amostral: 100% dos dias com matriz não PSD.

  **Esse período (2019–2023) é o holdout proposto abaixo, então o holdout não
  é virgem para construções do tipo M2.** Isso fica declarado. A separação
  dev/holdout ainda protege a escolha de hiperparâmetros, que é feita por
  regra automática no dev.
- **Nenhum dos itens novos rodou**: universo S&P 500 point-in-time nas
  regressões, M1, M3–M9, placebos P1–P6, normalizações, MCS, DM e QAP.

### 18.2 O que o roteiro pede e não dá para fazer (desvios declarados)

| item do roteiro | situação | motivo |
|---|---|---|
| 2.1 identificador permanente (CIK/PERMNO) | parcial | ticker + `graph_nodes.tickers`; o CIK da SEC é o atual |
| 2.2 rotulagem manual | **preparada, não executada** | a amostra estratificada e o cálculo das métricas ficam prontos; a rotulagem é humana (dois anotadores) |
| 2.2 `evidence_span` | ausente | o schema da extração não tem o campo; exigiria nova extração (GPU paga). Substituto: auditoria automática da presença literal dos agentes no texto da unidade |
| 5.1 retorno de deslistagem | ausente | o FNSPID não tem; retorno faltante conta 0, como nas seções 12–17 |
| 5.1 GICS | substituído | Fama-French 12/49 e SIC de 2 dígitos (SEC) |
| 5.2 covariância realizada intradiária; perda de Stein | ausentes | só há preço diário; com N > h a covariância do mês é singular e a perda de Stein não existe. A log-verossimilhança preditiva é a QLIKE com Σ_real = (1/h) Σ r r' (mesma ordenação) |
| market cap, book-to-market | substituídos | tamanho = log do volume financeiro mediano na janela; B/M fica de fora |
| 6.1 B7 com FF5 + momentum | substituído | fatores construídos no próprio universo: mercado, 12 indústrias FF, tamanho (volume financeiro) e momentum 12-1 |
| 6.1 B10 DCC; M7 DCC-X | fora | DCC composto com N ≈ 400 fica para depois; o B10 é EWMA |
| 6.3 grafos externos (Revere, TNIC, analistas) | fora | sem acesso |
| 7.1 holdout pós-cutoff do LLM | impossível | o FNSPID termina em 2023, antes do cutoff do Qwen2.5 |
| 7.1 anonimização e reextração | **preparada, não executada** | exige GPU (custo): só com confirmação explícita |
| 7.1 teste de eventos famosos | fora | sem lista de desfechos |
| 8 Dow 30, S&P 100 | substituídos | S&P 100 ≈ as 100 maiores do S&P por volume financeiro; sem Dow 30 |
| G4 surpresa de Kuttner | substituída | variação do yield de 2 anos (FRED DGS2) nos dias de FOMC (calendário do Fed) |

### 18.3 Dados comuns a tudo

- **Universo U_t.** Membros do S&P 500 no último registro estritamente antes
  de t (`data/reference/sp500`, fja05680, versão "(Updated)"), com retorno
  válido do FNSPID em ≥ 95% dos 252 pregões antes de t. São ~370–460 ações,
  ~350 depois de 2020-07 (tickers truncados). Antes de rodar, os setores
  (SIC → FF12/FF49) são completados para todos esses tickers pela etapa
  `sectors`. Sem setor, `same_*` = 0.
- **Datas.** t = 1º pregão de cada mês, de 2011-01 a 2023-12. A janela
  in-sample tem 252 pregões; o horizonte de carteira vai até o pregão
  anterior ao próximo t.
- **Grafo em t.** Eventos limpos (`news.graph_events`) com data em
  [t − 365 d, t). Peso de cada evento: sign·strength·0,5^(idade/τ). O τ
  padrão (meia-vida) é de 90 dias.
- **Pares e agentes.**
  - Aresta direta: w_ij = soma dos eventos entre i e j (os dois sentidos,
    todas as classes).
  - w⁺ e w⁻ são as somas dos eventos positivos e negativos, em módulo.
  - Por grupo de classe: w^(k).
  - Exposição: Ã_im = soma dos eventos entre a ação i e o nó m ∉ U_t (Fed,
    petróleo, empresas fora do universo...), com sinal. Pesa por IDF
    log(N/df_m), como na seção 4.1; as linhas são normalizadas.
- **Projeção bipartida:** g_ij = (ÃÃ')_ij, o cosseno com sinal. É PSD por
  construção; linha sem exposição fica com C^G_ii = 1 e zeros fora da
  diagonal.
- **Atenção.** m_i = nº de eventos de i na janela (inclui eventos com nós
  fora de U).
- **Normalizações (seção 2.4 do roteiro).** Aplicadas aos módulos; o sinal
  médio volta multiplicando:
  - bruta;
  - IDF (padrão da exposição);
  - grau: D^−½ |W| D^−½ (e D_s^−½ |A| D_a^−½ no bipartido);
  - PMI: max(PMI, 0) dos documentos em comum;
  - z contra o modelo de configuração: (w − k_i k_j/2m)/√(k_i k_j/2m).
- **Grupos de classe para G3.** São seis:
  - cadeia: `supply_chain`;
  - parceria: `partnership`;
  - concorrência: `competition`;
  - legal: `legal_regulatory`;
  - controle: `ma`, `ownership`;
  - impacto: `impact`.

  O resto vai para "outros", que entra como controle.

### 18.4 Hipóteses diádicas (G1, G2, G3, G6, G8, G9): teste e critério

**Painel.**
- Cada t, de 2011-01 a 2023-12, tem uma cross-section de pares i < j de U_t.
- y = z da correlação **residual** realizada nos h pregões a partir de t,
  com h ∈ {21, 63, 126}; o principal é h = 21.
- O resíduo vem do fator de mercado de U_t: b estimado na janela in-sample
  e aplicado à frente, como na seção 10.
- Regressores (padronizados na data):
  - sgnlog(w) = sign(w)·log(1+|w|);
  - g;
  - controles: z_hist residual (252 pregões), same_ff49, same_sic2,
    distância de tamanho, log(1+m_i)+log(1+m_j) e |log(1+m_i) − log(1+m_j)|.

**Inferência.**
- **Fama-MacBeth** com Newey-West, com L = 3 + h/21 defasagens.
- **QAP.** Em cada data, as linhas e colunas de todas as matrizes de notícia
  (w, g e atenção) são permutadas juntas. São 200 permutações (seed 7), cada
  uma refazendo o Fama-MacBeth.
- O p é o maior dos dois (NW unicaudal e QAP).

| | estatística principal | sucesso |
|---|---|---|
| **G1** | β de sgnlog(w), h = 21 | β > 0 com p < limite (18.8); positivo nos três subperíodos (2011–14, 2015–18, 2019–23); t > 2 só com pares intra-setor (FF12) e só com pares inter-setor |
| **G2** | β⁺ − β⁻, com log(1+w⁺) e log(1+w⁻) no lugar de sgnlog(w) | β⁺ − β⁻ > 0 com p < limite. O roteiro escreve a nula como β⁺ = −β⁻, mas, com w⁻ medido em módulo, "o sinal não importa" é β⁺ = β⁻; vale a segunda. A nula fraca β⁻ ≥ 0 também é reportada |
| **G3** | Wald da igualdade dos 6 β de grupo, com log(1+\|w^(k)\|) no lugar de sgnlog(w) e covariância NW da série de coeficientes | igualdade rejeitada com p < limite. Expectativas registradas agora: cadeia > 0; parceria ≥ 0 e pequena; legal ≤ 0; concorrência, controle e impacto sem sinal esperado. Os β individuais vão com Benjamini-Hochberg (exploratório) |
| G6 | interação sgnlog(w) × cobertura do par (log(1+m_i)+log(1+m_j)) e tercis de tamanho | descritiva; as duas direções do roteiro ficam registradas |
| **G8** | Δρ_{t→t+21} em Δsgnlog(w)_t (mudança contra t−1 mês), ρ_t e ρ_t − ρ̄_ij (média das 12 datas anteriores); reverso: Δsgnlog(w)_{t→t+1} em Δρ_{t−1→t} | b > 0 com p < limite **e** \|b padronizado\| > \|coeficiente reverso padronizado\| |
| **G9** | Fama-MacBeth por ação: beta de baixa futuro (63 pregões, dias de mercado < 0) em centralidade de autovetor de \|W\|+\|G\| em U_t, com controles: tamanho, beta passado e FF12 | coeficiente > 0 com p < limite. PageRank, grau, beta e co-assimetria são descritivos |

Os horizontes de 63 e 126 dias, as normalizações e os τ de 30, 180 e 365
dias são robustez, sem critério.

### 18.5 Avaliação walk-forward (G5, G7, G10–G13)

**Períodos.**
- O desenvolvimento (**dev**) vai de 2013-01 a 2018-12 (72 meses).
- O **holdout** vai de 2019-01 a 2023-12 (60 meses) e roda **uma vez**, com
  as escolhas do dev congeladas.
- Os coeficientes treinados (alvo do setor em R0, θ do M6) usam só os meses
  com período de carteira terminado antes de t, a partir de 2011.
- Embargo: zero dias. O grafo usa dt < t, e a data não tem hora.

**Estimadores.**

| código | Σ̂ | par |
|---|---|---|
| B0 | 1/N | — |
| B1 | amostral (pseudo-inversa) | — |
| B2 | LW identidade | — |
| B3 | LW correlação constante | — |
| B4 | LW fator único | — |
| **B5** | alvo de blocos setoriais: R_ij = correlação média do bloco (FF12_i, FF12_j), δ da fórmula da seção 4.3 | **referência principal** |
| B6 | LW não linear | — |
| B7 | fatores observáveis do universo (mercado, 12 indústrias FF, tamanho, momentum 12-1) + resíduo diagonal | M1 |
| B8 | POET: k por Bai-Ng (IC2, k ≤ 10), limiar adaptativo, C escolhido no dev ∈ {0,25; 0,5; 1} | M3 |
| B9 | fator de mercado + glasso na correlação residual, γ = 0, λ₀ escolhido no dev | M4 |
| B10 | EWMA (λ = 0,97, pseudo-inversa) | — |
| B11 | HRP com a covariância amostral | M8 |
| R0 | fator + resíduo setor (seção 10, β treinado no painel S&P) | elo com as seções 10–17 |
| **M2** | δ·T^G + (1−δ)·S, T^G = Δ C^G Δ, δ analítico (seção 4.3); τ e normalização escolhidos no dev | **modelo principal** |
| M2-abs, M2-bin | M2 com \|Ã\| e com 1[Ã ≠ 0] | G2 (carteira) |
| M2-δgmv, M2-δqlike | δ escolhido nos 12 meses anteriores, grade 0, 0,05, …, 1, pela variância realizada da GMV / pela QLIKE | escolha de δ |
| M2-res | fator + resíduo encolhido para C^G | R0 |
| M2+W | C^G = corr(G + α·K_β(W)), kernel de difusão no Laplaciano assinado, α e β do dev | — |
| M1 | fatores Fed (−ΔDGS2), petróleo (USO), China (FXI), dólar (UUP), ouro (GLD) + mercado; β_i ridge centrado em κ·ã_i,m (exposição com sinal ao nó), λ e κ escolhidos no dev | B7′ = mesmo modelo com MQO |
| M3 | POET com suporte do grafo: resíduo amostral só nos pares com w ≠ 0 ou g no quantil que iguala a densidade média do POET; projeção PSD | B8 e suporte placebo de mesma densidade |
| M4 | glasso ponderado, λ_ij = λ₀ exp(−γ\|w̃_ij\|), w̃ = g + sgnlog(w) reescalado para [−1, 1]; (λ₀, γ) do dev | B9 |
| M5 | Σ̂ = B^G F̂ B^G' + D̂, B^G = q autovetores do Laplaciano normalizado de C^G − I, q ∈ {3, 5, 10} do dev; e PCA-k + grafo-q contra PCA-(k+q) | — |
| M6 | C^G(θ, φ) = corr(Σ θ_k K(W^(k)) + Σ φ_k Ã^(k)Ã^(k)'), com θ, φ ≥ 0 por MQO não negativo diádico nos meses de treino | ablação por camada |
| M8 | comunidades (Louvain em G⁺ = max(G, 0)): modelo em blocos; HRP (a) com a distância de C^G, (b) com λC^G + (1−λ)ρ̂, λ do dev, (c) com bissecção restrita às comunidades no 1º nível | B11, FF12, clusters de correlação |

**Carteiras.**
- **GMV sem restrição (principal).**
- GMV long-only com teto de 5%.
- HRP.
- ERC (só com Σ̂ positiva definida).
- TE mínimo contra o SPY, long-only, teto de 5%.
- MVO com sinal fixo: μ = z transversal do momentum 12-1, γ = 5, long-only,
  teto de 5%.

As cinco últimas carteiras são descritivas e rodam para B0–B6, B8, R0, M2,
M3, M4 e M8.

**Configuração e métricas.**
- Custo de 10 bps sobre o giro contra os pesos derivados, como nas seções
  12–17.
- Métricas da seção 5 do roteiro:
  - vol.;
  - razão de previsão de risco;
  - estatística de viés da Barra;
  - Sharpe e Sortino;
  - MDD e CVaR 5%;
  - giro;
  - retorno líquido a 0, 5, 10 e 20 bps;
  - N efetivo e peso máximo;
  - κ;
  - QLIKE (principal estatística);
  - Frobenius em covariância e em correlação;
  - Spearman dos pares;
  - erro em pares com e sem aresta.

**Escolhas no dev (regra automática, registrada aqui).**

| modelo | o que se escolhe | grade | critério no dev |
|---|---|---|---|
| M2 | τ e normalização | τ ∈ {30, 90, 180, 365}, normalização ∈ {IDF, bruta, grau, PMI, z} | menor variância realizada da GMV |
| B9, M4 | λ₀ e γ (γ = 0 no B9) | λ₀ ∈ {0,02, 0,05, 0,1, 0,2}, γ ∈ {0, 0,5, 1, 2} | maior log-verossimilhança gaussiana fora da amostra |
| B8 | C | {0,25; 0,5; 1} | menor variância realizada da GMV |
| M5 | q | {3, 5, 10} | menor QLIKE |
| M1 | λ e κ | (*) | menor QLIKE |
| M2+W | α e β | (*) | menor variância realizada da GMV |
| M8 (b) | λ | {0,25; 0,5; 0,75} | menor variância realizada do HRP |

(*) As grades de M1 e M2+W ficam no código do commit da implementação,
antes de qualquer execução no dev.

Empates (diferença < 0,1%) ficam com o padrão: τ = 90 e IDF.

**Placebos (G10)** no M2 escolhido, R = 200 por data (seed 7):

| placebo | preserva |
|---|---|
| P1 | os valores não nulos de Ã sorteados em posições aleatórias |
| **P2** | rewiring do bipartido com grau de cada ação e de cada agente preservado (curveball), o peso viaja com a aresta |
| P3 | permutação das ações dentro do setor FF12 |
| P4 | sinais aleatórios |
| P5 | o grafo de t − 2 anos |
| P6 | só co-menção: cosseno dos vetores de documentos das marcações do FNSPID; e P6b, co-ocorrência nos artigos extraídos, sem tipo nem sinal |

### 18.6 Critérios das hipóteses de carteira (holdout)

| | estatística | sucesso |
|---|---|---|
| **G5** | razão de variâncias dos retornos diários líquidos da GMV M2 / B5 (bootstrap circular em blocos de 10 dias, 2.000 reamostragens, seed 7, bicaudal) | razão < 1 com **p < 0,01** (quinto desenho da H2 econômica: 0,05/5) **e** p ajustado (18.8) < 0,05. Os critérios do roteiro também são reportados: M2 no MCS a 90% (QLIKE mensal e r² diário da GMV) com o B5 fora, ou DM (HLN) significativo |
| G7 | regressão da diferença mensal de perda (QLIKE; variância da GMV no mês) B5 − M2 em tercis de vol. realizada do SPY nos 21 pregões antes de t (cortes do dev) e nos episódios 2018Q4, 2020Q1 e 2022; HAC | descritiva |
| **G10** | variância da GMV do M2 contra a distribuição dos 200 P2 | M2 abaixo do percentil 5 dos P2 (p empírico < 0,05) **e** razão M2/P6 < 1 com p < 0,05; P1, P3, P4 e P5 são descritivos |
| **G11** | ganho de log-verossimilhança fora da amostra do M4 sobre o B9 (média mensal, DM-HLN); e fração de pares com aresta e correlação parcial ≠ 0 contra a fração sem aresta (Fisher por mês, combinado) | DM > 0 com p < limite |
| **G12** | média, nos meses, de mean cos θ_ℓ entre B^G (q do dev) e os q primeiros autovetores da covariância amostral dos 63 pregões seguintes | acima do percentil 95 de 200 P2 |
| **G13** | razão de variâncias HRP-(a) / B11 | < 1 com p < limite; ARI/NMI de comunidades × FF12 × clusters de correlação e o modelo em blocos são descritivos |

### 18.7 Seção 2 e 7 do roteiro: extração e descritivas

- **Etapa 1 (validação).**
  - Auditoria automática: fração das tuplas limpas em que agent_a e agent_b
    aparecem no texto da unidade extraída, literalmente ou por token
    distintivo. O texto vem de `extraction_units`, copiado do Drive para
    `data/curated`.
  - Amostra estratificada para rotulagem: 25 tuplas por classe nas 16
    classes × 2 períodos (2011–2018, 2019–2023) = 800 tuplas na planilha.
    Mais 100 artigos inteiros para o recall.
  - Script que calcula as métricas da tabela 2.2 do roteiro quando a planilha
    voltar preenchida.
  - Sem critério até haver rótulos.
- **Etapa 2 (descritivas da seção 2.5).** Por ano:
  - nós e arestas;
  - distribuição de graus;
  - fração de arestas negativas por classe;
  - cobertura por quintil de tamanho;
  - Jaccard entre anos consecutivos;
  - fração intra-setor.

  Portão descritivo, "o grafo não é só setor reescrito": R² de g em
  same_ff49 + same_sic2 + same_ff12 < 0,5. Se falhar, segue-se e
  declara-se.
- **Vazamento (7.1).** Preparar o script de anonimização (máscaras
  `[EMPRESA_A]` em 300 unidades) para rodar na GPU só com autorização.

### 18.8 Múltiplos testes e robustez

- **Família confirmatória:** G1, G2, G3, G5, G8, G9, G10, G11, G12 e G13
  (10 testes, um p principal cada). Holm a 5% (FWER sob qualquer
  dependência), mais o limite próprio de G5. "p < limite" acima é p de Holm
  < 0,05. Romano-Wolf entra só dentro da comparação de estimadores do G5 (as
  perdas são bootstrapadas juntas). Hansen SPA e White RC testam "o melhor
  modelo com grafo contra B5" levando em conta todos os modelos com grafo.
- **Sharpe deflacionado** do M2, com o número de configurações avaliadas no
  dev (log em `logs/secao18_dev.log`).
- **Robustez (seção 8 do roteiro), uma dimensão por vez, na comparação
  principal M2 contra B5:**
  - janela de 126 e de 504 pregões;
  - rebalanceamento semanal e trimestral;
  - meia-vida;
  - normalização;
  - kernel (Gram, difusão, Higham de I + αW̃);
  - universo (S&P 100 aproximado, sem financeiras SIC 60–67, `news.universe`
    n = 100);
  - por ano, e antes e depois de 2020;
  - log contra retorno simples;
  - winsorização 1/99%;
  - custos de 0, 5 e 20 bps;
  - restrições (long-only com teto);
  - agentes: só nós sem ticker;
  - sem artigos de *market wrap* (unidade com ≥ 5 tickers distintos nos
    eventos);
  - bootstrap estacionário com bloco de Politis-White.

  Todas são reportadas; robusto = mesmo sinal na maioria.

### 18.9 G4 (exposição a agentes em dias de evento)

- **Dias de evento:** o dia do comunicado de cada reunião do FOMC, agendada
  ou extraordinária, de 2011 a 2023 (último dia da reunião, ou o 1º pregão
  depois dele). Fonte: calendário do Fed. Ficam de fora *notation vote*,
  conference call e reunião cancelada.
- **Surpresa:** ΔDGS2 do dia, do FRED.
- **Modelo:** r_i = a_i + b_i r_m + c (ã_i,Fed × surpresa) + d·surpresa + e,
  com o controle β̂_hist,Fed × surpresa.
- **Exploratório, fora da família:** a surpresa é um proxy diário, não a de
  Kuttner.

### 18.10 Implementação e ordem

- **Código:**
  - pacote `newsmanager/grafo/`, com etapas `python -m newsmanager.grafo
    <etapa>`;
  - testes sintéticos em `tests/test_grafo.py`:
    - placebos preservam graus e setores;
    - glasso ponderado satisfaz KKT;
    - POET, kernel de difusão PSD;
    - MCS, DM-HLN, Politis-White e Holm contra casos conhecidos;
  - scripts em `backtests/secao18/`.
- **Ordem:**
  1. setores completos;
  2. etapas 1 e 2;
  3. diádico;
  4. dev, com escolhas congeladas e registradas em
     `logs/secao18_escolhas.json`;
  5. holdout, uma vez;
  6. robustez.

  Antes do holdout, as escolhas do dev vão para uma nota nesta seção.

*Nota de implementação (2026-10-26, escrita no mesmo dia da seção, antes de
qualquer execução nos períodos avaliados).* São detalhes que a tabela 18.5
deixava em aberto, fixados no código:

- **M1.**
  - O mercado (média igual) entra como fator sem penalidade.
  - Prior: β_im = κ·dp_i(β̂_MQO,m)·ã_im / max_i|ã_im|.
  - Penalidade: λ_m = c·T·Var(f_m).
  - Grade: κ ∈ {0; 0,5; 1; 2}, c ∈ {0,1; 1; 10}.
- **M2+W.** C = corr(C^G + α·K_β(W/max|W|)), com α ∈ {0,25; 0,5; 1} e
  β ∈ {0,5; 1; 2}.
- **M4.** w̃ = (C^G − I) + sgnlog(w)/max|sgnlog(w)|, reescalado para máximo 1.
- **M3.** São os d pares de maior 10·1[w ≠ 0] + g, com d = nº de pares do
  POET com o C escolhido. O C do POET sobe ×1,25 até Σ̂ ser PD.
- **M5.** As cargas incluem uma coluna de mercado (1/√p) além dos q
  autovetores.
- **M6.**
  - 14 camadas: difusão (β = 1, W/max) de cada um dos 7 grupos de classe e
    Gram (IDF, com sinal) por tipo de agente (empresa, governo/banco central,
    commodity, país, moeda, macro, setor/índice).
  - c ≥ 0 por MQO não negativo na correlação residual realizada (ρ, não z),
    centrada por mês.
  - Alvo residual: R = nearest_corr(r̄_e + Σ c_k (L_k − L̄_k)) dentro do
    "fator + resíduo", como no R0.
- **MVO.** μ_i = 0,05·σ_i·z_i (Grinold e Kahn, IC = 0,05).
- **Σ̂ singulares.** B1 e B10 usam a pseudo-inversa na GMV. Também ficam com
  QLIKE indefinida e, por isso, fora do MCS de QLIKE. O ERC só roda com Σ̂ PD.
- **Placebos.** P1, P2 e P4 agem sobre Ã já ponderado, antes da
  normalização das linhas. O P5 é o snapshot em t − 730 dias com o mesmo
  U_t.
- **G8.** ρ_t é a correlação residual dos últimos 21 pregões da janela; ρ̄
  é a média (em z) dos 12 blocos de 21 pregões; y é padronizado por data.
- **G9.** A centralidade usa |W|/max|W| + |C^G − I|.
- **G12.** V_q são os autovetores da covariância amostral dos retornos
  totais nos 63 pregões após t.
- **Painel diádico.** A correlação futura exige ≥ 70% dos h pregões.
- **Teste de fumaça (sem olhar métricas).**
  - Walk-forward com todos os estimadores e as variantes de robustez em
    2012-06/07, que é só treino.
  - Painel diádico de 2011-01 a 2011-03, numa pasta separada, apagada em
    seguida.
  - O código só foi conferido quanto a rodar.
- **Testes sintéticos:** `tests/test_grafo.py`.

## Referências

- Anderson, T. W. (2003). *An Introduction to Multivariate Statistical Analysis*, 3ª ed. Wiley.
- Aronow, P.; Samii, C.; Assenova, V. (2015). Cluster-robust variance estimation for dyadic data. *Political Analysis* 23(4).
- Bai, J.; Ng, S. (2002). Determining the number of factors in approximate factor models. *Econometrica* 70(1).
- Bailey, D.; López de Prado, M. (2014). The deflated Sharpe ratio. *J. Portfolio Management* 40(5).
- Becquin, G.; Esmeir, S. (2023). Semantic Similarity Covariance Matrix Shrinkage. *Findings of EMNLP 2023*, 9977–9992. https://aclanthology.org/2023.findings-emnlp.668/
- Benjamini, Y.; Hochberg, Y. (1995). Controlling the false discovery rate. *JRSS-B* 57(1).
- Black, F.; Litterman, R. (1992). Global portfolio optimization. *Financial Analysts Journal* 48(5), 28–43.
- Blondel, V.; Guillaume, J.-L.; Lambiotte, R.; Lefebvre, E. (2008). Fast unfolding of communities in large networks. *J. Stat. Mech.* P10008.
- Bun, J.; Bouchaud, J.-P.; Potters, M. (2017). Cleaning large correlation matrices: tools from random matrix theory. *Physics Reports* 666.
- Cai, T.; Liu, W. (2011). Adaptive thresholding for sparse covariance matrix estimation. *JASA* 106.
- Chan, L.; Karceski, J.; Lakonishok, J. (1999). On portfolio optimization: forecasting covariances and choosing the risk model. *RFS* 12, 937–974.
- Choueifaty, Y.; Coignard, Y. (2008). Toward maximum diversification. *J. Portfolio Management* 35(1).
- Cohen, L.; Frazzini, A. (2008). Economic links and predictable returns. *Journal of Finance* 63(4).
- Cornuéjols, G.; Tütüncü, R. (2007). *Optimization Methods in Finance*. Cambridge University Press.
- DeMiguel, V.; Garlappi, L.; Uppal, R. (2009). Optimal versus naive diversification. *RFS* 22(5).
- Diebold, F.; Mariano, R. (1995). Comparing predictive accuracy. *JBES* 13(3).
- Elton, E.; Gruber, M. (1973). Estimating the dependence structure of share prices. *Journal of Finance* 28.
- Engle, R.; Colacito, R. (2006). Testing and valuing dynamic correlations for asset allocation. *JBES* 24(2).
- Fama, E.; MacBeth, J. (1973). Risk, return, and equilibrium: empirical tests. *JPE* 81(3).
- Fan, J.; Liao, Y.; Mincheva, M. (2013). Large covariance estimation by thresholding principal orthogonal complements. *JRSS-B* 75(4).
- Friedman, J.; Hastie, T.; Tibshirani, R. (2008). Sparse inverse covariance estimation with the graphical lasso. *Biostatistics* 9(3).
- Guo, L.; Peng, L.; Tao, Y.; Tu, J. Joint News, Attention Spillover, and Stock Returns. arXiv:1703.02715.
- Hansen, P. (2005). A test for superior predictive ability. *JBES* 23(4).
- Hansen, P.; Lunde, A.; Nason, J. (2011). The model confidence set. *Econometrica* 79(2).
- Harvey, D.; Leybourne, S.; Newbold, P. (1997). Testing the equality of prediction mean squared errors. *Int. J. Forecasting* 13(2).
- He, G.; Litterman, R. (1999). The intuition behind Black-Litterman model portfolios. Goldman Sachs Investment Management Research.
- Higham, N. (2002). Computing the nearest correlation matrix. *IMA J. Numer. Anal.* 22.
- Hoberg, G.; Phillips, G. (2016). Text-based network industries and endogenous product differentiation. *JPE* 124(5).
- Holm, S. (1979). A simple sequentially rejective multiple test procedure. *Scand. J. Statistics* 6(2).
- Jagannathan, R.; Ma, T. (2003). Risk reduction in large portfolios: why imposing the wrong constraints helps. *J. Finance* 58(4).
- Jorion, P. (1986). Bayes-Stein estimation for portfolio analysis. *JFQA* 21(3).
- Kan, R.; Zhou, G. (2007). Optimal portfolio choice with parameter uncertainty. *JFQA* 42(3).
- Kondor, R.; Lafferty, J. (2002). Diffusion kernels on graphs and other discrete input spaces. *ICML*.
- Krackhardt, D. (1988). Predicting with networks: nonparametric multiple regression analysis of dyadic data. *Social Networks* 10(4).
- Kunegis, J. et al. (2010). Spectral analysis of signed graphs for clustering, prediction and visualization. *SDM*.
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
- Newey, W.; West, K. (1987). A simple, positive semi-definite, heteroskedasticity and autocorrelation consistent covariance matrix. *Econometrica* 55(3).
- Patton, A. (2011). Volatility forecast comparison using imperfect volatility proxies. *J. Econometrics* 160(1).
- Peralta, G.; Zareei, A. (2016). A network approach to portfolio selection. *J. Empirical Finance* 38.
- Politis, D.; Romano, J. (1994). The stationary bootstrap. *JASA* 89(428).
- Politis, D.; White, H. (2004). Automatic block-length selection for the dependent bootstrap. *Econometric Reviews* 23(1); correção de Patton, Politis e White (2009).
- Pozzi, F.; Di Matteo, T.; Aste, T. (2013). Spread of risk across financial markets: better to invest in the peripheries. *Scientific Reports* 3, 1665.
- Qian, E.; Gorman, S. (2001). Conditional distribution in portfolio theory. *Financial Analysts Journal* 57(2), 44–51.
- Romano, J.; Wolf, M. (2005). Stepwise multiple testing as formalized data snooping. *Econometrica* 73(4).
- Scherbina, A.; Schlusche, B. (2016). Economic linkages inferred from news stories and the predictability of stock returns. AEI WP; (2020) Follow the leader, *Review of Finance* 24(1).
- Strona, G. et al. (2014). A fast and unbiased procedure to randomize ecological binary matrices with fixed row and column totals. *Nature Communications* 5, 4114.
- Tang, Y.; Zhou, Y.; Hong, M. (2019). News co-occurrences, stock return correlations, and portfolio construction implications. *JRFM* 12(1), 45.
- Tumminello, M.; Aste, T.; Di Matteo, T.; Mantegna, R. (2005). A tool for filtering information in complex systems. *PNAS* 102(30).
- White, H. (2000). A reality check for data snooping. *Econometrica* 68(5).
