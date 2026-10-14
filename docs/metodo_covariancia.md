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

## Referências

- Becquin, G.; Esmeir, S. (2023). Semantic Similarity Covariance Matrix Shrinkage. *Findings of EMNLP 2023*, 9977–9992. https://aclanthology.org/2023.findings-emnlp.668/
- Bun, J.; Bouchaud, J.-P.; Potters, M. (2017). Cleaning large correlation matrices: tools from random matrix theory. *Physics Reports* 666.
- Chan, L.; Karceski, J.; Lakonishok, J. (1999). On portfolio optimization: forecasting covariances and choosing the risk model. *RFS* 12, 937–974.
- Cohen, L.; Frazzini, A. (2008). Economic links and predictable returns. *Journal of Finance* 63(4).
- DeMiguel, V.; Garlappi, L.; Uppal, R. (2009). Optimal versus naive diversification. *RFS* 22(5).
- Elton, E.; Gruber, M. (1973). Estimating the dependence structure of share prices. *Journal of Finance* 28.
- Engle, R.; Colacito, R. (2006). Testing and valuing dynamic correlations for asset allocation. *JBES* 24(2).
- Friedman, J.; Hastie, T.; Tibshirani, R. (2008). Sparse inverse covariance estimation with the graphical lasso. *Biostatistics* 9(3).
- Guo, L.; Peng, L.; Tao, Y.; Tu, J. Joint News, Attention Spillover, and Stock Returns. arXiv:1703.02715.
- Higham, N. (2002). Computing the nearest correlation matrix. *IMA J. Numer. Anal.* 22.
- Hoberg, G.; Phillips, G. (2016). Text-based network industries and endogenous product differentiation. *JPE* 124(5).
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
- Scherbina, A.; Schlusche, B. (2016). Economic linkages inferred from news stories and the predictability of stock returns. AEI WP; (2020) Follow the leader, *Review of Finance* 24(1).
- Tang, Y.; Zhou, Y.; Hong, M. (2019). News co-occurrences, stock return correlations, and portfolio construction implications. *JRFM* 12(1), 45.
- Tumminello, M.; Aste, T.; Di Matteo, T.; Mantegna, R. (2005). A tool for filtering information in complex systems. *PNAS* 102(30).
