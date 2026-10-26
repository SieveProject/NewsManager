# Grafos de Notícias Extraídos por LLM para Estimação de Covariância e Seleção de Portfólios

**Ideias de modelagem, hipóteses testáveis e protocolo de avaliação**

---

## Sumário

0. [Visão geral e pergunta de pesquisa](#0-visão-geral-e-pergunta-de-pesquisa)
1. [Notação](#1-notação)
2. [Do corpus ao grafo: construção e validação](#2-do-corpus-ao-grafo-construção-e-validação)
3. [Modelos](#3-modelos)
   - M1 – Agentes como fatores observáveis com cargas a priori de notícias
   - M2 – Projeção bipartida como alvo de shrinkage
   - M3 – Arestas ação–ação no bloco residual (estilo Barra)
   - M4 – Graphical lasso com penalização ponderada por notícias
   - M5 – Fatores espectrais do grafo
   - M6 – Grafo multiplex por tipo de relação
   - M7 – Grafo dinâmico e correlação variante no tempo
   - M8 – Comunidades do grafo para HRP e estrutura em blocos
   - M9 – Centralidade como fator de risco sistêmico
4. [Hipóteses e desenho dos testes](#4-hipóteses-e-desenho-dos-testes)
5. [Protocolo de avaliação](#5-protocolo-de-avaliação)
6. [Baselines e controles](#6-baselines-e-controles)
7. [Vazamento de informação e vieses específicos de LLM](#7-vazamento-de-informação-e-vieses-específicos-de-llm)
8. [Robustez](#8-robustez)
9. [Inferência estatística e múltiplos testes](#9-inferência-estatística-e-múltiplos-testes)
10. [Roteiro experimental sugerido](#10-roteiro-experimental-sugerido)
11. [Referências](#11-referências)

---

## 0. Visão geral e pergunta de pesquisa

A partir de um grande corpus de notícias, um LLM extrai tuplas

$$
(\text{agente}_a,\ \text{agente}_b,\ \text{tipo de relação } k,\ \text{força } s \in [-1, 1],\ \text{data } t)
$$

onde os agentes podem ser ações (empresas) ou entidades econômicas não negociáveis diretamente (Fed, petróleo, China, reguladores, sindicatos etc.).

**Pergunta central:** a estrutura relacional extraída das notícias contém informação sobre a matriz de covariância futura dos retornos que **não** está contida (i) na covariância amostral histórica e (ii) em modelos de fatores padrão (mercado, setor, estilo)? E essa informação se traduz em portfólios melhores fora da amostra?

**Intuição estrutural.** O grafo é um multigrafo assinado, tipado e datado, com dois tipos de nós. Isso mapeia naturalmente nos componentes de um modelo de fatores:

$$
\Sigma = B\,F\,B^\top + D
$$

| Componente do grafo | Componente do modelo |
|---|---|
| Arestas ação–agente | Cargas $B$ (exposição a fatores macro/econômicos) |
| Arestas ação–ação | Estrutura fora da diagonal de $D$ (correlação residual) |
| Comunidades do grafo | Fatores em bloco ("indústrias implícitas") |
| Dinâmica das arestas | Variação temporal de $B_t$, $D_t$ |

---

## 1. Notação

| Símbolo | Significado |
|---|---|
| $N$ | Número de ações no universo (point-in-time) |
| $M$ | Número de agentes não-ação |
| $K$ | Número de tipos de relação |
| $r_t \in \mathbb{R}^N$ | Vetor de retornos (log ou simples, excesso sobre RF) no dia $t$ |
| $S_t$ | Covariância amostral numa janela de $T_w$ dias até $t$ |
| $\Sigma_{t+1:t+h}$ | Covariância realizada no horizonte de avaliação |
| $W^{(k)}_t \in \mathbb{R}^{N\times N}$ | Matriz de adjacência ação–ação, tipo $k$, assinada |
| $A^{(k)}_t \in \mathbb{R}^{N\times M}$ | Matriz de exposição ação–agente, tipo $k$, assinada |
| $L_t$ | Laplaciano (assinado) do grafo |
| $\hat\Sigma$ | Estimador de covariância genérico |
| $C$ | Matriz de correlação; $\Delta = \operatorname{diag}(\Sigma)^{1/2}$ |
| $\delta$ | Intensidade de shrinkage |

---

## 2. Do corpus ao grafo: construção e validação

A qualidade de tudo que vem depois depende desta etapa. Ela deve ser tratada como um experimento próprio, com métricas próprias.

### 2.1 Resolução de entidades

- Mapear menções a tickers com **identificador permanente** (CIK, PERMNO ou equivalente), não ticker, pois tickers mudam e são reutilizados.
- Tratar fusões, spin-offs, mudanças de nome com tabela de vigência por data.
- Agentes não-ação: criar um **vocabulário controlado** (ontologia fechada: `FED`, `ECB`, `OIL`, `CHINA`, `USD`, `TREASURY_YIELDS`, `REGULATOR_SEC`, ...). Agentes fora da ontologia vão para um bucket `OTHER` que é descartado ou analisado à parte.

### 2.2 Validação da extração (obrigatório)

Amostrar aleatoriamente $n \approx 500$–$1000$ tuplas, estratificadas por tipo de relação e período, e rotular manualmente (idealmente dois anotadores).

| Métrica | O que mede |
|---|---|
| Precisão de entidade | Agentes corretamente identificados |
| Precisão da relação | A relação existe de fato no texto |
| Acurácia do tipo $k$ | Classificação correta; reportar matriz de confusão |
| Acurácia do sinal | $\operatorname{sign}(s)$ correto |
| Correlação de Spearman da força | Entre $s$ do LLM e nota humana ordinal |
| Cohen's $\kappa$ | Concordância entre anotadores (teto de desempenho) |
| Recall (amostra de artigos) | Fração das relações reais capturadas, avaliado em ~100 artigos lidos integralmente |

Exigir no schema Pydantic um campo `evidence_span` (trecho literal do artigo que sustenta a relação). Isso permite auditoria automática (o trecho existe no artigo?) e reduz alucinação.

### 2.3 Agregação temporal

As tuplas são eventos. Para obter um grafo no tempo $t$, agregar com decaimento exponencial:

$$
w_{ij,t}^{(k)} = \sum_{e \in \mathcal{E}^{(k)}_{ij},\ t_e \le t} s_e \cdot e^{-(t - t_e)/\tau}
$$

com meia-vida $\tau \ln 2$ tratada como hiperparâmetro (grade: 30, 90, 180, 365 dias). Escolher $\tau$ **apenas** na janela de validação.

Usar a data de publicação, com regra de corte conservadora: notícia publicada após o fechamento do dia $t$ entra em $t+1$.

### 2.4 Normalização contra viés de cobertura

Grandes empresas aparecem muito mais nas notícias. Sem correção, o grafo mede atenção da mídia, não estrutura econômica. Opções (testar todas como variantes):

1. **PMI (pointwise mutual information)** de co-menção:
   $$\text{PMI}_{ij} = \log \frac{P(i,j)}{P(i)\,P(j)}$$
   e manter a força assinada média condicional: $\tilde w_{ij} = \bar s_{ij}\cdot \max(\text{PMI}_{ij}, 0)$.
2. **Normalização por grau** (simétrica): $\tilde W = D_{|W|}^{-1/2}\, W\, D_{|W|}^{-1/2}$, com $D_{|W|} = \operatorname{diag}(\sum_j |w_{ij}|)$.
3. **z-score contra um modelo nulo** de configuração (grafo aleatório com o mesmo grau esperado): $z_{ij} = (w_{ij} - \mathbb{E}_0[w_{ij}])/\sigma_0[w_{ij}]$.

### 2.5 Estatísticas descritivas a reportar

Número de nós/arestas por ano, distribuição de graus, fração de arestas negativas por tipo, cobertura por quintil de market cap, estabilidade temporal (correlação de Jaccard entre grafos de anos consecutivos), sobreposição com setores GICS (fração de arestas intra-setor).

---

## 3. Modelos

Para cada modelo: motivação, especificação, estimação, hipóteses associadas e riscos.

### Convenção para alvos de correlação

Muitos modelos produzem uma matriz de correlação alvo $C^{G}$ derivada do grafo. A conversão para covariância usa as variâncias amostrais (ou de um modelo GARCH univariado):

$$
T^{G} = \hat\Delta\, C^{G}\, \hat\Delta
$$

**Positividade semidefinida (PSD).** Uma matriz de adjacência assinada não é, em geral, PSD. Garantir PSD por uma das vias:

- **Kernel de difusão** sobre o Laplaciano assinado (Kunegis et al., 2010): $L_\sigma = \bar D - W$, com $\bar D = \operatorname{diag}(\sum_j |w_{ij}|)$, que é PSD. Então
  $$K_\beta = \exp(-\beta L_\sigma),\qquad C^{G} = \operatorname{diag}(K_\beta)^{-1/2} K_\beta \operatorname{diag}(K_\beta)^{-1/2}$$
  $K_\beta$ é sempre positiva definida (Kondor & Lafferty, 2002).
- **Projeção na correlação mais próxima** (Higham, 2002) de $I + \alpha \tilde W$.
- **Forma de Gram**: $C^G \propto A\Omega A^\top + \operatorname{diag}(\cdot)$, PSD por construção (ver M2).

---

### M1 – Agentes como fatores observáveis com cargas a priori de notícias

**Motivação.** Se a notícia diz que a empresa $i$ é afetada negativamente por altas de juros, isso é informação sobre o sinal e a magnitude de $\beta_{i,\text{Fed}}$, potencialmente antes que a regressão histórica perceba (ex.: mudança de modelo de negócio, nova dívida).

**Especificação.** Para cada agente $m$ com série de choques observável $f_{m,t}$:

| Agente | Proxy de fator |
|---|---|
| Fed | Variação do yield de 2 anos; surpresas de política monetária em dias de FOMC |
| Petróleo | Retorno do WTI/Brent |
| China | Retorno de FXI ou variação de USD/CNH |
| Dólar | Retorno do DXY |
| Crédito | Variação de spread HY/IG |
| Agentes sem série | Portfólio mimetizador (long exposições positivas, short negativas) |

A carga a priori vem do grafo: $\beta^{\text{news}}_{im} = \kappa_m \cdot \tilde a_{im,t}$, com escala $\kappa_m$ calibrada.

**Estimação.** Regressão ridge com centro no prior:

$$
\hat\beta_i = \arg\min_{\beta}\ \|r_i - X\beta\|_2^2 + \lambda\,\|\beta - \beta^{\text{news}}_i\|_2^2
\quad\Rightarrow\quad
\hat\beta_i = (X^\top X + \lambda I)^{-1}(X^\top r_i + \lambda \beta^{\text{news}}_i)
$$

Equivalente a um posterior bayesiano com prior gaussiano $\beta_i \sim \mathcal{N}(\beta^{\text{news}}_i, \sigma^2/\lambda\, I)$. Escolher $\lambda$ e $\kappa$ por validação temporal. Variante hierárquica: $\kappa_m$ por agente estimado por máxima verossimilhança marginal.

Covariância implícita: $\hat\Sigma = \hat B \hat F \hat B^\top + \hat D$, com $\hat F$ amostral (ou EWMA) dos fatores e $\hat D$ diagonal dos resíduos.

**Hipóteses associadas:** H4, H5.

**Riscos.** Fatores macro explicam pouca variância diária de ações individuais; o ganho pode aparecer só em dias de evento. Avaliar também condicional a dias de FOMC/CPI.

---

### M2 – Projeção bipartida como alvo de shrinkage

**Motivação.** Duas ações tendem a covariar se estão expostas aos mesmos agentes na mesma direção. Esse é o candidato mais limpo a contribuição principal, pois se encaixa diretamente num estimador consolidado (Ledoit–Wolf).

**Especificação.**

$$
G = \tilde A\, \Omega\, \tilde A^\top,\qquad C^{G} = \operatorname{diag}(G + \epsilon I)^{-1/2}\,(G + \epsilon I)\,\operatorname{diag}(G + \epsilon I)^{-1/2}
$$

Escolhas para $\Omega$ (variantes):
- $\Omega = I$: todos os agentes com peso igual.
- $\Omega = \hat F$: covariância empírica das séries de fatores dos agentes (combina estrutura de notícias com dados de mercado).
- $\Omega$ diagonal com pesos aprendidos por validação.

Opcionalmente somar o termo ação–ação: $C^{G} \leftarrow \text{corr}(G + \alpha\, K_\beta(W))$.

**Estimador final:**

$$
\hat\Sigma(\delta) = \delta\, T^{G} + (1 - \delta)\, S
$$

**Escolha de $\delta$** (comparar as três):
1. Fórmula analítica de Ledoit–Wolf generalizada para alvo arbitrário (minimiza perda de Frobenius esperada):
   $$\delta^\ast = \frac{\sum_{ij}\widehat{\operatorname{Var}}(s_{ij}) - \widehat{\operatorname{Cov}}(s_{ij}, t_{ij})}{\sum_{ij}(t_{ij} - s_{ij})^2}$$
   truncada em $[0,1]$. Atenção: o termo de covariância entre $S$ e $T^G$ é zero se $C^G$ não depende dos retornos (só de notícias), mas não se $T^G$ usa $\hat\Delta$ ou $\hat F$. Derivar com cuidado ou usar a aproximação.
2. Validação cruzada temporal minimizando a variância realizada do portfólio GMV.
3. Validação minimizando QLIKE.

**Hipóteses associadas:** H5, H6, H7, H10.

---

### M3 – Arestas ação–ação no bloco residual (estilo Barra)

**Motivação.** Isolar o que as notícias acrescentam **além** de setor e fatores conhecidos. É o confundidor principal: empresas do mesmo setor co-aparecem em notícias e co-movem por razões setoriais.

**Especificação.**

1. Ajustar modelo base: $r_t = B^{0} f_t + \varepsilon_t$, com $B^0$ = mercado + dummies setoriais (GICS nível 2 ou 3) + estilos (FF5 / momentum), ou $k$ fatores de PCA (POET; Fan, Liao & Mincheva, 2013).
2. Modelar a covariância residual $\Sigma_\varepsilon$ com estrutura induzida pelo grafo:
   $$\Sigma_\varepsilon = \hat\Delta_\varepsilon\,\big[(1-\gamma) I + \gamma\, C^{G}_{\text{res}}\big]\,\hat\Delta_\varepsilon$$
   ou por thresholding seletivo: manter $\hat\sigma^\varepsilon_{ij}$ amostral apenas onde $|w_{ij}| > 0$ e zerar o resto (POET com suporte informado).

**Teste-chave.** Comparar POET padrão (threshold adaptativo uniforme) contra POET com suporte dado pelo grafo, e contra suporte dado por um grafo placebo com mesma esparsidade.

**Hipóteses associadas:** H1, H2, H5.

---

### M4 – Graphical lasso com penalização ponderada por notícias

**Motivação.** Relações de cadeia de suprimento e concorrência são, conceitualmente, **ligações diretas**. A matriz de precisão $\Theta = \Sigma^{-1}$ codifica dependência condicional (correlação parcial), que é o objeto certo para ligações diretas.

**Especificação.**

$$
\hat\Theta = \arg\min_{\Theta \succ 0}\ -\log\det\Theta + \operatorname{tr}(S\Theta) + \sum_{i \ne j} \lambda_{ij}\,|\Theta_{ij}|,
\qquad \lambda_{ij} = \lambda_0 \exp(-\gamma\, |\tilde w_{ij}|)
$$

$\gamma = 0$ recupera o glasso padrão (Friedman, Hastie & Tibshirani, 2008). Selecionar $(\lambda_0, \gamma)$ por validação temporal (log-verossimilhança gaussiana fora da amostra).

Aplicar preferencialmente aos **resíduos** de um modelo de fatores (fatores + precisão esparsa residual), porque a precisão de retornos brutos não é esparsa devido ao fator mercado.

**Diagnóstico adicional.** Testar se o sinal da aresta prediz o sinal da correlação parcial: correlação parcial estimada $\hat\rho^{\text{par}}_{ij} = -\hat\Theta_{ij}/\sqrt{\hat\Theta_{ii}\hat\Theta_{jj}}$ vs $\operatorname{sign}(w_{ij})$.

**Hipóteses associadas:** H1, H2, H11.

---

### M5 – Fatores espectrais do grafo

**Motivação.** Os autovetores de baixa frequência do Laplaciano capturam "direções suaves" no grafo (grupos de empresas fortemente conectadas). Se essas direções coincidem com fatores de risco, servem como cargas estatísticas sem precisar estimar a partir de retornos ruidosos.

**Especificação.**

- $U_q$ = $q$ autovetores de menor autovalor (excluindo o trivial) de $L_\sigma$ normalizado; ou embeddings node2vec/spectral do grafo bipartido.
- Cargas: $B^{G} = U_q$; fatores estimados por regressão transversal diária $\hat f_t = (B^{G\top} B^G)^{-1} B^{G\top} r_t$.
- $\hat\Sigma = B^G \hat F B^{G\top} + \hat D$.

**Testes.**
- **Spanning:** regressão dos $k$ primeiros fatores de PCA (fora da amostra) nos fatores do grafo; reportar $R^2$ e ângulos principais entre subespaços: $\cos\theta_\ell = \sigma_\ell(U_q^\top V_k)$, com $V_k$ autovetores da covariância amostral.
- **Incremental:** o modelo PCA-$k$ + grafo-$q$ melhora fora da amostra sobre PCA-$(k+q)$? (comparação com mesmo número de parâmetros)

**Hipóteses associadas:** H5, H12.

---

### M6 – Grafo multiplex por tipo de relação

**Motivação.** Interpretabilidade econômica: quais tipos de relação geram co-movimento? Concorrência pode gerar correlação negativa residual (ganho de market share) ou positiva (choque de demanda comum); cadeia de suprimento deve gerar positiva.

**Especificação.**

$$
C^{G}(\theta) = \operatorname{corr}\Big(\sum_{k=1}^{K} \theta_k\, \mathcal{K}(W^{(k)}) + \sum_{k} \phi_k\, \tilde A^{(k)}\Omega \tilde A^{(k)\top}\Big)
$$

com $\mathcal{K}(\cdot)$ um kernel PSD (difusão). Estimar $\theta, \phi$ minimizando a perda de previsão de correlação residual na janela de treino, por exemplo:

$$
\min_{\theta,\phi}\ \sum_{t \in \text{treino}} \big\|C^{\text{res}}_{t+1:t+h} - C^{G}_t(\theta,\phi)\big\|_F^2
$$

ou, mais simples e interpretável, a regressão diádica de H3 (seção 4).

**Ablação.** Remover uma camada por vez e medir a queda de desempenho (importância por permutação de camada).

**Hipóteses associadas:** H3.

---

### M7 – Grafo dinâmico e correlação variante no tempo

**Motivação.** Notícias chegam antes de dados de retorno se acumularem. Mudanças no grafo podem antecipar mudanças de correlação (ex.: anúncio de aquisição, nova parceria de fornecimento, escalada de guerra comercial).

**Especificações (ordem crescente de complexidade).**

1. **Regressão diádica de mudanças** (base para H8):
   $$\Delta\rho_{ij,t\to t+h} = a + b\,\Delta \tilde w_{ij,t} + c\,\rho_{ij,t} + d\,(\rho_{ij,t} - \bar\rho_{ij}) + \text{controles} + u_{ij,t}$$
2. **HAR para correlações com covariável de notícias:**
   $$\rho_{ij,t+1} = \alpha + \beta_d \rho^{(d)}_{ij,t} + \beta_w \rho^{(w)}_{ij,t} + \beta_m \rho^{(m)}_{ij,t} + \gamma\, \tilde w_{ij,t} + \eta\, \Delta\tilde w_{ij,t} + \varepsilon$$
   com $\rho^{(d,w,m)}$ correlações realizadas diária/semanal/mensal (dados intradiários se disponíveis; senão janelas de retornos diários).
3. **DCC com covariáveis (DCC-X)** (Engle, 2002): na equação de correlação,
   $$Q_t = (1-a-b)\bar Q + a\, z_{t-1}z_{t-1}^\top + b\, Q_{t-1} + c\,(C^G_{t-1} - \bar C^G)$$
   Na prática, para $N=500$, usar DCC composto (composite likelihood) ou aplicar a subuniversos.
4. **Alvo de shrinkage dinâmico:** M2 com $C^G_t$ atualizado a cada rebalanceamento (o mais simples e o primeiro a testar).

**Hipóteses associadas:** H7, H8.

---

### M8 – Comunidades do grafo para HRP e estrutura em blocos

**Motivação.** HRP (López de Prado, 2016) depende de uma árvore de clusters estimada de correlações históricas ruidosas. Clusters derivados de notícias podem ser mais estáveis e antecipar mudanças.

**Especificação.**

- Detecção de comunidades no grafo ação–ação + projeção bipartida: Louvain/Leiden (para pesos positivos) ou clustering espectral assinado (Kunegis et al., 2010).
- Variantes de HRP:
  - (a) árvore a partir da distância $d_{ij} = \sqrt{\tfrac{1}{2}(1 - C^G_{ij})}$;
  - (b) distância combinada $d_{ij} = \sqrt{\tfrac12(1 - [\lambda C^G_{ij} + (1-\lambda)\hat\rho_{ij}])}$;
  - (c) bissecção restrita às comunidades de notícias no primeiro nível.
- Modelo em blocos: fatores = indicadores de comunidade (como dummies setoriais), $\hat\Sigma = B^{\text{com}} \hat F B^{\text{com}\top} + \hat D$.

**Comparação obrigatória:** comunidades de notícias vs GICS vs clusters de correlação histórica. Medir similaridade entre partições (ARI, NMI) e desempenho de portfólio.

**Hipóteses associadas:** H5, H13.

---

### M9 – Centralidade como fator de risco sistêmico

**Motivação.** Empresas centrais na rede de relações podem transmitir e receber choques, tendo maior beta em estresse e maior co-assimetria negativa.

**Especificação.** Calcular centralidade de autovetor (no grafo de valores absolutos), PageRank e grau ponderado. Construir um fator long–short alta–baixa centralidade e testar:
- se a centralidade prediz beta de mercado futuro e **beta de baixa** (downside beta) transversalmente;
- se o fator de centralidade explica variância adicional da covariância (spanning, como em M5).

**Hipóteses associadas:** H9.

---

## 4. Hipóteses e desenho dos testes

Cada hipótese tem: enunciado, nula, teste, critério de sucesso e controles. "Fora da amostra" significa sempre a janela de teste do protocolo da seção 5.

### Considerações gerais para regressões diádicas (H1–H3, H8)

- Para $N = 500$ há $\approx 124{,}750$ pares por período. Os pares **não são independentes**: pares que compartilham uma ação têm erros correlacionados (dependência diádica).
- Erros-padrão: usar **erros robustos diádicos** (Aronow, Samii & Assenova, 2015) ou cluster duplo (ação $i$ e ação $j$, Cameron, Gelbach & Miller, 2011) combinado com dependência temporal (Newey–West ou cluster por período).
- Alternativa não paramétrica: **teste QAP** (quadratic assignment procedure), permutando simultaneamente linhas e colunas da matriz do grafo (preserva a estrutura de dependência por nó) para gerar a distribuição nula do coeficiente.
- Alternativa com menos suposições: **Fama–MacBeth** por período (uma regressão transversal de pares por mês) com erros Newey–West na série de coeficientes.

### H1 – Notícias predizem correlação residual futura

- **Enunciado:** a força da aresta (normalizada) no tempo $t$ prediz a correlação residual entre $t+1$ e $t+h$, controlando por setor, tamanho e correlação passada.
- **Modelo:**
  $$\rho^{\text{res}}_{ij,t+1:t+h} = \alpha + \beta\,\tilde w_{ij,t} + \gamma\,\rho^{\text{res}}_{ij,t-T_w:t} + \delta\,\mathbb{1}[\text{mesmo setor}] + \zeta^\top x_{ij,t} + u_{ij,t}$$
  com $x_{ij}$: distância de tamanho $|\log ME_i - \log ME_j|$, distância de book-to-market, mesma bolsa, mesmo estado/sede, co-cobertura de analistas (se disponível), número de menções individuais (controle de atenção).
- **Nula:** $\beta = 0$.
- **Sucesso:** $\beta > 0$ significativo com erros diádicos/QAP, estável em subperíodos, e significativo também **só com pares intra-setor** e **só com pares inter-setor** separadamente.
- **Horizontes:** $h \in \{21, 63, 126\}$ dias.
- **Transformação:** usar Fisher-$z$ das correlações ($\operatorname{arctanh}\rho$) como variável dependente.

### H2 – O sinal importa

- **Enunciado:** arestas negativas estão associadas a correlação residual menor que arestas positivas de mesma magnitude.
- **Modelo:** substituir $\tilde w$ por $\tilde w^{+} = \max(\tilde w, 0)$ e $\tilde w^{-} = \max(-\tilde w, 0)$:
  $$\ldots + \beta^{+}\tilde w^{+}_{ij} + \beta^{-}\tilde w^{-}_{ij} + \ldots$$
- **Nula:** $\beta^{+} = -\beta^{-}$ (simetria; o sinal não carrega informação além da magnitude) ou, mais fraco, $\beta^- \ge 0$.
- **Teste de portfólio complementar:** M2 com grafo assinado vs não assinado ($|w|$) vs binário ($\mathbb{1}[w \ne 0]$).

### H3 – O tipo de relação importa

- **Modelo:** $\ldots + \sum_k \beta_k \tilde w^{(k)}_{ij} + \ldots$
- **Nula:** $\beta_1 = \beta_2 = \dots = \beta_K$ (teste de Wald com matriz de covariância robusta diádica).
- **Expectativas a priori (registrar antes de rodar):** cadeia de suprimento $> 0$; exposição comum a agente macro $> 0$; concorrência ambígua; litígio entre as partes $\le 0$; parceria $\ge 0$ e pequena.

### H4 – Exposição a agentes prediz sensibilidade realizada

- **Desenho:** estudo de evento em dias de FOMC (e CPI, payroll, OPEP para petróleo).
- **Modelo:**
  $$r_{i,\tau} = a_i + b_i\,r_{m,\tau} + c\,(\tilde a_{i,\text{Fed},t} \times \text{surpresa}_\tau) + d\,\text{surpresa}_\tau + e_{i,\tau}$$
  com surpresa monetária medida por variação de futuros de fed funds na janela do anúncio (estilo Kuttner) ou do yield de 2 anos.
- **Nula:** $c = 0$.
- **Teste de portfólio:** ordenar ações em quintis de exposição de notícias ao agente; o spread de retorno Q5–Q1 em dias de evento deve ter o sinal da surpresa.
- **Controle:** a exposição de notícias deve predizer sensibilidade **além** do beta histórico ao mesmo fator (incluir $\hat\beta^{\text{hist}}_{i,\text{Fed}} \times \text{surpresa}$).

### H5 – Estimadores com grafo superam estimadores padrão fora da amostra

- **Desenho:** protocolo walk-forward (seção 5) com todas as métricas.
- **Nula:** perda esperada igual entre o estimador com grafo e o melhor baseline.
- **Testes:** Diebold–Mariano sobre a série de perdas; Model Confidence Set; teste de diferença de variância/Sharpe de Ledoit–Wolf (seção 9).
- **Sucesso:** o estimador com grafo entra no MCS a 90% e o baseline correspondente (mesmo método, alvo sem grafo) não, ou a diferença é significativa no DM.

### H6 – Heterogeneidade por cobertura e tamanho

- **Enunciado alternativo (registrar as duas):** (a) ganhos maiores em small caps, onde a estimativa amostral é mais ruidosa; (b) ganhos menores em small caps, porque têm pouca cobertura.
- **Teste:** repetir H5 em subuniversos por tercil de market cap e por tercil de contagem de notícias; regressão diádica de H1 com interação $\tilde w_{ij} \times \text{cobertura}_{ij}$.

### H7 – Ganhos concentrados em regimes de estresse

- **Teste:** avaliar perdas condicionais a regime (tercis de VIX no início do período de teste; ou episódios 2018Q4, 2020Q1, 2022). Regressão da diferença de perda $d_t = L^{\text{base}}_t - L^{G}_t$ em dummies de regime, erros HAC.

### H8 – Mudanças no grafo antecedem mudanças de correlação

- **Teste:** regressão de M7(1). Teste reverso: $\Delta\tilde w_{ij,t+1:t+h}$ em $\Delta\rho_{ij,t-h:t}$. Se só a direção reversa é significativa, as notícias apenas descrevem co-movimento passado.
- **Sucesso:** $b > 0$ significativo e maior que o coeficiente da direção reversa.

### H9 – Centralidade e risco sistêmico

- **Teste:** Fama–MacBeth de beta futuro, beta de baixa e co-assimetria em centralidade, controlando por tamanho, setor, beta passado.

### H10 – A estrutura do grafo importa, não só a quantidade de notícias

- **Teste:** comparar M2 com o grafo real vs **grafos placebo** (seção 6.2). Calcular a posição do desempenho real na distribuição dos placebos (p-valor empírico).
- **Sucesso:** desempenho real acima do percentil 95 dos placebos com preservação de grau.

### H11 – Arestas de notícias correspondem a dependência condicional direta

- **Teste:** em M4, proporção de arestas de notícias com correlação parcial não nula estimada vs proporção em pares sem aresta (teste exato de Fisher / regressão logística com controles); e ganho de log-verossimilhança fora da amostra de $\gamma > 0$ vs $\gamma = 0$.

### H12 – Fatores espectrais do grafo capturam fatores de risco

- **Teste:** ângulos principais e $R^2$ de spanning (M5), comparados com os de fatores espectrais de grafos placebo.

### H13 – Comunidades de notícias são mais informativas que GICS para alocação

- **Teste:** HRP e modelo em blocos com comunidades vs GICS vs clusters de correlação (M8), métricas da seção 5.

---

## 5. Protocolo de avaliação

### 5.1 Desenho walk-forward

```
|------ estimação (T_w) ------|-- validação (V) --|-- teste (h) --|
                    → avança h dias e repete
```

- **Universo:** constituintes do S&P 500 point-in-time em cada data de rebalanceamento (sem viés de sobrevivência). Exigir histórico completo na janela de estimação; tratar deslistamentos no período de teste com o retorno de deslistamento (não descartar a ação).
- **Janela de estimação** $T_w \in \{126, 252, 504\}$ dias. Com $N=500$ e $T_w = 252$, $N/T_w \approx 2$: a covariância amostral é singular, que é exatamente o regime onde shrinkage e estrutura importam.
- **Rebalanceamento:** mensal (21 dias) como principal; semanal e trimestral como robustez.
- **Hiperparâmetros** ($\tau$, $\delta$, $\beta$, $\lambda$, $\gamma$, normalização): escolhidos **somente** na janela de validação anterior a cada período de teste, ou num bloco inicial de calibração congelado. Nunca no período de teste.
- **Separação dos dados:**
  - Período de desenvolvimento (ex.: primeiros 60% do tempo): toda a exploração.
  - Período de teste final (holdout, últimos 40%): rodar **uma vez**, com a especificação congelada e registrada.
- **Embargo:** um intervalo de alguns dias entre o fim da estimação e o início do teste para evitar sobreposição de informação via notícias publicadas com atraso.

### 5.2 Métricas estatísticas (qualidade da matriz)

Sendo $\Sigma^{\text{real}}$ uma proxy da covariância realizada no período de teste (amostral dos retornos diários do período, ou realizada intradiária se disponível):

| Métrica | Fórmula | Observação |
|---|---|---|
| Frobenius | $\|\hat\Sigma - \Sigma^{\text{real}}\|_F^2$ | Dominada por variâncias e grandes caps; reportar também em correlação |
| Frobenius em correlação | $\|\hat C - C^{\text{real}}\|_F^2$ | Isola a estrutura de dependência |
| Perda de Stein | $\operatorname{tr}(\hat\Sigma^{-1}\Sigma^{\text{real}}) - \log\det(\hat\Sigma^{-1}\Sigma^{\text{real}}) - N$ | Requer $\hat\Sigma$ invertível |
| QLIKE | $\log\det\hat\Sigma + \operatorname{tr}(\hat\Sigma^{-1}\Sigma^{\text{real}})$ | Robusta a ruído na proxy (Patton, 2011); escolher como métrica estatística principal |
| Log-verossimilhança preditiva | $\sum_{\tau \in \text{teste}} \log \phi(r_\tau; 0, \hat\Sigma)$ | Avalia diretamente nos retornos, sem proxy |
| Correlação par a par | Spearman entre $\hat\rho_{ij}$ e $\rho^{\text{real}}_{ij}$ (todos os pares) | Intuitiva; útil para comunicar |
| Erro em pares com aresta vs sem aresta | Frobenius restrito a subconjuntos | Mostra onde o grafo ajuda |

QLIKE e Stein exigem $\Sigma^{\text{real}}$ bem definida; com $N > h$ dias, a amostral de teste é singular. Soluções: usar apenas métricas que não exigem inversão de $\Sigma^{\text{real}}$ (QLIKE exige só $\hat\Sigma^{-1}$ — ok), ou avaliar em portfólios (seção 5.3), que é a abordagem preferida.

### 5.3 Métricas de portfólio (econômicas)

O teste mais limpo de qualidade de covariância é o **portfólio de variância mínima global (GMV)**, porque não depende de estimativas de retorno esperado (Engle & Colacito, 2006; Ledoit & Wolf, 2017).

**Portfólios a construir com cada $\hat\Sigma$:**

1. **GMV irrestrito:** $w = \hat\Sigma^{-1}\mathbf{1}/(\mathbf{1}^\top\hat\Sigma^{-1}\mathbf{1})$. **Métrica principal.**
2. **GMV long-only** (com teto por ativo, ex.: 5%): mais realista.
3. **HRP** (onde a covariância/árvore vem do estimador).
4. **Risk parity / equal risk contribution.**
5. **Mínimo tracking error** contra o S&P 500 com $n$ ativos (teste de covariância relativa).
6. **MVO** com sinal de retorno fixo e idêntico entre estimadores (ex.: momentum 12-1), para isolar o efeito da covariância. Secundário.

**Métricas:**

| Métrica | Uso |
|---|---|
| Volatilidade realizada anualizada | Principal para GMV |
| Razão de previsão de risco $\hat\sigma_p / \sigma_p^{\text{real}}$ | Calibração do risco previsto |
| Estatística de viés (Barra) | Desvio-padrão de $z_\tau = r_{p,\tau}/\hat\sigma_{p,\tau}$; ideal $\approx 1$, IC $1 \pm \sqrt{2/n}$ |
| Sharpe e Sortino | Secundário (dominado pelo retorno, ruidoso) |
| Max drawdown, CVaR 5% | Risco de cauda |
| Turnover $\sum_i |w_{i,t} - w_{i,t^-}|$ | Custo implícito |
| Retorno líquido com custos | 5, 10, 20 bps por unidade de turnover |
| Concentração | $1/\sum w_i^2$ (número efetivo de ativos), peso máximo |
| Número de condição de $\hat\Sigma$ | Estabilidade numérica |

### 5.4 Métricas de previsão de correlação (para M7 e H8)

- $R^2$ fora da amostra (Campbell–Thompson) da correlação prevista vs benchmark (correlação histórica):
  $$R^2_{OOS} = 1 - \frac{\sum (\rho^{\text{real}} - \hat\rho^{G})^2}{\sum(\rho^{\text{real}} - \hat\rho^{\text{hist}})^2}$$
- Teste de Clark–West para modelos aninhados.

---

## 6. Baselines e controles

### 6.1 Baselines de estimação (mínimo obrigatório)

| Código | Estimador | Referência |
|---|---|---|
| B0 | 1/N (pesos iguais) | DeMiguel, Garlappi & Uppal (2009) |
| B1 | Covariância amostral (pseudo-inversa) | — |
| B2 | Ledoit–Wolf, alvo identidade escalada | Ledoit & Wolf (2004a) |
| B3 | Ledoit–Wolf, alvo correlação constante | Ledoit & Wolf (2004b) |
| B4 | Ledoit–Wolf, alvo índice único (mercado) | Ledoit & Wolf (2003) |
| B5 | Alvo de blocos setoriais GICS (correlação constante intra e inter setor) | — |
| B6 | Shrinkage não linear analítico | Ledoit & Wolf (2020) |
| B7 | Modelo de fatores observáveis (FF5 + momentum + setores) | — |
| B8 | POET (PCA-$k$ + threshold residual) | Fan, Liao & Mincheva (2013) |
| B9 | Glasso padrão ($\gamma = 0$) | Friedman et al. (2008) |
| B10 | EWMA / DCC | RiskMetrics; Engle (2002) |
| B11 | HRP padrão | López de Prado (2016) |

**Regra de pareamento:** cada modelo com grafo deve ser comparado com seu par sem grafo (mesmo método, mesmo número de hiperparâmetros ajustados). Ex.: M2 vs B3/B4/B5; M3 vs B8; M4 vs B9; M7 vs B10; M8 vs B11.

### 6.2 Grafos placebo

Para cada data de rebalanceamento, gerar $R = 200$–$1000$ grafos nulos:

| Placebo | Preserva | Testa |
|---|---|---|
| P1 – Embaralhamento total de pesos | Distribuição de pesos | Qualquer estrutura |
| P2 – Rewiring com preservação de grau (configuration model / double edge swap) | Grau de cada nó | Estrutura além de "atenção" |
| P3 – Rewiring intra-setor | Grau + composição setorial | Informação além do setor |
| P4 – Inversão de sinais aleatória | Topologia | Informação do sinal |
| P5 – Grafo defasado (de $t - 2$ anos) | Estrutura antiga | Valor da atualização temporal |
| P6 – Grafo de co-menção pura (sem LLM: aresta = co-ocorrência no mesmo artigo) | Informação de co-ocorrência | Valor agregado do LLM (tipo e sinal) |

P6 é especialmente importante: responde se o LLM agrega valor sobre uma heurística barata.

### 6.3 Grafos de referência externos (se acessíveis)

- Dados de cadeia de suprimentos (Compustat Segments / FactSet Revere).
- TNIC de Hoberg & Phillips (similaridade textual de 10-K).
- Co-cobertura de analistas.

Comparar: o grafo de notícias acrescenta informação sobre essas redes? (incluí-las como controles em H1 e como alvos alternativos em M2).

---

## 7. Vazamento de informação e vieses específicos de LLM

### 7.1 Vazamento por conhecimento prévio do LLM

O LLM foi pré-treinado com dados posteriores a grande parte do período amostral. Ao extrair "força" e "sinal" de uma notícia de 2016, ele pode incorporar conhecimento de como a relação evoluiu depois (look-ahead).

**Mitigações e testes:**

1. **Restringir a tarefa:** o prompt pede apenas o que está **explícito** no texto, com `evidence_span` obrigatório; descartar tuplas cujo trecho não suporta a relação (verificação automática de presença literal + amostra manual).
2. **Holdout pós-cutoff:** verificar a data de corte de conhecimento do modelo extrator usado e reservar todo o período posterior como teste puro. Comparar desempenho pré vs pós-cutoff; uma queda grande após o cutoff indica vazamento.
3. **Teste de anonimização:** em uma subamostra, mascarar nomes de empresas/datas (`[EMPRESA_A]`, `[EMPRESA_B]`) e reextrair. Comparar sinal e força com a extração original; divergências sistematicamente alinhadas com o futuro são evidência de vazamento.
4. **Teste de eventos famosos:** verificar se notícias anteriores a eventos marcantes (ex.: falências, grandes fusões) recebem forças anormalmente "proféticas" em relação a notícias equivalentes sem desfecho marcante.
5. **Modelo de extração de menor conhecimento como robustez:** reextrair uma subamostra com um modelo menor/antigo ou com um extrator não-LLM (P6) e comparar os resultados de H1/H5.

### 7.2 Outros vieses

- **Viés de cobertura:** tratado em 2.4; controlar por número de menções em todas as regressões.
- **Viés de seleção da fonte:** um único veículo (WSJ) tem linha editorial e foco em grandes empresas. Reportar a cobertura por tamanho e setor.
- **Reflexividade:** notícias frequentemente reportam co-movimentos já ocorridos ("ações de bancos caem juntas"). Mitigação: H8 (direção temporal); excluir artigos de "market wrap" (classificador simples por palavras-chave ou pelo próprio LLM); testar com e sem eles.
- **Duplicação:** deduplicar notícias republicadas/atualizadas antes de agregar (já no pipeline); medir sensibilidade a limiar de similaridade.
- **Não estacionariedade do LLM:** se a extração usa versões diferentes do modelo ao longo do tempo, fixar versão e temperatura (0) e registrar.

---

## 8. Robustez

Reportar as conclusões principais sob:

| Dimensão | Variações |
|---|---|
| Janela de estimação | 126 / 252 / 504 dias |
| Frequência de rebalanceamento | semanal / mensal / trimestral |
| Meia-vida do decaimento | 30 / 90 / 180 / 365 dias |
| Normalização | PMI / grau / z-score / bruta |
| Kernel PSD | difusão / Higham / Gram |
| Universo | S&P 500 / S&P 100 / Dow 30 / excluindo financeiro |
| Subperíodos | por ano; pré/pós-2020; pré/pós-cutoff do LLM |
| Retornos | simples / log; com e sem winsorização (1%/99%) |
| Custos | 0 / 5 / 10 / 20 bps |
| Restrições | irrestrito / long-only / teto por ativo |
| Tratamento de agentes `OTHER` | descartar / incluir |
| Artigos de market wrap | com / sem |

Uma conclusão é robusta se o sinal e a significância (ou pertencimento ao MCS) se mantêm na grande maioria das variações. Reportar todas, inclusive as que falham.

---

## 9. Inferência estatística e múltiplos testes

### 9.1 Comparação de perdas entre estimadores

Seja $d_t = L(\hat\Sigma^{A}_t) - L(\hat\Sigma^{B}_t)$ a diferença de perda no período de teste $t$.

- **Diebold–Mariano (1995):** $DM = \bar d / \sqrt{\widehat{\operatorname{LRV}}(d)/n}$ com variância de longo prazo HAC (Newey–West). Com poucas observações, usar a correção de Harvey, Leybourne & Newbold (1997).
- **Model Confidence Set** (Hansen, Lunde & Nason, 2011): conjunto de modelos que contém o melhor com probabilidade $1-\alpha$, dado o universo de candidatos. Reportar o MCS a 75% e 90% para cada métrica. É a forma mais honesta de comparar muitos estimadores simultaneamente.

### 9.2 Diferenças de variância e Sharpe de portfólios

- **Variância:** teste de Ledoit & Wolf (2011) para diferença de log-variâncias, com bootstrap estacionário em blocos (robusto a caudas pesadas e autocorrelação).
- **Sharpe:** teste de Ledoit & Wolf (2008), também por bootstrap em blocos. O teste de Jobson–Korkie/Memmel assume normalidade e i.i.d., então não deve ser usado como principal.
- **Bootstrap estacionário** (Politis & Romano, 1994): tamanho médio de bloco escolhido pelo método de Politis & White (2004).

### 9.3 Múltiplos testes e snooping

Com muitos modelos, variantes e hiperparâmetros, a probabilidade de achar um "vencedor" por acaso é alta.

- **Pré-registro interno:** antes de rodar o holdout, registrar (arquivo versionado com hash/commit) a especificação principal, as hipóteses e os sinais esperados.
- **Contar os testes:** manter um log de todas as especificações testadas no período de desenvolvimento.
- **Reality Check de White (2000)** e **SPA de Hansen (2005):** testam se o melhor modelo supera o benchmark, levando em conta todos os modelos testados.
- **Romano–Wolf stepdown (2005):** controla a FWER entre as múltiplas hipóteses H1–H13 com dependência entre testes.
- **Deflated Sharpe Ratio** (Bailey & López de Prado, 2014): ajusta o Sharpe pelo número de tentativas, assimetria e curtose.
- **Benjamini–Hochberg** para o conjunto de coeficientes de tipo de relação em H3 (exploratório).

### 9.4 Poder estatístico

Com 10 anos de dados, 2–3 de aquecimento e rebalanceamento mensal, restam ~85–95 períodos de teste. Diferenças de volatilidade GMV de poucos pontos-base anualizados dificilmente serão significativas. Por isso:

- Avaliar também com **perdas diárias** dentro de cada período (mais observações, com HAC).
- Agregar evidências: regressões diádicas (H1–H3) têm muito mais poder e servem de mecanismo; métricas de portfólio mostram relevância econômica.
- Reportar **tamanho de efeito** com intervalos de confiança, não apenas p-valores.

---

## 10. Roteiro experimental sugerido

Ordem pensada para que cada etapa valide a seguinte e para que falhas apareçam cedo.

| Etapa | Entregável | Critério para seguir |
|---|---|---|
| 1. Validação da extração | Tabela de precisão/recall/κ (2.2) | Precisão de relação e sinal razoáveis (ex.: > 80%) |
| 2. Descritivas do grafo | Seção 2.5 + sobreposição com GICS | Grafo não é só "setor reescrito" |
| 3. H1 + H2 (diádico) | Coeficientes com erros diádicos/QAP | Mecanismo existe? Se $\beta \approx 0$ na correlação residual, revisar normalização antes de seguir |
| 4. Placebos P2, P3, P6 em H1 | Distribuição nula | Grafo real fora da distribuição |
| 5. M2 (alvo de shrinkage) vs B2–B6 | GMV + QLIKE + MCS | Ganho sobre B5 (setor) |
| 6. M6 / H3 (decomposição por tipo) | $\theta_k$, ablação | Interpretação econômica |
| 7. M3, M4, M8 | Extensões | Comparação pareada |
| 8. M7 / H8 (dinâmica) | Previsão de mudanças | Direção temporal |
| 9. H4, H9 (agentes macro, centralidade) | Estudos de evento | — |
| 10. Testes de vazamento (7.1) | Pré vs pós-cutoff, anonimização | Sem evidência forte de vazamento |
| 11. Holdout final (rodar uma vez) | Tabelas finais, robustez (8), inferência (9) | — |

**Núcleo mínimo para um trabalho completo:** etapas 1–6 + 10 + 11. As demais são extensões.

### Tabela-modelo de resultados (holdout)

| Estimador | Vol. GMV (a.a.) | Vol. GMV long-only | QLIKE | Viés Barra | Turnover | Sharpe líq. (10 bps) | No MCS 90%? | p (vs par) |
|---|---|---|---|---|---|---|---|---|
| B3 LW corr. constante | | | | | | | | — |
| B5 Blocos GICS | | | | | | | | — |
| B6 Não linear | | | | | | | | — |
| B8 POET | | | | | | | | — |
| M2 Alvo grafo (assinado) | | | | | | | | vs B3/B5 |
| M2 Alvo grafo (não assinado) | | | | | | | | vs M2 assinado |
| M2 Alvo placebo P2 (mediana) | | | | | | | | — |
| M2 Alvo co-menção P6 | | | | | | | | — |
| M3 POET + suporte do grafo | | | | | | | | vs B8 |
| ... | | | | | | | | |

---

## 11. Referências

**Estimação de covariância**
- Ledoit, O., & Wolf, M. (2003). Improved estimation of the covariance matrix of stock returns with an application to portfolio selection. *Journal of Empirical Finance*.
- Ledoit, O., & Wolf, M. (2004a). A well-conditioned estimator for large-dimensional covariance matrices. *Journal of Multivariate Analysis*.
- Ledoit, O., & Wolf, M. (2004b). Honey, I shrunk the sample covariance matrix. *Journal of Portfolio Management*.
- Ledoit, O., & Wolf, M. (2017). Nonlinear shrinkage of the covariance matrix for portfolio selection: Markowitz meets Goldilocks. *Review of Financial Studies*.
- Ledoit, O., & Wolf, M. (2020). Analytical nonlinear shrinkage of large-dimensional covariance matrices. *Annals of Statistics*.
- Fan, J., Liao, Y., & Mincheva, M. (2013). Large covariance estimation by thresholding principal orthogonal complements. *JRSS-B*.
- Friedman, J., Hastie, T., & Tibshirani, R. (2008). Sparse inverse covariance estimation with the graphical lasso. *Biostatistics*.
- Engle, R. (2002). Dynamic conditional correlation. *Journal of Business & Economic Statistics*.
- Higham, N. (2002). Computing the nearest correlation matrix. *IMA Journal of Numerical Analysis*.

**Grafos**
- Kondor, R. I., & Lafferty, J. (2002). Diffusion kernels on graphs and other discrete input spaces. *ICML*.
- Kunegis, J., et al. (2010). Spectral analysis of signed graphs for clustering, prediction and visualization. *SDM*.

**Portfólios**
- DeMiguel, V., Garlappi, L., & Uppal, R. (2009). Optimal versus naive diversification. *Review of Financial Studies*.
- López de Prado, M. (2016). Building diversified portfolios that outperform out of sample. *Journal of Portfolio Management*.
- Engle, R., & Colacito, R. (2006). Testing and valuing dynamic correlations for asset allocation. *Journal of Business & Economic Statistics*.

**Redes econômicas e texto**
- Cohen, L., & Frazzini, A. (2008). Economic links and predictable returns. *Journal of Finance*.
- Hoberg, G., & Phillips, G. (2016). Text-based network industries and endogenous product differentiation. *Journal of Political Economy*.
- Lee, C. M. C., Sun, S. T., Wang, R., & Zhang, R. (2019). Technological links and predictable returns. *Journal of Financial Economics*.
- Scherbina, A., & Schlusche, B. — trabalhos sobre fluxos de informação entre empresas via co-menção em notícias (verificar referência exata).

**Avaliação e inferência**
- Patton, A. (2011). Volatility forecast comparison using imperfect volatility proxies. *Journal of Econometrics*.
- Diebold, F., & Mariano, R. (1995). Comparing predictive accuracy. *Journal of Business & Economic Statistics*.
- Harvey, D., Leybourne, S., & Newbold, P. (1997). Testing the equality of prediction mean squared errors. *International Journal of Forecasting*.
- Hansen, P., Lunde, A., & Nason, J. (2011). The model confidence set. *Econometrica*.
- White, H. (2000). A reality check for data snooping. *Econometrica*.
- Hansen, P. (2005). A test for superior predictive ability. *Journal of Business & Economic Statistics*.
- Romano, J., & Wolf, M. (2005). Stepwise multiple testing as formalized data snooping. *Econometrica*.
- Ledoit, O., & Wolf, M. (2008). Robust performance hypothesis testing with the Sharpe ratio. *Journal of Empirical Finance*.
- Ledoit, O., & Wolf, M. (2011). Robust performance hypothesis testing with the variance. *Wilmott*.
- Politis, D., & Romano, J. (1994). The stationary bootstrap. *JASA*.
- Politis, D., & White, H. (2004). Automatic block-length selection for the dependent bootstrap. *Econometric Reviews*.
- Bailey, D., & López de Prado, M. (2014). The deflated Sharpe ratio. *Journal of Portfolio Management*.
- Clark, T., & West, K. (2007). Approximately normal tests for equal predictive accuracy in nested models. *Journal of Econometrics*.
- Campbell, J., & Thompson, S. (2008). Predicting excess stock returns out of sample. *Review of Financial Studies*.

**Inferência com dados diádicos**
- Cameron, A. C., Gelbach, J., & Miller, D. (2011). Robust inference with multiway clustering. *Journal of Business & Economic Statistics*.
- Aronow, P., Samii, C., & Assenova, V. (2015). Cluster-robust variance estimation for dyadic data. *Political Analysis*.
- Krackhardt, D. (1988). Predicting with networks: nonparametric multiple regression analysis of dyadic data (QAP). *Social Networks*.
