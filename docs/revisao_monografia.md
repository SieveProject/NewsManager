# Revisão da monografia: métodos, equações, modelos e aderência à nova linha de hipóteses

Data: 2026-11-10. Escopo: `monografia/` (todos os capítulos), conferida contra o código
(`newsmanager/grafo/`, `newsmanager/portfolio/`), os resultados gravados em `data/marts/grafo/`
e os documentos de método (`docs/metodo_covariancia.md`, `docs/secao18_resultados.md`).

Os diagnósticos novos desta revisão são **só de leitura** (nenhum estimador novo foi avaliado em
desempenho no período de teste) e estão em seis datas amostradas, salvo indicação. Os scripts
ficaram no scratchpad da sessão; os números abaixo podem ser refeitos com eles.

## Estado depois da reescrita (2026-11-10)

Por decisão do autor, a monografia passou a ter três hipóteses: H1a (exposição comum), H1b (tipo de
relação) e H2 (carteiras, desenhos D1 a D9). A centralidade (H1c) e o programa final de carteiras (D10)
saíram do texto. O que foi feito com cada achado desta revisão:

| achado | situação |
|---|---|
| F1, F2 (M2 não mede as notícias) | o programa final saiu do texto; uma limitação ("Testes não relatados") declara que ele existiu, por que não é relatado e que nada depende dele |
| F3 (número de condição) | fica como diagnóstico nos desenhos D2 e D9, onde é relatado |
| F4 (inferência de H1a e H1b) | QAP refeito com 1.000 permutações e estendido à exposição comum (`scripts/revisao_qap.py`), relatado como análise posterior; robustez de H1a por subperíodo, setor, horizonte e sem classes de ação; ressalva sobre a classe concorrência |
| F5 (p > N, classes duplicadas) | as carteiras usam só os universos de 50, 100 e 200 ações (N > n, sem classes duplicadas, conferido); o S&P 500 fica só nos testes de pares, com a cobertura de `g` relatada |
| E1, E2, E3, E7 | texto reescrito para descrever o que o código faz |
| E4, E6, E8 | deixaram de existir com a saída do programa final |
| E5, E9 | declarados no texto |
| contagem dos desenhos, p = 0,000, 70% contra 80%, Apêndice B | corrigidos |

Depois disso, ainda em 2026-11-10 e por decisão do autor, o texto foi enxugado para cerca de 60 páginas:
saíram as análises de robustez, as tabelas descritivas que não sustentam hipóteses e toda menção a testes
fora das três hipóteses (inclusive a limitação "Testes não relatados" e a correção de Holm sobre a família
de dez testes). H1b passou a ser relatada com o QAP de 1.000 permutações (p = 0,001; p = 0,003 com
Bonferroni para as três hipóteses; com Holm sobre os dez testes originais o p seria no máximo 0,01, de modo
que o veredito não muda). As robustezes calculadas nesta revisão continuam em `logs/revisao_qap/` e neste
documento, mas não no texto. A Seção 5 ganhou uma figura com um recorte real do grafo.

As seções abaixo ficam como registro da revisão original.

## 0. Resumo

1. **O teste confirmatório de H2 (M2 contra B5) não mede o conteúdo das notícias.** O alvo do M2
   é quase a matriz diagonal e é singular por um artefato de construção. O resultado "+11% de
   variância" é real, mas a leitura "as notícias pioram a carteira" não se sustenta (F1, F2).
2. **O número de condição do programa final existe nos dados e não está no texto.** No período de
   teste, o do M2 é cerca de 60 vezes pior que o do B5, em 60 de 60 meses (F3).
3. **H1a sobrevive ao teste que poderia derrubá-la**: sem os pares de classes de ação, o
   coeficiente da exposição comum vai de 0,0038 para 0,0035, com t = 11,2 (F4).
4. **A nova linha de hipóteses (estabilidade, fator de mercado via Fed, carteiras) não é a do
   texto atual e, com a evidência que já existe, as três partes saem negativas ou sem apoio**
   (seção 4). Dá para escrever a tese nessa linha, mas como investigação com resultado
   negativo e com as duas primeiras hipóteses declaradas como não pré-registradas.
5. Há ainda divergências entre texto e código, inconsistências de contagem e lacunas de
   validação (seções 2 e 3).

## 1. Falhas graves (afetam conclusões)

### F1. O alvo do M2 é quase diagonal e singular

O M2 é `Σ̂ = δ·Δ C^G Δ + (1−δ)·S` (Eq. 3.11, `Metodologia.tex:397`). Medido no universo do
S&P 500, com a configuração escolhida no desenvolvimento (τ = 180, normalização z):

| medida (5 datas de 2019 a 2023) | valor |
|---|---|
| média de `g_ij` fora da diagonal | +0,0004 a +0,0007 |
| média de `|g_ij|` | 0,0010 a 0,0015 |
| pares com `|g_ij| > 0,1` | 0,05% a 0,25% |
| correlação amostral média dos mesmos pares | 0,30 a 0,55 |
| δ do M2 | 0,033 a 0,074 |
| δ de um alvo diagonal (`R = I`) | igual ao do M2 até a 3ª casa |
| δ do B5 | 0,20 a 0,56 |
| distância relativa entre o Σ̂ do M2 e o do alvo diagonal (Frobenius) | 0,06% a 0,30% |
| autovalores de `C^G` exatamente nulos | 2 a 5 por data |

Dois problemas se somam:

- **Falta o modo de mercado.** `C^G` tem correlação média zero, e a correlação verdadeira média
  é 0,3 a 0,5. A fórmula de δ mede essa distância e devolve δ ≈ 0,05: o M2 é, na prática, a
  covariância amostral (singular, porque p ≈ 350–460 > N = 252) com 5% de um alvo diagonal. O
  conteúdo das notícias responde por 0,1% a 0,3% da matriz.
- **Classes de ação no mesmo nó.** O grafo une as classes de ação da mesma empresa
  (`Extracao.tex:193`), mas o universo do S&P 500 traz os dois *tickers*: GOOGL/GOOG, NWSA/NWS,
  UA/UAA, FOXA/FOX. As duas linhas de `Ã` são idênticas e `g = 1`. O mesmo acontece por
  coincidência com ações ligadas a uma única entidade (DVA/VRSN, LYB/CF). O alvo afirma
  correlação 1 para esses pares, a amostra já dá 0,99, e a combinação fica quase singular
  justamente nessas direções, que a GMV sem restrição explora com posições compradas e vendidas
  grandes.

Consequências visíveis nos resultados já gravados:

- número de condição mediano no teste: M2 = 142.935, alvo diagonal ≈ 1.600–4.500, B5 = 1.704;
- giro de 52 vezes o patrimônio por ano e previsão de risco de 0,26 (`ResultadosCarteiras.tex:336`);
- variância do M2 com o grafo real (1,017e-4) a 1% da mediana dos placebos P2 (1,006e-4): os
  placebos têm o mesmo defeito, então o teste de placebo não tem poder para ver conteúdo;
- M2-abs e M2-bin são **o mesmo estimador** sob a normalização z (o código usa 1 nos dois
  casos), e por isso dão resultados idênticos; a Tabela 3.3 os lista como variantes distintas
  (`Metodologia.tex:412`);
- na robustez, a desvantagem do M2 some onde o artefato não age: universo das 100 mais citadas
  (1,003; sem classes duplicadas e com grafo mais denso), alvo de Higham (1,050; positivo
  definido) e carteira sem venda a descoberto (0,972).

**Correção.** No texto: relatar δ, o número de condição e a média de `g_ij`; dizer que o M2,
como especificado, não testa o conteúdo das notícias. No método (análise posterior, rotulada):
manter uma classe por empresa no universo e usar um alvo que contenha o modo de mercado (o
resíduo ao fator de mercado, como no D2, ou `C^G` combinado com o alvo de blocos). Qualquer
número novo de desempenho no período de 2019–2023 é posterior ao teste e tem de ser rotulado.

### F2. Frases que atribuem às notícias o que é da construção do estimador

- `ResultadosCarteiras.tex:270` e `Conclusao.tex:42`: "o M2 teve 11% mais variância" lido como
  efeito das notícias. É efeito de F1.
- `Metodologia.tex:119` (parágrafo de H2, escrito na revisão anterior): "`C^G` é um alvo
  candidato natural, por ser uma matriz de correlação válida por construção". É positiva
  semidefinida, mas singular e sem modo de mercado; não é um bom alvo para a covariância total.
- `ResultadosCarteiras.tex:369`: "o problema do sinal não explica o desempenho do M2" está
  certo, mas pelo motivo errado: nada do conteúdo explica.
- `ResultadosCarteiras.tex:327`: "o placebo P3 é melhor que o grafo real" deve ser lido com a
  mesma ressalva.

Os desenhos D2, D3 e D4–D8 não têm esse defeito (alvo calibrado por regressão sobre a correlação
residual, universo sem classes duplicadas) e continuam válidos: neles as notícias empatam com o
setor. O D10 não acrescenta evidência sobre o conteúdo.

### F3. O número de condição do programa final não é relatado

`Metodologia.tex:613` diz que ele é registrado mês a mês, e o capítulo 7 só o mostra no D2 e no
D9. Os dados do teste estão em `data/marts/grafo/wf/holdout/mensal.parquet`:

| par (com notícias / sem) | número de condição mediano | leitura |
|---|---|---|
| M2 / B5 | 142.935 / 1.704 | 60× pior, em 60 de 60 meses (artefato de F1) |
| M2-res / R0 | 84.857 / 2.291 | pior (mesmo artefato) |
| M4 / B9 | 5.418 / 1.259 | 4× pior |
| M6 / R0 | 3.382 / 2.291 | pior |
| M1 / B7′ | 537 / 535 | igual |
| M8 (blocos) / FF12 (blocos) | 1.152 / 1.424 | 19% melhor; variância igual (0,998) |
| D2, n = 50, 100, 200 | +6%, +19%, +45% | pior |
| D9 (Black-Litterman) | 137 / 183 | 25% melhor; variância 14% pior |

Além disso, entre os 27 estimadores com número de condição finito, a correlação de postos entre
o número de condição e a volatilidade realizada da GMV é 0,05 (p = 0,82). O B7′ tem o melhor
condicionamento (535) e uma das piores carteiras (17,6%); o M4 tem condicionamento ruim (5.418)
e uma das melhores (14,4%). **Nesta amostra, o número de condição não ordena as carteiras.**

### F4. Inferência de H1a, H1b e H1c

- **H1a não tem p de permutação.** O QAP foi calculado só para o coeficiente da aresta direta;
  o da exposição comum tem apenas o t de Fama-MacBeth (11,1). O texto da seção 6.3 dá a entender
  que a inferência dupla vale para todos. Correção: rodar o QAP para `g` ou dizer que não foi.
- **H1a é robusta às classes duplicadas** (teste desta revisão): sem os pares com `g ≥ 0,999`
  (2,1 por mês), coeficiente 0,00345 e t = 11,2; sem os pares com `g ≥ 0,5`, 0,00329 e t = 11,3.
  Vale relatar.
- **H1b está no piso de resolução do QAP.** Com 200 permutações, o menor p possível é 0,005, e
  0,005 × 9 = 0,045 é o p de Holm relatado. O veredito depende do número de permutações.
  Correção: declarar; se for refeito com mais permutações, rotular como posterior.
- **H1b se apoia na classe menos confiável.** A concorrência é o grupo de maior efeito e também a
  classe em que a relação existe em só 64% das tuplas anotadas (`Extracao.tex:368`); a classe
  correta é 62% no geral, e parte do erro é do mapeamento por expressões regulares. O texto deve
  ligar as duas coisas.
- **Magnitude.** 0,0038 em escala z por desvio-padrão de `g` e 0,3 ponto de R² são efeitos
  estatisticamente fortes e economicamente pequenos. O texto diz isso; o resumo e as
  contribuições deveriam dizer também.

### F5. O universo "S&P 500" e o regime p > N

- O universo tem de 344 a 474 ações (faltam empresas grandes e, depois de 2020-07, os *tickers*
  truncados), com N = 252: **a covariância amostral é singular em todo o programa final**.
- Quase todos os estimadores subestimam o risco da GMV sem restrição de 2 a 4 vezes (previsão
  de risco de 0,26 a 0,43, inclusive no B5; a exceção é o M5, com 0,84). O teste principal de H2 foi feito num regime em que o
  erro de estimação domina qualquer estimador. O texto deve dizer isso e relatar a alavancagem
  bruta das carteiras.
- As classes de ação duplicadas afetam todos os estimadores no universo do S&P 500, não só o M2.
- Na exposição comum, só 13% a 40% dos pares do S&P 500 têm `g ≠ 0`. O texto dá a cobertura do
  universo das 100 mais citadas (74,7%) e não a do universo principal.

## 2. Divergências entre texto, equações e código

| # | onde | texto | código | efeito |
|---|---|---|---|---|
| E1 | `Metodologia.tex:184` | IDF = log(p/df_m) | numerador = nós com *ticker* no grafo da janela | constante por coluna; pequeno |
| E2 | `Metodologia.tex:215` | grau: D^{-1/2}\|W\|D^{-1/2} | colunas de Ã divididas pela raiz do grau da entidade | descrição errada |
| E3 | `Metodologia.tex:250` | R² "na seção transversal de cada mês" | 1 − ΣSSE/ΣSST empilhado | redação |
| E4 | `Metodologia.tex:412` | M2-abs e M2-bin distintos | idênticos sob z e PMI | ver F1 |
| E5 | Seção 2.2.1 | S com 1/(N−1) | shrinkage usa 1/N | irrelevante; uniformizar |
| E6 | Eq. 2.7 | QLIKE | dividida por p; vira NaN se cond > 1e14 | dizer que B1, B6 e B10 ficam fora |
| E7 | `Revisao.tex:183` | d = √(2(1−ρ)) | √((1−ρ)/2) | mesma árvore |
| E8 | `Metodologia.tex:486` | Markowitz com Bayes-Stein | vale para D4; a carteira de média-variância do programa final usa o momento e não é relatada | sem efeito no texto; não misturar se for relatada |
| E9 | `Dados.tex:155` | retorno ausente = 0 | idem, e dia sem retorno vira linha nula na janela de **todos** os estimadores | só o B6 foi corrigido |

As Equações 2.1–2.7, 3.1, 3.3–3.10 conferem com a implementação. A fórmula de δ (3.6) confere
com o `covCor.m` de Ledoit e Wolf, com a correlação de cada par no lugar da constante.

## 3. Inconsistências e lacunas do texto

- **Contagem dos desenhos.** `Introducao.tex:139` fala em nove desenhos e um programa;
  `Conclusao.tex:10` fala em dez desenhos **e** um programa (conta o D10 duas vezes);
  `ResultadosCarteiras.tex:426` fala em dez desenhos.
- **p = 0,000** nas Tabelas 7.1 e 7.3 (`ResultadosCarteiras.tex:37,41,44,124`): trocar por
  "< 0,001".
- **Amplitude contra profundidade.** O texto defende investigar poucas hipóteses a fundo, mas
  ainda descreve 11 estimadores com notícias, 13 referências, 6 placebos e 10 desenhos. Com a
  linha nova, M3, M5, M6, M8, P1, P3–P6 e os desenhos D4–D8 deixam de ter função.
- **Validação da extração.** Precisão medida por outro LLM em 567 tuplas, sem anotador humano,
  sem recall; teste de anonimização com n = 82; 434 *tickers* sem verificação. Tudo declarado,
  mas a frase "quatro em cada cinco tuplas" convive com "cerca de 70% eram relações claras" na
  amostra de 40 (`Extracao.tex:140`): usar uma medida só ou explicar a diferença.
- **Vieses de dados declarados só em parte**: setor atual aplicado ao passado
  (`Dados.tex:195`); modelo de linguagem treinado depois do período de teste; período de teste
  já visto num *notebook* exploratório.
- **Referências a conferir**: Acemoglu et al. (2012), Ang, Chen e Xing (2006) e Ross (1976),
  acrescentadas de memória na revisão anterior; as duas entradas marcadas "conferir" no
  `referencias.bib`.
- **Apêndice B** ainda é um marcador `\pendente`.

## 4. A linha de hipóteses proposta, contra a evidência existente

Linha proposta: (N1) o grafo melhora a estabilidade da matriz (número de condição); (N2) existe
um fator de mercado descrito pelo grafo, por nós como o Fed/FOMC; (N3) a melhora da covariância
melhora as carteiras no backtest.

**O texto atual não segue essa linha.** Ele está organizado em H1a (exposição comum), H1b (tipo
de relação), H1c (centralidade) e H2 (carteira). O número de condição aparece como diagnóstico,
e o teste do FOMC foi retirado na revisão anterior.

### N1. Estabilidade

- A evidência existente é contrária em quase todas as comparações (tabela de F3). Onde o
  condicionamento melhora (D9, blocos por comunidades), a carteira não melhora.
- O número de condição não ordena as carteiras nesta amostra (ρ de postos = 0,05).
- Problema conceitual: qualquer encolhimento para a identidade melhora o número de condição sem
  informação nenhuma. A hipótese só tem conteúdo se a comparação for contra um alvo sem notícias
  com a mesma intensidade, ou contra placebos, e se a estabilidade for medida também no que
  importa para a carteira (giro, razão entre risco previsto e realizado).
- Protocolo: o pré-registro diz que o número de condição "é descritivo, **não** é critério de
  sucesso" (`docs/metodo_covariancia.md`, adendo da seção 10). Promovê-lo a hipótese é posterior
  aos resultados e tem de ser declarado.

### N2. Fator de mercado descrito pelo grafo

- Ambiguidade a resolver: "fator de mercado" (o modo comum a todas as ações) ou "fator
  monetário" (a carga de cada ação em choques do Fed)? São hipóteses diferentes.
- Modo comum: o primeiro autovetor de `C^G` explica 1% a 2% do traço e tem cosseno de 0,08 a
  0,33 com a primeira componente principal dos retornos, que explica 33% a 56% e tem cosseno de
  0,94 a 0,99 com o vetor de pesos iguais. O grafo não descreve o modo de mercado.
- Nó do Fed: liga-se a 8% a 16% das ações do S&P 500 em cada janela (33% em 2023). A correlação
  de postos entre a exposição ao Fed e o beta de mercado vai de −0,04 a +0,26. Petróleo: 6% a
  10% das ações; China: 5% a 12%.
- Testes já feitos: reação ao FOMC, coeficiente +0,0029 com t = 1,8, não confirmada, com medida
  de surpresa grosseira (variação diária do título de dois anos); fatores observáveis com
  *prior* do grafo (M1), 1,6% pior que sem *prior*, e o *prior* muda a QLIKE em 10⁻⁵; fatores
  espectrais (M5), 52% pior que componentes principais; alinhamento espectral, 0,236 contra
  0,178 dos placebos (significativo e modesto).
- Leitura: nas notícias, o Fed aparece ligado a índices, títulos, dólar e bancos, e não a cada
  empresa. A amostra de 36% do corpus agrava a esparsidade.

### N3. Carteiras

- Rejeitada nos desenhos bem especificados (D2, D3, D4–D8, D9). O D10 não informa (F1).
- A cadeia N1 → N3 não tem apoio: melhor condicionamento não deu melhor carteira.

### O que a linha nova exige

1. **Decidir o estatuto das hipóteses.** N1 e N2 não foram pré-registradas como confirmatórias.
   Ou entram como exploratórias, com os dados já existentes, ou exigem um desenho novo, e não há
   período de teste intocado (o corpus acaba em 2023 e 2019–2023 já foi usado).
2. **Corrigir o veículo do teste** (F1): sem isso, N1 e N3 seriam avaliadas com um estimador
   defeituoso.
3. **Reescrever** Introdução, Metodologia (hipóteses, testes, métricas), os dois capítulos de
   resultados, a Conclusão, o Resumo e o guia de equações. O material do FOMC e do M1 volta ao
   texto; H1b e H1c saem ou viram apoio.
4. **Aceitar o resultado provável**: com o que já foi medido, N1 sai rejeitada (ou "não melhora"),
   N2 sem apoio e N3 rejeitada. A tese fica coerente como diagnóstico de por que o grafo não
   ajuda, mas não como confirmação.

## 5. Ordem sugerida das correções

1. Corrigir as frases de F2 e relatar δ, número de condição e cobertura de `g` (F1, F3, F5).
2. Uniformizar a contagem dos desenhos e os p = 0,000 (seção 3).
3. Corrigir E1, E2, E3, E4 e E8 no texto.
4. Acrescentar a robustez de H1a sem classes duplicadas e a ressalva do QAP em H1b (F4).
5. Decidir a linha de hipóteses (seção 4) antes de qualquer reescrita maior.
6. Só então: análise posterior rotulada com o universo sem classes duplicadas e um alvo com modo
   de mercado.
