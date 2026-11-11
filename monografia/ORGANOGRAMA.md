# Organograma da monografia

Lógica do texto: **H1 (as notícias preveem a correlação) ⇒ H2 (isso melhora a carteira)**.
Três hipóteses: H1a (exposição comum), H1b (tipo de relação) e H2 (carteiras).
Os capítulos 6 e 7 respondem cada uma. Texto conciso: cerca de 52 páginas de corpo, 62 com referências
e apêndices. Sem análises de robustez nem menção a testes fora das três hipóteses.

```
Monografia.tex
├── Pré-textuais ─ Capa · Resumo · Agradecimentos · Sumário · Listas de figuras e tabelas
│
├── 1 Introdução  (Introducao/, ~5 p)
│     contexto · pergunta e hipóteses · objetivos · contribuições · organização
│
├── 2 Fundamentação Teórica e Revisão Bibliográfica  (Revisao/, ~12 p)
│     carteiras e Σ · estimação de Σ (amostral, fatores, shrinkage linear e não linear,
│     esparsidade, Black-Litterman) · redes em finanças · vínculos e notícias ·
│     texto → Σ (Tang; Becquin e Esmeir; Lu et al.) · LLMs · inferência · lacuna
│
├── 3 Metodologia  (Metodologia/, ~12 p)
│     pré-registro · hipóteses H1a, H1b e H2 · atributos de par · previsão de correlação
│     e testes de pares · estimadores com notícias (alvo, fator+resíduo, BL) ·
│     referências e placebo · carteiras · desenhos D1–D9 · métricas e testes
│
├── 4 Dados  (Dados/, ~7 p)
│     FNSPID (ingestão, curadoria) · painel de preços · universos · setores e índice de referência
│
├── 5 Extração de Relações e Construção do Grafo  (Extracao/, ~10 p)
│     tarefa e prompt · escolha do modelo · execução · filtro de ruído ·
│     ligação a tickers · grafo dinâmico (figura de exemplo) · validação · descritivas
│
├── 6 Resultados I: as Notícias e a Correlação Futura  (ResultadosCorrelacao/, ~8 p)   ← H1
│     correlação total · residual e amostra ampliada · canais: exposição comum (H1a),
│     tipo e sinal (H1b) · síntese
│
├── 7 Resultados II: as Notícias e as Carteiras  (ResultadosCarteiras/, ~12 p)        ← H2
│     D1 alvo-prior · D2 fator+resíduo · D3 long-only · D4–D8 otimizadores · D9 BL · síntese
│
├── 8 Discussão e Conclusões  (Conclusao/, ~5 p)
│     síntese · mecanismos · sinal do LLM × correlação · literatura · limitações · futuros
│
├── Referências  (referencias.bib, natbib autor-ano)
│
└── Apêndices  (Apendices/)
      A Prompt e esquema · B Reprodutibilidade
```

## Fontes de cada capítulo

| capítulo | de onde vem o conteúdo |
|---|---|
| 1 | `docs/guia/guia_tese.tex` §2 |
| 2 | guia §3; `docs/metodo_covariancia.md` §2; roteiro §11 |
| 3 | guia §6; método §4–5, 10, 12, 14, 16, 18 |
| 4 | guia §4; `README.md` partes 1 e 4 |
| 5 | guia §5; `README.md` partes 2 e 3; `docs/secao18_resultados.md` §2–3 |
| 6 | método §8, 11, 13; resultados da seção 18, §4; `scripts/revisao_qap.py` |
| 7 | método §9, 11, 13, 15, 17 |
| 8 | guia §7–9 |

## Convenções

- Um `.tex` por capítulo, figuras em `<Capitulo>/Figuras/`.
- Rótulos: `cap:`, `sec:`, `fig:`, `tab:`, `eq:`, `ap:` (ex.: `\ref{sec:testes-pares}`).
- `\pendente{...}` marca em vermelho o que falta escrever; para a versão final, troque a
  definição em `Monografia.tex` por `\newcommand{\pendente}[1]{}`.
- Citações: `\citet{chave}` (Autor (ano)) e `\citep{chave}` ((Autor, ano)); chaves em `referencias.bib`.
