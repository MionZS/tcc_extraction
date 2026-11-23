# Auditoria Reconciliada de Novembro/2026 (Fontes Únicas Verificadas)

- **Execução `DAILY_RULES`:** `20261123_110449_017_daily_rules` (Status: `validated`)
- **Execução `OOT_MULTIVIEW`:** `20261123_110515_695_oot_multiview` (Status: `validated`)
- **Git Commit:** `d13d8f8`

## 1. Reconciliação das 7 Divergências Históricas
| Divergência Anterior | Valor Anterior | Valor Auditado Reproduzido | Causa-Raiz Comprovada |
|:---|:---|:---|:---|
| Série 19/11 vs Regra Calibrada | 29.966 | **4,978** (calibrado) vs **28,368** (regra legada) | Regra antiga `current_imb_max >= 0.45` sem materialidade rotulava ~45% da rede; filtro calibrado corrige o critério. |
| OOT 19/11 = 0 candidatos | 0 | **6,333** candidatos OOT | Bug de interseção vazia entre `MultiViewAnomalyPipeline` e `actionable_cats` + sobrescrita de colunas em `classified_ucs_latest`. |
| Texto = 412 vs CSV ~30 mil | 412 | Média `DAILY_RULES` = **5,387.7**; OOT 19/11 = **6,333** | Texto antigo possuía literal hardcoded `412` desvinculado dos pipelines reais. |
| Janelas 1.248.501 vs 1.226.702 | 1.226.702 | **1,248,501** janelas (`19` cortes) | Literal hardcoded `1.226.702` substituído pela soma exata das partições diárias avaliadas. |
| Ablação M4=242 vs M5=274 | M5 > M4 | M4 = **4,390**, M5 sequencial = **4,390** | M5 ignorava filtro M2 (`cov >= 70%`). Funil corrigido garante inclusão estrita $M_5 \subseteq M_4$. |

## 2. Resultado Principal OOT (`OOT_MULTIVIEW` em 2026-11-19)
- População total avaliada: **65,686** UCs únicas
- Elegíveis limpas: **62,857** | Quarentena telecom: **2,829**
- Candidatos à vistoria (política congelada): **6,333** | Baseline IsolationForest (limiar congelado `0.643`): **3,122**

| Categoria Diagnóstica OOT | UCs |
|:---|---:|
| `OPERACAO_REGULAR` | 44,560 |
| `MUDANCA_CLASSE_COMPORTAMENTAL` | 12,100 |
| `ANOMALIA_ELETRICA_SUSPEITA_FRAUDE_PNT` | 4,155 |
| `GARGALO_COLETIVO_RF_MESH` | 1,740 |
| `POSSIVEL_GD_NAO_CADASTRADA` | 1,485 |
| `ANOMALIA_COMUNICACAO_ISOLADA` | 764 |
| `FALHA_METROLOGICA_MEDIDOR_TRAVADO` | 748 |
| `INICIACAO_PERFIL_UC_NOVA` | 71 |
| `TRANSICAO_TROCA_MEDIDOR` | 63 |

## 3. Série Diária `DAILY_RULES` (2026-11-01 a 2026-11-19)
| Corte | População Total | Elegíveis Limpos | Quarentena | Transição Hardware | Candidatos Calibrados | Entradas | Saídas | Permanência | Regra Legada (Diag.) |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2026-11-01 | 65,778 | 62,839 (95.5%) | 2,745 | 194 | 6,404 | +6404 | -0 | 0 | 28,479 |
| 2026-11-02 | 65,765 | 62,838 (95.5%) | 2,750 | 177 | 6,728 | +1122 | -798 | 5606 | 28,506 |
| 2026-11-03 | 65,767 | 62,856 (95.6%) | 2,729 | 182 | 6,403 | +563 | -888 | 5840 | 28,515 |
| 2026-11-04 | 65,766 | 62,988 (95.8%) | 2,597 | 181 | 5,863 | +561 | -1101 | 5302 | 28,547 |
| 2026-11-05 | 65,773 | 63,062 (95.9%) | 2,530 | 181 | 5,450 | +505 | -918 | 4945 | 28,564 |
| 2026-11-06 | 65,705 | 63,069 (96.0%) | 2,460 | 176 | 5,327 | +612 | -735 | 4715 | 28,368 |
| 2026-11-07 | 65,705 | 63,067 (96.0%) | 2,472 | 166 | 5,262 | +625 | -690 | 4637 | 28,356 |
| 2026-11-08 | 65,717 | 63,075 (96.0%) | 2,470 | 172 | 5,053 | +733 | -942 | 4320 | 28,353 |
| 2026-11-09 | 65,690 | 63,154 (96.1%) | 2,372 | 164 | 5,543 | +1064 | -574 | 4479 | 28,393 |
| 2026-11-10 | 65,694 | 63,166 (96.2%) | 2,355 | 173 | 5,567 | +800 | -776 | 4767 | 28,388 |
| 2026-11-11 | 65,690 | 63,163 (96.2%) | 2,343 | 184 | 5,202 | +567 | -932 | 4635 | 28,369 |
| 2026-11-12 | 65,688 | 63,164 (96.2%) | 2,333 | 191 | 5,059 | +635 | -778 | 4424 | 28,395 |
| 2026-11-13 | 65,688 | 63,163 (96.2%) | 2,334 | 191 | 5,121 | +659 | -597 | 4462 | 28,424 |
| 2026-11-14 | 65,684 | 63,163 (96.2%) | 2,338 | 183 | 5,089 | +645 | -677 | 4444 | 28,429 |
| 2026-11-15 | 65,658 | 63,152 (96.2%) | 2,327 | 179 | 4,737 | +603 | -955 | 4134 | 28,430 |
| 2026-11-16 | 65,671 | 63,125 (96.1%) | 2,366 | 180 | 4,775 | +637 | -599 | 4138 | 28,415 |
| 2026-11-17 | 65,689 | 63,092 (96.0%) | 2,397 | 200 | 4,876 | +680 | -579 | 4196 | 28,406 |
| 2026-11-18 | 65,687 | 63,075 (96.0%) | 2,403 | 209 | 4,930 | +650 | -596 | 4280 | 28,396 |
| 2026-11-19 | 65,686 | 62,986 (95.9%) | 2,497 | 203 | 4,978 | +636 | -588 | 4342 | 28,368 |

## 4. Textos Prontos Derivados dos Artefatos (Método, Resultados, Estudos de Caso, Limitações)
- **Método:** A avaliação temporal utilizou 19 cortes diários (2026-11-01 a 2026-11-19, totalizando 1,248,501 janelas `UC × cutoff`) para monitoramento `DAILY_RULES` e validação fora do tempo (`OOT_MULTIVIEW`) com modelo multi-visão ajustado sobre 189,341 janelas limpas (2026-08-31, 2026-09-30, 2026-10-31) e congelado antes da inferência em 2026-11-19.
- **Resultados:** Na inferência congelada OOT de 2026-11-19, das 65,686 UCs ativas (62,857 elegíveis; 2,829 em quarentena), o comitê especializado priorizou **6,333 candidatos operacionais à vistoria** sob score médio 0.2944 (P95 = 0.9115), eliminando a dependência do lote (`max_abs_shift = 0.0` sob normalização congelada).
- **Estudos de Caso (`908***00`):** A auditoria longitudinal e a correção do ciclo de vida confirmam que `908***00` possui histórico anterior no mesmo medidor (`DATA_INSTALACAO = 2026-10-31` no CIS foi atualização cadastral administrativa, não medidor novo). Com retração de consumo FA (18,05 -> 5,72 kWh/d, -68,3%) e 12 dias de injeção reversa RA ativa (14–31 kWh/d, OOT score=0.2944), a UC é corretamente classificada como `POSSIVEL_GD_NAO_CADASTRADA` e priorizada para vistoria.
- **Limitações:** Na ausência de laudos independentes de inspeção em campo (TOI), todas as ocorrências denotam hipóteses diagnósticas e candidatos operacionais, vedada qualquer estimativa de Precision@K, redução real de falsos positivos ou fraude confirmada sem validação física.
