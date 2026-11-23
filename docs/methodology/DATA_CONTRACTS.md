# Contratos Canônicos de Dados e Unidades Analíticas

## 1. Grãos Analíticos Oficiais
| Entidade | Grão / Chave Primária | Política de Deduplicação & Validade |
|:---|:---|:---|
| Medição | `NIO × timestamp × canal` | Deduplicação por último timestamp; cadência via `CADENCE_MINUTES` |
| Vínculo | `UC × NIO × [DATA_INSTALACAO, DATA_REMOCAO]` | Respeita trocas intradia e resets; `DATA_INSTALACAO <= cutoff_date` |
| Série UC | `UC × timestamp × canal` | Reconstruída após resolução temporal NIO -> UC |
| Dia comportamental | `UC × REPORT_DAY` | 1 linha por UC/dia (maior `INTERVAL_COVERAGE` em trocas) |
| Janela experimental | `UC × cutoff_date × WINDOW_DAYS` | Estritamente `(cutoff_date - 30d, cutoff_date]` |
| Score | `run_id × model × UC × cutoff_date` | Armazenado em Parquet versionado sob `experiments/runs/<run_id>/` |
| Hipótese diagnóstica | `UC × cutoff_date × categoria × versão` | Interpretação física/estatística sem confirmação causal |
| Candidato | `UC × cutoff_date × política` | Subconjunto aprovado pela política operacional |
| Inspeção / Resultado | `inspecao_id` / `inspecao_id × resultado` | **Pendência externa** (aguarda integração TOI de campo) |

## 2. Taxonomia Obrigatória de Resultados
1. **Anomalia estatística:** Desvio extremo em relação à distribuição de referência (`score >= frozen_threshold`).
2. **Anomalia de qualidade:** Falha de entrega ou insuficiência amostral (`INTERVAL_COVERAGE < 0.70`, quarentena telecom).
3. **Hipótese diagnóstica:** Interpretação elétrica/operacional compatível (e.g. `POSSIVEL_GD_NAO_CADASTRADA`, `ANOMALIA_ELETRICA_SUSPEITA_FRAUDE_PNT`), **jamais** confirmação de fraude sem TOI.
4. **Candidato operacional:** UC elegível aprovada pelos critérios de materialidade, persistência e acionabilidade.
5. **Positivo confirmado:** Somente após inspeção física independente (não disponível no Data Lake local).

## 3. Catálogo Formal de KPIs (Seção 3.2)
| `metric_id` | Nome | Grão | Numerador | Denominador | Filtros | Validação |
|:---|:---|:---|:---|:---|:---|:---|
| `evaluated_windows` | Janelas Avaliadas (UC × Cutoff) | `UC × cutoff_date` | `COUNT(*) em features de janela` | `N/A (contagem absoluta)` | `REPORT_DAY in (cutoff - 30d, cutoff]` | `N_total == N_eligible + N_quarantine + N_other` |
| `distinct_ucs` | UCs Únicas | `UC (id__uc_id normalizado)` | `COUNT(DISTINCT id__uc_id)` | `População ativa elegível ou total do cadastro` | `Por corte ou dedup no último corte` | `distinct_ucs <= evaluated_windows` |
| `eligible_ucs` | UCs Elegíveis (Telemetria Limpa) | `UC × cutoff_date` | `COUNT(UC WHERE is_eligible_clean == True)` | `N_total no corte` | `reliable_days >= min_reliable_days AND avg_coverage >= min_window_coverage` | `E_t ∩ Q_t == ∅` |
| `quarantine_ucs` | UCs em Quarentena Telecom | `UC × cutoff_date` | `COUNT(UC WHERE telecom_status LIKE 'QUARANTINE%')` | `N_total no corte` | `NOT is_eligible_clean AND NOT is_other` | `E_t + Q_t + O_t == N_total` |
| `statistical_anomaly_ucs` | UCs com Anomalia Estatística | `UC × cutoff_date` | `COUNT(UC WHERE is_eligible_clean AND is_statistical_anomaly)` | `N_eligible (apenas elegíveis)` | `is_eligible_clean == True AND score >= frozen_threshold` | `0 <= statistical_anomaly_ucs <= eligible_ucs` |
| `diagnostic_hypothesis_ucs` | UCs com Hipótese Diagnóstica | `UC × cutoff_date` | `COUNT(UC WHERE categoria_diagnostica IN actionable_hypotheses)` | `N_eligible no corte` | `is_eligible_clean == True` | `Mutualmente exclusivo por categoria primária` |
| `candidate_ucs` | Candidatos à Vistoria (Campo) | `UC × cutoff_date` | `COUNT(UC WHERE is_candidato_vistoria == True)` | `N_eligible no corte` | `is_eligible_clean AND hypothesis_actionable AND materiality AND persistence` | `C_t == I_t ∪ P_t; C_{t-1} == O_t ∪ P_t` |
| `queue_entries` | Entradas na Fila (I_t) | `UC × cutoff_date` | `|C_t \ C_{t-1}|` | `N/A ou |C_{t-1}| para taxa de renovação` | `t > t_0` | `I_t ∩ P_t == ∅` |
| `queue_exits` | Saídas da Fila (O_t) | `UC × cutoff_date` | `|C_{t-1} \ C_t|` | `|C_{t-1}| para taxa de saída` | `t > t_0` | `O_t ∩ P_t == ∅` |
| `queue_persistence` | Permanência na Fila (P_t) | `UC × cutoff_date` | `|C_t ∩ C_{t-1}|` | `|C_{t-1}| para taxa de retenção` | `t > t_0` | `C_t == I_t ∪ P_t AND C_{t-1} == O_t ∪ P_t` |

## 4. Schema ML (`id__`, `meta__`, `x__`, `y__`)
- `id__*`: Identificadores (`id__uc_id`), nunca usados como features.
- `meta__*`: Metadados temporais (`meta__cutoff_date`, `meta__split`, `meta__data_quality`, `meta__is_clean_for_train`, `meta__feeder`). Informações futuras proibidas.
- `x__*`: Features puras de entrada; variáveis de comunicação/ciclo de vida são excluídas da matriz elétrica via `select_pure_electrical_features`.
