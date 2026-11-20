# Relatório de Análise Exploratória de Dados Real (EDA — Copel Araucária)

Todas as figuras e tabelas abaixo derivam estritamente dos dados reais locais.

## Figura EDA 01: Distribuição de UCs Únicas por Alimentador
- **Arquivo:** `01_ucs_por_alimentador.png` (vetorial: `.svg`) | **Tabela Origem:** `01_ucs_por_alimentador.csv`
- **Método:** Contagem deduplicada por `id__uc_id` agrupada por `meta__feeder`.
- **Interpretação:** Total de 19 alimentadores mapeados; maior circuito concentra 8,810 UCs.
- **Limitações:** UCs sem vínculo geográfico aparecem agrupadas como 'Sem Alimentador'.

## Figura EDA 02: Relação Histórica UC × NIOs
- **Arquivo:** `02_ucs_nios_historicos.png` (vetorial: `.svg`) | **Tabela Origem:** `02_ucs_nios_historicos.csv`
- **Método:** Contagem de `NIO` distintos por `UC` em `meter_installation_history`.
- **Interpretação:** Identificadas 1,288 UCs com troca/múltiplos medidores históricos, justificando a reconciliação temporal UC–NIO.
- **Limitações:** Restrito ao histórico de instalações disponível na extração CIS/ORCA.

## Figura EDA 03: Cobertura Temporal da População
- **Arquivo:** `03_cobertura_temporal_populacao.png` (vetorial: `.svg`) | **Tabela Origem:** `03_cobertura_temporal_populacao.csv`
- **Método:** Histograma de `x__avg_interval_coverage` na última janela de cada UC.
- **Interpretação:** Mediana de cobertura de 99.3%, evidenciando cauda degradada abaixo de 70%.
- **Limitações:** Mede completude de entrega de pacotes, não acurácia metrológica.

## Figura EDA 04: Calendário Real de Integridade das Partições
- **Arquivo:** `04_calendario_integridade_real.png` (vetorial: `.svg`) | **Tabela Origem:** `04_calendario_integridade_real.csv`
- **Método:** Varredura das partições Hive em `output/features/uc_day_features`.
- **Interpretação:** Continuidade verificada em 102 dias entre 2026-08-02 e 2026-11-12.
- **Limitações:** Tamanho em disco reflete volume comprimido Zstd por dia.

## Figura EDA 05: Distribuição de Consumo Diário
- **Arquivo:** `05_distribuicao_consumo_diario.png` (vetorial: `.svg`) | **Tabela Origem:** `05_distribuicao_consumo_diario.csv`
- **Método:** Quantis e histogramas linear e logarítmico de `x__hist_fa_sum_median`.
- **Interpretação:** Forte assimetria à direita: mediana 4.94 kWh/dia vs P99 55.06 kWh/dia, justificando uso de RobustScaler e estimadores não-paramétricos.
- **Limitações:** Representa energia ativa direta (FA) sem somar energia reativa.

## Figura EDA 06: Curvas de Carga Horárias Normalizadas
- **Arquivo:** `06_curvas_carga_por_perfil.png` (vetorial: `.svg`) | **Tabela Origem:** `06_curvas_carga_por_perfil.csv`
- **Método:** Medianas horárias reais extraídas de `ami_interval` e normalizadas por soma unitária.
- **Interpretação:** Evidencia diversidade entre picos noturnos residenciais, carga contínua e perfis diurnos.
- **Limitações:** Amostra restrita às UCs com partições intervalares locais disponíveis.

## Figura EDA 07: Variabilidade e Dispersão Robusta (MAD vs CV)
- **Arquivo:** `07_variabilidade_dispersao_robusta.png` (vetorial: `.svg`) | **Tabela Origem:** `07_variabilidade_dispersao_robusta.csv`
- **Método:** Dispersão bivariada de `x__hist_fa_sum_median` vs `x__hist_fa_sum_mad` colorida por `x__fa_cv_window`.
- **Interpretação:** Demonstra que UCs com mesmo consumo mediano exibem amplitudes de MAD e CV radicalmente distintas.
- **Limitações:** Eixos truncados no gráfico para legibilidade do núcleo populacional.

## Figura EDA 08: Diferença entre Magnitude e Formato
- **Arquivo:** `08_magnitude_vs_morfologia.png` (vetorial: `.svg`) | **Tabela Origem:** `08_magnitude_vs_morfologia.csv`
- **Método:** Comparação horária direta de duas UCs reais com energia diária equivalente (~19-20 kWh/dia).
- **Interpretação:** Refuta regras unidimensionais baseadas apenas em kWh mensal; exige métricas de forma e dispersão.
- **Limitações:** Caso ilustrativo em par contrastante específico.

## Figura EDA 09: Distribuição de Dias Atípicos Consecutivos
- **Arquivo:** `09_distribuicao_dias_atipicos.png` (vetorial: `.svg`) | **Tabela Origem:** `09_distribuicao_dias_atipicos.csv`
- **Método:** Contagem de `DIAS_ANOMALIA_SEGUIDOS > 0` na população deduplicada.
- **Interpretação:** Permite separar eventos transitórios de desvios persistentes prioritários para campo.
- **Limitações:** Limitado ao horizonte da janela de 30 dias.

## Figura EDA 10: Distribuição dos Scores por Especialista
- **Arquivo:** `10_distribuicao_scores_especialistas.png` (vetorial: `.svg`) | **Tabela Origem:** `10_distribuicao_scores_especialistas.csv`
- **Método:** Densidade empírica dos scores [0,1] dos detectores do `CompositeAnomalyDetector`.
- **Interpretação:** Mostra diferenças de calibração entre escalas lineares (PCA), densidade (LOF) e caudas (ECOD/COPOD).
- **Limitações:** Normalização min-max altera amplitude relativa conforme extremos do lote.

## Figura EDA 11: Concentração de Anomalias por Alimentador
- **Arquivo:** `11_concentracao_anomalias_alimentador.png` (vetorial: `.svg`) | **Tabela Origem:** `11_concentracao_anomalias_alimentador.csv`
- **Método:** Razão `ucs_anomalas / total_ucs` por alimentador (mín. 20 UCs).
- **Interpretação:** Evidencia heterogeneidade espacial entre circuitos, motivando modelos contextualizados por alimentador.
- **Limitações:** Taxa reflete anomalias detectadas pelo modelo, não fraudes confirmadas em campo.

## Figura EDA 12: Diferenças entre Populações Elegíveis e Inelegíveis
- **Arquivo:** `12_elegiveis_vs_nao_elegiveis.png` (vetorial: `.svg`) | **Tabela Origem:** `12_elegiveis_vs_nao_elegiveis.csv`
- **Método:** Comparação estatística entre UCs com cobertura >= 70% e < 70%.
- **Interpretação:** Quantifica o perfil das UCs isoladas pelo gating de comunicação para evitar viés de seleção silencioso.
- **Limitações:** UCs com cobertura muito baixa podem ter estimativa de consumo enviesada pelos poucos dias válidos.
