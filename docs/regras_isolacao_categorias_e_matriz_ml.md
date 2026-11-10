# Delimitação de Regras de Isolação, Contraste Espacial de Comunicação, Mudança de Classe e Matriz "Onde Usar ML vs. Regras"

Este documento consolida a arquitetura analítica implementada em [`src/models/category_reporting.py`](file:///d:/Projects/tcc_extraction/src/models/category_reporting.py) e [`src/models/composite.py`](file:///d:/Projects/tcc_extraction/src/models/composite.py), respondendo diretamente a quatro questões fundamentais do projeto:

1. **O que exatamente acontece com UCs novas (1ª instalação), medidores trocados em UCs existentes e medidores que comunicam mal?**
2. **Como provamos pela proximidade (transformador/vizinhança geográfica) que uma falha de comunicação é anomalia individual do medidor e não falha da rede RF Mesh?**
3. **Como agrupamos consumidores parecidos em classes e listamos as UCs que mudaram de classe ao longo do tempo?**
4. **Onde faz sentido usar Machine Learning (PyOD) e onde faz sentido usar Regras Determinísticas, Físicas ou Estatística Espacial?**

---

## 1. Delimitação Estrita dos 4 Regimes de Isolação e Ciclo de Vida

Antes de qualquer inferência de fraude ou anomalia elétrica, cada janela (`UC × CUTOFF_DATE`) passa por uma triagem determinística e topológica que separa **4 situações operacionais distintas**:

| Situação Operacional | Código da Categoria | Regra Matemática e Cadastral Exata | O que acontece com a UC no Pipeline? | Relatório Próprio Exportado |
| :--- | :--- | :--- | :--- | :--- |
| **1. UC Nova (Primeira Instalação de Medidor)** | `INICIACAO_PERFIL_UC_NOVA` | `N_METERS_HISTORICAL == 1` (apenas 1 `NIO` na história da `UC` em `meter_installation_history`) **E** (`METER_AGE_DAYS <= 30` ou `CUTOFF_DATE - INICIO_UC <= 30`). | **Não vai para a lista de fraude nem para a quarentena genérica de telecom.** Entra na fila de **Iniciação de Perfil**. Se já possuir $\ge 7$ dias confiáveis na janela, recebe uma **Classe Comportamental Preliminar** (`C1` a `C6`) para validar se o porte inicial bate com a tarifa contratada. | `output/reports/categorias/relatorio_01_iniciacao_perfil_ucs_novas.csv` |
| **2. Medidor Trocado em UC Existente** | `TRANSICAO_TROCA_MEDIDOR` | `N_METERS_HISTORICAL >= 2` (já teve medidor anterior na mesma `UC`) **E** `0 < METER_AGE_DAYS <= 30` (`METER_CHANGED_30D == True`). | **Isolada em Transição de Hardware.** Evita que o degrau causado pela substituição física do medidor contamine o treino. Compara o consumo antes vs. depois da troca: se o consumo subiu fortemente após o novo medidor entrar, sinaliza **submedição/fraude no medidor antigo removido**. | `output/reports/categorias/relatorio_02_transicao_troca_medidor.csv` |
| **3A. Medidor Comunica Mal com Vizinhos que Comunicam Bem** | `ANOMALIA_COMUNICACAO_ISOLADA` | `AVG_INTERVAL_COVERAGE < 0.60` (ou `< 15` dias confiáveis), `METER_AGE_DAYS > 30`, **E** os vizinhos do mesmo transformador (`POSTO_OPERACIONAL`) ou célula geográfica de $250\text{ m}$ (`LATITUDE`/`LONGITUDE`) têm **`PEER_MEDIAN_COVERAGE >= 0.80`** com **`COVERAGE_DEFICIT_VS_PEERS >= 0.30`**. | **Anomalia Individual de Comunicação Confirmada por Proximidade.** Como os vizinhos imediatos entregam $\ge 80\%$ dos pacotes no mesmo período, fica provado que a rede RF Mesh do local está íntegra e o defeito (ou sabotagem de antena/gaiola metálica) está **naquele medidor específico**. | `output/reports/categorias/relatorio_03_comunicacao_isolada_vizinhanca_boa.csv` |
| **3B. Gargalo Coletivo de Malha RF Mesh** | `GARGALO_COLETIVO_RF_MESH` | `AVG_INTERVAL_COVERAGE < 0.60` **E** a mediana dos vizinhos do mesmo transformador/rua também é baixa (`PEER_MEDIAN_COVERAGE < 0.80` ou déficit $< 0.30$). | **Isolada como Problema de Infraestrutura de Rede.** Não gera ordem de serviço comercial na UC; vai para o relatório de manutenção de repetidores/concentradores (DCU) da equipe de Telecom. | `output/reports/categorias/relatorio_03b_gargalo_coletivo_malha_mesh.csv` |

---

## 2. Separação Total entre Comunicação e Detector de Anomalias Elétricas (PyOD)

### Por que o detector antigo misturava comunicação com perfil elétrico?
Anteriormente, a matriz $\mathbf{X}$ enviada ao `CompositeAnomalyDetector` selecionava todas as colunas `x__*`, incluindo `x__avg_interval_coverage`, `x__coverage_reliable_days`, `x__alarm_count_window`, `x__avg_alarm_latency_sec` e `x__meter_age_days`. Mesmo filtrando janelas com cobertura $\ge 60\%$, um medidor com cobertura de $62\%$ recebia score alto de anomalia no Isolation Forest/ECOD apenas por ter menos pacotes que a mediana de $98\%$.

### Como funciona agora (`select_pure_electrical_features`)
Em [`src/models/composite.py`](file:///d:/Projects/tcc_extraction/src/models/composite.py), a função [`select_pure_electrical_features`](file:///d:/Projects/tcc_extraction/src/models/composite.py#L46-L65) **bloqueia estritamente** qualquer variável de cobertura, latência, alarme ou idade de medidor. O detector de anomalias elétricas enxerga **exclusivamente Consumo, Tensão e Corrente**:
- **Consumo Ativo e Dispersão**: `x__hist_fa_sum_median` ($\text{kWh/dia}$), `x__hist_fa_sum_mad` (dispersão robusta intra-janela), `x__fa_cv_window` (coeficiente de variação robusto $\text{MAD}/\text{Mediana}$), `x__trend_fa_slope` (inclinação linear em $\text{kWh/dia}^2$), `x__hist_load_factor_mean` (fator de carga médio), `x__zero_consumption_flag`.
- **Corrente e Equilíbrio entre Fases**: `x__hist_current_imbalance_max` (desbalanço P90 de corrente apenas em intervalos sob carga $\bar{I} \ge 1,0\text{ A}$).
- **Tensão e Injeção Reversa**: `x__hist_voltage_imbalance_max` (desbalanço P95 em fases energizadas $\ge 50\text{ V}$), `x__hist_ra_reversal_days`, `x__hist_ra_reversal_ratio_max`, `x__tensao_base`, `x__pot_inst_kva`.

### Estamos usando PyOD? Melhorias Implementadas no Detector de Anomalias
O pacote `CompositeAnomalyDetector` em [`src/models/composite.py`](file:///d:/Projects/tcc_extraction/src/models/composite.py) foi expandido para operar um **Comitê de 6 Algoritmos da Família PyOD** (compatível nativamente com `pyod` se instalado no ambiente e com implementação vetorial determinística em NumPy/SciPy/Scikit-Learn):
1. **Isolation Forest (`iso_forest`, peso $0,22$)**: Isola anomalias multivariadas não-lineares (ex: queda de consumo combinada com aumento de desbalanço de corrente).
2. **LOF — Local Outlier Factor (`lof`, peso $0,20$)**: Compara a densidade local de cada UC contra seus $k=20$ vizinhos de mesmo porte elétrico.
3. **ECOD — Empirical Cumulative Distribution Outlier Detection (`ecod`, peso $0,18$, Li et al. TKDE 2022)**: Calcula a probabilidade empírica não-paramétrica nas caudas esquerda e direita de cada grandeza elétrica.
4. **COPOD — Copula-Based Outlier Detection (`copod`, peso $0,18$, Li et al. ICDM 2020)**: Modela a cópula empírica multivariada com **correção de assimetria (*skewness-corrected tail*)**, ideal para variáveis elétricas positivamente assimétricas como consumo e desbalanço de fase.
5. **HBOS — Histogram-Based Outlier Score (`hbos`, peso $0,10$, Goldstein & Dengel 2012)**: Estimador rápido de densidade por histogramas dinâmicos com suavização de Laplace, altamente sensível a modos raros (travamento ou degrau).
6. **Robust PCA Reconstruction Error (`pca`, peso $0,12$)**: Mede o erro ortogonal de reconstrução fora do subespaço linear normal de operação elétrica.

---

## 3. Classificação de Consumidores Parecidos e Relatório de Mudança de Classe no Tempo

Cada janela (`UC × CUTOFF_DATE`) é classificada em um dos **6 Arquétipos Comportamentais de Consumo (`CLASSE_COMPORTAMENTAL`)**:
- **`C1_MINIMO_OU_DESOCUPADO`**: Mediana $< 0,50\text{ kWh/dia}$ (imóvel vago, uso esporádico ou medidor travado).
- **`C2_RESIDENCIAL_BAIXO_CONSUMO`**: $0,50 \le \text{Mediana} < 3,50\text{ kWh/dia}$ ($\approx 15\text{ a }105\text{ kWh/mês}$).
- **`C3_RESIDENCIAL_PADRAO`**: $3,50 \le \text{Mediana} < 10,00\text{ kWh/dia}$ ($\approx 105\text{ a }300\text{ kWh/mês}$).
- **`C4_COMERCIAL_OU_ELEVADO`**: $10,00 \le \text{Mediana} < 25,00\text{ kWh/dia}$ ($\approx 300\text{ a }750\text{ kWh/mês}$).
- **`C5_ALTA_CARGA_CONTINUA`**: $\text{Mediana} \ge 25,00\text{ kWh/dia}$ ($> 750\text{ kWh/mês}$).
- **`C6_PROSUMIDOR_INJECAO_REVERSA`**: `RA_REVERSAL_RATIO_MAX > 0.05` (atuação como microgerador / injeção reversa na rede).

Ao comparar a classe de cada UC nos 4 momentos consolidados (`2026-08-31` $\to$ `2026-09-30` $\to$ `2026-10-31` $\to$ `2026-11-05`), a função [`build_temporal_class_transition_report`](file:///d:/Projects/tcc_extraction/src/models/category_reporting.py#L288-L382) gera o relatório dedicado **`relatorio_04_mudanca_classe_comportamental.csv`** contendo apenas as UCs que mudaram de classe, com a coluna `TRAJETORIA_CLASSES_TEMPORAL` e a hipótese automática:
- **`QUEDA_ABRUPTA_CLASSE_SUSPEITA_BYPASS_OU_DESOCUPACAO`**: Caiu $\ge 1$ degrau de classe com queda $> 45\%$ em $\text{kWh/dia}$ sem troca de medidor.
- **`ASCENSAO_CLASSE_TROCA_TITULARIDADE_OU_NOVO_EQUIPAMENTO`**: Subiu de classe (candidata direta à validação com a data de troca de titularidade ou aquisição de ar-condicionado/bomba de calor/EV).
- **`NOVO_PROSUMIDOR_ENTRADA_GD_SOLAR`**: Passou de classe consumidora (`C1`..`C5`) para `C6_PROSUMIDOR_INJECAO_REVERSA`.
- **`MUDANCA_CLASSE_ASSOCIADA_TROCA_MEDIDOR`**: Mudou de classe exatamente após a substituição do medidor.

---

## 4. Matriz Metodológica: Onde Faz Sentido Ser ML vs. Regra Determinística / Espacial

| # | Categoria / Relatório (`output/reports/categorias/`) | Faz Sentido Usar ML? | Método Recomendado e Implementado | Por que essa escolha é a mais rigorosa? |
| :-: | :--- | :---: | :--- | :--- |
| **01** | `relatorio_01_iniciacao_perfil_ucs_novas` *(UC Nova / 1ª Instalação)* | **NÃO para isolar** / **SIM para classificar porte após 7d** | **Regra Determinística Cadastral** (`N_METERS == 1` e `IDADE <= 30d`) + Arquétipo Preliminar | A data de ligação da UC e o fato de ser o 1º medidor são fatos exatos de banco de dados. Aplicar detector de anomalia em UC com 10 dias de vida só geraria falso alarme por falta de histórico. |
| **02** | `relatorio_02_transicao_troca_medidor` *(Troca de Medidor em UC Existente)* | **NÃO para isolar** / **Estatística Longitudinal Antes×Depois** | **Regra de Ciclo de Vida** (`N_METERS >= 2` e `IDADE <= 30d`) + Delta de Consumo Pós-Troca | A troca física do equipamento é registrada no histórico de instalação. O valor analítico está em medir o salto $\Delta\text{kWh}$ antes vs. depois da troca para provar submedição do medidor antigo. |
| **03A** | `relatorio_03_comunicacao_isolada_vizinhanca_boa` *(Medidor Comunica Mal c/ Vizinhos Bons)* | **NÃO precisa de ML caixa-preta** | **Estatística Espacial/Topológica de Vizinhança** (Mediana do `POSTO_OPERACIONAL` ou Grid $250\text{ m} \ge 80\%$ vs. UC $< 60\%$) | O que prova que um medidor tem defeito individual de comunicação é o **contraste direto com seus vizinhos de mesmo transformador/rua**. Uma métrica espacial explícita (`COVERAGE_DEFICIT_VS_PEERS >= 0.30`) é 100% auditável pela engenharia de telecom. |
| **03B** | `relatorio_03b_gargalo_coletivo_malha_mesh` *(Falha Coletiva de Malha)* | **NÃO** | **Agregação Topológica por Transformador / Alimentador** (`PEER_MEDIAN_COVERAGE < 0.80`) | Se o transformador inteiro está com baixa cobertura, trata-se de queda de concentrador/repetidor RF Mesh — uma regra topológica direta resolve sem complexidade desnecessária. |
| **04** | `relatorio_04_mudanca_classe_comportamental` *(Mudança de Classe no Tempo)* | **SIM (Classificação de Perfil + Transição de Regime)** | **Classificação de Arquétipos de Carga (`C1`..`C6`) + Matriz de Transição Temporal entre Cutoffs** | Permite agrupar dezenas de milhares de UCs em perfis interpretáveis e listar exatamente quem mudou de comportamento entre Agosto, Setembro, Outubro e Novembro/2026. |
| **05** | `relatorio_05_anomalias_eletricas_fraude_pnt` *(Anomalia Elétrica Pura / Fraude / Bypass)* | **SIM — Essencial (Core de ML Não-Supervisionado)** | **Comitê PyOD de 6 Modelos (`IsoForest + LOF + ECOD + COPOD + HBOS + PCA`) sobre Features Elétricas Puras** | É aqui que o Machine Learning é insubstituível: fraudes reais combinam queda parcial de consumo, alteração na dispersão diária (`MAD`/`CV`), fator de carga atípico e desbalanço de corrente entre fases sob carga ($\ge 1\text{ A}$). Nenhum `if/else` fixo captura essas interações multivariadas. |
| **06** | `relatorio_06_gd_nao_cadastrada_reversao` *(Possível GD Não Cadastrada)* | **HÍBRIDO (Regra Física + Score de Cauda COPOD/ECOD)** | **Gatilho Físico de Fluxo Reverso (`RA_RATIO >= 0.08`) + Score de Intensidade PyOD** | A existência de fluxo reverso (`RA > 0`) sem cadastro de GD é uma lei física; já o score de cauda (ECOD/COPOD) ajuda a ranquear a severidade e consistência da injeção solar ao longo dos dias. |
| **07** | `relatorio_07_falha_metrologica_medidor_travado` *(Medidor Travado / Consumo Zero c/ Tensão)* | **HÍBRIDO (Regra Física + Densidade Local LOF/HBOS)** | **Regra Metrológica (`FA_MEDIAN < 0.05`, `U_L1 > 100V`, Cobertura Limpa) + Score PyOD** | A condição física (tensão normal na rede + comunicação ativa + consumo travado em zero) delimita o candidato, e o score LOF/HBOS confirma a discrepância frente ao circuito local. |
| **Nível 1** | Classificador de Perturbações de Tensão (`PRODIST Módulo 8`) | **OPCIONAL / REDUNDANTE para rotulagem estrita** | **Regra Normativa PRODIST M8** (ou ML apenas para probabilidades suaves / previsão) | Como os limites de afundamento, elevação e corte de tensão são definidos por lei pela ANEEL ($0,80\text{ pu}$, $1,10\text{ pu}$, etc.), a fórmula determinística já atinge 100% de exatidão regulatória; o foco de ML deve permanecer no **Detector de Anomalias Elétricas (Categoria 05)** e na **Transição Comportamental (Categoria 04)**. |

