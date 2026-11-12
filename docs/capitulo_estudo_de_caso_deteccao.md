# Estudo de Caso e Anatomia da Detecção: Como o Pipeline AMI Captura Quedas de Consumo e Picos Anômalos em UCs Reais

> **Nota de Transparência Metodológica e LGPD:**  
> Em conformidade com a Lei Geral de Proteção de Dados (LGPD) e com os protocolos operacionais da distribuidora, **todos os identificadores reais de Unidades Consumidoras (UCs) e NIOs apresentados em capítulos públicos e figuras do artigo são mascarados** pela função [`censor_uc_id`](file:///d:/Projects/tcc_extraction/src/utils/censorship.py#L15-L32) (ex.: `100****70`), enquanto a tabela interna de inspeção operacional fica restrita ao ambiente local (`output/reports/private_examples_lookup.json`) e ao modo operador do Cockpit Web.  
> Além disso, registra-se uma evolução importante de rigor científico: uma versão exploratória anterior de `acompanhamento_semanal_outlier.png` utilizava uma projeção sintética ilustrativa em torno da mediana diária da UC. O script [`scripts/find_eda_examples.py`](file:///d:/Projects/tcc_extraction/scripts/find_eda_examples.py) e o novo extrator [`src/eda/uc_timeseries.py`](file:///d:/Projects/tcc_extraction/src/eda/uc_timeseries.py) foram integralmente reescritos para consultar **exclusivamente séries temporais reais do Data Lake** (`ami_interval` horário de 7 dias e `uc_day_features` diário de 30 dias), eliminando qualquer perturbação artificial e substituindo marcadores genéricos por gatilhos estatísticos auditáveis (CUSUM, Z-Score robusto via MAD e cobertura de telecomunicação).

---

## 1. Ato I — A Armadilha da Média: Por que Duas UCs com o Mesmo Consumo Faturado São Completamente Diferentes

Considere dois medidores inteligentes reais da rede de Araucária extraídos automaticamente pelo script [`find_eda_examples.py`](file:///d:/Projects/tcc_extraction/scripts/find_eda_examples.py) (ver [`comparacao_duas_ucs_mesmo_consumo.png`](file:///d:/Projects/tcc_extraction/docs/figures/eda/exemplos/comparacao_duas_ucs_mesmo_consumo.png) e [`acompanhamento_semanal_comparacao_par.png`](file:///d:/Projects/tcc_extraction/docs/figures/eda/exemplos/acompanhamento_semanal_comparacao_par.png)). No sistema comercial legado de faturamento mensal, ambos encerram a janela de 30 dias com praticamente a mesma energia ativa diária mediana (~`20 kWh/dia`):

1. **UC Estável `934***50` (Mediana = `20,382 kWh/dia`, `MAD = 0,051`, `CV = 0,0025`, Fator de Carga `0,973`, Alimentador Fonte Nova):**  
   Sua curva de carga horária real de 7 dias (`168h` extraídas de `ami_interval`) exibe uma linha praticamente plana em torno de `0,85 kWh/h` 24 horas por dia, 7 dias por semana ($FC \approx 1{,}0$). Essa é a assinatura física característica de cargas contínuas ininterruptas (como estações rádio-base de telecomunicação, refrigeração industrial ou servidores).
2. **UC Volátil `116****57` (Mediana = `19,224 kWh/dia`, `MAD = 16,564`, `CV = 0,8572`, Fator de Carga `0,150`, Alimentador CSN):**  
   Embora a **mediana diária de faturamento seja virtualmente idêntica** à da primeira UC, sua dispersão robusta (`MAD`) é **324 vezes maior**. No acompanhamento semanal contínuo (`168h`), a curva real oscila de consumo próximo a zero na madrugada para picos comerciais diurnos que atingem **`7,03 kWh/h`**, além de apresentar desequilíbrio de corrente máximo elevado (`1,0`).

> **Lição de engenharia:** A média e a mediana isoladas comprimem 288 leituras diárias de 5 minutos em um único escalar e **apagam a assinatura física do consumo**. Somente ao parear tendência central robusta (`HIST_FA_SUM_MEDIAN`) com métricas de dispersão resistente a outliers (`MAD`, `FA_CV_WINDOW`, `IQR` intradiário) e perfil horário real é possível distinguir uma carga plana contínua de um perfil comercial pulsante ou irregular.

---

## 2. Ato II — Storytelling da Detecção: A História de Como uma UC Entra na Fila de Anomalias (`acompanhamento_semanal_outlier.png`)

Para responder à pergunta central operacional — *"Qual combinação matemática de features e algoritmos permitiu detectar simultaneamente a queda de consumo e os picos isolados nessa UC e promovê-la à lista de anomalias?"* — acompanhamos a jornada completa de uma leitura desde o medidor em campo até o score final no Cockpit Web.

### 2.1 O Que Acontece no Gráfico de Acompanhamento Semanal (e O Que Significa Cada Alarme)

No painel atualizado de [`plot_real_outlier_timeline`](file:///d:/Projects/tcc_extraction/scripts/find_eda_examples.py) (e na visualização interativa do Cockpit Web), a série da UC suspeita é decomposta em dois painéis sincronizados:

- **Painel Superior (Energia Ativa + Baseline + Bandas):**
  - **Linha sólida escura:** consumo real medido (`FA_INTERVAL` agregado por hora em 7 dias / `168h` via `ami_interval`, ou diário `FA_INTERVAL_SUM` via `uc_day_features`).
  - **Linha tracejada azul (`Baseline Mediana Histórica`):** expectativa robusta para aquela hora do dia (`HOD`) ou dia na janela de referência (até 28 dias confiáveis anteriores).
  - **Faixa sombreada azul (`Banda Operacional MAD ±3.5σ`):** envelope estatístico esperado calculado com a escala robusta `1.4826 × MAD`, dentro do qual ~99% das oscilações normais da própria UC residem.
  - **Faixa âmbar e linha vertical tracejada-ponto (`Alarme CUSUM — Início da Queda`):** marca exatamente o ponto de mudança estrutural (*change-point*) em que a mediana móvel pós-evento despenca de forma sustentada em relação ao patamar anterior. **Não se trata de um evento genérico sem explicação:** é o ponto ótimo que maximiza a excursão acumulada do algoritmo CUSUM (`_detect_step_change_cusum`).
  - **Triângulos vermelhos (`Pico Anômalo — Z-Score MAD > +3.5`):** instantes individuais que rompem o teto da banda operacional daquela UC mesmo depois de descontada a sazonalidade horária.
  - **Regiões cinzentas (`Gap Telecom — Cobertura < 70%`):** janelas em que o rádio RF Mesh falhou em entregar ao menos 8 dos 12 intervalos de 5 minutos esperados na hora (ou `< 70%` no dia).
- **Painel Inferior (Qualidade de Comunicação em %):**  
  Exibe hora a hora (ou dia a dia) o percentual real de entregas da memória de massa contra o limiar operacional de `70%`. Isso responde imediatamente à dúvida do operador sobre *por que* um eventual vale na curva não gerou falso alarme: se a barra inferior está vermelha (`< 70%`), houve perda de comunicação; se está verde (`≥ 70%`) e o consumo caiu, a queda é **fisicamente real**.

---

## 3. A Composição de Técnicas: Como o Sistema Captura Queda, Picos e Condena a UC

Nenhum detector isolado capturaria ambos os fenômenos com baixa taxa de falsos positivos em `262.849` avaliações reais de Araucária. A detecção resulta da orquestração em **quatro camadas sequenciais e complementares**:

### Camada 1 — Gating de Telecomunicação & Contexto Espacial (O Filtro Anti-Falso-Positivo)
Antes que qualquer algoritmo estatístico avalie consumo zero ou queda de patamar, [`RetroactiveDataCleaner`](file:///d:/Projects/tcc_extraction/src/datasets/retroactive_cleaner.py) e [`ElectricalAnomalyDetector`](file:///d:/Projects/tcc_extraction/src/models/specialized_detectors.py) aplicam duas barreiras de integridade:

1. **Cobertura mínima diária (`INTERVAL_COVERAGE >= 0.70`):** dias com perda superior a `30%` dos 288 intervalos diários esperados são excluídos do cálculo de `HIST_FA_SUM_MEDIAN`, `HIST_FA_SUM_MAD` e `TREND_FA_SLOPE`.
2. **Quarentena de janela (`AVG_INTERVAL_COVERAGE < 0.60` ou `COVERAGE_RELIABLE_DAYS < 15`):** se o medidor está incomunicável na maior parte do mês — ou se sofreu apagão local — a UC vai para `TELECOM_QUARANTINE` (`10.422` janelas no snapshot atual, conforme [`model_training_summary_latest.json`](file:///d:/Projects/tcc_extraction/output/reports/model_training_summary_latest.json)), ficando **proibida de contaminar o treinamento do modelo de ML** (`meta__is_clean_for_train = False`).

Somente quando a UC prova estar **comunicando normalmente** (enquanto seu consumo cai ou oscila) ela avança para a acusação elétrica.

### Camada 2 — Estatísticas Robustas: O Mecanismo Exato Para "Picos" vs "Quedas"

| Fenômeno na Curva | Estatística / Feature Calculada | Fórmula / Critério Real no Código | Papel na Detecção |
| :--- | :--- | :--- | :--- |
| **Baseline Imune a Picos** | `x__hist_fa_sum_median` e `x__hist_fa_sum_mad` | $\tilde{E} = \text{mediana}(E_d)$, $\text{MAD} = \text{mediana}(\|E_d - \tilde{E}\|)$ sobre dias com cobertura $\ge 0.70$ | A mediana não é puxada pelos próprios picos extremos, permitindo que o pico "sobressaia" em relação à base |
| **Detecção de Picos Isolados** | Modified Z-Score (`z_self`) + `FA_CV_WINDOW` + Tukey Fences em `log1p` | $z_i = \frac{x_i - \tilde{E}}{\max(1.4826 \cdot \text{MAD},\; 0.10|\tilde{E}|,\; 0.05)} > 3.5$ ([`dataset_eda.py`](file:///d:/Projects/tcc_extraction/src/eda/dataset_eda.py) e [`uc_timeseries.py`](file:///d:/Projects/tcc_extraction/src/eda/uc_timeseries.py)) | Dispara alerta pontual mesmo em medidores de baixo consumo base; evita explosão numérica quando $\text{MAD} \approx 0$ |
| **Detecção de Queda Sustentada (Degrau)** | `x__trend_fa_slope` (Slope OLS) + `_detect_step_change_cusum` | Regressão linear no calendário de dias confiáveis ($\text{slope} < 0$) + ponto de corte $t^*$ com máxima excursão acumulada $\sum (y_t - \bar{y})$ e $\Delta\%_{\text{mediana}} \le -25\%$ a $-45\%$ | Identifica o momento exato em que a carga afunda e permanece em um novo patamar inferior (assinatura típica de desvio parcial por *bypass* ou alteração clandestina de medidor) |
| **Evidência Elétrica Auxiliar** | `x__hist_current_imbalance_max`, `x__hist_voltage_imbalance_max`, `x__hist_load_factor_mean` | P90/Máximo de desbalanço de corrente/tensão e fator de carga na janela | Um *bypass* de fase reduz `FA_INTERVAL_SUM` (queda), mas gera **assimetria violenta de corrente entre fases** ou distorção do fator de carga |

### Camada 3 — O Ensemble Não-Supervisionado PyOD + Guardrails Físicos PRODIST ([`src/models/composite.py`](file:///d:/Projects/tcc_extraction/src/models/composite.py))

Os vetores puramente elétricos (`select_pure_electrical_features`) — já livres de variáveis administrativas e de cobertura para que o modelo não "aprenda" que falta de sinal é fraude — alimentam o [`CompositeAnomalyDetector`](file:///d:/Projects/tcc_extraction/src/models/composite.py), que combina **6 detectores de naturezas matemáticas distintas** via normalização quantil-robusta e ponderação fixa:

1. **Isolation Forest (`0.22`):** cria partições recursivas aleatórias no espaço multidimensional. UCs que combinam **slope fortemente negativo + CV/MAD alto** são isoladas na raiz da árvore em pouquíssimos cortes (*short path length*).
2. **LOF — Local Outlier Factor (`0.20`):** compara a densidade local da UC com os seus vizinhos de consumo semelhante. Uma UC que tem `HIST_FA_SUM_MEDIAN = 15 kWh/dia`, mas `TREND_FA_SLOPE < -0.5` e `MAD` 5× maior que os pares, fica isolada em região de baixíssima densidade local.
3. **ECOD (`0.18`) & COPOD (`0.18`):** métodos livres de hiperparâmetros baseados em funções de distribuição empírica e cópulas. A força dessa dupla é enxergar **ambas as caudas simultaneamente**: a cauda esquerda extrema do `trend_fa_slope` (queda abrupta) **e** a cauda direita extrema de `fa_cv_window` / `hist_fa_sum_mad` / `current_imbalance_max` (picos e assimetria).
4. **HBOS (`0.10`):** histogramas independentes por feature que pontuam rapidamente variáveis com valores raros na rede de Araucária.
5. **Robust PCA (`0.12`):** projeta o comportamento normal nos principais componentes lineares e mede o **erro de reconstrução**. Quando uma UC mantém consumo mediano razoável, mas sua correlação interna entre energia ativa, fator de carga e corrente se rompe (por exemplo, consumo despenca com corrente ou pico atípico), a projeção PCA falha em reconstruí-la e eleva o erro residual.
6. **Guardrails Físicos PRODIST Módulo 8 (`+0.15` de bônus direto no score):** se além do desvio estatístico houver subtensão/sobretensão severa (`v_pu < 0.70` ou `> 1.25`), desequilíbrio de tensão crítico (`> 20%`) ou alta razão de fluxo reverso (`> 0.85`), o score recebe um reforço determinístico.

Por fim, todas as UCs elegíveis são ranqueadas pelo `ANOMALY_SCORE`, e as pertencentes ao quantil superior definido pela taxa de contaminação (`100 × (1 - contamination)`) recebem `IS_ANOMALY = True` (`2.629` ocorrências no resumo de treinamento vigente em [`model_training_summary_latest.json`](file:///d:/Projects/tcc_extraction/output/reports/model_training_summary_latest.json)).

### Camada 4 — Taxonomia Causal e Entrada na Fila Operacional ([`src/models/category_reporting.py`](file:///d:/Projects/tcc_extraction/src/models/category_reporting.py))

Após `IS_ANOMALY = True`, uma camada explicável de pós-classificação traduz o vetor z-lift ([`AnomalyEvidenceMiner`](file:///d:/Projects/tcc_extraction/src/models/evidence_miner.py)) e as regras causais (`apply_delimited_isolation_and_categories`) na etiqueta acionável exibida à engenharia da COPEL:

- **`DEGRAU_DESCENDENTE_CONSUMO` / `QUEDA_TENDENCIA_CONSUMO_SLOPE_NEGATIVO`**: quando a queda contínua (`slope`) ou degrau inter-cutoffs (`DELTA_CONSUMO_KWH_DIA <= -45%`) domina a explicação da anomalia.
- **`DESEQUILIBRIO_CARGA_OU_BYPASS` / `ANOMALIA_ELETRICA_SUSPEITA_FRAUDE_PNT`**: quando o desequilíbrio entre fases (`CURRENT_IMBALANCE_MAX`) ou a soma multivariada do ensemble acende o alerta de Perda Não-Técnica (PNT).
- **`ANOMALIA_MULTIVARIADA`**: quando o consenso dos 6 detectores PyOD é atingido pela interação não linear de alta dispersão (`MAD` / picos), variação abrupta de fator de carga e deslocamento vetorial no espaço das features.
- **`POSSIVEL_GD_NAO_CADASTRADA`** (`704` casos no snapshot): quando `RA_REVERSAL_DAYS` / `RA_REVERSAL_RATIO_MAX` revela energia injetada de volta na rede sem registro de microgeração.
- **`MEDIDOR_TRAVADO_OU_CONSUMO_ZERO`** (`3` casos no snapshot): consumo nulo com medidor energizado e comunicando.

### Caso Empírico Selecionado (`acompanhamento_semanal_outlier.png`)
No snapshot real processado (`cutoff = 2026-11-04`), o buscador automático isolou a **UC `685***20`** (Alimentador Polonia, cobertura `99,49%`), classificada como **`DESEQUILIBRIO_CARGA_OU_BYPASS`** com **`ANOMALY_SCORE = 0,6212`**. Mesmo mantendo uma mediana histórica alta (`81,076 kWh/dia`, `MAD = 24,459`), a UC apresenta simultaneamente tendência descendente acentuada (`Slope = -1,3682 kWh/dia`) e severo desequilíbrio de corrente (`0,798`), revelando perda de simetria entre fases compatível com derivação parcial sob carga.

---

## 4. Novo Produto no Cockpit Web e Marimo Notebook: Candidatos a Vistoria, Persistência e Gráfico Duplo On-Demand

Para transformar toda essa pipeline matemática em uma ferramenta operacional imediata para os engenheiros da COPEL no servidor web ([`scripts/run_web_cockpit.py`](file:///d:/Projects/tcc_extraction/scripts/run_web_cockpit.py)) e no notebook analítico ([`notebooks/eda_dataset_marimo.py`](file:///d:/Projects/tcc_extraction/notebooks/eda_dataset_marimo.py)), foram implementadas quatro capacidades centrais:

1. **Categoria `CANDIDATOS_A_VISTORIA` & Métrica `DIAS_ANOMALIA_SEGUIDOS`:**  
   - O pipeline ([`category_reporting.py`](file:///d:/Projects/tcc_extraction/src/models/category_reporting.py)) e o script [`enrich_candidate_vistorias.py`](file:///d:/Projects/tcc_extraction/scripts/enrich_candidate_vistorias.py) isolam exclusivamente anomalias físicas acionáveis (`IS_CANDIDATO_VISTORIA = True`) e quantificam quantos **dias consecutivos** a UC sustentou o desvio ativo (`DIAS_ANOMALIA_SEGUIDOS`), evitando o despacho de equipes por eventos transitórios de 1 único dia.
2. **Dashboard Analítico de Anomalias Detectadas (`GET /api/v1/dashboard`):**  
   - Consolidado no topo da tela principal, mostra a contagem exata e o percentual de todas as UCs detectadas em cada categoria diagnóstica (`CATEGORIA_DIAGNOSTICA`), bem como o botão de atalho rápido **`🎯 Candidatos a Vistoria (Campo)`** e os **Top Alimentadores**.
3. **Visualizador Duplo Interativo On-Demand (`GET /api/v1/meter/<uc>/timeseries` e `plot_uc_interactive`):**  
   - Sem pré-gerar imagens estáticas para 260k UCs: ao clicar em **`📈 Gráfico & Raio-X`** no Cockpit (ou selecionar qualquer UC no Marimo Notebook), o sistema consulta o Data Lake real sob demanda e renderiza **dois gráficos sincronizados**:
     - *Gráfico Superior:* Consumo real (`7d Horário`, `14d Horário` ou `30d Diário`), Mediana Baseline, Banda Operacional MAD (±3.5σ), início de degrau CUSUM e Picos (`z > +3.5`).
     - *Gráfico Inferior:* Diagnóstico de **Dados Faltantes / Cobertura Telecom (%)** contra o limiar de 70%.
4. **Navegação Contínua entre Candidatos & Modo LGPD:**  
   - No painel lateral (*Drawer*), os botões **`◀ Anterior (←)`**, **`Próxima (→) ▶`** e os **atalhos de teclado (`←` / `→`)** permitem revisar os gráficos dos candidatos um a um sem fechar a janela, além do botão **`🔒 LGPD: Censurado / 🔓 Visível`**.

---

## 5. Como Reproduzir as Figuras Reais e Abrir o Novo Cockpit Web (Windows CMD)

Execute os comandos abaixo no Prompt de Comando (CMD) dentro de `d:\Projects\tcc_extraction`:

```cmd
:: 1. Regenerar os exemplos reais (Par Contrastante + Acompanhamento Real) com censura LGPD
uv run python scripts/find_eda_examples.py

:: 2. Enriquecer o dataset com IS_CANDIDATO_VISTORIA e DIAS_ANOMALIA_SEGUIDOS
uv run python scripts/enrich_candidate_vistorias.py

:: 3. Subir o Cockpit Web com Dashboard de Candidatos e Gráficos Duplos On-Demand
uv run python scripts/run_web_cockpit.py --port 8050
```

