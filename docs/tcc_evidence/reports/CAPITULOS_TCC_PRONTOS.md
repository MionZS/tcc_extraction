# Capítulos Metodológicos e Experimentais Prontos para Incorporação ao TCC

**Título:** *Pattern Analysis for Anomaly Localization in Distribution Networks with Machine Learning Approaches*

---

## Capítulo X — Metodologia de Engenharia de Dados e Fusão Multi-Visão

### X.1 Entidade Analítica Canônica e Blindagem Temporal
Toda a modelagem projeta fenômenos elétricos intervalares (5 min), contexto cadastral/topológico e eventos de alarme na entidade canônica $\text{UC} \times \text{cutoff\_date}$. Nenhuma medição com $\text{REPORT\_DAY} > \text{cutoff\_date}$ participa do cálculo de atributos móveis, garantindo ausência estrita de vazamento temporal (*temporal leakage*).

### X.2 Formalização Matemática da Fusão de Especialistas (Fiel ao Código)
A arquitetura executa quatro camadas sequenciais:

1. **Blindagem de Subespaço (`select_pure_electrical_features`):** Exclusão determinística de variáveis de comunicação e ciclo de vida (`coverage`, `latency`, `alarm`, `meter_age`, `meter_changed`).
2. **Comitê Elétrico Não-Supervisionado (`CompositeAnomalyDetector`):** Combina seis famílias ortogonais padronizadas por `RobustScaler` e normalizadas em $[0, 1]$:
   $$S_{\text{comp}}(x) = \text{clip}\left(\frac{0.22\,s_{\text{IF}} + 0.20\,s_{\text{LOF}} + 0.18\,s_{\text{ECOD}} + 0.18\,s_{\text{COPOD}} + 0.10\,s_{\text{HBOS}} + 0.12\,s_{\text{PCA}}}{1.0} + 0.15\cdot \mathbb{I}_{\text{PRODIST}}(x),\; 0,\; 1\right)$$
3. **Gating de Qualidade (`ElectricalAnomalyDetector`):** Observações com `data_quality < 0.70` recebem `is_reliable = False` e não são declaradas anomalias elétricas.
4. **Roteamento Hierárquico (`MoERouter`) e Filtro Espúrio (`filter_spurious_anomalies`):** Combina modelos por alimentador, município e global conforme tamanho amostral do circuito; eventos coletivos no mesmo transformador ($\ge 60\%$ dos pares anômalos) sofrem atenuação ($\times 0.35$) e falhas não isoladas ($\times 0.40$).

---

## Capítulo Y — Resultados Experimentais na Rede de Araucária

Na campanha analisada sobre **66,099 UCs únicas** (**262,849 janelas** `UC × cutoff_date`), o sistema identificou **932 UCs com anomalia detectada** (1.41%) no último corte temporal e priorizou **274 candidatos a vistoria** por persistência temporal consecutiva.

Os resultados completos de ablação ($M_0 \to M_5$), sensibilidade de gating ($50\%$ a $90\%$) e caracterização exploratória encontram-se auditados nas tabelas CSV e figuras vetoriais SVG/PNG anexas no diretório de evidências.
