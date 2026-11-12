# Apostila TCC: Machine Learning e Detecção de Anomalias em Redes Inteligentes (AMI)
## Módulo 3 — Modelos Não-Supervisionados de Anomalia, Regras Regulatórias (PRODIST) e Arquitetura em Camadas

---

### 1. Taxonomia de Anomalias em Smart Grids

1. **Anomalia Pontual / Global**: Uma única leitura horária fisicamente impossível ou com magnitude extrema (ex.: corrente de 1000 A em ramal monofásico residencial).
2. **Anomalia Contextual**: Um valor aparentemente plausível isoladamente, mas anômalo dado o contexto (ex.: consumo diurno de 15 kWh em uma escola em pleno domingo ou durante férias escolares).
3. **Anomalia Coletiva / Estrutural**: Uma sequência de observações que, em conjunto, desviam da série temporal de base (ex.: degrau repentino de 60% com queda de consumo pós-intervenção ou manipulação de TC).

---

### 2. Algoritmos de Detecção Não-Supervisionada de Outliers

#### 2.1 Isolation Forest (iForest)
- **Princípio**: As anomalias são "poucas e diferentes", logo são mais fáceis de isolar em partições aleatórias do espaço ortogonal de atributos.
- **Estrutura**: Floresta de árvores binárias aleatórias ($iTree$). Em cada nó, seleciona-se aleatoriamente um atributo e um valor de corte uniforme entre o mínimo e máximo.
- **Escore de Anomalia $s(x, n)$**:
  $$s(x, n) = 2^{-\frac{\mathbb{E}(h(x))}{c(n)}}$$
  onde $h(x)$ é o comprimento do caminho percorrido até isolar a amostra $x$, e $c(n)$ é o comprimento médio de uma árvore de busca binária não-sucedida:
  $$c(n) = 2 \ln(n - 1) + 0.5772156649 - \frac{2(n - 1)}{n}$$
  - $s \to 1$: Alta probabilidade de anomalia (isolado rapidamente nos primeiros níveis).
  - $s < 0.5$: Comportamento perfeitamente normal e denso.

#### 2.2 Local Outlier Factor (LOF)
- Mede o desvio de densidade local de uma amostra em relação aos seus $k$-vizinhos mais próximos ($k$-NN).
- **Densidade de Atingibilidade Local (LRD)**:
  $$\text{lrd}_k(p) = \frac{|N_k(p)|}{\sum_{o \in N_k(p)} \text{reach-dist}_k(p, o)}$$
- **LOF Score**:
  $$\text{LOF}_k(p) = \frac{\sum_{o \in N_k(p)} \frac{\text{lrd}_k(o)}{\text{lrd}_k(p)}}{|N_k(p)|}$$
  - Se $\text{LOF} \approx 1$: A amostra tem densidade similar à de seus vizinhos.
  - Se $\text{LOF} \gg 1$: A amostra está em uma região de densidade substancialmente inferior aos seus vizinhos (outlier local).

#### 2.3 Erro de Reconstrução por PCA (Residual Analysis)
Projeta-se a curva de 24 horas normalizada no subespaço latente gerado pelos $k$ primeiros autovetores ($V_k$) e calcula-se a norma do resíduo não-capturado:
$$\hat{x} = V_k V_k^T (x - \mu) + \mu$$
$$r_i = \|x_i - \hat{x}_i\|_2^2 = \sum_{h=1}^{24} (x_{i,h} - \hat{x}_{i,h})^2$$
Curvas com padrões aberrantes (inversão horária, picos espúrios de alta frequência) apresentam resíduos de reconstrução ordens de magnitude superiores.

---

### 3. Regras Físicas e Regulatórias (PRODIST Módulo 8 — Qualidade da Energia)

O modelo analítico deve cruzar os escores puramente estatísticos com os limites estipulados pela ANEEL (**PRODIST Módulo 8**):

#### 3.1 Níveis de Tensão em Regime Permanente (em p.u. - por unidade)
Para tensão nominal $V_{\text{nom}} = 127 \text{ V}$ (monofásica) ou $220 \text{ V}$ (bifásica/trifásica):
$$V_{pu} = \frac{V_{\text{med}}}{V_{\text{nom}}}$$
- **Faixa Adequada**: $0.92 \le V_{pu} \le 1.05$ ($117 \text{ V} \le V \le 133 \text{ V}$).
- **Faixa Precária**: $0.87 \le V_{pu} < 0.92$ ou $1.05 < V_{pu} \le 1.08$.
- **Faixa Crítica**: $V_{pu} < 0.87$ ou $V_{pu} > 1.08$.

#### 3.2 Fator de Desequilíbrio de Tensão (PVUR - Phase Voltage Unbalance Ratio)
Para ramais polifásicos:
$$\text{PVUR} = \frac{\max(|V_A - V_{\text{avg}}|, |V_B - V_{\text{avg}}|, |V_C - V_{\text{avg}}|)}{V_{\text{avg}}} \times 100\%$$
- Limite regulatório PRODIST: $\text{PVUR} \le 3.0\%$.

#### 3.3 Regras de Consistência Causal Física
- **Reversão de Energia Ativa sem GD**: $\sum \text{RA} > 0.5 \text{ kWh} \land \text{tipo\_ligacao} \ne \text{'GD'}$. Sinaliza instalação solar clandestina ou medidor invertido.
- **Desequilíbrio Severo de Corrente**: Corrente nula em uma das fases com consumo ativo contínuo na outra fase.

---

### 4. Arquitetura em Camadas e Consenso de Vizinhança Espacial

Para erradicar falsos positivos gerados por instabilidades da rede de telecomunicações ou distúrbios sistêmicos da subestação, propõe-se uma esteira de 6 camadas:

```
[ Camada 0: Telemetria Bruta AMI ]
              │
              ▼
[ Camada 1: Score de Telecom / Estabilidade Mesh (S_comm) ]
   └─ Se C < 0.70 ou G_max > 48h ──► [ FLAG: QUARENTENA_TELECOM ] (Não penalizar UC)
              │ (C >= 0.70)
              ▼
[ Camada 2: Detecção de Mudança de Regime (CPD / CUSUM) ]
   └─ Identifica data de inflexão t* e magnitude do degrau Delta
              │
              ▼
[ Camada 3: Motor de Anomalia Elétrica (S_elec) ]
   ├─ Isolation Forest (Densidade Global de Features)
   ├─ LOF (Densidade Local por Grupo de Consumo)
   └─ PCA Residual Error (Distorção do Perfil 24h)
              │
              ▼
[ Camada 4: Regras Físicas & Regulatórias (PRODIST Módulo 8) ]
   ├─ Reversão Ativa (Possível GD Clandestina)
   ├─ Tensão Crítica (< 0.87 pu ou > 1.08 pu)
   └─ Desbalanço de Fases
              │
              ▼
[ Camada 5: Consenso de Vizinhança Espacial e Topológica ]
   └─ Compara evento com vizinhos no mesmo Transformador / Alimentador:
      R_neighbor = (# vizinhos anômalos no tempo t) / (# vizinhos totais)
      - Se R_neighbor > 0.50: Distúrbio Sistêmico de Rede / Queda de Fase a Montante.
      - Se R_neighbor < 0.10: Anomalia Local Restrita à UC (Alta Suspeita de Fraude/Defeito).
              │
              ▼
[ Camada 6: Matriz Diagnóstica Causal (Contrato de Saída do Cockpit) ]
   ├─ NORMAL / REGULAR
   ├─ ANOMALIA_CONFIRMADA (Fraude / Perda Não-Técnica Provável)
   ├─ POSSIVEL_GD_CLANDESTINA
   ├─ QUARENTENA_TELECOM (Dado Insuficiente / Falha de Rádio)
   ├─ AVARIA_MEDIDOR / FALHA_MEDICAO
   └─ DISTURBIO_SISTEMICO_REDE (Afundamento Coletivo)
```

