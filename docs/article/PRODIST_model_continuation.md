# Relatório Técnico-Científico (Continuação): Integração do PRODIST Módulo 8 com Mudança de Regime e Calibração Regulatória

> **Documento de Continuação:** Complemento normativo e metodológico a [`docs/article/PRODIST_model.md`](file:///d:/Projects/tcc_extraction/docs/article/PRODIST_model.md)  
> **Normas Técnicas Aplicáveis:** ANEEL PRODIST Módulo 8 (Qualidade da Energia Elétrica em Regime Permanente), Módulo 5 (Sistemas de Medição) e IEEE Std 1159  
> **Status:** Especificação Regulatória Consolidada e Integrada ao Pipeline Multivisão

---

## 7. O Rigor Regulatório do PRODIST Módulo 8: Desfazendo Simplificações

A literatura acadêmica frequentemente comete o equívoco de reduzir o PRODIST Módulo 8 a um teste booleano pontual: *"se $V < 0.93\text{ pu}$ ou $V > 1.05\text{ pu}$, a rede está violada"*. Essa simplificação ignora a própria natureza estatística da norma brasileira estabelecida pela ANEEL.

### 7.1 Indicadores Oficiais de Conformidade de Tensão: DRP e DRC
No PRODIST M8, a conformidade de tensão em regime permanente não é determinada por leituras isoladas, mas por **campanhas de medição com duração de 168 horas consecutivas (7 dias)** divididas em $N = 1.008$ intervalos integrados de 10 minutos:

1. **Classificação das Leituras Individuais:**
   Cada intervalo de medição $i$ ($1 \le i \le 1.008$) é classificado em função da tensão nominal de fornecimento contratada ($V_n$):
   - **Tensão Adequada:** $V_i \in [V_{\text{min\_adeq}}, V_{\text{max\_adeq}}]$
   - **Tensão Precária:** $V_i \in [V_{\text{min\_prec}}, V_{\text{min\_adeq}}) \cup (V_{\text{max\_adeq}}, V_{\text{max\_prec}}]$
   - **Tensão Crítica:** $V_i < V_{\text{min\_prec}}$ ou $V_i > V_{\text{max\_prec}}$

   *Tabela de Referência para Conexões em Baixa Tensão Secundária (127V / 220V):*
   | Faixa de Atendimento | Tensão Fase-Neutro (127V) | Tensão Fase-Fase (220V) | Faixa em p.u. |
   | :--- | :--- | :--- | :--- |
   | **Adequada** | $117\text{V} \le V \le 133\text{V}$ | $202\text{V} \le V \le 231\text{V}$ | $0.921 \le V \le 1.047\text{ pu}$ |
   | **Precária** | $110\text{V} \le V < 117\text{V}$ ou $134\text{V} \le V \le 135\text{V}$ | $191\text{V} \le V < 202\text{V}$ ou $232\text{V} \le V \le 233\text{V}$ | $0.866 \le V < 0.921$ ou $1.047 < V \le 1.063\text{ pu}$ |
   | **Crítica** | $V < 110\text{V}$ ou $V > 135\text{V}$ | $V < 191\text{V}$ ou $V > 233\text{V}$ | $V < 0.866\text{ pu}$ ou $V > 1.063\text{ pu}$ |

2. **Cálculo dos Indicadores Semanais:**
   - **Duração Relativa da Transgressão de Tensão Precária ($DRP$):**
     $$DRP = \frac{n_{\text{prec}}}{1008} \times 100\% \quad (\text{Limite Regulatório: } DRP \le 3.0\%)$$
   - **Duração Relativa da Transgressão de Tensão Crítica ($DRC$):**
     $$DRC = \frac{n_{\text{crit}}}{1008} \times 100\% \quad (\text{Limite Regulatório: } DRC \le 0.5\%)$$

### 7.2 A Transposição do PRODIST para a Telemetria AMI de 15 Minutos
Em redes inteligentes AMI com amostragem de 15 minutos (96 pontos/dia), **não se pode aplicar a campanha formal de 10 minutos sem adaptação metodológica explícita**.
O sistema implementa uma **aproximação de triagem preliminar de regime permanente**:
- Em vez de gerar laudos regulatórios punitivos imediatos, calcula a **Taxa de Transgressão Diária e Janelada**:
  $$\text{DRP}_{\text{AMI}} = \frac{\sum_{t=1}^M \mathbf{1}[V(t) \in \text{Precária}]}{M} \times 100\%, \quad \text{DRC}_{\text{AMI}} = \frac{\sum_{t=1}^M \mathbf{1}[V(t) \in \text{Crítica}]}{M} \times 100\%$$
- Uma excursão pontual ($V < 117\text{V}$ em apenas um intervalo isolado de 15 min) é catalogada como **perturbação transitória**, sem acionar o classificador de não conformidade de regime permanente.
- Apenas quando $\text{DRP}_{\text{AMI}} > 3\%$ ou $\text{DRC}_{\text{AMI}} > 0.5\%$ na janela acumulada de 7 a 30 dias é que a UC é sinalizada para despacho de equipe de regulação de rede.

---

## 8. Desequilíbrio de Tensão e Componentes Simétricas

O PRODIST Módulo 8 estabelece que a conformidade de desequilíbrio de tensão em sistemas trifásicos é definida pelo **Fator de Desequilíbrio de Tensão ($FD\%$)**:
$$FD\% = \frac{V_2}{V_1} \times 100\%$$
onde $V_2$ é a magnitude da componente de sequência negativa e $V_1$ é a componente de sequência positiva, com limite normativo de $FD\% \le 2.0\%$ (ou $3.0\%$ em pontos específicos da rede).

### Limitação de Medição em Redes AMI e Proxy Adotado
Os medidores inteligentes de distribuição registram as magnitudes eficazes de tensão ($U_{L1}, U_{L2}, U_{L3}$), mas não transmitem os ângulos fasoriais relativos de fase ($\theta_{12}, \theta_{23}$) para o sistema central MDM devido a restrições de largura de banda da rede de comunicação. Sem os ângulos fasoriais, o cálculo analítico exato de Fortescue das componentes de sequência simétrica é impossível.

Diante dessa restrição física, o ecossistema adota a formulação padronizada pelo IEEE e NEMA: o **PVUR (*Phase Voltage Unbalance Rate*)**:
$$\text{PVUR}(t) = \frac{\max_{k \in \{L1, L2, L3\}} |V_k(t) - \bar{V}(t)|}{\bar{V}(t)} \times 100\%$$
onde $\bar{V}(t) = \frac{1}{K}\sum_{k=1}^K V_k(t)$ sobre as fases ativas da instalação.

* **Tratamento de Fases:** Em instalações monofásicas ou bifásicas atendidas por rede trifásica, o PVUR não é calculado sobre fases ausentes. Mapeia-se explicitamente o cadastro da UC (`PHASE_TYPE` e `FASE_CIRCUITO`), descartando fases inexistentes e evitando alarmes espúrios de desbalanço.
* **Classificação de Caráter Temporal:**
  - **Desequilíbrio Dinâmico/Intermitente:** Ocorre de forma concentrada em horários de pico noturno (carga desigual de chuveiros e condicionadores de ar monofásicos em ramais de baixa tensão).
  - **Desequilíbrio Crônico/Permanente:** Persiste ao longo de 24 horas contínuas, incluindo o período da madrugada (carga mínima). Indica neutro frouxo/rompido na rede secundária, derivação clandestina de alta potência ou quebra de derivação interna no transformador.

---

## 9. Acoplamento entre PRODIST Determinístico e o Mecanismo de Mudança de Regime (CPD)

A interface entre a conformidade de tensão PRODIST e a detecção de mudança de regime em potência ativa opera em regime de mútua validação:

```
                            EVENTO DE QUEBRA NO CPD (t*)
                                          │
                  ┌───────────────────────┴───────────────────────┐
                  ▼                                               ▼
     [Tensão Normal e Equilibrada]               [Tensão com Salto ou Desbalanço]
     • Faixa Adequada em todas as fases          • Salto coincidente de tensão em t*
     • PVUR < 2.0%                               • Ou PVUR explodindo em t*
                  │                                               │
                  ▼                                               ▼
     [Suspeita Comercial / Carga]                    [Perturbação de Circuito Físico]
     • Consumo caiu sem alteração de V           • Corroborar com UCs do mesmo trafo:
     • Provável alteração de consumo,             - Se geral: manobra / LTC / regulação
       by-pass direto ou desocupação              - Se individual: neutro solto / defeito
```

1. **Validação Cruzada de Causa:**
   Se uma UC sofre uma queda severa no consumo registrado ($FA \downarrow 60\%$) exatamente no instante $t^*$, mas os registros de tensão ($U_{L1..L3}$) mantêm-se estritamente na faixa adequada e o desbalanço é inferior a $1\%$, descarta-se a hipótese de avaria elétrica de ramal. O sinal aponta com alta probabilidade para intervenção no circuito de corrente do medidor (desvio/jumper) ou redução de atividade do imóvel.
2. **Identificação de Defeito de Medição:**
   Se a queda de consumo coincide com o zeramento de um canal de tensão ($U_{L1} \to 0\text{V}$) em uma instalação trifásica enquanto as demais fases e as UCs vizinhas mantêm tensão nominal, identifica-se de forma determinística a queima do elemento sensor de potencial ou a abertura do fusível de medição do medidor inteligente.

Essa arquitetura integrada garante que cada modelo atue dentro do seu domínio de competência: as regras físicas do PRODIST delimitam a conformidade do produto elétrico entregue, enquanto o CPD e os modelos estatísticos identificam anomalias na utilização desse produto.
