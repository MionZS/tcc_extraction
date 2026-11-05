# Panorama Geral do Projeto: Estado Atual, Evolução e Roadmap de Detecção Especializada

**Projeto:** Sistema de Extração, Normalização e Inteligência Analítica em Smart Grid (Copel Distribuição — Araucária/PR)  
**Natureza do Documento:** Documento Operacional e Estratégico Adjacente (Visão de Engenharia & Roadmap)  
**Data:** 05 de Novembro de 2026  

---

## 1. Onde Estamos Hoje? (O "Pé Atual" do Projeto)

Concluímos com sucesso a fase de **infraestrutura de dados, saneamento e integridade matemática**. O sistema deixou de ser um script de extração experimental e tornou-se uma plataforma robusta de engenharia de dados e aprendizado de máquina em escala real para o município de Araucária (`63.356` medidores inteligentes):

```text
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                                 STATUS ATUAL DO SISTEMA                                │
├────────────────────────────────────────────────────────────────────────────────────────┤
│ 1. DATA LAKE LOCAL INTEGRADO                                                           │
│    • CIS (Comercial/Cadastral) + GEO (Topologia Elétrica de Alimentadores)             │
│    • MDM/ORCA (Telemetria 5-min: intervalos, instantâneos de tensão/corrente, registros)│
│    • Alarmes AMI de medidores refinados                                                │
│                                                                                        │
│ 2. SANEAMENTO RETROATIVO & CURADORIA (src/datasets/retroactive_cleaner.py)             │
│    • Reconciliação canônica NIO (8/10 dígitos) ↔ UC ↔ Alimentador: 100% resolvida      │
│    • Filtragem de ruído físico: desequilíbrio de corrente sob carga (I >= 1.0A)        │
│    • Isolamento da malha RF Mesh: dias com pacote dropado (cov < 0.70) expurgados      │
│    • Janelas desacopladas (0% sobreposição: mensais 30d / quinzenais 15d)              │
│    • Portão de Integridade: fit limpo em X_clean vs Quarentena Telecom                 │
│                                                                                        │
│ 3. SUÍTE INTERATIVA TUI (uv run main.py)                                               │
│    • Menu de extração com auto-backfill e saneamento offline                           │
│    • Pré-voo automático contra datasets legados/degenerados                            │
│    • Treinamento e governança (Model Registry com audit trail imutável)                │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Como Evoluímos até Aqui? (Da Teoria ao Diagnóstico Empírico)

Para entender por que as decisões recentes foram tomadas, vale recapitular a evolução cronológica:

1. **Fase 1 — Extração Bruta e Descoberta:**  
   Descobrimos que a extração de dados brutos continha armadilhas silenciosas entre sistemas legados (a supressão de zeros à esquerda do serial `NIO` no MDM que impedia a correspondência com a `UC` no CIS/GEO).
2. **Fase 2 — O Diagnóstico dos Gráficos com 50% e 100% de Anomalias:**  
   Ao rodar o treinamento sobre todos os alimentadores, alguns circuitos apresentavam 100% de anomalia. Auditamos e provamos:
   - Apenas 6 medidores casavam por colisão numérica acidental com o campo `UC`;
   - O dataset continha apenas 1 único dia de medição projetado em uma janela de 30 dias, resultando em `MAD = 0` e `Slope = 0` para 100% das amostras. Os modelos estavam particionando ruído de ponto flutuante;
   - Quedas de pacotes da rede mesh simulavam "queda brusca de consumo" (fraude aparente).
3. **Fase 3 — A Mudança de Paradigma (Saneamento Retroativo e Desacoplamento):**  
   Em vez de tentar forçar modelos supervisionados (que não possuem dados anotados sincrônicos no início da irregularidade) ou janelas móveis diárias (que violam a independência estatística e causam auto-agrupamento no LOF), criamos o módulo de **Curadoria Retroativa**, que limpa as distorções físicas e temporais preservando a integridade dos dados brutos do Data Lake.

---

## 3. O Que Ainda Falta Fazer? (Objetivos Práticos e de Negócio)

Com a base matematicamente saneada, agora podemos avançar para a **especialização dos diagnósticos**. Uma anomalia em rede elétrica não é genérica: ela tem causa física e comercial específica.

Abaixo está o mapeamento dos objetivos analíticos que devemos atender:

### A. Detecção de Fraude / Desvio (Perdas Não-Técnicas — PNT)
* **Assinatura Típica:** Queda abrupta ou degrau descendente no consumo ativo ($FA$), aumento de consumo fantasma ou reativo sem correspondência, assimetria severa de corrente entre fases apenas durante o regime de carga ($I \ge 1,0\text{ A}$), presença de alarmes de abertura de tampa, tamper magnético ou inversão de fase.
* **O que falta:** Cruzar as janelas consolidadas com a densidade de alarmes operacionais e calcular o escore de desvio $S_{\text{tamper}}$ e $S_{\text{elec}}$ no Pipeline Multi-Visão.

### B. Micro e Minigeração Distribuída (GD) Não Cadastrada
* **Assinatura Típica:**
  - Fluxo reverso significativo de energia ativa ($RA$) durante o dia (especialmente entre 10h e 16h em dias ensolarados);
  - Fator de carga diurno atípico ou "curva da banheira" (duck curve) na medição de 5 minutos, onde o consumo líquido cai bruscamente ao meio-dia e sobe no início da noite;
  - Tensão de fase elevada no horário de pico solar em comparação com os vizinhos do mesmo transformador (injeção de potência ativa na rede secundária).
* **O que temos no CIS:** Já temos os campos `GERACAO_PROPRIA`, `MODALIDADE_GERACAO`, `TIPO_GD`, `DATA_INICIO_GD` e `BENEFICIARIA_GD`.
* **Como detectar a não cadastrada:** Identificar UCs com $RA > 0$ consistente ou curva diurna invertida cujo status no cadastro CIS seja `GERACAO_PROPRIA = 'N'` e `TIPO_GD IS NULL`. Trata-se de uma anomalia com fortíssima evidência física e cadastral.

### C. Indícios de Falha Metrológica no Medidor (Quebra / Defeito)
* **Assinatura Típica:**
  - Leitura de tensão normal, mas corrente identicamente zero ou travada;
  - Desconexão entre os pulsos intervalares e o registrador acumulado ($FA_{\text{interval}} \ll \Delta \text{Registrador}$);
  - Alarmes recorrentes de erro de memória (NVRAM, watchdog, relógio interno dessincronizado);
  - Desbalanceamento crônico de medição que não se reflete nas UCs vizinhas do mesmo ramal.
* **O que falta:** Regras de consistência intervalar vs registrador na camada diária para isolar falha metrológica de corte administrativo.

### D. Mudança de Perfil de Carga e Utilização de Novos Equipamentos
* **Assinatura Típica:**
  - Aumento expressivo da potência de pico noturna (ex.: carregamento de veículo elétrico — EV — residencial, gerando saltos de 7 kW a 11 kW contínuos por 4–8 horas);
  - Introdução de climatização pesada (ar-condicionado) visível por correlação térmica;
  - Transição de perfil unimodal (comercial diurno) para bimodal ou residencial.
* **O que falta:** Extrair métricas de forma da curva diária (razão de pico/base, horário do pico máximo, fator de carga horário).

### E. Troca de Titularidade como Validador / Ground Truth de Mudança de Perfil
* **A Ideia:** A troca de titularidade da UC (mudança de cliente/inquilino/proprietário) é um evento registrado no sistema comercial que frequentemente acarreta mudança abrupta e legítima de perfil (número de moradores diferente, hábitos diferentes, período de reforma ou desocupação temporária).
* **Como Usar:**
  - Extrair o histórico de movimentação cadastral de clientes (`num_cli_uee` e datas de alteração contratual no CIS);
  - **Validação de Modelos:** Uma UC que sofreu troca de titularidade no meio de uma janela ou entre duas janelas consecutivas deve ser detectada pelos algoritmos de mudança de regime estrutural (ex.: *Change-Point Detection* / *CUSUM*);
  - **Filtro de Falso Positivo:** Se a detecção acusar "anomalia comportamental", mas houver uma troca de titularidade recente, o sistema classifica como **Mudança Legítima de Titularidade (Perfil Alterado)** em vez de fraude/suspeita, reduzindo drasticamente custos de fiscalização desnecessária.

---

## 4. Onde Encaixar Cada Análise no Sistema?

A tabela abaixo organiza como essas vertentes se articulam com a arquitetura que construímos:

| Vertente de Análise | Indicadores Eletrotécnicos e Cadastrais | Camada de Software | Tipo de Modelo / Regra |
| :--- | :--- | :--- | :--- |
| **Fraude / Desvio** | Queda abrupta de $FA$, desequilíbrio de corrente sob carga, alarmes de tamper | `uc_window_features` + `spurious_filter` | Comitê Composto + MoE + Filtro Espúrio |
| **GD Não Cadastrada** | $RA > 0$ diurno, pico de tensão solar, `GERACAO_PROPRIA == 'N'` | `meter_day_features` + `RetroactiveDataCleaner` | Regra Determinística + ECOD univariado sobre $RA$ |
| **Falha de Medidor** | Divergência Intervalar $\times$ Registrador, alarmes de hardware | `meter_day_features` + `alarm_extractor` | Regra de Auditoria Metrológica |
| **Equipamentos (EV/AC)** | Pico noturno persistente, fator de forma da curva de 5 min | `uc_day_features` | Decomposição Espectral / Agrupamento de Perfil |
| **Troca de Titularidade** | Histórico de `CLIENTE` no CIS, descontinuidade de média e MAD | `uc_context` + `DatasetSplitter` | *Change-Point Detection* / Validador de Benchmarking |

---

## 5. Próximos Passos Sugeridos

Para materializar essas vertentes na prática:

1. **Ativar Indicadores de GD e Reversão na Matriz de Treinamento:**  
   Garantir que as métricas `HIST_RA_REVERSAL_DAYS`, `HIST_RA_REVERSAL_RATIO_MAX` e a flag cadastral `GERACAO_PROPRIA` componham uma visão específica de **Detecção de GD Irregular** no Menu da TUI.
2. **Incorporar o Histórico de Troca de Titularidade:**  
   Consultar a tabela de vínculos contratuais do CIS (`cad_uc_ee` / histórico de clientes) para gerar o atributo `DATA_ULTIMA_TROCA_TITULARIDADE` e `DIAS_DESDE_TROCA_TITULARIDADE`.
3. **Executar a Curadoria Offline (`[1] -> [5]`) e Avaliar os Especialistas:**  
   Rodar a reconstrução multi-janela mensal sobre o Data Lake local atual para gerar o primeiro conjunto verdadeiramente estável de scores e comparar os alimentadores sem as distorções anteriores.

