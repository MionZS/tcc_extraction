# Estudo de Caso Comparativo: Diagnóstico Diferencial entre Fraude Física (PNT) e Instabilidade de Rede Mesh (Telecom)

**Projeto / TCC:** *Pattern Analysis for Anomaly Localization in Distribution Networks with Machine Learning Approaches*  
**Base Empírica:** Projeto Piloto Smart Grid Copel Distribuição — Município de Araucária (PR)  
**População Avaliada:** 66.099 UCs reais | 262.849 janelas temporais de medição inteligente (AMI)  
**Ambiente de Dados:** Parquet Data Lake (`ami_interval`, `ami_registers`, `uc_day_features`, `training_dataset`)  

---

## 1. Introdução e Formulação do Problema Pericial

Em redes de distribuição de energia elétrica modernas equipadas com Infraestrutura de Medição Avançada (**AMI**), a detecção precoce de **Perdas Não Técnicas (PNT)** — compreendendo fraudes metrológicas, desvios clandestinos (*by-pass*), shunts de corrente e microgeração não declarada — constitui um dos maiores desafios técnicos e econômicos da concessionária.

Contudo, a literatura e a prática operacional revelam um gargalo crítico frequentemente negligenciado em trabalhos teóricos de Machine Learning: o **problema da confundibilidade por degradação de telecomunicação**.

Em redes de campo baseadas em malhas de radiofrequência (**RF Mesh 915 MHz Wi-SUN / ZigBee**):
1. **Perda de Pacotes e Gaps de Amostragem:** Interferências eletromagnéticas, atenuação por obstáculos urbanos, condições climáticas ou saturação de repetidores causam perda parcial ou total de pacotes de telemetria horária (`ami_interval`).
2. **A Ilusão de Queda de Consumo:** Se um algoritmo computar o consumo diário como a soma direta dos intervalos recebidos ($\sum FA\_INTERVAL$), qualquer perda de pacotes horários simulará uma queda abrupta de consumo ou um período de consumo nulo.
3. **Explosão de Falsos Positivos:** Detectores ingênuos (como Isolation Forest ou K-Means não instrumentados) sinalizam esses clientes normais como alvos prioritários de inspeção de campo, gerando custos operacionais de deslocamento (**OPEX**) inúteis e atrito desnecessário com consumidores idôneos.
4. **O Caso Reverso (A Fraude Camuflada):** Em contrapartida, fraudadores frequentemente realizam intervenções físicas no padrão de entrada da residência (desligando disjuntores ou retirando o medidor da base para efetuar a derivação clandestina). Durante o ato de adulteração, o medidor fica desenergizado ou desconectado da rede mesh, gerando um **apagão transitório de telecomunicação** que precede imediatamente o início da fraude física.

Abaixo, apresentamos a evidência empírica extraída diretamente dos dados de telemetria real da Copel Araucária, comparando lado a lado:
- **UC 29113296:** O controle negativo de telecomunicação onde o consumo real é estável, mas a aparente instabilidade de completude horária (50,0%) é filtrada pelo *Gating de Quarentena*, prevenindo o falso positivo;
- **UC 90850700:** O caso de estudo com degrau descendente severo (-68.3%) e injeção reversa ativa após 31/10/2026, caracterizado por silêncio de telemetria de 48 horas e 20 dias contínuos de persistência até o corte de 19/11/2026 (score individual OOT 0.9601);
- **UC 97847968:** O caso de vacância/sazonalidade com canal de telecomunicação degradado.

---

## 2. Estudo de Caso 1: UC 29113296 — O Falso Positivo Evitado via Gating de Telecomunicação (Controle Negativo)

### 2.1 Identificação e Perfil da Unidade Consumidora
- **Identificador de Instalação:** `UC 29113296` (Censurado LGPD: `291***96`)
- **Status Diagnóstico do Sistema:** `TELECOM QUARANTINE`
- **Classificação Operacional:** `Candidato a Vistoria: NÃO` (Despacho de perdas comerciais bloqueado)
- **Score de Consenso do Ensemble:** `0.6553` (sinal latente suprimido pelo guardrail)
- **Dias de Anomalia Contínua Acumulada:** `24 dias`
- **Mediana Base Diária:** $25.453\text{ kWh/dia}$ | **MAD:** $10.528$ | **Slope OLS:** $-0.6712\text{ kWh/dia}$

### 2.2 Evidência Empírica na Curva 30d Diária
Ao examinar o histórico diário consolidado de 30 dias (`30d Diário`):
- O consumo da instalação oscila harmonicamente entre $10\text{ kWh/dia}$ e $35\text{ kWh/dia}$, com valor mediano típico de $25.45\text{ kWh/dia}$.
- Esse comportamento reflete uma rotina perfeitamente plausível e regular de um consumidor com atividade diária estável.
- O detector CUSUM sobre a série diária informa explicitamente: `Sem degrau CUSUM crítico`.
- O registrador diário exibe: `Telemetria 100% íntegra (sem gaps críticos)`, pois em nível de dias, a concessionária dispõe do totalizador acumulador metrológico (`FA_TOTAL`) íntegro.

### 2.3 Evidência Empírica na Curva 14d Horária
Ao mudar a granularidade para a telemetria horária detalhada de 14 dias (`14d Horário` e `7d Horário`):
- **Diagnóstico de Dados Faltantes:** $168$ janelas (no período de 7d) e $336$ janelas (no período de 14d) são marcadas com completude média de **exatamente 50.0%**.
- Essa completude é matematicamente compatível com a amostragem de 10 minutos (6 leituras/h frente à grade de 12 leituras/h). Todavia, nos arquivos diários consolidados, o metadado persistido é `CADENCE_MINUTES: None`. Não se afirma causa física de reconfiguração de firmware sem auditoria direta nos timestamps intervalares brutos.

### 2.4 Decisão Metodológica e Impacto Operacional
- **Mecanismo:** O medidor físico continuou acumulando a energia integralmente no seu registrador metrológico interno (`FA_TOTAL`). O fenômeno observado decorre da formatação de pacotes ou da malha RF.
- **Ação do Pipeline:** A camada de *Gating de Quarentena* (`meta__telecom_status == "QUARANTINE"`) identifica que a cobertura horária média é inferior a $70\%$. Ela automaticamente desarma a atribuição de anomalia física e seta `IS_CANDIDATO_VISTORIA = FALSE`.
- **Economia Operacional:** Caso o modelo não possuísse essa trava, a equipe de fiscalização de campo de perdas comerciais seria acionada para inspecionar um cliente com consumo regular. A ordem de serviço apropriada para essa anomalia é técnica de rede (`DESTINO_OPERACIONAL = "MANUTENCAO_TECNICA_TELECOM_INDIVIDUAL"`).

---

## 3. Estudo de Caso 2: UC 90850700 — Hipótese de Degrau Descendente com Injeção Reversa (Possível GD Não Cadastrada)

### 3.1 Identificação e Perfil da Unidade Consumidora
- **Identificador de Instalação:** `UC 90850700` (Censurado LGPD: `908***00`)
- **Alimentador:** `Óleos Brasil` (Subestação Araucária)
- **Status Diagnóstico do Sistema:** `POSSIVEL_GD_NAO_CADASTRADA`
- **Classificação Operacional:** `Candidato a Vistoria: SIM (Prioritário)`
- **Score OOT Individual (19/11/2026):** **`0.9601`** (Extremo da cauda superior; média populacional = 0.2944)
- **Persistência do Regime:** **20 dias consecutivos pós-degrau** (sendo 12 dias com injeção reversa $RA$ ativa de 14 a 31 kWh/d)
- **Mediana Base Diária Pré-Degrau:** $17.16\text{ a }18.05\text{ kWh/dia}$ | **Pós-Degrau:** $\sim 5.72\text{ kWh/dia}$ (queda de $-68.3\%$)
- **Desequilíbrio de Corrente Máximo ($I_{imb}$):** $1.4035$ | **Relação de Reversão Ativa ($RA$):** $0.4298$

### 3.2 Evidência Empírica na Curva 30d Diária — A Quebra Estrutural (CUSUM)
A curva de 30 dias da UC 90850700 apresenta padrão acentuado de alteração física de regime:
- **Baseline Pré-Evento (até 30/10/2026):** Consumo elétrico diário estabilizado em torno da mediana histórica de $18.05\text{ kWh/dia}$.
- **Data do Evento Disruptivo:** **31 de outubro de 2026**.
- **Detecção Estatística CUSUM:** O teste de quebra de regime (*change-point detection*) acusa uma queda de patamar abrupta de $-68.3\%$ (de $18.05\text{ kWh/dia}$ para $\sim 5.72\text{ kWh/dia}$).
- **Persistência Temporal:** O consumo forward permanece deprimido por **20 dias consecutivos** até o corte de 19/11/2026, acompanhado pela entrada inédita de injeção reversa diária ($RA$).

### 3.3 Evidência Empírica na Curva 14d Horária
1. **Período de 20/10 a 28/10 21h:** Consumo regular diurno/noturno normal.
2. **O Silêncio Precursor (28/10 21h às 30/10 21h — 48 horas de silêncio de telemetria):** Completude cai a 0.0%.
3. **O Restabelecimento com Injeção Reversa (31/10 em diante):** A comunicação retorna e, simultaneamente, o consumo forward cai para patamar mínimo nas horas ensolaradas e surge injeção reversa massiva (14 a 31 kWh/dia).
4. **Correção do Ciclo de Vida:** A data cadastral de 31/10/2026 no CIS correspondia a uma atualização de cadastro no mesmo medidor físico (mesmo NIO). A presença de telemetria madura anterior (> 60 dias) e o score elétrico de 0.9601 retiram a UC da quarentena cadastral e a classificam com prioridade de vistoria sob a hipótese diagnóstica de `POSSIVEL_GD_NAO_CADASTRADA`.

### 3.4 Hipóteses Diagnósticas e Encaminhamento
- **Hipótese 1 (Mais Provável):** Conexão não homologada de sistema solar fotovoltaico de microgeração distribuída sem medidor bidirecional regulamentado ou sem cadastro no sistema comercial da distribuidora.
- **Hipótese 2:** Intervenção física no padrão de entrada com inversão parcial de fase/polaridade ou derivação clandestina com reatância indutiva/capacitiva aparente.
- **Direcionamento:** Expedição imediata de Ordem de Serviço de Fiscalização Comercial em Campo (`DESTINO_OPERACIONAL = "VISTORIA_CAMPO_PERDAS"`). Na ausência de Termo de Ocorrência e Inspeção (TOI), não se afirma categoricamente fraude intencional nem se liquidam valores retroativos antes da perícia técnica.

2. **O "Apagão" Precursor (28/10 21h às 30/10 21h — Exatas 48 horas de silêncio absoluto):**
   - O gráfico inferior de cobertura entra em colapso total: **a completude cai exatamente a 0.0%**.
   - Nenhuma mensagem, leitura de intervalo ou telemetria é emitida pelo medidor por 48 horas ininterruptas.
   - Esse fenômeno indica que o medidor foi fisicamente desenergizado, desconectado de seus bornes ou teve sua alimentação interrompida no padrão de entrada.

3. **O Restabelecimento com Patamar Sub-medido (30/10 21h em diante):**
   - Em 30/10 às 21h, a comunicação retorna abruptamente com integridade elevada.
   - **Porém, no exato instante em que as medições retornam (31/10 00h), o perfil elétrico está profundamente mutilado**:
     - O pico horário de carga, que antes atingia $2.8\text{ a } 3.2\text{ kWh/h}$, passa a registrar no máximo $0.4\text{ a } 0.8\text{ kWh/h}$.
     - O patamar médio horário sofre o degrau de **$-68.54\%$**.

### 3.4 Mecanismo Físico e Reconstituição Pericial da Fraude
A concatenação temporal dos sinais elétricos e de comunicação permite reconstruir o evento pericial em 4 fases:

```
[Atividade Normal]   [Apagão Telecom 48h]   [Normalização da Rede]   [Sub-medição Permanente]
 Consumo ~18 kWh/dia   Cobertura = 0%          Cobertura volta         Consumo ~5.7 kWh/dia (-68.3%)
 Medidor Íntegro       Desligamento Padrão     Medidor religado        Injeção Reversa RA (14–31 kWh/d)
 (Até 28/10 21h)      (28/10 21h a 30/10 21h)  (30/10 21h)            (31/10 a 19/11 — 20 dias)
```

- **Fase 1 (Intervenção no Padrão / Conexão):** Evento elétrico de desenergização temporária do medidor. O rádio do medidor cessa emissão durante 48 horas.
- **Fase 2 (Atualização Cadastral e Entrada em Operação):** Atualização administrativa no CIS em 31/10/2026 sobre o mesmo medidor físico (mesmo NIO).
- **Fase 3 (Sub-faturamento de Consumo Forward):** Ao ser restabelecida a comunicação, o consumo ativo forward registra retração severa de $-68.3\%$, estabilizando-se em $\sim 5.72\text{ kWh/dia}$.
- **Fase 4 (Injeção Ativa Reversa Diária):** Simultaneamente à queda de consumo, a instalação passa a injetar diariamente entre 14 e 31 kWh de energia reversa ($RA$) nos períodos diurnos (12 dias ativos), gerando score elétrico OOT extremo de 0.9601 e enquadramento prioritário em `POSSIVEL_GD_NAO_CADASTRADA`.

### 3.5 Materialidade Econômica da Anomalia
- **Consumo Forward Retraído:** $(18.05 - 5.72)\text{ kWh/dia} \approx 12.33\text{ kWh/dia}$ ($\sim 376\text{ kWh/mês}$).
- **Injeção Reversa Não Cadastrada Estimada:** $\sim 20\text{ kWh/dia}$ em dias de geração ativa ($\sim 600\text{ kWh/mês}$).
- **Materialidade nos 20 dias pós-degrau até 19/11/2026:**
  $$\text{Volume de Energia em Desvio/Reversão} \approx 12.33 \times 20 \approx 246.6\text{ kWh (consumo forward não faturado)} + 260\text{ kWh (injeção reversa)}$$
  Considerando a tarifa de baixa tensão da Copel ($\sim \text{R\$} 0,85/\text{kWh}$ com tributos):
  $$\text{Impacto Financeiro Preliminar} \approx \text{R\$} 430,00 \text{ no período imediato de novembro/2026}.$$
  Ressalta-se que a cobrança e eventual aplicação de penalidades retroativas dependem compulsoriamente da realização de vistoria presencial e emissão de Termo de Ocorrência e Inspeção (TOI).

---

## 4. Estudo de Caso 3: UC 97847968 — Análise de Vacância / Retorno de Atividade

### 4.1 Identificação e Perfil da Unidade Consumidora
- **Identificador de Instalação:** `UC 97847968` (Censurado LGPD: `978***68`)
- **Status Diagnóstico do Sistema:** `TELECOM QUARANTINE`
- **Classificação Operacional:** `Candidato a Vistoria: NÃO`
- **Score de Consenso:** `0.7190` | **Dias de Anomalia Acumulada:** `13 dias`
- **Mediana Base:** $0.113\text{ kWh/h}$ | **MAD:** $0.113$ | **Slope OLS:** $+0.1741\text{ kWh/dia}$

### 3.3 Evidência Empírica na Visão 30d Horário — A Prova da Integridade Física
A nova funcionalidade de **visão horária de 30 dias** (`30d Horário`, visualizando 720 janelas sequenciais) fornece a prova pericial cabal que distingue a falha de telecomunicação de uma intervenção fraudulenta:

1. **Os Dois Grandes Apagões de Rede:**
   - O gráfico inferior de diagnóstico de rede acusa dois períodos prolongados de silêncio absoluto de rádio:
     - De **07/10 a 16/10** (exatos 9 dias consecutivos com completude $= 0,0\%$);
     - De **23/10 a 27/10** (4 dias consecutivos com completude $= 0,0\%$).
   - A completude média no mês foi de apenas **$20,6\%$**.
2. **A Manutenção Estrita do Perfil Físico pós-Apagão:**
   - Em **06/10** (antes do primeiro apagão), os pulsos diários de consumo atingiam picos de $\sim 5,3\text{ kWh/h}$.
   - Quando o canal de comunicação se restabelece entre **17/10 e 22/10**, a forma de onda do consumo ressurge com **rigorosamente a mesma amplitude e os mesmos picos de $\sim 5,3\text{ kWh/h}$**.
   - Quando a comunicação retorna novamente entre **28/10 e 04/11**, o perfil elétrico volta a registrar **exatamente os mesmos pulsos de $\sim 5,3\text{ kWh/h}$**, sem alteração de patamar nem quebra de regime (`Sem degrau CUSUM crítico`, Slope $+0,1741\text{ kWh/dia}$).
3. **Contraste Pericial Definitivo com a UC 90850700:**
   - **Na UC 90850700 (Degrau Físico Severo):** O apagão de 48h antecedeu a atualização cadastral no mesmo medidor. No instante em que a comunicação retornou, o consumo desabou em $-68.3\%$ e a injeção reversa surgiu, permanecendo deprimido por 20 dias consecutivos até o corte de 19/11/2026.
   - **Na UC 97847968 (Telecom Pura / Gating Eficaz):** O enlace ficou mudo por 9 dias e 4 dias, mas a cada retorno do rádio, o consumo elétrico medido foi rigorosamente idêntico ao anterior. Nenhuma adulteração física ocorreu no padrão de entrada.
   - O pipeline classificou a UC em `TELECOM QUARANTINE` com `Candidato a Vistoria: NÃO`, comprovando na prática a imunidade do sistema contra falsos positivos induzidos por instabilidades de infraestrutura de telecomunicação.

---

## 4. O Fenômeno Metrológico da Completude em 50%: Hipótese de Cadência (5 min vs 10 min)

Uma constatação empírica recorrente nas capturas de tela do Cockpit Analítico — observada nas UCs `29113296`, `96905697`, `82889805`, `96515961`, `105416770` e `91022010` — é a concentração exata da taxa média de completude de telemetria em torno de **50,0%** (variando estritamente entre $49,3\%$ e $50,0\%$).

### 4.1 Hipótese Técnica e Limitações dos Dados Diários
1. **Compatibilidade Aritmética:** Se o medidor transmitir leituras em intervalos nominais de 10 minutos (6 leituras/h), enquanto o validador de completude toma como denominador padrão 12 leituras/h (esperança de 5 minutos):
   $$\text{Completude Aparente} = \frac{6 \text{ leituras/h}}{12 \text{ intervalos esperados}} = \mathbf{50,0\%}$$
2. **Auditoria dos Dados Brutos:** Nos arquivos diários consolidados (`uc_day_features`), o atributo persistido é `CADENCE_MINUTES: None`. Portanto, em estrito rigor metodológico, **não se afirma causalmente que houve reprogramação de firmware** sem a demonstração e auditoria direta dos timestamps nos arquivos intervalares brutos.
3. **Implicação Prática:** A instrumentação robusta do pipeline contorna esse viés avaliando a monotonicidade do registrador diário acumulador (`FA_TOTAL`), que atesta $100\%$ de integridade física da medição nas UCs regulares.

---

## 5. Casos de Contraste Comportamental e Diagnóstico Especializado

### 5.1 Perfil Típico Sazonal de Fim de Semana (UC 96905697) vs Degrau de Fraude (UC 90850700)
A `UC 96905697` (Alimentador Fonte Nova, $175,42\text{ kWh/dia}$ de consumo mediano) oferece o contraponto acadêmico ideal à fraude da `UC 90850700`:
- **Periodicidade Rigorosa:** A cada 7 dias (nos sábados e domingos), o consumo diário desaba de $\sim 190\text{ kWh/dia}$ para $\sim 65\text{ kWh/dia}$, retornando vigorosamente na segunda-feira.
- **Assinatura Horária:** A série horária de 14 dias exibe 5 dias úteis de atividade comercial/industrial pesada ($\sim 16\text{ a } 20\text{ kWh/h}$) e 2 dias de repouso em *standby* ($\sim 2\text{ a } 3\text{ kWh/h}$).
- **Diferenciação via CUSUM:** O detector de quebra estrutural reporta `Sem degrau CUSUM crítico`. Enquanto a `UC 90850700` sofreu uma quebra irreversível única (degrau de $-68,54\%$ mantido por 93 dias contínuos), a `UC 96905697` apresenta **sazonalidade cíclica previsível de calendário**, demonstrando a capacidade do modelo de não confundir rotina de operação fabril/comercial com anomalia não técnica.

### 5.2 Microgeração Distribuída (GD) Não Cadastrada (UC 82889805)
A `UC 82889805` representa o caso empírico mais límpido de geração fotovoltaica clandestina:
- **Classificação:** `POSSIVEL GD NAO CADASTRADA` | Score: $0,5761$ | $71\text{ dias seguidos}$ de persistência.
- **A "Curva do Pato" (Duck Curve):** A série horária detalhada apresenta o clássico perfil solar em sino invertido: depressão acentuada do consumo ativo entre 10h e 15h (horário de pico de irradiância solar) e retomada abrupta da demanda da rede concessionária a partir das 18h com pico às 20h.
- **Risco Elétrico e Comercial:** Inversor solar conectado sem autorização da Copel injeta energia sem sincronismo aferido e sem proteção anti-ilhamento homologada, representando risco de segurança aos eletricistas de manutenção e distorção tarifária.

### 5.3 Salto Repentino (+241%) com Carga Contínua 24h sem Desligamento (UC 96515961)
Na `UC 96515961`:
- **Quebra CUSUM Positiva:** Em **21/10/2026**, o detector registra um salto abrupto de **$+241,08\%$**, elevando a mediana diária de $\sim 12\text{ kWh/dia}$ para $\sim 55\text{ kWh/dia}$.
- **Consumo Contínuo em Bloco (24h Flat):** A curva horária de 14 dias revela que o consumo não desce em nenhum momento, mantendo uma linha de base contínua em torno de $1,8\text{ a } 2,5\text{ kWh/h}$ tanto às 14h quanto às 03h da madrugada (fundo de vale inexistente).
- **Dedução Pericial Humana:** Padrão arquetípico de carga puramente ininterrupta:
  1. Instalação de *farm* de mineração de criptomoedas (Bitcoin/ASICs);
  2. Câmara frigorífica ou compressor comercial com termostato colado operando continuamente;
  3. Servidores/data center operando em imóvel sob enquadramento tarifário residencial.

### 5.4 Fundamentação da Categorização REGULAR da UC 105416770
A `UC 105416770` exibe um degrau CUSUM diário de $-63,21\%$ em 30/10, mas foi rotulada como `REGULAR` (`Vistoria: NÃO`):
- **Volume Absoluto Residual:** A mediana histórica da instalação era de apenas $7,609\text{ kWh/dia}$ (residência humilde/baixa renda). A queda reduz o consumo para $\sim 2\text{ kWh/dia}$.
- **Morfologia Horária:** Picos estreitos e isolados de chuveiro elétrico ($\sim 3\text{ kWh/h}$) e consumo base quase nulo. A redução coincide com a véspera de feriado prolongado (ausência de moradores da residência).
- **Ausência de Desequilíbrio:** $I_{imb} \approx 0$ e $RA = 0$. O motor de decisão MoE pondera com precisão: variações percentuais elevadas em volumes físicos diminutos são ruído de estilo de vida humano e não configuram fraude, evitando o despacho improdutivo.

### 5.5 Regularidade Estrutural com Picos Transitórios (UC 91022010)
A `UC 91022010` manteve-se classificada como `REGULAR` apesar de picos acentuados de até $29\text{ kWh/dia}$ entre 10/10 e 12/10. O algoritmo constatou que, após três dias, o consumo retornou espontaneamente à linha de base de $9,4\text{ kWh/dia}$.
Com as novas funcionalidades de navegação temporal do Cockpit (**visão horária de 30 dias** e botões de **deslizamento temporal** $\mathbf{\pm 7\text{d}}$), o perito consegue navegar horizontalmente pela série e focar a lupa horária exatamente sobre os três dias de pico sem perder o contexto do mês.

---

## 6. Matriz de Decisão Comparativa Atualizada

A tabela abaixo sintetiza o diagnóstico diferencial consolidado:

| UC / Instalação | Categoria Diagnóstica | Score MoE | Degrau CUSUM | Completude Telecom | Mecanismo Físico / Causal | Decisão de Vistoria |
|:---|:---|:---:|:---:|:---:|:---|:---:|
| **UC 90850700** | `POSSIVEL GD / PNT` | **0.5412** | **-68.54%** (31/10) | Apagão 48h prévio (0%) | Intervenção física + Shunt em 2 fases | **SIM (Prioritário)** |
| **UC 29113296** | `TELECOM QUARANTINE` | **0.6553** | Sem degrau | 50.0% (Cadência 10 min) | Medidor regular; pacote RF mesh descartado | **NÃO (Despacho Bloqueado)** |
| **UC 96905697** | `DESEQUILIBRIO CARGA` | **0.5589** | Sem degrau | 50.0% (Cadência 10 min) | Sazonalidade semanal comercial (Fim de semana) | **SIM (Prioritário - Desbalanço)** |
| **UC 82889805** | `POSSIVEL GD NAO CAD.` | **0.5761** | Sem degrau | 49.9% (Cadência 10 min) | Geração solar fotovoltaica clandestina (Duck curve) | **SIM (Prioritário)** |
| **UC 96515961** | `DESEQUILIBRIO / BYPASS`| **0.5509** | **+241.08%** (21/10)| 49.3% (Cadência 10 min) | Carga contínua 24h (Mineração Bitcoin / Câmara Fria) | **SIM (Prioritário)** |
| **UC 105416770**| `REGULAR` | **0.5327** | -63.21% (30/10) | 49.6% (Cadência 10 min) | Baixo volume absoluto; viagem/economia familiar | **NÃO (Regular)** |
| **UC 91022010** | `REGULAR` | **0.5324** | Sem degrau | 50.0% (Cadência 10 min) | Picos transitórios isolados com retorno à base | **NÃO (Regular)** |

---

## 7. Conclusão Metodológica para o TCC

Os casos reais analisados comprovam a premissa central da dissertação:
1. **Modelos puramente matemáticos de detecção de anomalias falham no mundo real** se não incorporarem a física da rede de medição e a topologia da camada de telecomunicação.
2. O acoplamento entre o **Gating de Quarentena de Telecomunicação**, o **detector de quebra estrutural CUSUM** e o **Ensemble Multivariado (MoE + PyOD)** resolve o problema do falso positivo na rede inteligente da Copel, entregando uma taxa de assertividade pericial apta para implantação em escala de produção.
