# Relatório consolidado de funcionalidades e capítulos do TCC

**Projeto:** `tcc_pipeline_simple`  
**Data da análise:** 20/11/2026  
**Documento-base:** `docs/TCC_ARTIGO_FINAL_REVISADO_2026-11-18.md`  
**Escopo:** código-fonte, scripts, consultas SQL, testes, documentação técnica, figuras e texto acadêmico.

## 1. Resumo executivo

Projeto apresenta duas entregas principais já materializadas:

1. **Plataforma de engenharia de dados AMI**, cobrindo extração, normalização, persistência, auditoria, construção de datasets e execução operacional.
2. **Metodologia multivisão de detecção de anomalias**, combinando comunicação, comportamento, grandezas elétricas, eventos físicos, contexto temporal e topológico.

Texto-base mais recente possui aproximadamente **2.868 linhas, 21.819 palavras e 161 mil caracteres**. Estrutura cobre problema, objetivos, fundamentação, arquitetura, pipeline, EDA, modelagem, demonstração experimental, estudo de caso, interfaces, operacionalização, avaliação econômica, governança, resultados, limitações e conclusão.

Situação global:

| Dimensão | Estado | Diagnóstico |
|---|---:|---|
| Estrutura acadêmica | Avançada | Todos os blocos centrais existem |
| Fundamentação metodológica | Avançada | Argumento coerente e limites bem delimitados |
| Engenharia de dados | Implementada | Módulos, scripts, SQLs e testes disponíveis |
| Modelagem multivisão | Implementada parcialmente | Núcleo presente; validação populacional incompleta |
| Demonstração experimental | Pronta com ressalvas | Figuras e métricas presentes; regressões recentes no demonstrador |
| Estudos de caso | Prontos como evidência analítica | Não equivalem à confirmação de fraude |
| TUI | Implementada | Fluxos operacionais documentados |
| Cockpit Web | Protótipo funcional | Papel complementar; não deve ser apresentado como produto plenamente validado |
| Validação automatizada | Boa, porém não verde | 219/229 testes aprovados; 10 falhas |
| Validação de campo | Pendente | Protocolo definido; resultados externos ainda ausentes |
| Avaliação econômica | Modelo pronto | Parâmetros reais e resultados ex post pendentes |
| Fechamento editorial | Pendente | Figuras ausentes, bibliografia e formatação final |

**Conclusão:** TCC está tecnicamente maduro para consolidação final. Principal risco não está na ausência de conteúdo, mas na diferença entre **metodologia implementada**, **evidência computacional demonstrada** e **eficácia operacional ainda não comprovada em campo**.

## 2. Base documental adotada

### 2.1 Documento estrutural principal

`docs/TCC_ARTIGO_FINAL_REVISADO_2026-11-18.md`

Escolhido por ser versão completa mais recente encontrada. Incorpora:

- correções do orientador;
- arquitetura multivisão;
- engenharia do pipeline;
- EDA;
- demonstração de ML;
- sensibilidade do gating;
- estudo de caso;
- TUI e cockpit;
- funil operacional;
- avaliação econômica;
- resultados, limitações e conclusão.

### 2.2 Documentos complementares processados

- `README.md` — execução e arquitetura operacional;
- `docs/MAPA_CORRECOES_TCC_ORIENTADOR_2026-11-18.md` — rastreabilidade das correções;
- `docs/artigo_arquitetura_e_modelos_especializados.md` — detalhamento multivisão;
- `docs/artigo_estudo_de_caso_completo.md` — casos empíricos;
- `docs/artigo_cockpit_analitico_ami.md` — cockpit e investigação;
- `docs/panorama_evolucao_e_roadmap.md` — evolução e lacunas;
- `docs/article/*` — fundamentos, dificuldades, PRODIST, modelagem e operação;
- `docs/data_contracts.md`, `docs/feature_catalog.md` e documentos de datasets — governança semântica;
- código em `src/`, `scripts/` e `queries/`;
- suíte em `tests/`;
- figuras em `docs/figures/`.

## 3. Inventário técnico

| Grupo | Quantidade | Linhas aproximadas |
|---|---:|---:|
| Módulos Python em `src/` | 56 | 15.749 |
| Scripts Python | 19 | 9.039 |
| Arquivos de teste | 39 | 4.319 |
| Consultas SQL | 13 | — |
| Figuras ML presentes | 10 | — |
| Figuras EDA presentes | 3 | — |

Organização principal:

- `src/datasets/` — normalização, schemas, partições, writers, splitter e saneamento retroativo;
- `src/features/` — features elétricas, temporais, topológicas, por medidor, UC e circuito;
- `src/models/` — detectores especializados, ensemble, MoE, evidências, categorias e registry;
- `src/events/` — extração e categorização de alarmes;
- `src/eda/` — análise exploratória e séries por UC;
- `src/checks/` — integridade, cobertura, reset, nomenclatura e disponibilidade;
- `src/automation/` — execução diária e health check;
- `src/visualization/` — continuidade e diagnósticos;
- `src/tui/` — interface operacional Rich;
- `scripts/` — orquestração, treino, benchmark, demonstração e cockpit;
- `queries/` — CIS, GEO, MDM e fontes de rótulos.

## 4. Relatório funcional

### 4.1 Extração e integração de fontes

**Estado: implementada, com dependência ambiental.**

Funcionalidades identificadas:

- consulta às fontes Oracle;
- integração CIS, GEO e MDM/ORCA;
- seleção por alimentador;
- extração por lista de NIOs;
- processamento por data;
- lotes de medidores;
- paralelismo de consultas;
- sincronização por dias faltantes;
- recuperação de lacunas;
- consultas de intervalos, instantâneos e registradores;
- extração de alarmes;
- integração meteorológica;
- persistência de contexto cadastral e topológico.

Evidências:

- `src/db.py`;
- `src/pipeline.py`;
- `scripts/run_araucaria_full.py`;
- `scripts/run_feeder_pipeline.py`;
- `scripts/run_total_pipeline.py`;
- SQLs em `queries/`.

Restrições:

- execução real exige credenciais, rede e disponibilidade das bases;
- existem caminhos de pipeline sobrepostos;
- revisão antiga apontou riscos de cache e duplicação, parcialmente mitigados por evolução posterior, mas não revalidados integralmente nesta análise.

### 4.2 Normalização e arquitetura do Data Lake

**Estado: implementada.**

Funcionalidades:

- camadas semânticas de contexto, medições, features, model input, modelos e relatórios;
- formatos Parquet e CSV conforme fluxo;
- particionamento por data;
- conversão de formatos;
- normalização de identificadores;
- detecção de cadência;
- interpretação de séries serializadas;
- escrita incremental;
- separação entre UC e NIO;
- histórico de instalação de medidores;
- schema explícito;
- contratos de dados;
- manifestos de execução.

Evidências:

- `src/datasets/normalize.py`;
- `src/datasets/partitioning.py`;
- `src/datasets/writers.py`;
- `src/datasets/schemas.py`;
- `src/run_manifest.py`;
- `docs/data_contracts.md`.

### 4.3 Saneamento retroativo

**Estado: implementado, com regressão detectada.**

Funcionalidades:

- reconstrução de features diárias;
- reconciliação UC–NIO;
- normalização de NIOs de 8/10 dígitos;
- janelas mensais e quinzenais desacopladas;
- filtro de cobertura mínima;
- quarentena de telecomunicação;
- distinção entre ruído de comunicação e sinal elétrico;
- tratamento de desequilíbrio apenas sob carga;
- reconstrução de datasets históricos;
- geração de relatórios de limpeza.

Evidência principal: `src/datasets/retroactive_cleaner.py`.

Problema atual:

- teste de reconciliação esperava 3 linhas elegíveis e obteve 0;
- treinamento sequencial falha quando recebe somente dataset model input, pois saneamento exige partições de medição.

### 4.4 Qualidade, continuidade e auditoria

**Estado: implementada.**

Funcionalidades:

- calendário de disponibilidade;
- cobertura de períodos;
- integridade de partições;
- análise de tamanho de arquivos;
- rastreamento diário de UCs;
- detecção de resets de registrador;
- verificação de outputs;
- health check;
- auditoria de continuidade;
- backfill inteligente;
- nomenclatura diária padronizada.

Evidências: `src/checks/`, `src/automation/health_check.py`, `src/visualization/continuity.py`.

### 4.5 Engenharia de features

**Estado: implementada.**

Grãos suportados:

- medidor-dia;
- UC-dia;
- UC-janela;
- perfil de circuito;
- vizinhança/topologia;
- grid temporal.

Features identificadas:

- energia ativa direta e reversa;
- cobertura intervalar;
- mediana e MAD históricos;
- inclinação temporal;
- fator de carga;
- desequilíbrio de tensão;
- desequilíbrio de corrente;
- reversão de fluxo;
- idade e troca do medidor;
- alarmes por janela;
- comparação com pares;
- atributos temporais e ambientais.

Evidências: `src/features/*`, `docs/feature_catalog.md`.

### 4.6 Construção de datasets de aprendizado

**Estado: implementada.**

Funcionalidades:

- construção de matriz de treinamento;
- separação temporal;
- normalização de nomes e tipos;
- cutoff explícito;
- prevenção de target leakage;
- separação de features, metadados, identificadores e targets;
- persistência versionada;
- conversão entre formatos;
- definição de grãos.

Evidências:

- `src/datasets/build_training_dataset.py`;
- `src/datasets/dataset_splitter.py`;
- testes de cutoff, leakage e grains.

### 4.7 Detecção de anomalias multivisão

**Estado: implementada parcialmente; núcleo funcional.**

Especialistas:

| Especialista | Objetivo | Estado |
|---|---|---:|
| Comunicação | Separar ausência/degradação de telemetria | Implementado |
| Elétrico | Detectar raridade em grandezas físicas | Implementado |
| Comportamental | Perfis, recorrência e mudanças | Implementado |
| Tamper/eventos | Incorporar alarmes físicos | Implementado |
| Topológico | Corroborar vizinhança e eventos coletivos | Implementado |
| Alimentador | Modelos locais/hierárquicos | Implementado |

Componentes:

- `CompositeAnomalyDetector`;
- Isolation Forest;
- LOF;
- ECOD;
- COPOD;
- HBOS;
- PCA;
- K-Means;
- especialistas por subespaço;
- roteador MoE;
- treino hierárquico;
- filtro de falso positivo;
- fusão de scores;
- score de consenso;
- categorias diagnósticas.

Evidências: `src/models/composite.py`, `specialized_detectors.py`, `moe_router.py`, `hierarchical_trainer.py`, `spurious_filter.py`.

Limite científico:

- detectores produzem raridade e compatibilidade diagnóstica;
- não produzem confirmação causal de fraude;
- superioridade operacional depende de ablação real e inspeção independente.

### 4.8 Classificação de distúrbios de tensão

**Estado: implementada, com duas regressões de teste.**

Funcionalidades:

- normalização por unidade;
- baseline determinístico inspirado no PRODIST;
- classificador supervisionado;
- comparação baseline × ML;
- análise de divergências;
- diagnóstico de fonte de ruído;
- confiança e limiar de abstenção;
- relatórios de discrepância.

Problemas atuais:

- linha totalmente nula é removida, contrariando teste de flag `IS_NULL_L1`;
- caso limítrofe de desbalanceamento retorna `UNCLASSIFIED_DIVERGENCE`.

### 4.9 Mineração de evidências e categorias

**Estado: implementada, com incompatibilidades recentes.**

Funcionalidades:

- agrupamento dos anômalos;
- hipóteses causais;
- categorias de GD, bypass, comunicação e mudança de classe;
- transições comportamentais;
- relatórios por categoria;
- candidatos a vistoria;
- persistência temporal das ocorrências.

Problemas atuais:

- teste esperava 8 relatórios; implementação gera 9;
- nome de cluster de injeção reversa diverge do contrato do teste.

### 4.10 Automação e execução diária

**Estado: implementada.**

Funcionalidades:

- runner diário;
- sincronização incremental;
- checks antes/depois da execução;
- manifesto;
- campanhas por alimentador;
- pipeline total;
- agregação de alarmes;
- enriquecimento de candidatos;
- treino sequencial;
- scripts CMD.

### 4.11 TUI

**Estado: implementada e documentada.**

Áreas funcionais:

- extração;
- sincronização;
- saneamento offline;
- geração de datasets;
- estatísticas do Data Lake;
- calendário;
- auditoria;
- treinamento;
- comparação de modelos;
- registry;
- configurações.

Evidências: `src/tui/tui_app.py`, `docs/TUI_INSTRUCTIONS.md`, `docs/article/tui_functionalities.md`.

### 4.12 Cockpit Web

**Estado: protótipo funcional.**

Funcionalidades documentadas:

- KPI cards;
- tabela de candidatos;
- filtros dinâmicos;
- seleção de colunas;
- presets investigativos;
- painel lateral por UC;
- evidências e histórico;
- visão de carteira;
- execução por `scripts/run_web_cockpit.py`.

Limites:

- não há evidência de validação completa de UX ou implantação corporativa;
- interface é complementar à TUI;
- falha no teste do treinamento sequencial impede afirmar fluxo integrado totalmente estável.

### 4.13 Model Registry e rastreabilidade

**Estado: implementada para escopo acadêmico.**

Funcionalidades:

- registro de modelos;
- persistência de artefatos;
- metadados de treinamento;
- comparação de versões;
- semente aleatória;
- audit trail;
- versionamento de datasets e cutoffs.

Evidências: `src/models/registry.py`, `src/run_manifest.py`.

### 4.14 Avaliação operacional e econômica

**Estado: formalizada no texto; validação real pendente.**

Elementos prontos:

- funil operacional;
- ciclo de vida de casos;
- abstenção;
- backlog;
- capacidade de inspeção;
- ranking em duas filas;
- coortes O/A/N;
- política piloto de até 40%;
- rendimento@K;
- valor esperado;
- custo de oportunidade;
- cenários e sensibilidade;
- protocolo de piloto de campo.

Elementos pendentes:

- positivos confirmados em campo;
- custos reais consolidados;
- taxa de confirmação;
- valor recuperado;
- comparação ex post;
- teste estatístico da meta de +5 pontos percentuais.

## 5. Situação dos capítulos

Critérios:

- **Pronto:** conteúdo suficiente e coerente; resta revisão editorial.
- **Pronto com ressalvas:** conteúdo sólido, mas depende de correção técnica, figura ou delimitação.
- **Parcial:** estrutura existe, porém faltam resultados ou evidências essenciais.
- **Prospectivo:** apresenta protocolo/modelo, não resultado comprovado.

| Capítulo/bloco | Estado | Evidência atual | Ação restante |
|---|---:|---|---|
| Introdução | Pronto | Contexto, motivação, contribuições e delimitação | Revisão linguística |
| Problema e objetivos | Pronto | Problema, objetivo geral, 8 objetivos, rastreabilidade e hipótese | Conferir aderência final com conclusão |
| Fundamentação metodológica | Pronto | Estatística robusta, anomalias, CPD, perfis e topologia | Consolidar referências |
| Evolução metodológica | Pronto com ressalvas | Baseline M0 e arquitetura M1–M5 | Completar ablação real quando disponível |
| Arquitetura de dados | Pronto | UC/NIO, camadas, temporalidade, Polars e continuidade | Inserir figura ausente |
| Engenharia do pipeline | Pronto com ressalvas | Fontes, lotes, persistência, normalização, manifestos | Atualizar após correção das regressões |
| EDA | Pronto com ressalvas | Integridade, UC–NIO, topologia, grandezas e comportamento | Acrescentar tabelas quantitativas populacionais |
| Arquitetura multivisão | Pronto com ressalvas | Especialistas, fusão, gating e MoE | Inserir figura comportamental ausente |
| Demonstração experimental de ML | Pronto com ressalvas | 10 figuras, métodos e runtime | Corrigir demonstrador e teste E2E |
| Sensibilidade do gating | Parcial | Ensaio sintético e protocolo real | Figura ausente; executar população real |
| Saídas e compatibilidade | Pronto | Contratos e interpretação das saídas | Revisão editorial |
| Estudo de caso | Pronto com ressalvas | Casos mascarados e narrativa analítica | Corrigir três referências de figuras |
| Aplicação operacional e TUI | Pronto com ressalvas | Funcionalidades mapeadas | Validar comandos contra snapshot final |
| Operacionalização da detecção | Prospectivo sólido | Funil, filas, backlog e drift | Dados reais de operação |
| Avaliação econômica | Prospectivo sólido | Fórmulas e cenários | Custos e retornos reais |
| Governança e interfaces | Pronto com ressalvas | Contratos, catálogo, feedback e cockpit | Fechar feedback de inspeção |
| Resultados e discussão | Parcial | Resultados de plataforma, experimento e casos | Inserir métricas reais consolidadas |
| Limitações e riscos | Pronto | Limites técnicos, estatísticos e operacionais explícitos | Revisão final |
| Conclusão | Pronto com ressalvas | Síntese coerente sem promessas indevidas | Ajustar após resultados finais |

## 6. Capítulos efetivamente prontos

Considerando conteúdo técnico e coerência argumentativa, podem ser tratados como prontos para revisão final:

1. Introdução.
2. Problema de pesquisa e objetivos.
3. Fundamentação metodológica.
4. Arquitetura de dados.
5. Engenharia do pipeline.
6. Arquitetura analítica multivisão.
7. Saídas como padrões de compatibilidade.
8. Aplicação operacional e TUI.
9. Limitações e riscos metodológicos.

Capítulos quase prontos, dependentes de correções objetivas:

1. Evolução metodológica e comparação com baseline.
2. EDA.
3. Demonstração experimental.
4. Estudo de caso.
5. Governança e cockpit.
6. Conclusão.

Capítulos não fechados como resultado científico definitivo:

1. Sensibilidade do gating na população real.
2. Operacionalização com resultados de inspeção.
3. Avaliação econômica ex post.
4. Resultados e discussão populacionais.

## 7. Validação automatizada

Comando executado:

```cmd
.venv\Scripts\python.exe -m pytest
```

Resultado:

| Métrica | Valor |
|---|---:|
| Testes aprovados | 219 |
| Testes falhos | 10 |
| Total | 229 |
| Taxa de aprovação | 95,6% |
| Warnings | 2 |
| Duração | 37,91 s |

Falhas:

| Área | Falha | Impacto |
|---|---|---|
| Category reporting | 9 relatórios gerados contra 8 esperados | Contrato desatualizado ou categoria extra não documentada |
| Demo ML | assinatura de `load_real_benchmark` divergente | Teste e script incompatíveis |
| Demo ML E2E | CLI não aceita argumentos esperados | Demonstrador não reproduzível pelo contrato atual |
| Evidence miner | nome de cluster divergente | Relatório semântico inconsistente |
| Pipeline E2E | falha no fluxo integral | Impede declarar suíte totalmente estável |
| Retroactive cleaner | elegibilidade 0 contra 3 | Risco em gate de treino |
| Sequential training | exige medições mesmo com model input pronto | Acoplamento operacional excessivo |
| Voltage classifier | linha nula removida | Perda de evidência de ausência de telemetria |
| Voltage diagnosis | limítrofe não classificado | Diagnóstico de divergência incompleto |
| Configuração | override parcial por ambiente falha | Contrato de configuração inconsistente |

Interpretação:

- cobertura de testes é material e valiosa;
- taxa alta não autoriza afirmar estabilidade total;
- falhas concentram-se em integrações e contratos recentes;
- correção deve preceder geração da versão acadêmica definitiva.

## 8. Figuras e artefatos

### 8.1 Presentes

- 10 figuras do demonstrador de ML;
- 3 figuras EDA de estudos de caso;
- `method_summary.csv`;
- `metrics.csv`;
- metadados dos exemplos EDA.

### 8.2 Referenciados, porém ausentes

| Referência no artigo | Estado |
|---|---:|
| `figures/app_architecture.pdf` | Ausente |
| `figures/behavior.pdf` | Ausente |
| `figures/gating_sensitivity_synthetic.png` | Ausente |
| `figures/comparacao_duas_ucs_mesmo_consumo.png` | Caminho incorreto |
| `figures/acompanhamento_semanal_comparacao_par.png` | Caminho incorreto |
| `figures/acompanhamento_semanal_outlier.png` | Caminho incorreto |

Os três últimos arquivos existem em `docs/figures/eda/exemplos/`. Exigem correção do caminho, não regeneração.

### 8.3 Observação

Arquivo `10_case_studies.png` existe, mas não está referenciado diretamente na versão principal.

## 9. Pendências acadêmicas

### Críticas

1. Consolidar bibliografia e referências formais.
2. Corrigir seis referências de figuras.
3. Corrigir dez testes falhos.
4. Garantir consistência entre resultados narrados e outputs reproduzíveis.
5. Diferenciar explicitamente:
   - ensaio sintético;
   - dados AMI reais mascarados;
   - resultado de inspeção de campo;
   - hipótese econômica.

### Importantes

1. Gerar tabela quantitativa da população analisada.
2. Registrar período exato dos dados.
3. Registrar número de UCs, NIOs, alimentadores, dias e janelas válidas.
4. Reportar taxa de exclusão por cobertura.
5. Reportar distribuição de candidatos por especialista.
6. Executar ablação M0–M5 sobre base real estável.
7. Medir estabilidade Top-K entre cutoffs.
8. Atualizar resultados após correção do saneamento retroativo.

### Editoriais

1. Normalizar codificação UTF-8 na cadeia de geração.
2. Padronizar “não”, “visão”, “detecção” e demais acentos no output final.
3. Remover âncoras convertidas do Pandoc quando desnecessárias.
4. Padronizar títulos, numeração e referências cruzadas.
5. Revisar uso de inglês técnico.
6. Conferir siglas na primeira ocorrência.
7. Inserir lista de símbolos e abreviações.

## 10. Pendências técnicas priorizadas

### P0 — bloqueiam versão final reproduzível

1. Corrigir testes do demonstrador ML.
2. Corrigir elegibilidade do saneamento retroativo.
3. Corrigir pipeline E2E.
4. Corrigir classificador de tensão.
5. Corrigir caminhos/gerar figuras ausentes.
6. Executar suíte até `229 passed`.

### P1 — fecham resultados científicos

1. Congelar snapshot do dataset.
2. Executar pipeline final.
3. Gerar manifestos e hashes.
4. Rodar ablação real M0–M5.
5. Rodar sensibilidade do gating.
6. Consolidar KPIs populacionais.
7. Atualizar capítulo de resultados.

### P2 — fortalecem defesa

1. Preparar demonstração curta da TUI.
2. Preparar cockpit com dados mascarados.
3. Gerar fluxograma final da arquitetura.
4. Gerar tabela “objetivo → método → evidência → limite”.
5. Preparar respostas sobre ausência de ground truth.

## 11. Matriz objetivo–evidência–situação

| Objetivo | Evidência disponível | Situação |
|---|---|---:|
| Integrar fontes AMI e cadastrais | Pipeline, SQLs, contratos e Data Lake | Atendido |
| Preservar temporalidade UC–NIO | Histórico e saneamento retroativo | Atendido com regressão |
| Construir features multivisão | Módulos `src/features/` | Atendido |
| Detectar padrões anômalos | Detectores e ensemble | Atendido computacionalmente |
| Localizar e caracterizar casos | Categorias, topologia e estudos | Atendido analiticamente |
| Comparar abordagem original e refinada | Baseline, figuras e desenho M0–M5 | Parcial |
| Disponibilizar fluxo operacional | TUI, scripts e cockpit | Atendido como protótipo |
| Avaliar impacto operacional/econômico | Protocolo e fórmulas | Pendente de campo |

## 12. Formulação segura para resultados

Afirmações sustentadas:

- plataforma integra e organiza fontes heterogêneas;
- arquitetura preserva vínculo UC–NIO e cutoff temporal;
- detectores respondem de formas distintas ao mesmo conjunto;
- gating reduz população elegível e pode excluir casos;
- estudos reais mascarados demonstram necessidade de análise temporal e contextual;
- TUI e cockpit materializam fluxo de investigação;
- metodologia gera prioridades auditáveis.

Afirmações ainda não sustentadas:

- redução comprovada de fraude;
- aumento realizado de 5 p.p. no rendimento;
- economia efetivamente recuperada;
- superioridade estatística definitiva sobre processo atual;
- causalidade entre anomalia e perda não técnica;
- generalização para toda concessão sem validação externa.

## 13. Sequência recomendada de fechamento

1. Corrigir regressões.
2. Fechar figuras.
3. Congelar dataset.
4. Executar pipeline final.
5. Rodar suíte completa.
6. Executar ablação.
7. Executar sensibilidade.
8. Consolidar resultados.
9. Atualizar discussão.
10. Atualizar conclusão.
11. Consolidar bibliografia.
12. Gerar PDF final.

## 14. Parecer final

Projeto já ultrapassou estágio de protótipo simples. Há plataforma modular, camada de qualidade, arquitetura temporal, múltiplos especialistas, interfaces operacionais, documentação extensa e suíte robusta.

TCC pode ser considerado:

- **pronto em arquitetura e metodologia**;
- **quase pronto em demonstração computacional**;
- **parcial em resultados populacionais comparativos**;
- **prospectivo em impacto operacional e econômico**;
- **pendente em fechamento editorial e regressões técnicas**.

Prioridade correta: não ampliar escopo. Fechar reprodutibilidade, figuras, métricas reais e consistência entre código, testes e narrativa.
