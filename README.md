# TCC Extraction Pipeline

Este repositório contém o pipeline de extração, processamento e modelagem de dados de medição inteligente (AMI) para fins de detecção de anomalias, focado no alimentador **Fonte Nova** da subestação Araucária.

## 1. Arquitetura Conceitual e de Dados

O projeto adota um princípio de separação semântica das fontes de dados:

* **AMI (Medições) $\rightarrow$ Fenômeno**: Dados de séries temporais provenientes do MDM (intervalos, instantâneos e registradores).
* **Cadastro + Medidor + GEO $\rightarrow$ Contexto**: Perfil da Unidade Consumidora (UC), histórico temporal de instalações de medidores e topologia de rede extraída do GIS/GEO.
* **Alarmes $\rightarrow$ Eventos**: Eventos operacionais vinculados ao NIO (Número de Instalação da Obra) com cálculo de latência.
* **Unidade Analítica Final para ML**: Tudo é projetado e consolidado na matriz final baseada na chave: `UC x cutoff_date`.

## 2. Estrutura do Datalake

Em vez de gerar arquivos CSV desorganizados, o pipeline agora grava tabelas particionadas no formato **Parquet**, otimizadas para leitura e processamento analítico. A estrutura no diretório `output/` é a seguinte:

```text
output/
├── context/
│   ├── uc_context/
│   ├── meter_installation_history/
│   └── electrical_hierarchy/
├── measurements/
│   ├── ami_interval/
│   ├── ami_instantaneous/
│   └── ami_registers/
├── events/
│   └── alarm_events/
├── features/
│   ├── uc_day_features/
│   └── uc_window_features/ (ex: 30 dias)
└── model_input/
    └── training_dataset.parquet (formatado com x__*, id__*, meta__*, y__*)
```

## 3. Escopo: População do Alimentador Fonte Nova

Historicamente, o CIS extraía todos os medidores de Araucária (~71k), e os testes eram feitos em amostras aleatórias (ex: 200 UCs) que muitas vezes não possuíam dados topológicos completos (pois a query do GEO tinha um filtro fixo para o Alimentador Fonte Nova).

**Decisão Atual:** Para garantir consistência e performance, o pipeline roda **100% da população de um alimentador específico (Fonte Nova)**. 
- Extrai dados de aproximadamente 1.500 a 3.000 UCs.
- Tempo de execução rápido (~2 a 5 minutos) desde a extração até o treinamento.

## 4. Como Usar o Pipeline (Automação e Treinamento)

O pipeline ponta a ponta foi automatizado. Ele cruza os dados do CIS com o GEO, extrai o MDM apenas para os medidores relevantes, normaliza as camadas no datalake e aciona o treinamento do modelo baseline de detecção de anomalias (`IsolationForest` + `KMeans`).

### Configuração

1. Crie o arquivo `config.json` a partir do `config_example.json` na raiz do projeto, preenchendo as credenciais dos bancos de dados Oracle (ORCA e CIS).
2. Garanta que as dependências Python estejam instaladas (gerenciadas pelo `uv` via `pyproject.toml` / `uv.lock`).

### Execução Simples

Para rodar o pipeline completo e treinar o modelo, basta executar o script CMD fornecido:

```cmd
run_feeder.cmd
```
*(Ou dê um clique duplo no arquivo `run_feeder.cmd` no Windows)*

Isso executará o pipeline com o padrão de extração (`--days-back 1`).

### Execução Manual / Avançada

Você também pode chamar o script Python diretamente a partir da raiz do projeto, o que permite passar parâmetros adicionais:

```bash
python scripts/run_feeder_pipeline.py --days-back 1
```

O que acontece nos bastidores:
1. Executa `queries/geo_feeder_direct.sql` para buscar as UCs reais do alimentador Fonte Nova no GEO.
2. Cruza CIS + GEO do alimentador.
3. Extrai do MDM apenas os medidores que compõem este alimentador.
4. Salva as partições Parquet e cria a matriz consolidada `training_dataset.parquet`.
5. Aciona o `scripts/train_anomaly_model.py` para treinar o modelo e gerar os scores/ranking de anomalias.

O resultado do treinamento e as previsões do modelo podem ser inspecionados no diretório `output/model_input/`.
