# Política de Decisão e Encaminhamento Operacional de Vistorias

## 1. Princípio de Separação de Filas Operacionais

O pipeline desacopla estritamente as ordens de serviço em dois destinos operacionais mutuamente exclusivos:

1. **Fila de Vistoria de Campo Comercial / Perdas Não Técnicas (`DESTINO_OPERACIONAL = "VISTORIA_CAMPO_PERDAS"`):**
   - **Flag:** `IS_CANDIDATO_VISTORIA = True`
   - **Critérios:** Telemetria confiável (`COVERAGE_RELIABLE_DAYS >= 15` e `AVG_INTERVAL_COVERAGE >= 0.70`), UC não em iniciação de ciclo de vida (`is_new_uc_initiation = False`), score elétrico $\ge 0.5226$, e hipótese física acionável (`ANOMALIA_ELETRICA_SUSPEITA_FRAUDE_PNT`, `POSSIVEL_GD_NAO_CADASTRADA`, `FALHA_METROLOGICA_MEDIDOR_TRAVADO`).
   - Falhas de telecomunicação nunca entram nesta fila.

2. **Fila de Manutenção Técnica de Telecomunicações (`DESTINO_OPERACIONAL = "MANUTENCAO_TECNICA_TELECOM_*"`):**
   - **Flag:** `IS_CANDIDATO_MANUTENCAO_TELECOM = True`
   - **Subcategorias:**
     - `ANOMALIA_COMUNICACAO_ISOLADA`: OS individual no medidor/antena (contraste espacial com vizinhança saudável).
     - `GARGALO_COLETIVO_RF_MESH`: Chamado coletivo de infraestrutura/concentrador DCU.
     - `TELECOM_QUARANTINE`: Quarentena lógica em software.

## 2. Reconciliação Contratual de Limiares e Populações

- **Elegibilidade de Cobertura: 70% (Produção / OOT) vs 60% (Série Diária):** O despacho de campo e o funil de ablação $M_2$ operam com limiar estrito de **70% de cobertura** no intervalo (`x__avg_interval_coverage >= 0.70`), retendo 62,857 UCs (95.7%) e enviando 2,829 UCs para quarentena técnica. O limiar de 60% é tolerância exploratória aplicada exclusivamente na série diária (`DAILY_RULES`) para amortecer quedas de fim de semana.
- **Diferença de 22 UCs (6.333 OOT vs 6.311 Gating 70%):** O pipeline `OOT_MULTIVIEW` emprega modelo congelado e pós-processamento por regras físicas e de negócio (`apply_delimited_isolation_and_categories`), descartando 394 falsos positivos estatísticos benignos e resgatando 416 anomalias físicas com score em cauda marginal ($416 - 394 = +22$).
- **Destino das 1.943 UCs fora do Funil Topológico ($6.333 - 4.390 = 1.943$):** São instalações com anomalia física confirmada e cobertura $\ge 70\%$, porém sem alimentador cadastrado corporativamente (`meta__feeder is null`). São interrompidas em $M_3$ (análise topológica), mas permanecem ativas na fila geral de vistorias individuais.
- **Igualdade $M_4 = M_5$ (4.390 UCs):** Na coorte de 19/11/2026, todas as UCs anômalas contextualizadas atenderam à elegibilidade de negócio de $M_5$ (sem instalações em quarentena cadastral de início de ciclo de vida).
