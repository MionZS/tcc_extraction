# Política de Decisão e Encaminhamento Operacional de Vistorias

## 1. Princípio de Separação de Filas Operacionais

O pipeline desacopla estritamente as ordens de serviço em dois destinos operacionais mutuamente exclusivos:

1. **Fila de Vistoria de Campo Comercial / Perdas Não Técnicas (`DESTINO_OPERACIONAL = "VISTORIA_CAMPO_PERDAS"`):**
   - **Flag:** `IS_CANDIDATO_VISTORIA = True`
   - **Critérios:** Telemetria confiável (`COVERAGE_RELIABLE_DAYS >= 15` e `AVG_INTERVAL_COVERAGE >= 0.60`), UC não em iniciação, score elétrico $\ge 0.5226$, e hipótese física acionável (`ANOMALIA_ELETRICA_SUSPEITA_FRAUDE_PNT`, `POSSIVEL_GD_NAO_CADASTRADA`, `FALHA_METROLOGICA_MEDIDOR_TRAVADO`).
   - Falhas de telecomunicação nunca entram nesta fila.

2. **Fila de Manutenção Técnica de Telecomunicações (`DESTINO_OPERACIONAL = "MANUTENCAO_TECNICA_TELECOM_*"`):**
   - **Flag:** `IS_CANDIDATO_MANUTENCAO_TELECOM = True`
   - **Subcategorias:**
     - `ANOMALIA_COMUNICACAO_ISOLADA`: OS individual no medidor/antena (contraste espacial com vizinhança saudável).
     - `GARGALO_COLETIVO_RF_MESH`: Chamado coletivo de infraestrutura/concentrador DCU.
     - `TELECOM_QUARANTINE`: Quarentena lógica em software.
