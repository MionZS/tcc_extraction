# Política de Decisão e Encaminhamento Operacional

1. **Separação Qualidade vs Física:** Janelas com `INTERVAL_COVERAGE < 0.70` em mais de 15 dias ou média de janela `< 0.60` são direcionadas a categorias de telecomunicação (`GARGALO_COLETIVO_RF_MESH`, `ANOMALIA_COMUNICACAO_ISOLADA` ou `TELECOM_QUARANTINE`), nunca à fila de fraude elétrica.
2. **Isolamento de Ciclo de Vida:** Instalações com idade `<= 30d` são isoladas em `INICIACAO_PERFIL_UC_NOVA` ou `TRANSICAO_TROCA_MEDIDOR`.
3. **Candidatos à Vistoria (`IS_CANDIDATO_VISTORIA`):** Requer cumulativamente: telemetria elegível, não ser UC nova, hipótese acionável (`ANOMALIA_ELETRICA_SUSPEITA_FRAUDE_PNT`, `POSSIVEL_GD_NAO_CADASTRADA`, `FALHA_METROLOGICA_MEDIDOR_TRAVADO`, `ANOMALIA_COMUNICACAO_ISOLADA`) e `IS_ANOMALY == True` frente ao limiar congelado `0.5226`.
