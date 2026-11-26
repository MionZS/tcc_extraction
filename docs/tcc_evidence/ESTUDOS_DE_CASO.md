# Estudos de Caso Completos sob Arquitetura Multivisão Integrada (P1 — Seção 20)

- **Execução:** `20261126_172513_p1_multiview_eval` | **Figura:** `docs/tcc_evidence/p1_multiview/figures/fig_p1_17_trilhas_estudos_de_caso.pdf`


### UC `908***00` (NIO-435***63 — Circuito F-13)

- **Cobertura OOT (19/11):** `81,48%` (`25` dias confiáveis) | **Elegibilidade:** `True`
- **Grandezas elétricas (30d):** FA mediana = `5,717` kWh/dia; RA máx = `0,5069` (`12` dias); slope = `-0,4434` kWh/dia²
- **Escores individuais normalizados:** IF=`0,804`, LOF=`0,919`, ECOD=`0,884`, COPOD=`0,974`, HBOS=`1,000`, PCA=`1,000`
- **Especialistas P1:** `S_comm=0,5558` | `S_elec(25/11)=1,0000` | `MoE=1,0000` | `S_behavior=0,3908` | `CPD=True (-69.06% em 2026-10-31)`
- **Regras físicas:** `REGRA_GD_REVERSA|DESBAL_CORRENTE>45%` | **Evidência topológica:** `MOE_F=1.000|G=1.000|W_F=0.88`
- **Decisão 25/11 vs. Multivisão P1:** Categoria = `POSSIVEL_GD_NAO_CADASTRADA`; Candidata = `True`; Rank V2 (25/11) = `1` → Rank V5 (P1) = `574`
- **Trilha auditável:** `s_elec=1.000; moe=1.000; CPD_QUEDA(-69.1%@2026-10-31); FISICA(REGRA_GD_REVERSA|DESBAL_CORRENTE>45%); ALARMES_TAMPER_INDISPONIVEIS_NOV`

### UC `291***96` (NIO-423***54 — Circuito F-04)

- **Cobertura OOT (19/11):** `3,33%` (`1` dias confiáveis) | **Elegibilidade:** `False`
- **Grandezas elétricas (30d):** FA mediana = `28,001` kWh/dia; RA máx = `0,5139` (`1` dias); slope = `0,0000` kWh/dia²
- **Escores individuais normalizados:** IF=`0,563`, LOF=`0,695`, ECOD=`0,718`, COPOD=`0,558`, HBOS=`0,611`, PCA=`1,000`
- **Especialistas P1:** `S_comm=1,0000` | `S_elec(25/11)=0,8236` | `MoE=0,9721` | `S_behavior=0,1095` | `CPD=None (None% em None)`
- **Regras físicas:** `DESBAL_CORRENTE>45%` | **Evidência topológica:** `MOE_F=0.993|G=0.824|W_F=0.88`
- **Decisão 25/11 vs. Multivisão P1:** Categoria = `ANOMALIA_COMUNICACAO_ISOLADA`; Candidata = `False`; Rank V2 (25/11) = `None` → Rank V5 (P1) = `None`
- **Trilha auditável:** `QUARENTENA_COBERTURA(cov=0.03<0.70)`

### UC `685***20` (NIO-437***86 — Circuito F-14)

- **Cobertura OOT (19/11):** `99,49%` (`30` dias confiáveis) | **Elegibilidade:** `True`
- **Grandezas elétricas (30d):** FA mediana = `82,599` kWh/dia; RA máx = `0,0625` (`9` dias); slope = `1,3274` kWh/dia²
- **Escores individuais normalizados:** IF=`1,000`, LOF=`0,175`, ECOD=`1,000`, COPOD=`1,000`, HBOS=`1,000`, PCA=`1,000`
- **Especialistas P1:** `S_comm=0,0443` | `S_elec(25/11)=0,9850` | `MoE=0,9981` | `S_behavior=1,0000` | `CPD=False (0.0% em None)`
- **Regras físicas:** `DESBAL_CORRENTE>45%` | **Evidência topológica:** `MOE_F=1.000|G=0.985|W_F=0.88`
- **Decisão 25/11 vs. Multivisão P1:** Categoria = `ANOMALIA_ELETRICA_SUSPEITA_FRAUDE_PNT`; Candidata = `True`; Rank V2 (25/11) = `304` → Rank V5 (P1) = `376`
- **Trilha auditável:** `s_elec=0.985; moe=0.998; BEH_ALTO(1.00); FISICA(DESBAL_CORRENTE>45%); ALARMES_TAMPER_INDISPONIVEIS_NOV`


## Impacto da Arquitetura Integrada sobre as Interpretações Anteriores

1. **UC `908***00`:** A integração automática do CPD confirma em lote o degrau de `-69,06%` exatamente em `2026-10-31`, alinhado à injeção reversa (`RA` em 12 dias, `3,89 a 31,27 kWh/dia`), mantendo-a no topo prioritário com escore `1,0000` tanto no global quanto no MoE. Hipótese alternativa: instalação de microgeração solar sem conclusão cadastral no CIS.
2. **UC `291***96`:** Preservada estritamente em quarentena técnica (`ANOMALIA_COMUNICACAO_ISOLADA`, cobertura `3,33%`, `1` dia confiável), com `CPD = NULL` (inavaliável) e `rank = NULL`. Preserva-se a ressalva de não atribuir estabilidade ao consumo médio anterior nem inferir vínculo comercial após `DATA_REMOCAO = 2026-06-11` sem verificação cadastral/campo.
3. **UC `685***20`:** Mantida como elegível de altíssima qualidade (`99,49%` à cadência nativa de 10 min, 144 intervalos/dia; 100% na grade de 10 min vs. 50% aparente na grade fixa de 5 min), com `S_elec = 0,9850`, `MoE = 0,9981`, `S_behavior = 1,0000` e `CPD = ESTAVEL` (anomalia estrutural contínua, não degrau recente).
