# Aplicação Operacional sob Capacidade Restrita K = 2.000/mês (Parte C — Seção 21)

- **Execução:** `20261126_172513_p1_multiview_eval` | **Figura:** `docs/tcc_evidence/p1_multiview/figures/fig_p1_18_composicao_carteiras_operacionais_k2000.pdf`

> **Fluxo operacional prospectivo:** `Telemetria → Qualidade → Evidências → Hipóteses → Priorização → Análise humana (Cockpit) → Eventual vistoria → Resultado futuro de campo`. A política de 40% de substituição e a hipótese de +5 p.p. de *yield* permanecem estritamente **cenários de planejamento**, nunca resultados medidos.

## 1. Separação em Filas Operacionais (OOT Completo: 65.686 UCs)

| Fila Operacional | UCs no OOT | Destino Recomendado |
| --- | ---: | --- |
| 1. Comunicação (Quarentena Telecom: Isolada + Coletiva) | 2.739 | Equipe Telecom / Automação RF Mesh (NÃO fiscalização comercial) |
| 2. Ciclo de Vida Cadastral (UC Nova / Troca Medidor) | 136 | Maturação cadastral / auditoria pós-troca (NÃO fiscalização comercial) |
| 3. Manutenção Metrológica (Medidor Travado / FA~0) | 515 | Verificação de ocupação e calibração metrológica |
| 4. Possível GD Não Cadastrada (Injeção Reversa RA) | 1.451 | Auditoria documental CIS / parecer de acesso microgeração |
| 5. Análise Técnica de Possível PNT (Anomalia Elétrica) | 4.087 | Análise humana (Cockpit) -> eventual vistoria de campo sob K=2.000/mês |
| 6. Monitoramento Comportamental / Operação Regular | 56.758 | Acompanhamento passivo (sem despacho) |

## 2. Comparação de Políticas de Priorização sob Capacidade Fixa (K = 2.000)

| Política de Priorização | Jaccard vs 25/11 | Trocas | Fila PNT | Fila GD | Fila Metrol. | FA Med. (kWh/d) | Cob. Média | % c/ CPD>=30% |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| P1: Escore Elétrico (OOT 25/11) | 1,0000 | 0 | 744 | 1.227 | 29 | 13,64 | 96,84% | 7,60% |
| P2: Prioridade Multivisão (V5) | 0,7190 | 327 | 936 | 1.024 | 40 | 17,85 | 96,77% | 9,90% |
| P3: Persistência + CPD | 0,6793 | 382 | 750 | 1.245 | 5 | 13,13 | 97,23% | 12,25% |
| P4: Materialidade Energética | 0,5486 | 583 | 1.187 | 797 | 0 | 32,08 | 97,05% | 7,20% |
| P5: Qualidade da Evidência | 0,6786 | 383 | 956 | 998 | 46 | 18,42 | 98,06% | 9,50% |
| P6: Contexto Topológico (MoE) | 0,7197 | 326 | 931 | 1.038 | 31 | 17,94 | 96,83% | 7,60% |
