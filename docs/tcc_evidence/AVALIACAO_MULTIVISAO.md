# Avaliação Integrada da Arquitetura Multivisão e MoE Topológico (P1 — Seções 12 e 18)

- **Execução:** `20261126_172513_p1_multiview_eval` | Tempo total: `44,89 s`
- **Contrato de saída gerado:** `experiments/runs/20261126_172513_p1_multiview_eval/multiview_evidence_oot.parquet` (`65.686` UCs, anonimizado LGPD, `NULL` preservado quando componente indisponível/inelegível)
- **Figuras:** `fig_p1_09_arquitetura_multivisao_executada.pdf` e `fig_p1_15_comparacao_global_vs_contextual_moe.pdf`

## 1. Desempenho dos Especialistas no OOT (65.686 UCs)

- **`S_elec` (congelado):** `11,04 s`
- **`S_behavior` (distância PCA+KMeans congelado):** `0,12 s`
- **`CPD` (CUSUM sobre 30 dias reais de `uc_day_features`):** `8,96 s` — `63.719` UCs avaliáveis (`1.967` inavaliáveis com `< 8` dias confiáveis → `NULL`), `8.801` quebras detectadas (`2.758` quedas `>= 30%`).
- **`MoE` Hierárquico sem vazamento (16 especialistas por alimentador treinados estritamente em `<= 31/10/2026`):** `15,69 s`.

## 2. Comparação Global × Contextual MoE (População Comum Elegível com Alimentador: `58.820` UCs)

- **Spearman Global × MoE:** `0,9670`
- **Deslocamento absoluto de percentil:** média `5,14` p.p., P95 `15,85` p.p.
- **Alteração no Top-2.000:** Jaccard = `0,7528` (**`282` substituições**; `1.718` comuns).
- **Filtro de eventos coletivos no OOT real (`grid_event_threshold = 0,60`):** `0` transformadores atingiram 60% simultâneo no corte estático de 19/11 (efeito coletivo validado em perturbação controlada na Seção 18/Base B).

| Circuito (LGPD) | Treino Limpo | OOT (19/11) | GD Treino (%) | w_feeder | τ Alimentador | Spearman vs Global |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Circuito F-02 | 15.160 | 5.098 | 1,69% | 0,875 | 0,4613 | 0,9452 |
| Circuito F-03 | 11.923 | 4.045 | 1,15% | 0,875 | 0,4771 | 0,9616 |
| Circuito F-04 | 25.931 | 8.753 | 2,06% | 0,875 | 0,5071 | 0,9813 |
| Circuito F-05 | 15.282 | 5.181 | 4,47% | 0,875 | 0,4991 | 0,9760 |
| Circuito F-06 | 22.327 | 7.575 | 1,08% | 0,875 | 0,4737 | 0,9654 |
| Circuito F-07 | 295 | 187 | 4,07% | 0,647 | 0,6508 | 0,8829 |
| Circuito F-08 | 19.438 | 6.545 | 2,88% | 0,875 | 0,5083 | 0,9713 |
| Circuito F-09 | 10.599 | 3.573 | 1,45% | 0,875 | 0,4596 | 0,9810 |
| Circuito F-11 | 10.502 | 3.531 | 4,95% | 0,875 | 0,5102 | 0,9607 |
| Circuito F-13 | 13.747 | 4.641 | 2,11% | 0,875 | 0,5130 | 0,9771 |
| Circuito F-14 | 12.058 | 4.083 | 1,30% | 0,875 | 0,4610 | 0,9689 |
| Circuito F-16 | 5.977 | 2.040 | 2,33% | 0,875 | 0,4678 | 0,9641 |
| Circuito F-17 | 518 | 210 | 3,28% | 0,731 | 0,6534 | 0,9742 |
| Circuito F-18 | 749 | 1.537 | 1,20% | 0,806 | 0,4849 | 0,9270 |
| Circuito F-19 | 7.074 | 2.400 | 1,29% | 0,875 | 0,4487 | 0,9686 |
| Circuito F-22 | 4.766 | 1.606 | 1,85% | 0,875 | 0,4608 | 0,9750 |
