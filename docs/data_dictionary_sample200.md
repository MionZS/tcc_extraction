# Dicionário de dados — ARAUCARIA sample joined output

Arquivo-alvo esperado: `output/refined/reports/sample200/araucaria_model_input_sample_200_YYYYMMDD.csv`

> Formato simples: cada coluna aparece com sua descrição, sem tabela e sem coluna extra.
> 
> Observação: os campos `total` são documentados sem sufixo, porque representam soma acumulada de incrementais e não uma leitura de janela de 5 minutos.

## Colunas de contexto CIS

- `smart`: flag derivada que indica se o medidor foi classificado como smart.
- `uc`: identificador da unidade consumidora.
- `nio`: identificador do medidor/ativo usado no join.
- `cliente`: identificador do cliente associado à UC.
- `qtd_tens_lig_uee`: código/quantidade de tensão ligada da UEE.
- `disjuntor`: código do disjuntor/proteção.
- `lat`: latitude da instalação.
- `LONG`: longitude da instalação.
- `municipio`: município cadastrado.
- `tipo_medidor`: descrição do tipo de medidor.
- `cod_subtipo_medidor`: código do subtipo do medidor.
- `situacao_uc`: situação atual da unidade consumidora.
- `data_situacao_uc`: data da situação atual da UC.
- `data_instalacao_medidor`: data de instalação do medidor.
- `data_retirada_medidor`: data de retirada do medidor.
- `tipo_fase`: configuração/tipo de fase elétrica.
- `sub_grupo`: subgrupo tarifário.
- `classe_consumo`: classe de consumo.
- `tipo_entrega`: tipo/modalidade de entrega.
- `tarifa_branca`: flag de tarifa branca.
- `baixa_renda`: flag de baixa renda.
- `modalidade_geracao`: modalidade de geração distribuída.
- `geracao_propria`: flag de geração própria.
- `tipo_gd`: tipo de geração distribuída.
- `data_inicio_gd`: data de início da GD.
- `data_fim_gd`: data de fim da GD.
- `beneficiaria_gd`: flag de beneficiária de GD.
- `inicio_beneficiaria`: início da condição de beneficiária GD.
- `fim_beneficiaria`: fim da condição de beneficiária GD.

## Colunas ORCA / MDM de série temporal

- `dia`: data de referência da extração ORCA.
- `fa_interval`: série JSON intervalar de 5 minutos da energia ativa forward.
- `ra_interval`: série JSON intervalar de 5 minutos da energia ativa reverse.
- `i_l1_avg`: corrente média de 5 minutos na fase L1.
- `i_l2_avg`: corrente média de 5 minutos na fase L2.
- `i_l3_avg`: corrente média de 5 minutos na fase L3.
- `u_l1_avg`: tensão média de 5 minutos na fase L1.
- `u_l2_avg`: tensão média de 5 minutos na fase L2.
- `u_l3_avg`: tensão média de 5 minutos na fase L3.
- `u_l1`: tensão instantânea na fase L1.
- `u_l2`: tensão instantânea na fase L2.
- `u_l3`: tensão instantânea na fase L3.
- `i_instant_l1`: corrente instantânea na fase L1.
- `i_instant_l2`: corrente instantânea na fase L2.
- `i_instant_l3`: corrente instantânea na fase L3.
- `r_q1_interval`: energia reativa intervalar no quadrante Q1.
- `r_q2_interval`: energia reativa intervalar no quadrante Q2.
- `r_q3_interval`: energia reativa intervalar no quadrante Q3.
- `r_q4_interval`: energia reativa intervalar no quadrante Q4.
- `fa_total`: energia ativa forward total acumulada.
- `fa_t1_total`: componente T1 da energia ativa forward total acumulada.
- `fa_t2_total`: componente T2 da energia ativa forward total acumulada.
- `fa_t3_total`: componente T3 da energia ativa forward total acumulada.
- `fa_t4_total`: componente T4 da energia ativa forward total acumulada.
- `ra_total`: energia ativa reverse total acumulada.
- `ra_t1_total`: componente T1 da energia ativa reverse total acumulada.
- `ra_t2_total`: componente T2 da energia ativa reverse total acumulada.
- `ra_t3_total`: componente T3 da energia ativa reverse total acumulada.
- `ra_t4_total`: componente T4 da energia ativa reverse total acumulada.
- `fa_md`: série JSON de 5 minutos da métrica FA MD.
- `fa_md_t1`: série JSON de 5 minutos da métrica FA MD T1.
- `fa_md_t2`: série JSON de 5 minutos da métrica FA MD T2.
- `fa_md_t3`: série JSON de 5 minutos da métrica FA MD T3.
- `fa_md_t4`: série JSON de 5 minutos da métrica FA MD T4.

## Legenda dos prefixos e sufixos

- `fa`: forward active.
- `ra`: reverse active.
- `i`: current / corrente.
- `u`: voltage / tensão.
- `r_q1` a `r_q4`: energia reativa por quadrante.
- `md`: maximum demand.
- `l1`, `l2`, `l3`: fases/canais elétricos.
- `t1`, `t2`, `t3`, `t4`: subperíodos ou faixas tarifárias usadas pelo sistema de origem.
- `instant`: medida instantânea no momento exato do timestamp.
- `avg`: média das medições dentro da janela de 5 minutos.
- `interval`: série intervalar agregada em janelas de 5 minutos.
- `total`: soma acumulada do incremental; não é uma leitura instantânea de 5 minutos.
