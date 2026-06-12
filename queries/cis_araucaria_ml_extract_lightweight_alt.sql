-- CIS - extração final otimizada para machine learning
-- Alternativa A
--
-- Complementos adicionados para teste:
--   - QTD_TENS_LIG_UEE
--   - DISJUNTOR
--   - LAT/LONG

WITH medidores_araucaria AS (
    SELECT DISTINCT
        r.cod_un_cons_reu AS uc,
        r.num_eqip_reu AS nio
    FROM rel_equip_uc r
    JOIN REDEDES.cad_uc_ee c
        ON c.cod_un_cons_uee = r.cod_un_cons_reu
    JOIN tab_localidade tl
        ON c.cod_loc_uee = tl.cod_loc_loc
    JOIN tab_municipio mu
        ON tl.cod_mun_loc = mu.cod_mun_mun
    WHERE r.dta_reti_reu IS NULL
      AND UPPER(mu.nom_mun_mun) = 'ARAUCARIA'
),

meters_araucaria_base AS (
    SELECT
        t.cod_un_cons_reu AS uc,
        t.num_eqip_reu AS nio,
        t.dta_ins_reu AS data_instalacao_medidor,
        t.dta_reti_reu AS data_retirada_medidor,
        c.cod_situ_uee AS situacao_uc,
        c.dta_situ_uee AS data_situacao_uc,
        c.cod_tipo_fase_uee AS tipo_fase,
        c.cod_gru_tens_fat_uee || c.cod_sub_gru_fat_uee AS sub_grupo,
        c.cod_clas_cons_uee AS classe_consumo,
        c.cod_tipo_entg_uee AS tipo_entrega,
        c.cod_tipo_tar_fat_uee,
        c.num_cli_uee AS cliente,
        c.qtd_tens_lig_uee AS qtd_tens_lig_uee,
        c.cod_tipo_disj_uee AS cod_tipo_disj_uee,
        ste.des_sub_tipo_eqip_ste AS tipo_medidor,
        e.cod_sub_tipo_eqip_emd AS cod_subtipo_medidor,
        ccord.num_coory_xxx AS lat,
        ccord.num_coorx_xxx AS lon,
        mu.nom_mun_mun AS municipio
    FROM rel_equip_uc t
    JOIN medidores_araucaria ma
        ON ma.uc = t.cod_un_cons_reu
       AND ma.nio = t.num_eqip_reu
    JOIN REDEDES.cad_uc_ee c
        ON c.cod_un_cons_uee = t.cod_un_cons_reu
    JOIN tab_localidade tl
        ON c.cod_loc_uee = tl.cod_loc_loc
    JOIN tab_municipio mu
        ON tl.cod_mun_loc = mu.cod_mun_mun
    LEFT JOIN REDEDES.cad_equip_med e
        ON e.num_eqip_emd = t.num_eqip_reu
    LEFT JOIN REDEDES.cad_coordenada_uc_ee ccord
        ON ccord.cod_un_cons_xxx = c.cod_un_cons_uee
    LEFT JOIN REDEDES.tab_sub_tipo_equip ste
        ON ste.cod_sub_tipo_eqip_ste = e.cod_sub_tipo_eqip_emd
    WHERE t.dta_reti_reu IS NULL
),

meters_araucaria_classified AS (
    SELECT
        b.*,
        CASE
            WHEN (
                TO_NUMBER(b.nio) BETWEEN 41000000 AND 46000000
                OR UPPER(NVL(b.tipo_medidor, ' ')) LIKE '%INTEL%'
            )
            AND UPPER(NVL(b.tipo_medidor, ' ')) NOT LIKE '%QUALI%'
            THEN 1
            ELSE 0
        END AS smart
    FROM meters_araucaria_base b
),

population_uc AS (
    SELECT DISTINCT uc
    FROM meters_araucaria_classified
),

population_uc_cliente AS (
    SELECT DISTINCT uc, cliente
    FROM meters_araucaria_classified
    WHERE cliente IS NOT NULL
),

cte_cge AS (
    SELECT
        cge.cod_un_cons_utg AS uc,
        MAX(cge.cod_modl_dis_utg) AS modalidade_geracao
    FROM cad_uc_ger_energia cge
    JOIN population_uc p
        ON p.uc = cge.cod_un_cons_utg
    WHERE cge.dta_fim_vign_utg IS NULL
      AND (cge.cod_situ_utg = 'AT' OR cge.cod_situ_utg IS NULL)
      AND cge.qtd_pot_ger_utg <= 25000
    GROUP BY cge.cod_un_cons_utg
),

cte_mmgd AS (
    SELECT *
    FROM (
        SELECT
            kcg.cod_un_cons_kcg AS uc,
            kcd.cod_tipo_css_ger_kcd AS tipo_gd,
            kcg.dta_inic_vign_kcg AS data_inicio_gd,
            kcg.dta_fim_vign_kcg AS data_fim_gd,
            ROW_NUMBER() OVER (
                PARTITION BY kcg.cod_un_cons_kcg
                ORDER BY
                    CASE WHEN kcg.dta_fim_vign_kcg IS NULL THEN 1 ELSE 2 END,
                    kcg.dta_fim_vign_kcg DESC,
                    kcg.dta_inic_vign_kcg DESC
            ) AS rn
        FROM rel_classif_mmgd kcg
        LEFT JOIN rel_det_classif_mmgd kcd
            ON kcd.num_seq_kcd = kcg.num_seq_kcg
        JOIN population_uc p
            ON p.uc = kcg.cod_un_cons_kcg
    )
    WHERE rn = 1
),

cte_benef AS (
    SELECT
        x.cod_un_cons_ben_xbg AS uc,
        x.num_cli_ben_xbg AS cliente,
        x.dta_inic_vign_xbg AS inicio_beneficiaria,
        x.dta_fim_vign_xbg AS fim_beneficiaria
    FROM (
        SELECT
            b.*,
            ROW_NUMBER() OVER (
                PARTITION BY b.cod_un_cons_ben_xbg, b.num_cli_ben_xbg
                ORDER BY
                    NVL(b.dta_fim_vign_xbg, DATE '2999-12-31') DESC,
                    b.dta_inic_vign_xbg DESC
            ) AS rn
        FROM rel_micro_ger_benef b
        JOIN population_uc_cliente p
            ON p.uc = b.cod_un_cons_ben_xbg
           AND p.cliente = b.num_cli_ben_xbg
        WHERE b.con_val_xbg = 'AT'
          AND (b.dta_fim_vign_xbg IS NULL OR b.dta_fim_vign_xbg >= TRUNC(SYSDATE, 'MM'))
          AND b.dta_inic_vign_xbg <= TRUNC(SYSDATE, 'MM')
    ) x
    WHERE x.rn = 1
)

SELECT
    c.smart AS "SMART",
    c.uc AS "UC",
    c.nio AS "NIO",
    c.cliente AS "CLIENTE",
    c.qtd_tens_lig_uee AS "QTD_TENS_LIG_UEE",
    c.cod_tipo_disj_uee AS "DISJUNTOR",
    c.lat AS "LAT",
    c.lon AS "LONG",
    c.municipio AS "MUNICIPIO",
    c.tipo_medidor AS "TIPO_MEDIDOR",
    c.cod_subtipo_medidor AS "COD_SUBTIPO_MEDIDOR",
    c.situacao_uc AS "SITUACAO_UC",
    TO_CHAR(c.data_situacao_uc, 'YYYY-MM-DD') AS "DATA_SITUACAO_UC",
    TO_CHAR(c.data_instalacao_medidor, 'YYYY-MM-DD') AS "DATA_INSTALACAO_MEDIDOR",
    TO_CHAR(c.data_retirada_medidor, 'YYYY-MM-DD') AS "DATA_RETIRADA_MEDIDOR",
    c.tipo_fase AS "TIPO_FASE",
    c.sub_grupo AS "SUB_GRUPO",
    c.classe_consumo AS "CLASSE_CONSUMO",
    c.tipo_entrega AS "TIPO_ENTREGA",
    CASE WHEN c.cod_tipo_tar_fat_uee = '05' THEN 'S' ELSE 'N' END AS "TARIFA_BRANCA",
    CASE WHEN c.classe_consumo IN ('9109', '9111', '9113', '9114', '9116', '9115') THEN 'S' ELSE 'N' END AS "BAIXA_RENDA",
    cge.modalidade_geracao AS "MODALIDADE_GERACAO",
    CASE WHEN cge.uc IS NOT NULL THEN 'S' ELSE 'N' END AS "GERACAO_PROPRIA",
    mmgd.tipo_gd AS "TIPO_GD",
    TO_CHAR(mmgd.data_inicio_gd, 'YYYY-MM-DD') AS "DATA_INICIO_GD",
    TO_CHAR(mmgd.data_fim_gd, 'YYYY-MM-DD') AS "DATA_FIM_GD",
    CASE WHEN ben.uc IS NOT NULL THEN 'S' ELSE 'N' END AS "BENEFICIARIA_GD",
    TO_CHAR(ben.inicio_beneficiaria, 'YYYY-MM-DD') AS "INICIO_BENEFICIARIA",
    TO_CHAR(ben.fim_beneficiaria, 'YYYY-MM-DD') AS "FIM_BENEFICIARIA"
FROM meters_araucaria_classified c
LEFT JOIN cte_cge cge
    ON cge.uc = c.uc
LEFT JOIN cte_mmgd mmgd
    ON mmgd.uc = c.uc
LEFT JOIN cte_benef ben
    ON ben.uc = c.uc
   AND ben.cliente = c.cliente;
