-- meter_day_metadata.sql — Daily snapshot of meter and UC metadata
--
-- Grain: NIO × report_day
-- Purpose: temporal photograph of the installation at the reference instant.
-- Source: CIS (cad_uc_ee, rel_equip_uc, cad_equip_med, tab_sub_tipo_equip)
--         + GEO (alimentador, subestação, poste)
--
-- Bind expected:
--   :DAYS_BACK -> number of days to go back
--   :REPORT_DAY -> ISO date string (YYYY-MM-DD) for the report day

WITH params AS (
    SELECT
        :DAYS_BACK AS days_back,
        TO_DATE(:REPORT_DAY, 'YYYY-MM-DD') AS report_day
    FROM dual
),
-- Base meter population in Araucária
meters_araucaria AS (
    SELECT DISTINCT
        r.cod_un_cons_reu AS uc,
        r.num_eqip_reu AS nio,
        TRUNC(r.dta_ins_reu) AS installation_date,
        TRUNC(r.dta_reti_reu) AS removal_date,
        e.cod_sub_tipo_eqip_emd AS meter_subtype_code,
        ste.des_sub_tipo_eqip_ste AS meter_type,
        c.cod_situ_uee AS service_status,
        c.cod_tipo_fase_uee AS phase_type,
        c.cod_gru_tens_fat_uee || c.cod_sub_gru_fat_uee AS subgroup,
        c.cod_clas_cons_uee AS consumer_class,
        c.qtd_tens_lig_uee AS installed_kva,
        mu.nom_mun_mun AS municipality,
        ccord.num_coory_xxx AS lat,
        ccord.num_coorx_xxx AS lon,
        ag.num_gedis_almg AS feeder_id,
        se.nome_se AS substation_name
    FROM rel_equip_uc r
    JOIN REDEDES.cad_uc_ee c
        ON c.cod_un_cons_uee = r.cod_un_cons_reu
    JOIN tab_localidade tl
        ON c.cod_loc_uee = tl.cod_loc_loc
    JOIN tab_municipio mu
        ON tl.cod_mun_loc = mu.cod_mun_mun
    LEFT JOIN REDEDES.cad_equip_med e
        ON e.num_eqip_emd = r.num_eqip_reu
    LEFT JOIN REDEDES.tab_sub_tipo_equip ste
        ON ste.cod_sub_tipo_eqip_ste = e.cod_sub_tipo_eqip_emd
    LEFT JOIN REDEDES.cad_coordenada_uc_ee ccord
        ON ccord.cod_un_cons_xxx = c.cod_un_cons_uee
    LEFT JOIN REDEDES.cad_pste_sist_extn psx
        ON psx.num_pste_psx = c.num_pste_uee
        AND psx.cod_situ_psx = 'AT'
    LEFT JOIN GDG.POSTO_TRANSFORMADOR pt
        ON pt.num_oper_posto = c.numero_posto_uee
    LEFT JOIN GDG.TRECHO_PRIMARIO tr
        ON pt.num_geo_trecho_prim_posto = tr.num_seq_geo
    LEFT JOIN GDG.ALIMENTADOR_GEO ag
        ON tr.num_geo_alm_trprim = ag.num_seq_geo
    LEFT JOIN SNAP_USER.ALIMENTADOR al
        ON al.num_gedis_alim = ag.num_gedis_almg
    LEFT JOIN SNAP_USER.SUBESTACAO se
        ON al.num_seq_se_alim = se.num_seq_se
    WHERE UPPER(mu.nom_mun_mun) = 'ARAUCARIA'
      AND (r.dta_reti_reu IS NULL OR r.dta_reti_reu >= TRUNC(CURRENT_DATE - p.days_back))
)
SELECT
    p.report_day                                            AS "REPORT_DAY",
    nio                                                     AS "NIO",
    uc                                                      AS "UC_KEY",
    NULL                                                    AS "METER_SERIAL_KEY",
    NULLIF(TRIM(meter_type), '')                            AS "METER_TYPE",
    NULLIF(meter_subtype_code, '')                          AS "METER_SUBTYPE",
    NULLIF(TRIM(phase_type), '')                            AS "PHASE_TYPE",
    NULLIF(TRIM(service_status), '')                        AS "SERVICE_STATUS",
    installation_date                                       AS "INSTALLATION_DATE",
    removal_date                                            AS "REMOVAL_DATE",
    NULLIF(TRIM(consumer_class), '')                        AS "CONSUMER_CLASS",
    NULLIF(TRIM(subgroup), '')                              AS "SUBGROUP",
    installed_kva                                           AS "INSTALLED_KVA",
    NULLIF(TRIM(feeder_id), '')                             AS "FEEDER_ID",
    NULLIF(TRIM(substation_name), '')                       AS "SUBSTATION_ID",
    NULL                                                    AS "NOMINAL_VOLTAGE",
    NULLIF(TRIM(municipality), '')                          AS "MUNICIPALITY",
    lat                                                     AS "LATITUDE",
    lon                                                     AS "LONGITUDE"
FROM meters_araucaria
CROSS JOIN params p
WHERE
    -- Only meters that were installed on or before the report day
    (installation_date IS NULL OR installation_date <= p.report_day)
    -- And not removed before the report day
    AND (removal_date IS NULL OR removal_date >= p.report_day)
ORDER BY nio;
