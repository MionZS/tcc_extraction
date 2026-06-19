-- GEO / Hierarquia de alimentacao da UC
-- Versão HARDCODED para teste com 2 UCs específicas.
-- Inclui coordenadas da UC (CAD_COORDENADA_UC_EE) e do poste (cad_pste_sist_extn).
--
-- UCs hardcoded: 94438722, 110275977

WITH selected_ucs AS (
    SELECT '94438722' AS uc_key FROM DUAL
    UNION ALL
    SELECT '110275977' AS uc_key FROM DUAL
),
geo_base AS (
    SELECT DISTINCT
        UC.ISN_UC               AS "UC",
        UC.TIPO_SIT_UC          AS "SITUACAO",
        UC.NUMERO_POSTO_UC      AS "POSTO_OPERACIONAL",
        PT.POT_INST_POSTO       AS "POT_INST_KVA",
        AG.NUM_GEDIS_ALIMG      AS "GEDIS_ALIMENTADOR",
        AL.NOME_ALIM            AS "ALIMENTADOR",
        AL.TENSAO_OPER_ALIM     AS "TENSAO_ALIMENTADOR",
        SE.NOME_SE              AS "SUBESTACAO",
        SE.SIGLA_SE             AS "SIGLA_SE",
        SE.TENSAO_NOM_SE        AS "TENSAO_SE",
        SE.CAR_SE               AS "CAR_SE",
        MU.NOME_MUN             AS "MUNICIPIO_SE",
        MU.COD_MUN              AS "COD_MUN_SE",
        -- Coordenadas da UC (lat/lon)
        COORD.NUM_COORY_XXX     AS "LAT_UC",
        COORD.NUM_COORX_XXX     AS "LONG_UC",
        -- Coordenadas do poste (UTM SIRGAS 2000)
        PSX.num_coorx_psx       AS "COORD_X_POSTE",
        PSX.num_coory_psx       AS "COORD_Y_POSTE",
        -- Conversao para WGS84 (latitude/longitude) 
        SDO_CS.TRANSFORM(
            SDO_GEOMETRY(
                2001,
                29192,
                SDO_POINT_TYPE(
                    PSX.num_coorx_psx,
                    PSX.num_coory_psx,
                    NULL
                ),
                NULL,
                NULL
            ),
            4326
        ).SDO_POINT.Y           AS "LAT_POSTE_WGS84",
        SDO_CS.TRANSFORM(
            SDO_GEOMETRY(
                2001,
                29192,
                SDO_POINT_TYPE(
                    PSX.num_coorx_psx,
                    PSX.num_coory_psx,
                    NULL
                ),
                NULL,
                NULL
            ),
            4326
        ).SDO_POINT.X           AS "LONG_POSTE_WGS84"
    FROM CIS.UC_ENERGIA UC
        JOIN selected_ucs SU
            ON SU.uc_key = NULLIF(LTRIM(TRIM(TO_CHAR(UC.ISN_UC)), '0'), '')
        JOIN GDG.POSTO_TRANSFORMADOR PT
            ON UC.NUMERO_POSTO_UC = PT.NUM_OPER_POSTO
        LEFT JOIN GDG.TRECHO_PRIMARIO TR
            ON PT.NUM_GEO_TRECHO_PRIM_POSTO = TR.NUM_SEQ_GEO
        LEFT JOIN GDG.ALIMENTADOR_GEO AG
            ON TR.NUM_GEO_ALIM_TRPRIM = AG.NUM_SEQ_GEO
        LEFT JOIN SNAP_USER.ALIMENTADOR AL
            ON AL.NUM_GEDIS_ALIM = AG.NUM_GEDIS_ALIMG
           AND AG.NUM_ALIM_ALIMG = AL.NUM_SEQ_ALIM
        LEFT JOIN SNAP_USER.SUBESTACAO SE
            ON AL.NUM_SEQ_SE_ALIM = SE.NUM_SEQ_SE
        LEFT JOIN CIS.MUNICIPIO MU
            ON SE.COD_MUN_SE = MU.COD_MUN
        -- Coordenadas da UC (via REDEDES)
        LEFT JOIN REDEDES.CAD_COORDENADA_UC_EE COORD
            ON COORD.COD_UN_CONS_XXX = UC.ISN_UC
        -- Coordenadas do poste (sistema externo)
        LEFT JOIN cad_pste_sist_extn PSX
            ON PSX.num_pste_psx = UC.NUMERO_POSTO_UC
           AND PSX.cod_situ_psx = 'AT'
    WHERE UC.NUMERO_POSTO_UC NOT LIKE '%PTMUN'
      AND UC.TIPO_SIT_UC IN ('LG', 'CR', 'DS')
)
SELECT
    UC,
    SITUACAO,
    POSTO_OPERACIONAL,
    POT_INST_KVA,
    GEDIS_ALIMENTADOR,
    ALIMENTADOR,
    TENSAO_ALIMENTADOR,
    SUBESTACAO,
    SIGLA_SE,
    TENSAO_SE,
    CAR_SE,
    MUNICIPIO_SE,
    COD_MUN_SE,
    -- Coordenadas
    LAT_UC,
    LONG_UC,
    COORD_X_POSTE,
    COORD_Y_POSTE,
    LAT_POSTE_WGS84,
    LONG_POSTE_WGS84
FROM geo_base
ORDER BY UC;
