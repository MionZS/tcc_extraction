-- ORCA / MDM - uma linha por NIO por dia
--
-- Saida:
--   - 1 linha por meter_asset_no + dia
--   - cada medida vira uma coluna JSON
--   - horarios sem dado NULL sao omitidos do JSON
--   - valores 0 sao preservados
--
-- Bind esperado:
--   :DAYS_BACK -> numero de dias para voltar
--   :UCS       -> SYS.ODCIVARCHAR2LIST com a lista de medidores/NIOs

WITH params AS (
    SELECT :DAYS_BACK AS days_back
    FROM dual
),
selected_meters AS (
    SELECT DISTINCT
        NULLIF(LTRIM(REGEXP_REPLACE(TRIM(COLUMN_VALUE), '[^0-9]', ''), '0'), '') AS meter_asset_no
    FROM TABLE(CAST(:UCS AS SYS.ODCIVARCHAR2LIST))
    WHERE NULLIF(LTRIM(REGEXP_REPLACE(TRIM(COLUMN_VALUE), '[^0-9]', ''), '0'), '') IS NOT NULL
),
catalogue AS (
    SELECT
        a.data_id,
        NULLIF(LTRIM(TRIM(a.meter_asset_no), '0'), '') AS meter_asset_no
    FROM AMI.a_data_catalogue a
    JOIN selected_meters sm
        ON sm.meter_asset_no = NULLIF(LTRIM(TRIM(a.meter_asset_no), '0'), '')
),
time_grid AS (
    SELECT
        TRUNC(CURRENT_DATE - p.days_back)
        + NUMTODSINTERVAL((LEVEL - 1) * 5, 'MINUTE') AS tv
    FROM dual
    CROSS JOIN params p
    CONNECT BY LEVEL <= 288
),
base AS (
    SELECT
        a.meter_asset_no,
        TRUNC(t.tv) AS dia,
        TO_CHAR(t.tv, 'HH24:MI') AS hhmi,

        f.fa_interval,
        r.ra_interval,

        c.i_l1_avg,
        c.i_l2_avg,
        c.i_l3_avg,

        v.u_l1_avg,
        v.u_l2_avg,
        v.u_l3_avg,

        i.u_l1,
        i.u_l2,
        i.u_l3,

        ii.i_l1 AS i_instant_l1,
        ii.i_l2 AS i_instant_l2,
        ii.i_l3 AS i_instant_l3,

        q.r_q1_interval,
        q.r_q2_interval,
        q.r_q3_interval,
        q.r_q4_interval,

        d.fa,
        d.fa_t1,
        d.fa_t2,
        d.fa_t3,
        d.fa_t4,

        rr.ra AS ra_total,
        rr.ra_t1 AS ra_t1_total,
        rr.ra_t2 AS ra_t2_total,
        rr.ra_t3 AS ra_t3_total,
        rr.ra_t4 AS ra_t4_total,

        m.fa_md,
        m.fa_md_t1,
        m.fa_md_t2,
        m.fa_md_t3,
        m.fa_md_t4
    FROM catalogue a
    CROSS JOIN time_grid t
    LEFT JOIN AMI.biz_pub_data_f_energy_c f
        ON f.data_id = a.data_id
       AND f.tv = t.tv
    LEFT JOIN AMI.biz_pub_data_r_energy_c r
        ON r.data_id = a.data_id
       AND r.tv = t.tv
    LEFT JOIN AMI.biz_pub_data_current c
        ON c.data_id = a.data_id
       AND c.tv = t.tv
    LEFT JOIN AMI.biz_pub_data_voltage v
        ON v.data_id = a.data_id
       AND v.tv = t.tv
    LEFT JOIN AMI.biz_pub_data_voltage_instant i
        ON i.data_id = a.data_id
       AND i.tv = t.tv
    LEFT JOIN AMI.biz_pub_data_current_instant ii
        ON ii.data_id = a.data_id
       AND ii.tv = t.tv
    LEFT JOIN AMI.biz_pub_data_q_energy_c q
        ON q.data_id = a.data_id
       AND q.tv = t.tv
    LEFT JOIN AMI.biz_pub_data_f_energy_d d
        ON d.data_id = a.data_id
       AND d.tv = t.tv
    LEFT JOIN AMI.biz_pub_data_r_energy_d rr
        ON rr.data_id = a.data_id
       AND rr.tv = t.tv
    LEFT JOIN AMI.biz_pub_data_f_md_d m
        ON m.data_id = a.data_id
       AND m.tv = t.tv
),
slot_values AS (
    SELECT
        meter_asset_no,
        dia,
        hhmi,

        MAX(fa_interval) AS fa_interval,
        MAX(ra_interval) AS ra_interval,

        MAX(i_l1_avg) AS i_l1_avg,
        MAX(i_l2_avg) AS i_l2_avg,
        MAX(i_l3_avg) AS i_l3_avg,

        MAX(u_l1_avg) AS u_l1_avg,
        MAX(u_l2_avg) AS u_l2_avg,
        MAX(u_l3_avg) AS u_l3_avg,

        MAX(u_l1) AS u_l1,
        MAX(u_l2) AS u_l2,
        MAX(u_l3) AS u_l3,

        MAX(i_instant_l1) AS i_instant_l1,
        MAX(i_instant_l2) AS i_instant_l2,
        MAX(i_instant_l3) AS i_instant_l3,

        MAX(r_q1_interval) AS r_q1_interval,
        MAX(r_q2_interval) AS r_q2_interval,
        MAX(r_q3_interval) AS r_q3_interval,
        MAX(r_q4_interval) AS r_q4_interval,

        MAX(fa) AS fa,
        MAX(fa_t1) AS fa_t1,
        MAX(fa_t2) AS fa_t2,
        MAX(fa_t3) AS fa_t3,
        MAX(fa_t4) AS fa_t4,

        MAX(ra_total) AS ra_total,
        MAX(ra_t1_total) AS ra_t1_total,
        MAX(ra_t2_total) AS ra_t2_total,
        MAX(ra_t3_total) AS ra_t3_total,
        MAX(ra_t4_total) AS ra_t4_total,

        MAX(fa_md) AS fa_md,
        MAX(fa_md_t1) AS fa_md_t1,
        MAX(fa_md_t2) AS fa_md_t2,
        MAX(fa_md_t3) AS fa_md_t3,
        MAX(fa_md_t4) AS fa_md_t4,

        CASE
            WHEN MAX(fa_interval) IS NOT NULL OR
                 MAX(ra_interval) IS NOT NULL OR
                 MAX(i_l1_avg) IS NOT NULL OR
                 MAX(i_l2_avg) IS NOT NULL OR
                 MAX(i_l3_avg) IS NOT NULL OR
                 MAX(u_l1_avg) IS NOT NULL OR
                 MAX(u_l2_avg) IS NOT NULL OR
                 MAX(u_l3_avg) IS NOT NULL OR
                 MAX(u_l1) IS NOT NULL OR
                 MAX(u_l2) IS NOT NULL OR
                 MAX(u_l3) IS NOT NULL OR
                 MAX(i_instant_l1) IS NOT NULL OR
                 MAX(i_instant_l2) IS NOT NULL OR
                 MAX(i_instant_l3) IS NOT NULL OR
                 MAX(r_q1_interval) IS NOT NULL OR
                 MAX(r_q2_interval) IS NOT NULL OR
                 MAX(r_q3_interval) IS NOT NULL OR
                 MAX(r_q4_interval) IS NOT NULL OR
                 MAX(fa) IS NOT NULL OR
                 MAX(fa_t1) IS NOT NULL OR
                 MAX(fa_t2) IS NOT NULL OR
                 MAX(fa_t3) IS NOT NULL OR
                 MAX(fa_t4) IS NOT NULL OR
                 MAX(ra_total) IS NOT NULL OR
                 MAX(ra_t1_total) IS NOT NULL OR
                 MAX(ra_t2_total) IS NOT NULL OR
                 MAX(ra_t3_total) IS NOT NULL OR
                 MAX(ra_t4_total) IS NOT NULL OR
                 MAX(fa_md) IS NOT NULL OR
                 MAX(fa_md_t1) IS NOT NULL OR
                 MAX(fa_md_t2) IS NOT NULL OR
                 MAX(fa_md_t3) IS NOT NULL OR
                 MAX(fa_md_t4) IS NOT NULL
            THEN 1
            ELSE 0
        END AS has_data_slot
    FROM base
    GROUP BY meter_asset_no, dia, hhmi
),
daily AS (
    SELECT
        meter_asset_no,
        dia,
        MAX(has_data_slot) AS has_data,

        JSON_OBJECTAGG(KEY hhmi VALUE fa_interval ABSENT ON NULL RETURNING CLOB) AS fa_interval_5m,
        JSON_OBJECTAGG(KEY hhmi VALUE ra_interval ABSENT ON NULL RETURNING CLOB) AS ra_interval_5m,

        JSON_OBJECTAGG(KEY hhmi VALUE i_l1_avg ABSENT ON NULL RETURNING CLOB) AS i_l1_avg_5m,
        JSON_OBJECTAGG(KEY hhmi VALUE i_l2_avg ABSENT ON NULL RETURNING CLOB) AS i_l2_avg_5m,
        JSON_OBJECTAGG(KEY hhmi VALUE i_l3_avg ABSENT ON NULL RETURNING CLOB) AS i_l3_avg_5m,

        JSON_OBJECTAGG(KEY hhmi VALUE u_l1_avg ABSENT ON NULL RETURNING CLOB) AS u_l1_avg_5m,
        JSON_OBJECTAGG(KEY hhmi VALUE u_l2_avg ABSENT ON NULL RETURNING CLOB) AS u_l2_avg_5m,
        JSON_OBJECTAGG(KEY hhmi VALUE u_l3_avg ABSENT ON NULL RETURNING CLOB) AS u_l3_avg_5m,

        JSON_OBJECTAGG(KEY hhmi VALUE u_l1 ABSENT ON NULL RETURNING CLOB) AS u_l1_5m,
        JSON_OBJECTAGG(KEY hhmi VALUE u_l2 ABSENT ON NULL RETURNING CLOB) AS u_l2_5m,
        JSON_OBJECTAGG(KEY hhmi VALUE u_l3 ABSENT ON NULL RETURNING CLOB) AS u_l3_5m,

        JSON_OBJECTAGG(KEY hhmi VALUE i_instant_l1 ABSENT ON NULL RETURNING CLOB) AS i_instant_l1_5m,
        JSON_OBJECTAGG(KEY hhmi VALUE i_instant_l2 ABSENT ON NULL RETURNING CLOB) AS i_instant_l2_5m,
        JSON_OBJECTAGG(KEY hhmi VALUE i_instant_l3 ABSENT ON NULL RETURNING CLOB) AS i_instant_l3_5m,

        JSON_OBJECTAGG(KEY hhmi VALUE r_q1_interval ABSENT ON NULL RETURNING CLOB) AS r_q1_interval_5m,
        JSON_OBJECTAGG(KEY hhmi VALUE r_q2_interval ABSENT ON NULL RETURNING CLOB) AS r_q2_interval_5m,
        JSON_OBJECTAGG(KEY hhmi VALUE r_q3_interval ABSENT ON NULL RETURNING CLOB) AS r_q3_interval_5m,
        JSON_OBJECTAGG(KEY hhmi VALUE r_q4_interval ABSENT ON NULL RETURNING CLOB) AS r_q4_interval_5m,

        JSON_OBJECTAGG(KEY hhmi VALUE fa ABSENT ON NULL RETURNING CLOB) AS fa_total_5m,
        JSON_OBJECTAGG(KEY hhmi VALUE fa_t1 ABSENT ON NULL RETURNING CLOB) AS fa_t1_total_5m,
        JSON_OBJECTAGG(KEY hhmi VALUE fa_t2 ABSENT ON NULL RETURNING CLOB) AS fa_t2_total_5m,
        JSON_OBJECTAGG(KEY hhmi VALUE fa_t3 ABSENT ON NULL RETURNING CLOB) AS fa_t3_total_5m,
        JSON_OBJECTAGG(KEY hhmi VALUE fa_t4 ABSENT ON NULL RETURNING CLOB) AS fa_t4_total_5m,

        JSON_OBJECTAGG(KEY hhmi VALUE ra_total ABSENT ON NULL RETURNING CLOB) AS ra_total_5m,
        JSON_OBJECTAGG(KEY hhmi VALUE ra_t1_total ABSENT ON NULL RETURNING CLOB) AS ra_t1_total_5m,
        JSON_OBJECTAGG(KEY hhmi VALUE ra_t2_total ABSENT ON NULL RETURNING CLOB) AS ra_t2_total_5m,
        JSON_OBJECTAGG(KEY hhmi VALUE ra_t3_total ABSENT ON NULL RETURNING CLOB) AS ra_t3_total_5m,
        JSON_OBJECTAGG(KEY hhmi VALUE ra_t4_total ABSENT ON NULL RETURNING CLOB) AS ra_t4_total_5m,

        JSON_OBJECTAGG(KEY hhmi VALUE fa_md ABSENT ON NULL RETURNING CLOB) AS fa_md_5m,
        JSON_OBJECTAGG(KEY hhmi VALUE fa_md_t1 ABSENT ON NULL RETURNING CLOB) AS fa_md_t1_5m,
        JSON_OBJECTAGG(KEY hhmi VALUE fa_md_t2 ABSENT ON NULL RETURNING CLOB) AS fa_md_t2_5m,
        JSON_OBJECTAGG(KEY hhmi VALUE fa_md_t3 ABSENT ON NULL RETURNING CLOB) AS fa_md_t3_5m,
        JSON_OBJECTAGG(KEY hhmi VALUE fa_md_t4 ABSENT ON NULL RETURNING CLOB) AS fa_md_t4_5m
    FROM slot_values
    GROUP BY meter_asset_no, dia
)
SELECT
    meter_asset_no AS "NIO",
    TO_CHAR(dia, 'YYYY-MM-DD') AS "DIA",
    fa_interval_5m,
    ra_interval_5m,
    i_l1_avg_5m,
    i_l2_avg_5m,
    i_l3_avg_5m,
    u_l1_avg_5m,
    u_l2_avg_5m,
    u_l3_avg_5m,
    u_l1_5m,
    u_l2_5m,
    u_l3_5m,
    i_instant_l1_5m,
    i_instant_l2_5m,
    i_instant_l3_5m,
    r_q1_interval_5m,
    r_q2_interval_5m,
    r_q3_interval_5m,
    r_q4_interval_5m,
    fa_total_5m,
    fa_t1_total_5m,
    fa_t2_total_5m,
    fa_t3_total_5m,
    fa_t4_total_5m,
    ra_total_5m,
    ra_t1_total_5m,
    ra_t2_total_5m,
    ra_t3_total_5m,
    ra_t4_total_5m,
    fa_md_5m,
    fa_md_t1_5m,
    fa_md_t2_5m,
    fa_md_t3_5m,
    fa_md_t4_5m
FROM daily
WHERE has_data = 1
ORDER BY dia, meter_asset_no;