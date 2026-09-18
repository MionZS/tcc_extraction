-- meter_register_snapshot.sql — Accumulated energy and demand register snapshots
--
-- Grain: NIO × timestamp_utc
-- Cadence: typically 6 h (00:00, 06:00, 12:00, 18:00)
-- Columns: FA_TOTAL, FA_T[1-4]_TOTAL, RA_TOTAL, RA_T[1-4]_TOTAL, FA_MD, FA_MD_T[1-4]
--
-- Bind expected:
--   :DAYS_BACK -> number of days to go back
--   :NIO_LIST  -> SYS.ODCIVARCHAR2LIST with NIOs

WITH params AS (
    SELECT :DAYS_BACK AS days_back FROM dual
),
selected_meters AS (
    SELECT DISTINCT
        NULLIF(LTRIM(REGEXP_REPLACE(TRIM(COLUMN_VALUE), '[^0-9]', ''), '0'), '') AS meter_asset_no
    FROM TABLE(CAST(:NIO_LIST AS SYS.ODCIVARCHAR2LIST))
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
time_grid_6h AS (
    -- 4 snapshots per day: 00:00, 06:00, 12:00, 18:00
    SELECT
        TRUNC(CURRENT_DATE - p.days_back)
        + NUMTODSINTERVAL((LEVEL - 1) * 6, 'HOUR') AS tv
    FROM dual
    CROSS JOIN params p
    CONNECT BY LEVEL <= 4
)
SELECT
    a.meter_asset_no                                      AS "NIO",
    TRUNC(t.tv)                                            AS "REPORT_DAY",
    CAST(t.tv AS TIMESTAMP)                                AS "TIMESTAMP_UTC",
    TO_CHAR(t.tv, 'YYYY-MM-DD"T"HH24:MI:SS')              AS "TIMESTAMP_UTC_ISO",
    d.fa                                                   AS "FA_TOTAL",
    d.fa_t1                                                AS "FA_T1_TOTAL",
    d.fa_t2                                                AS "FA_T2_TOTAL",
    d.fa_t3                                                AS "FA_T3_TOTAL",
    d.fa_t4                                                AS "FA_T4_TOTAL",
    rr.ra                                                  AS "RA_TOTAL",
    rr.ra_t1                                               AS "RA_T1_TOTAL",
    rr.ra_t2                                               AS "RA_T2_TOTAL",
    rr.ra_t3                                               AS "RA_T3_TOTAL",
    rr.ra_t4                                               AS "RA_T4_TOTAL",
    m.fa_md                                                AS "FA_MD",
    m.fa_md_t1                                             AS "FA_MD_T1",
    m.fa_md_t2                                             AS "FA_MD_T2",
    m.fa_md_t3                                             AS "FA_MD_T3",
    m.fa_md_t4                                             AS "FA_MD_T4",
    CASE
        WHEN d.fa IS NOT NULL OR
             rr.ra IS NOT NULL OR
             m.fa_md IS NOT NULL
        THEN 0
        ELSE NULL
    END                                                    AS "QUALITY_CODE"
FROM catalogue a
CROSS JOIN time_grid_6h t
LEFT JOIN AMI.biz_pub_data_f_energy_d d
    ON d.data_id = a.data_id AND d.tv = t.tv
LEFT JOIN AMI.biz_pub_data_r_energy_d rr
    ON rr.data_id = a.data_id AND rr.tv = t.tv
LEFT JOIN AMI.biz_pub_data_f_md_d m
    ON m.data_id = a.data_id AND m.tv = t.tv
WHERE
    d.fa IS NOT NULL OR
    d.fa_t1 IS NOT NULL OR
    d.fa_t2 IS NOT NULL OR
    d.fa_t3 IS NOT NULL OR
    d.fa_t4 IS NOT NULL OR
    rr.ra IS NOT NULL OR
    rr.ra_t1 IS NOT NULL OR
    rr.ra_t2 IS NOT NULL OR
    rr.ra_t3 IS NOT NULL OR
    rr.ra_t4 IS NOT NULL OR
    m.fa_md IS NOT NULL OR
    m.fa_md_t1 IS NOT NULL OR
    m.fa_md_t2 IS NOT NULL OR
    m.fa_md_t3 IS NOT NULL OR
    m.fa_md_t4 IS NOT NULL
ORDER BY a.meter_asset_no, t.tv;
