-- meter_interval.sql — Interval energy, reactive energy, current & voltage averages
--
-- Grain: NIO × timestamp_utc
-- Cadence: 5 min (288 slots/day)
-- Columns: FA_INTERVAL, RA_INTERVAL, I_L[123]_AVG, U_L[123]_AVG, R_Q[1-4]_INTERVAL
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
time_grid AS (
    SELECT
        TRUNC(CURRENT_DATE - p.days_back)
        + NUMTODSINTERVAL((LEVEL - 1) * 5, 'MINUTE') AS tv
    FROM dual
    CROSS JOIN params p
    CONNECT BY LEVEL <= 288
)
SELECT
    a.meter_asset_no                                      AS "NIO",
    TRUNC(t.tv)                                            AS "REPORT_DAY",
    CAST(t.tv AS TIMESTAMP)                                AS "TIMESTAMP_UTC",
    TO_CHAR(t.tv, 'YYYY-MM-DD"T"HH24:MI:SS')              AS "TIMESTAMP_UTC_ISO",
    CASE
        WHEN f.fa_interval IS NOT NULL THEN 5
        WHEN r.ra_interval IS NOT NULL THEN 5
        ELSE NULL
    END                                                    AS "CADENCE_MINUTES",
    f.fa_interval                                          AS "FA_INTERVAL",
    r.ra_interval                                          AS "RA_INTERVAL",
    c.i_l1_avg                                             AS "I_L1_AVG",
    c.i_l2_avg                                             AS "I_L2_AVG",
    c.i_l3_avg                                             AS "I_L3_AVG",
    v.u_l1_avg                                             AS "U_L1_AVG",
    v.u_l2_avg                                             AS "U_L2_AVG",
    v.u_l3_avg                                             AS "U_L3_AVG",
    q.r_q1_interval                                        AS "R_Q1_INTERVAL",
    q.r_q2_interval                                        AS "R_Q2_INTERVAL",
    q.r_q3_interval                                        AS "R_Q3_INTERVAL",
    q.r_q4_interval                                        AS "R_Q4_INTERVAL",
    CASE
        WHEN f.fa_interval IS NOT NULL OR
             r.ra_interval IS NOT NULL OR
             c.i_l1_avg IS NOT NULL OR
             c.i_l2_avg IS NOT NULL OR
             c.i_l3_avg IS NOT NULL OR
             v.u_l1_avg IS NOT NULL OR
             v.u_l2_avg IS NOT NULL OR
             v.u_l3_avg IS NOT NULL OR
             q.r_q1_interval IS NOT NULL OR
             q.r_q2_interval IS NOT NULL OR
             q.r_q3_interval IS NOT NULL OR
             q.r_q4_interval IS NOT NULL
        THEN 0
        ELSE NULL
    END                                                    AS "QUALITY_CODE",
    0                                                      AS "IS_ESTIMATED"
FROM catalogue a
CROSS JOIN time_grid t
LEFT JOIN AMI.biz_pub_data_f_energy_c f
    ON f.data_id = a.data_id AND f.tv = t.tv
LEFT JOIN AMI.biz_pub_data_r_energy_c r
    ON r.data_id = a.data_id AND r.tv = t.tv
LEFT JOIN AMI.biz_pub_data_current c
    ON c.data_id = a.data_id AND c.tv = t.tv
LEFT JOIN AMI.biz_pub_data_voltage v
    ON v.data_id = a.data_id AND v.tv = t.tv
LEFT JOIN AMI.biz_pub_data_q_energy_c q
    ON q.data_id = a.data_id AND q.tv = t.tv
WHERE
    f.fa_interval IS NOT NULL OR
    r.ra_interval IS NOT NULL OR
    c.i_l1_avg IS NOT NULL OR
    c.i_l2_avg IS NOT NULL OR
    c.i_l3_avg IS NOT NULL OR
    v.u_l1_avg IS NOT NULL OR
    v.u_l2_avg IS NOT NULL OR
    v.u_l3_avg IS NOT NULL OR
    q.r_q1_interval IS NOT NULL OR
    q.r_q2_interval IS NOT NULL OR
    q.r_q3_interval IS NOT NULL OR
    q.r_q4_interval IS NOT NULL
ORDER BY a.meter_asset_no, t.tv;
