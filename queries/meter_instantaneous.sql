-- meter_instantaneous.sql — Instantaneous voltage & current snapshots
--
-- Grain: NIO × timestamp_utc
-- Cadence: 15 or 60 min (varies by meter)
-- Columns: U_L[123], I_INSTANT_L[123]
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
        WHEN i.u_l1 IS NOT NULL OR i.u_l2 IS NOT NULL OR i.u_l3 IS NOT NULL THEN 15
        ELSE NULL
    END                                                    AS "CADENCE_MINUTES",
    i.u_l1                                                 AS "U_L1",
    i.u_l2                                                 AS "U_L2",
    i.u_l3                                                 AS "U_L3",
    ii.i_l1                                                AS "I_INSTANT_L1",
    ii.i_l2                                                AS "I_INSTANT_L2",
    ii.i_l3                                                AS "I_INSTANT_L3",
    CASE
        WHEN i.u_l1 IS NOT NULL OR i.u_l2 IS NOT NULL OR i.u_l3 IS NOT NULL OR
             ii.i_l1 IS NOT NULL OR ii.i_l2 IS NOT NULL OR ii.i_l3 IS NOT NULL
        THEN 0
        ELSE NULL
    END                                                    AS "QUALITY_CODE"
FROM catalogue a
CROSS JOIN time_grid t
LEFT JOIN AMI.biz_pub_data_voltage_instant i
    ON i.data_id = a.data_id AND i.tv = t.tv
LEFT JOIN AMI.biz_pub_data_current_instant ii
    ON ii.data_id = a.data_id AND ii.tv = t.tv
WHERE
    i.u_l1 IS NOT NULL OR
    i.u_l2 IS NOT NULL OR
    i.u_l3 IS NOT NULL OR
    ii.i_l1 IS NOT NULL OR
    ii.i_l2 IS NOT NULL OR
    ii.i_l3 IS NOT NULL
ORDER BY a.meter_asset_no, t.tv;
