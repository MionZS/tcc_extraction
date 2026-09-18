-- label_communication_failure.sql — Detect communication failures from MDM data
--
-- A NIO has a communication failure on report_day if it has zero
-- telemetry points in the MDM system for that day.
--
-- Output: one row per NIO per day with the HAS_MDM_DATA flag.
--
-- Bind expected:
--   :REPORT_DAY -> ISO date string (YYYY-MM-DD)

WITH params AS (
    SELECT TO_DATE(:REPORT_DAY, 'YYYY-MM-DD') AS report_day FROM dual
),
catalogue AS (
    SELECT data_id, meter_asset_no
    FROM AMI.a_data_catalogue
),
presence AS (
    SELECT
        c.meter_asset_no,
        TRUNC(f.tv) AS dia,
        COUNT(*) AS point_count,
        COUNT(f.fa_interval) AS fa_count,
        COUNT(r.ra_interval) AS ra_count
    FROM catalogue c
    JOIN AMI.biz_pub_data_f_energy_c f
        ON f.data_id = c.data_id
    JOIN params p
        ON TRUNC(f.tv) = p.report_day
    LEFT JOIN AMI.biz_pub_data_r_energy_c r
        ON r.data_id = c.data_id AND r.tv = f.tv
    GROUP BY c.meter_asset_no, TRUNC(f.tv)
)
SELECT
    'NIO'                                                    AS "ENTITY_TYPE",
    meter_asset_no                                           AS "ENTITY_ID",
    dia                                                      AS "REFERENCE_START",
    dia                                                      AS "REFERENCE_END",
    2                                                        AS "PREDICTION_HORIZON",
    'falha_comunicacao'                                      AS "LABEL_TYPE",
    CASE WHEN fa_count = 0 AND ra_count = 0 THEN 1 ELSE 0 END AS "LABEL_VALUE",
    'mdm'                                                    AS "LABEL_SOURCE",
    0.9                                                      AS "CONFIDENCE",
    ''                                                       AS "REVIEWER_ID",
    SYSTIMESTAMP                                             AS "CREATED_AT",
    'v1'                                                     AS "LABEL_VERSION",
    CASE
        WHEN fa_count = 0 AND ra_count = 0 THEN 'No telemetry on ' || TO_CHAR(dia, 'YYYY-MM-DD')
        ELSE 'Telemetry present: ' || point_count || ' points, ' || fa_count || ' FA, ' || ra_count || ' RA'
    END                                                      AS "NOTES"
FROM presence
ORDER BY meter_asset_no;
