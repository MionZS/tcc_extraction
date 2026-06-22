-- ORCA / MDM - timegrid (uma linha por NIO por intervalo de 5 min)
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
)
SELECT
    a.meter_asset_no AS "NIO",
    TO_CHAR(t.tv, 'DD/MM/YYYY HH24:MI:SS') AS "Data Time",
    -- Dados totalizador (+)
    d.fa           AS "Active energy (+) TOTAL",
    d.fa_t1        AS "tariff_1",
    d.fa_t2        AS "tariff_2",
    d.fa_t3        AS "tariff_3",
    d.fa_t4        AS "tariff_4",
    -- Dados totalizador (-)
    rr.ra           AS "Active energy (-) TOTAL",
    rr.ra_t1        AS "tariff_1_rev",
    rr.ra_t2        AS "tariff_2_rev",
    rr.ra_t3        AS "tariff_3_rev",
    rr.ra_t4        AS "tariff_4_rev",
    -- Dados incremental ativo
    f.fa_interval  AS "Active energy (+) incremental",
    r.ra_interval  AS "Active Energy (-) incremental",
    -- Dados tens?o
    i.u_l1     AS "Voltage in L1",
    i.u_l2     AS "Voltage in L2",
    i.u_l3     AS "Voltage in L3",
    v.u_l1_avg     AS "Voltage in L1 AVG",
    v.u_l2_avg     AS "Voltage in L2 AVG",
    v.u_l3_avg     AS "Voltage in L3 AVG",
    -- Dados corrente
    ii.i_l1     AS "Current in L1",
    ii.i_l2     AS "Current in L2",
    ii.i_l3     AS "Current in L3",
    c.i_l1_avg     AS "Current in L1 AVG",
    c.i_l2_avg     AS "Current in L2 AVG",
    c.i_l3_avg     AS "Current in L3 AVG",
    -- Dados Reativa
    q.r_q1_interval AS "Energy incremental RQI",
    q.r_q2_interval AS "Energy incremental RQII",
    q.r_q3_interval AS "Energy incremental RQIII",
    q.r_q4_interval AS "Energy incremental RQIV",
    -- Dados de demanda
    m.fa_md AS "Demanda_Maxima_Diaria_kW",
    m.fa_md_t1 AS "Demanda_Maxima_Patamar_1_kW",
    m.fa_md_t2 AS "Demanda_Maxima_Patamar_2_kW",
    m.fa_md_t3 AS "Demanda_Maxima_Patamar_3_kW",
    m.fa_md_t4 AS "Demanda_Maxima_Patamar_4_kW"
FROM
    catalogue a
    CROSS JOIN time_grid t
    LEFT JOIN AMI.biz_pub_data_f_energy_c f ON a.data_id = f.data_id AND f.tv = t.tv
    LEFT JOIN AMI.biz_pub_data_r_energy_c r ON a.data_id = r.data_id AND r.tv = t.tv
    LEFT JOIN AMI.biz_pub_data_current c ON a.data_id = c.data_id AND c.tv = t.tv
    LEFT JOIN AMI.biz_pub_data_voltage v ON a.data_id = v.data_id AND v.tv = t.tv
    LEFT JOIN AMI.biz_pub_data_voltage_instant i ON a.data_id = i.data_id AND i.tv = t.tv
    LEFT JOIN AMI.biz_pub_data_current_instant ii ON a.data_id = ii.data_id AND ii.tv = t.tv
    LEFT JOIN AMI.biz_pub_data_q_energy_c q ON a.data_id = q.data_id AND q.tv = t.tv
    LEFT JOIN AMI.biz_pub_data_f_energy_d d ON a.data_id = d.data_id AND d.tv = t.tv
    LEFT JOIN AMI.biz_pub_data_r_energy_d rr ON a.data_id = rr.data_id AND rr.tv = t.tv
    LEFT JOIN AMI.biz_pub_data_f_md_d m ON a.data_id = m.data_id AND m.tv = t.tv
WHERE (
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
    f.fa_interval IS NOT NULL OR
    r.ra_interval IS NOT NULL OR
    v.u_l1_avg IS NOT NULL OR
    v.u_l2_avg IS NOT NULL OR
    v.u_l3_avg IS NOT NULL OR
    i.u_l1 IS NOT NULL OR
    i.u_l2 IS NOT NULL OR
    i.u_l3 IS NOT NULL OR
    ii.i_l1 IS NOT NULL OR
    ii.i_l2 IS NOT NULL OR
    ii.i_l3 IS NOT NULL OR
    c.i_l1_avg IS NOT NULL OR
    c.i_l2_avg IS NOT NULL OR
    c.i_l3_avg IS NOT NULL OR
    q.r_q1_interval IS NOT NULL OR
    q.r_q2_interval IS NOT NULL OR
    q.r_q3_interval IS NOT NULL OR
    q.r_q4_interval IS NOT NULL OR
    m.fa_md IS NOT NULL OR
    m.fa_md_t1 IS NOT NULL OR
    m.fa_md_t2 IS NOT NULL OR
    m.fa_md_t3 IS NOT NULL OR
    m.fa_md_t4 IS NOT NULL
)
ORDER BY a.meter_asset_no, t.tv;