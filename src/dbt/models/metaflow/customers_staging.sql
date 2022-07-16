
WITH latest_etl AS (
    SELECT
        "ETL_ID"
    FROM "EXPLORATION_DB"."HM_RAW"."CUSTOMERS"
    ORDER BY "ETL_TIMESTAMP" DESC
    LIMIT 1
)
SELECT 
    -- get the columns we need based on NVIDIA previous experiments
    COALESCE(NULLIF(cd."RAW_DATA":"Active",''),0.0)::FLOAT AS ACTIVE, 
    COALESCE(NULLIF(cd."RAW_DATA":"FN",''), 0.0)::FLOAT AS FN,
    COALESCE(NULLIF(cd."RAW_DATA":"age",''), 0.0)::FLOAT AS AGE,
    cd."RAW_DATA":"club_member_status"::VARCHAR AS club_member_status,
    cd."RAW_DATA":"customer_id"::VARCHAR AS customer_id,
    cd."RAW_DATA":"fashion_news_frequency"::VARCHAR AS fashion_news_frequency,
    cd."RAW_DATA":"postal_code"::VARCHAR AS postal_code
FROM 
    "EXPLORATION_DB"."HM_RAW"."CUSTOMERS" as cd
JOIN 
    latest_etl as le ON le.ETL_ID=cd.ETL_ID
ORDER BY CUSTOMER_ID
