
WITH latest_etl AS (
    SELECT
        "ETL_ID"
    FROM "EXPLORATION_DB"."HM_RAW"."TRANSACTIONS_TRAIN"
    ORDER BY "ETL_TIMESTAMP" DESC
    LIMIT 1
)
SELECT 
    -- get the columns we need based on NVIDIA previous experiments
    cd."RAW_DATA":"article_id"::INT AS ARTICLE_ID, 
    cd."RAW_DATA":"customer_id"::VARCHAR AS customer_id,
    cd."RAW_DATA":"price"::FLOAT AS price,
    cd."RAW_DATA":"sales_channel_id"::INT  as sales_channel_id,
    cd."RAW_DATA":"t_dat"::DATETIME as t_dat
FROM 
     "EXPLORATION_DB"."HM_RAW"."TRANSACTIONS_TRAIN" as cd
JOIN 
    latest_etl as le ON le.ETL_ID=cd.ETL_ID
ORDER BY t_dat ASC

