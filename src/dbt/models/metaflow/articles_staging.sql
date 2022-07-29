WITH latest_etl AS (
    SELECT
        "ETL_ID"
    FROM "EXPLORATION_DB"."HM_RAW"."ARTICLES"
    ORDER BY "ETL_TIMESTAMP" DESC
    LIMIT 1
)
SELECT 
    -- get the columns we need based on NVIDIA previous experiments
    cd."RAW_DATA":"article_id"::INT AS ARTICLE_ID, 
    cd."RAW_DATA":"product_code"::INT AS PRODUCT_CODE, 
    cd."RAW_DATA":"product_type_no"::INT AS PRODUCT_TYPE_NO, 
    cd."RAW_DATA":"product_group_name"::VARCHAR AS PRODUCT_GROUP_NAME, 
    cd."RAW_DATA":"graphical_appearance_no"::INT AS graphical_appearance_no, 
    cd."RAW_DATA":"colour_group_code"::INT AS colour_group_code, 
    cd."RAW_DATA":"perceived_colour_value_id"::INT AS perceived_colour_value_id, 
    cd."RAW_DATA":"perceived_colour_master_id"::INT AS perceived_colour_master_id, 
    cd."RAW_DATA":"department_no"::INT AS department_no,
    cd."RAW_DATA":"index_code"::VARCHAR AS index_code, 
    cd."RAW_DATA":"index_group_no"::INT AS index_group_no, 
    cd."RAW_DATA":"section_no"::INT AS section_no, 
    cd."RAW_DATA":"garment_group_no"::INT AS garment_group_no
FROM 
    "EXPLORATION_DB"."HM_RAW"."ARTICLES" as cd
JOIN 
    latest_etl as le ON le.ETL_ID=cd.ETL_ID