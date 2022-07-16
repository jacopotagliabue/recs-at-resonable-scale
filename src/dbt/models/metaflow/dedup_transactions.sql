SELECT 
    ARTICLE_ID, 
    customer_id,
    last_value(price) over (partition by ARTICLE_ID, CUSTOMER_ID order by t_dat ASC) as price,
    last_value(sales_channel_id) over (partition by ARTICLE_ID, CUSTOMER_ID order by t_dat) as sales_channel_id,
    last_value(t_dat) over (partition by ARTICLE_ID, CUSTOMER_ID order by t_dat ASC) as t_dat
FROM 
    {{ ref('transactions_staging') }} as ts
GROUP BY 
   ARTICLE_ID, 
   customer_id,
   price,
   sales_channel_id,
   t_dat
ORDER BY 
    ARTICLE_ID, 
    customer_id
