SELECT 
    i_s.S3_URL AS S3_URL,
    cd.*
FROM 
    {{ ref('articles_staging') }} as cd
LEFT JOIN
    {{ ref('images_staging') }} AS i_s
    ON i_s.ARTICLE_ID=cd.ARTICLE_ID