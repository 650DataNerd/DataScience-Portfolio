#Create Database
CREATE DATABASE ecommerce_db;
#use the db
USE ecommerce_db;

#verify imported data
SELECT * FROM cleaned_ecommerce_dataset LIMIT 10;

#check rows and columns
SELECT COUNT(*) AS total_rows FROM cleaned_ecommerce_dataset;
SHOW COLUMNS FROM cleaned_ecommerce_dataset;

#summary statistics
SELECT
	MIN(actual_price) AS min_price,
    MAX(actual_price) AS max_price,
    AVG(actual_price) AS avg_price,
    MIN(selling_price) AS min_selling_price,
    MAX(selling_price) AS max_selling_price,
     AVG(selling_price) AS avg_selling_price,
    MIN(discount) AS min_discount,
    MAX(discount) AS max_discount,
    AVG(discount) AS avg_discount,
    MIN(average_rating) AS min_rating,
    MAX(average_rating) AS max_rating,
    AVG(average_rating) AS avg_rating
FROM cleaned_ecommerce_dataset;

#Product Distribution by Category
SELECT 
    category, 
    COUNT(*) AS product_count 
FROM cleaned_ecommerce_dataset
GROUP BY category 
ORDER BY product_count DESC;

#Popular Brands
SELECT
brand, 
    COUNT(*) AS product_count 
FROM cleaned_ecommerce_dataset
GROUP BY brand 
ORDER BY product_count DESC
LIMIT 10;

#correlation insights
#average discount by category
SELECT 
    category, 
    AVG(discount) AS avg_discount 
FROM cleaned_ecommerce_dataset
GROUP BY category 
ORDER BY avg_discount DESC;

#average rating by discount ranges
SELECT 
    CASE 
        WHEN discount < 10 THEN 'Low (<10%)'
        WHEN discount BETWEEN 10 AND 30 THEN 'Medium (10-30%)'
        WHEN discount > 30 THEN 'High (>30%)'
    END AS discount_range,
    AVG(average_rating) AS avg_rating
FROM cleaned_ecommerce_dataset
GROUP BY discount_range
ORDER BY avg_rating DESC;

#top 5products with highest discounts
SELECT 
    title, 
    actual_price, 
    selling_price, 
    discount 
FROM cleaned_ecommerce_dataset
ORDER BY discount DESC 
LIMIT 5;

#TREND
SELECT DISTINCT crawled_at 
FROM cleaned_ecommerce_dataset;

DESCRIBE cleaned_ecommerce_dataset;
#backup table
CREATE TABLE cleaned_ecommerce_dataset_backup AS
SELECT * FROM cleaned_ecommerce_dataset;

#alter table to DATETIME
#add new column
ALTER TABLE cleaned_ecommerce_dataset 
ADD crawled_at_datetime DATETIME;

#populate new column

SET SQL_SAFE_UPDATES = 0;

UPDATE cleaned_ecommerce_dataset
SET crawled_at_datetime = STR_TO_DATE(crawled_at, '%d/%m/%Y, %H:%i:%s');

SET SQL_SAFE_UPDATES = 1;

#MOST REVENUE
SELECT
category,
SUM(selling_price) AS total_revenue
FROM cleaned_ecommerce_dataset
GROUP BY category
ORDER BY total_revenue desc
LIMIT 10;

#Underperforming categories
SELECT 
    title, 
    brand, 
    category, 
    actual_price, 
    selling_price, 
    discount
FROM cleaned_ecommerce_dataset
WHERE selling_price < (actual_price * 0.5)
ORDER BY actual_price DESC
LIMIT 10;

#best deals(hih discounts + ratings
SELECT 
    title, 
    brand, 
    category, 
    average_rating, 
    discount
FROM cleaned_ecommerce_dataset
WHERE average_rating > 4.5
ORDER BY discount DESC
LIMIT 10;

#top sellers
SELECT 
    seller, 
    COUNT(*) AS total_products, 
    SUM(selling_price) AS total_revenue
FROM cleaned_ecommerce_dataset
GROUP BY seller
ORDER BY total_revenue DESC
LIMIT 10;

#get multivariable revenue analysis btn category, revenue and brand
SELECT 
    category, 
    brand, 
    SUM(selling_price) AS total_revenue
FROM cleaned_ecommerce_dataset
GROUP BY category, brand
ORDER BY total_revenue DESC
LIMIT 10;

#revenue contribution by sub-category
SELECT 
    sub_category, 
    COUNT(*) AS product_count, 
    SUM(selling_price) AS total_revenue, 
    AVG(selling_price) AS avg_price
FROM cleaned_ecommerce_dataset
GROUP BY sub_category
ORDER BY total_revenue DESC
LIMIT 10;

