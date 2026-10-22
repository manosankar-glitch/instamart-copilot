
-- Q1. Overall order status summary
SELECT
    status,
    COUNT(*) AS order_count,
    ROUND(SUM(order_amount), 2) AS total_amount
FROM orders
GROUP BY status
ORDER BY status;


-- Q2. Revenue by product category
SELECT
    p.category,
    COUNT(*) AS completed_orders,
    ROUND(SUM(o.order_amount), 2) AS revenue
FROM orders o
JOIN products p ON o.product_id = p.product_id
WHERE o.status = 'completed'
GROUP BY p.category
ORDER BY revenue DESC;


-- Q3. Revenue by city
SELECT
    c.city,
    COUNT(*) AS completed_orders,
    ROUND(SUM(o.order_amount), 2) AS revenue
FROM orders o
JOIN customers c ON o.customer_id = c.customer_id
WHERE o.status = 'completed'
GROUP BY c.city
ORDER BY revenue DESC;


-- Q4. Top 10 customers by completed-order spend
SELECT
    c.customer_id,
    c.city,
    COUNT(*) AS completed_orders,
    ROUND(SUM(o.order_amount), 2) AS total_spend
FROM orders o
JOIN customers c ON o.customer_id = c.customer_id
WHERE o.status = 'completed'
GROUP BY c.customer_id, c.city
ORDER BY total_spend DESC
LIMIT 10;


-- Q5. Average delivery time by product category
SELECT
    p.category,
    COUNT(*) AS completed_orders,
    ROUND(AVG(o.delivery_minutes), 2) AS avg_delivery_minutes
FROM orders o
JOIN products p ON o.product_id = p.product_id
WHERE o.status = 'completed'
GROUP BY p.category
ORDER BY avg_delivery_minutes DESC;


-- Q6. Monthly completed-order revenue
SELECT
    substr(order_ts, 1, 7) AS order_month,
    COUNT(*) AS completed_orders,
    ROUND(SUM(order_amount), 2) AS revenue
FROM orders
WHERE status = 'completed'
GROUP BY substr(order_ts, 1, 7)
ORDER BY order_month;


-- Q7. Cancellation rate by city
SELECT
    c.city,
    COUNT(*) AS total_orders,
    SUM(CASE WHEN o.status = 'cancelled' THEN 1 ELSE 0 END)
        AS cancelled_orders,
    ROUND(
        100.0 * SUM(CASE WHEN o.status = 'cancelled' THEN 1 ELSE 0 END)
        / COUNT(*),
        2
    ) AS cancellation_rate_pct
FROM orders o
JOIN customers c ON o.customer_id = c.customer_id
GROUP BY c.city
ORDER BY cancellation_rate_pct DESC;
