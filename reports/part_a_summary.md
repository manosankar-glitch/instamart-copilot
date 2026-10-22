# Part A — Data Audit, Cleaning and SQL Analysis

## Raw data quality audit

| Check | Result |
|---|---:|
| Raw order rows | 13084 |
| Exact duplicate order rows | 119 |
| Pending raw order rows | 656 |
| Distinct raw order IDs | 12309 |
| Nonnumeric or missing order amounts | 120 |
| Negative delivery durations | 106 |
| Blank delivery durations | 2541 |
| Blank customer ages | 61 |
| Customer ages outside 10-100 | 6 |
| Distinct raw city spellings | 24 |
| Distinct nonblank raw gender spellings | 10 |
| Blank customer genders | 30 |

## Clean dataset validation summary

| Metric | Result |
|---|---:|
| Clean customers | 1200 |
| Clean products | 40 |
| Clean orders | 12309 |
| Completed orders | 11110 |
| Cancelled orders | 1199 |
| Completed revenue (INR) | 2998668.80 |
| Missing order amounts | 0 |
| Completed orders missing delivery time | 0 |
| Cancelled orders missing delivery time | 1199 |

## Cleaning decisions

- Removed exact duplicate order rows.
- Converted order timestamps and numeric fields to usable types.
- Kept the latest timestamped record for each order ID.
- Normalized city names and mapped gender values to male, female, or unknown.
- Treated customer ages outside 10–100 as missing and filled missing ages using city medians.
- Recovered missing order amounts using product price, quantity, and discount.
- Treated negative delivery durations as missing.
- Filled missing delivery times for completed orders using the completed-order median for each product category.
- Kept delivery durations missing for cancelled orders.

## SQL findings

- Completed-order revenue was INR 2,998,668.80 across 11,110 completed orders.
- Electronics had the highest completed-order revenue among product categories.
- Mumbai had the highest cancellation rate among the six cities.
- Personal Care had the highest average delivery time among the listed product categories.
- Monthly completed-order revenue varied across April–September 2025.

These are descriptive findings from the supplied dataset; they do not establish why revenue or cancellation patterns differ.

## Reproduction

Run `python build_database.py`, then `python validate_data.py`, then `python run_sql_queries.py`, and finally `python part_a_analysis.py`.
