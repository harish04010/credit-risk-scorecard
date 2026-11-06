-- Roll-rate loss forecasting, the classic method for card portfolios.
--
-- Each account's repayment status is observed for six months. The status
-- codes aren't consistent across months: code 1 ("one month late") appears
-- 3,688 times in September and almost never before it, where the codes jump
-- from 0 to 2. So codes 1 and 2 are merged into one "1-2" bucket everywhere,
-- and every month-to-month step compares like with like.
--
-- Buckets: current (paid in full, revolving or no balance), 1-2, 3-4 and 5+
-- months past due. 5+ is treated as the charge-off stage: card issuers
-- charge off at 180 days.
--
-- Run from the project root:  python -m src.roll

create or replace table accounts as
select * from read_csv('results/sql_input.csv', header = true);

create or replace table status_long as
with long as (
    select ID as account_id,
           unnest([6, 5, 4, 3, 2, 1]) as month_index,          -- 6 = April ... 1 = September
           unnest([PAY_6, PAY_5, PAY_4, PAY_3, PAY_2, PAY_1]) as raw_status,
           unnest([BILL_AMT6, BILL_AMT5, BILL_AMT4, BILL_AMT3, BILL_AMT2, BILL_AMT1]) as balance
    from accounts
)
select account_id, month_index, greatest(balance, 0) as balance,
       case when raw_status <= 0 then 'current'
            when raw_status <= 2 then '1-2'
            when raw_status <= 4 then '3-4'
            else '5+' end as bucket
from long;

-- Balance-weighted roll rates: of the dollars in a bucket one month, what
-- share sits in each bucket the next? Dollars, not accounts, because losses are dollars.
create or replace table roll_matrix as
with t as (
    select a.bucket as from_bucket, b.bucket as to_bucket,
           count(*) as accounts, sum(a.balance) as balance
    from status_long a
    join status_long b on a.account_id = b.account_id and b.month_index = a.month_index - 1
    group by all
)
select from_bucket, to_bucket, accounts, balance,
       accounts / sum(accounts) over (partition by from_bucket) as account_rate,
       balance / nullif(sum(balance) over (partition by from_bucket), 0) as dollar_rate
from t;

create or replace table book as
select bucket, count(*) as accounts, sum(balance) as balance
from status_long where month_index = 1
group by bucket;
