select
    month,
    sum(hr_z1_secs) / 3600.0 as zone_1,
    sum(hr_z2_secs) / 3600.0 as zone_2,
    sum(hr_z3_secs) / 3600.0 as zone_3,
    sum(hr_z4_secs) / 3600.0 as zone_4,
    sum(hr_z5_secs) / 3600.0 as zone_5,
    sum(hr_z6_secs) / 3600.0 as zone_6,
    sum(hr_z7_secs) / 3600.0 as zone_7
from {{ ref('stg_activities') }}
group by month
order by month
