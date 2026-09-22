select
    week,
    sum(distance_km) as distance_km,
    count(*) as runs,
    sum(icu_training_load) as training_load,
    avg(pace_min_per_km) as avg_pace
from {{ ref('stg_activities') }}
group by week
order by week
