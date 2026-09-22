-- Easy-effort cohort is RPE <= 3, not the name-based "Easy Run" tag -- that
-- label only exists on a small, recent subset of runs. See mart docs.

with easy as (
    select
        *,
        60.0 / pace_min_per_km as speed_kmh
    from {{ ref('stg_activities') }}
    where icu_rpe <= 3
      and average_heartrate is not null
      and pace_min_per_km is not null
),

with_efficiency as (
    select
        *,
        speed_kmh / average_heartrate as efficiency
    from easy
)

select
    month,
    avg(average_heartrate) as avg_hr,
    avg(pace_min_per_km) as avg_pace,
    avg(efficiency) as efficiency,
    count(*) as runs
from with_efficiency
group by month
order by month
