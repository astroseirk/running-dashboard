with bucketed as (
    select
        *,
        case
            when distance_km between 4.8 and 5.3 then '5K'
            when distance_km between 9.7 and 10.3 then '10K'
            when distance_km between 20.5 and 21.6 then 'Half Marathon'
            when distance_km between 41.5 and 43.5 then 'Marathon'
        end as pb_distance
    from {{ ref('stg_activities') }}
    where pace_min_per_km is not null
),

ranked as (
    select
        *,
        row_number() over (partition by pb_distance order by pace_min_per_km asc) as rn
    from bucketed
    where pb_distance is not null
)

select
    pb_distance as distance,
    cast(start_date_local as date) as date,
    name as run,
    pace_min_per_km,
    moving_time_min
from ranked
where rn = 1
order by
    case pb_distance
        when '5K' then 1
        when '10K' then 2
        when 'Half Marathon' then 3
        when 'Marathon' then 4
    end
