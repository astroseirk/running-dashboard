-- Cleans the raw intervals.icu export: keeps running activities only, casts
-- numeric columns, derives distance/pace, classifies workout type from the
-- title, and drops sub-1km GPS/pause glitches. Mirrors src/data.py exactly
-- so both the dashboard's pandas path and this dbt path agree.

with source as (
    select * from {{ ref('activities') }}
),

typed as (
    select
        id,
        try_cast(start_date_local as timestamp) as start_date_local,
        name,
        type,
        try_cast(moving_time as double) as moving_time_sec,
        try_cast(distance as double) as distance_m,
        try_cast(total_elevation_gain as double) as total_elevation_gain_m,
        try_cast(average_heartrate as double) as average_heartrate,
        try_cast(average_cadence as double) as average_cadence,
        try_cast(icu_training_load as double) as icu_training_load,
        try_cast(icu_rpe as double) as icu_rpe,
        try_cast(icu_fatigue as double) as icu_fatigue,
        try_cast(icu_fitness as double) as icu_fitness,
        try_cast(hr_z1_secs as double) as hr_z1_secs,
        try_cast(hr_z2_secs as double) as hr_z2_secs,
        try_cast(hr_z3_secs as double) as hr_z3_secs,
        try_cast(hr_z4_secs as double) as hr_z4_secs,
        try_cast(hr_z5_secs as double) as hr_z5_secs,
        try_cast(hr_z6_secs as double) as hr_z6_secs,
        try_cast(hr_z7_secs as double) as hr_z7_secs
    from source
    where type = 'Run'
),

filtered as (
    -- below ~1km, GPS/pause glitches (near-zero distance with real elapsed
    -- time, producing absurd paces) are indistinguishable from genuine short
    -- runs, so drop them -- same threshold as src/data.py.
    select * from typed where distance_m >= 1000
)

select
    *,
    distance_m / 1000.0 as distance_km,
    moving_time_sec / 60.0 as moving_time_min,
    (moving_time_sec / 60.0) / (distance_m / 1000.0) as pace_min_per_km,
    date_trunc('week', start_date_local) as week,
    date_trunc('month', start_date_local) as month,
    case
        when lower(name) like '%race%' then 'Race'
        when lower(name) like '%interval%'
            or lower(name) like '%repeat%'
            or lower(name) like '%400%'
            or lower(name) like '%800%'
            or lower(name) like '%vo2%' then 'Intervals'
        when lower(name) like '%tempo%' then 'Tempo'
        when lower(name) like '%long run%' then 'Long Run'
        when lower(name) like '%easy%' then 'Easy Run'
        else 'Other'
    end as workout_type
from filtered
