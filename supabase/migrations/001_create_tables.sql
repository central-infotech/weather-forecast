-- ============================================================
-- 001_create_tables.sql
-- Weather Forecast schema: tables, indexes, and RLS policies
-- ============================================================

-- ----------------------------------------------------------
-- 1. forecast_runs - metadata for each forecast execution
-- ----------------------------------------------------------
create table forecast_runs (
  id               uuid        primary key default gen_random_uuid(),
  executed_at      timestamptz not null    default now(),
  initial_data_source text     not null,   -- GFS / ECMWF
  ensemble_size    int4        not null    default 50,
  models_used      text[],                 -- e.g. {'model_a','model_b'}
  status           text        not null    default 'completed'
);

comment on table  forecast_runs is 'One row per forecast execution run.';
comment on column forecast_runs.initial_data_source is 'Upstream data source (GFS or ECMWF).';

-- ----------------------------------------------------------
-- 2. forecasts - individual location/date predictions
-- ----------------------------------------------------------
create table forecasts (
  id                uuid   primary key default gen_random_uuid(),
  target_date       date   not null,
  location          text   not null,
  latitude          float8 not null,
  longitude         float8 not null,
  weather           text   not null,        -- e.g. 晴れ / 曇り / 雨 / 雪
  temp_max          float4,
  temp_min          float4,
  precipitation_prob int2,
  confidence        float4,
  model_agreement   float4,
  humidity          float4,
  wind_speed        float4,
  pressure          float4,
  created_at        timestamptz not null default now(),
  run_id            uuid   not null references forecast_runs (id) on delete cascade
);

comment on table forecasts is 'Weather forecast per location and target date.';

-- ----------------------------------------------------------
-- 3. calendar_sync_log - tracks Google Calendar sync status
-- ----------------------------------------------------------
create table calendar_sync_log (
  id              uuid        primary key default gen_random_uuid(),
  user_id         uuid        not null references auth.users (id),
  forecast_id     uuid        not null references forecasts (id) on delete cascade,
  google_event_id text        not null,
  synced_at       timestamptz not null default now(),
  status          text        not null default 'synced'
);

comment on table calendar_sync_log is 'Audit log linking forecasts to Google Calendar events.';

-- ============================================================
-- Indexes
-- ============================================================

-- Fast lookup by date + location (most common query pattern)
create index idx_forecasts_target_date_location
  on forecasts (target_date, location);

-- Foreign-key lookup
create index idx_forecasts_run_id
  on forecasts (run_id);

-- Per-user queries
create index idx_calendar_sync_log_user_id
  on calendar_sync_log (user_id);

-- Per-forecast queries
create index idx_calendar_sync_log_forecast_id
  on calendar_sync_log (forecast_id);

-- ============================================================
-- Row Level Security
-- ============================================================

-- Enable RLS on every table
alter table forecast_runs     enable row level security;
alter table forecasts         enable row level security;
alter table calendar_sync_log enable row level security;

-- ----------------------------------------------------------
-- forecast_runs: read-only for authenticated users
-- ----------------------------------------------------------
create policy "Authenticated users can read forecast_runs"
  on forecast_runs
  for select
  to authenticated
  using (true);

-- ----------------------------------------------------------
-- forecasts: read-only for authenticated users
-- ----------------------------------------------------------
create policy "Authenticated users can read forecasts"
  on forecasts
  for select
  to authenticated
  using (true);

-- ----------------------------------------------------------
-- calendar_sync_log: users can only access their own rows
-- ----------------------------------------------------------
create policy "Users can read own sync logs"
  on calendar_sync_log
  for select
  to authenticated
  using (user_id = auth.uid());

create policy "Users can insert own sync logs"
  on calendar_sync_log
  for insert
  to authenticated
  with check (user_id = auth.uid());

create policy "Users can update own sync logs"
  on calendar_sync_log
  for update
  to authenticated
  using (user_id = auth.uid())
  with check (user_id = auth.uid());
