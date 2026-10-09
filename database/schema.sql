PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS events (
    event_id TEXT PRIMARY KEY,
    season INTEGER NOT NULL,
    round INTEGER NOT NULL,
    race TEXT NOT NULL,
    event_name TEXT,
    official_event_name TEXT,
    circuit TEXT,
    country TEXT,
    location TEXT,
    event_date TEXT,
    race_laps INTEGER,
    season_status TEXT NOT NULL DEFAULT 'historical',
    UNIQUE(season, round, race)
);

CREATE TABLE IF NOT EXISTS laps (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    event_id TEXT NOT NULL REFERENCES events(event_id),
    Season INTEGER NOT NULL,
    Round INTEGER,
    Race TEXT NOT NULL,
    EventName TEXT,
    OfficialEventName TEXT,
    Circuit TEXT,
    Country TEXT,
    Location TEXT,
    EventDate TEXT,
    RaceLaps INTEGER,
    Driver TEXT NOT NULL,
    DriverName TEXT,
    DriverNumber TEXT,
    Team TEXT,
    Compound TEXT NOT NULL,
    LapNumber INTEGER NOT NULL,
    TyreLife REAL,
    Stint REAL,
    LapTime REAL,
    IsAccurate TEXT,
    Deleted TEXT,
    TrackStatus TEXT,
    FreshTyre TEXT,
    PitInTime TEXT,
    PitOutTime TEXT,
    Position REAL
);

CREATE TABLE IF NOT EXISTS event_drivers (
    event_id TEXT NOT NULL REFERENCES events(event_id),
    driver_code TEXT NOT NULL,
    driver_name TEXT,
    driver_number TEXT,
    team TEXT,
    PRIMARY KEY(event_id, driver_code)
);

CREATE TABLE IF NOT EXISTS pit_transitions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    season INTEGER,
    round INTEGER,
    race TEXT,
    circuit TEXT,
    team TEXT,
    driver TEXT,
    pit_lap INTEGER,
    current_lap INTEGER,
    race_laps INTEGER,
    race_progress REAL,
    previous_tyre TEXT,
    previous_tyre_age REAL,
    new_tyre TEXT,
    average_lap_time_before REAL,
    average_lap_time_after REAL
);

-- Pre-2018 source records are retained for circuit summaries only. They never
-- enter the driver's model-training or live-prediction lap query.
CREATE TABLE IF NOT EXISTS circuit_history_laps (
    event_id TEXT NOT NULL REFERENCES events(event_id) ON DELETE CASCADE,
    season INTEGER NOT NULL,
    round INTEGER NOT NULL,
    race TEXT NOT NULL,
    circuit TEXT,
    event_date TEXT,
    race_laps INTEGER,
    driver TEXT NOT NULL,
    lap_number INTEGER NOT NULL,
    lap_time REAL,
    position REAL,
    compound TEXT NOT NULL DEFAULT 'UNKNOWN',
    tyre_life REAL,
    stint INTEGER,
    source TEXT NOT NULL,
    PRIMARY KEY(event_id, driver, lap_number)
);

CREATE TABLE IF NOT EXISTS circuit_history_pit_stops (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    event_id TEXT NOT NULL REFERENCES events(event_id) ON DELETE CASCADE,
    driver TEXT NOT NULL,
    pit_lap INTEGER NOT NULL,
    stop_number INTEGER,
    duration_seconds REAL,
    source TEXT NOT NULL,
    UNIQUE(event_id, driver, stop_number, pit_lap)
);

CREATE TABLE IF NOT EXISTS tyre_performance (
    compound TEXT PRIMARY KEY,
    data_points INTEGER,
    compound_offset REAL,
    offset_source TEXT,
    degradation_rate REAL,
    degradation_source TEXT,
    stints_used INTEGER,
    degradation_std REAL,
    r_squared REAL
);

CREATE TABLE IF NOT EXISTS tyre_degradation (
    compound TEXT NOT NULL,
    tyre_life REAL NOT NULL,
    lap_time REAL,
    PRIMARY KEY(compound, tyre_life)
);

CREATE TABLE IF NOT EXISTS weather_observations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    event_id TEXT REFERENCES events(event_id),
    season INTEGER,
    race TEXT,
    circuit TEXT,
    country TEXT,
    observed_at TEXT,
    temperature REAL,
    precipitation REAL,
    rain REAL,
    precipitation_probability REAL,
    wind_speed REAL,
    wind_direction REAL,
    weather_code INTEGER,
    weather TEXT,
    condition TEXT,
    impact TEXT,
    source TEXT
);

CREATE TABLE IF NOT EXISTS weather_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    event_id TEXT NOT NULL REFERENCES events(event_id),
    kind TEXT NOT NULL,
    retrieved_at TEXT NOT NULL,
    source TEXT,
    payload_json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS data_quality_issues (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    checked_at TEXT NOT NULL,
    issue_code TEXT NOT NULL,
    issue_count INTEGER NOT NULL,
    details TEXT
);

CREATE TABLE IF NOT EXISTS app_metadata (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS historical_feature_cache (
    cache_key TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    payload_json TEXT NOT NULL
);

-- Cumulative fingerprints are refreshed during offline imports. The dashboard
-- can resolve cutoff-safe bundle versions with one indexed lookup instead of
-- scanning historical laps during a prediction request.
CREATE TABLE IF NOT EXISTS event_data_versions (
    event_date TEXT PRIMARY KEY,
    cumulative_version TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_events_season ON events(season, event_date, round);
CREATE INDEX IF NOT EXISTS idx_events_circuit ON events(circuit, season);
CREATE INDEX IF NOT EXISTS idx_laps_event_driver ON laps(event_id, Driver, LapNumber);
CREATE INDEX IF NOT EXISTS idx_laps_date_circuit ON laps(EventDate, Circuit);
CREATE INDEX IF NOT EXISTS idx_laps_compound_age ON laps(Compound, TyreLife);
CREATE INDEX IF NOT EXISTS idx_pit_transition_event ON pit_transitions(season, race, driver);
CREATE INDEX IF NOT EXISTS idx_weather_event_date ON weather_observations(event_id, observed_at);
CREATE INDEX IF NOT EXISTS idx_laps_cutoff_context ON laps(EventDate, Season, Circuit, Driver, Compound);
CREATE INDEX IF NOT EXISTS idx_laps_circuit_compound_age ON laps(Circuit, Compound, TyreLife, EventDate);
CREATE INDEX IF NOT EXISTS idx_event_data_versions_date ON event_data_versions(event_date);
CREATE INDEX IF NOT EXISTS idx_pit_season_round_race ON pit_transitions(season, round, race);
CREATE INDEX IF NOT EXISTS idx_pit_circuit_season ON pit_transitions(circuit, season, pit_lap);
CREATE INDEX IF NOT EXISTS idx_circuit_history_event_driver ON circuit_history_laps(event_id, driver, lap_number);
CREATE INDEX IF NOT EXISTS idx_circuit_history_date_circuit ON circuit_history_laps(event_date, circuit);
CREATE INDEX IF NOT EXISTS idx_circuit_history_pits_event ON circuit_history_pit_stops(event_id, driver, pit_lap);
