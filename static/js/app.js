const $ = (selector) => document.querySelector(selector);
const state = {
  event: null, context: null, driver: null, lastPrediction: null,
  seasons: [], seasonStates: [], activeSeason: null, seasonContext: null, raceOptions: [], drivers: [], metrics: null, quality: null,
  weather: null, modelBundleAvailable: false, modelBundleStatus: 'UNKNOWN', driverSelectionConfirmed: false,
  predictionState: 'UNKNOWN', eventStateTimer: null, eventStatePending: false, predictionBusy: false,
  activePreset: 'choose', eventRequest: 0, raceOptionsRequest: 0,
};
const TYRE = { SOFT: '#ed5966', MEDIUM: '#efcc59', HARD: '#e5e8eb', INTERMEDIATE: '#71d09f', WET: '#5aaee1' };
const SVG_NS = 'http://www.w3.org/2000/svg';

async function api(url, options = {}) {
  let response;
  try {
    response = await fetch(url, { headers: { 'Content-Type': 'application/json' }, ...options });
  } catch {
    throw new Error('The local strategy service could not be reached. Check that Flask is running.');
  }
  let data = {};
  try { data = await response.json(); }
  catch { throw new Error('The service returned an unreadable response. Retry the request.'); }
  if (!response.ok) throw new Error(data.error || 'The request could not be completed.');
  return data;
}

function escapeHtml(value) {
  return String(value ?? '').replace(/[&<>"']/g, (char) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[char]));
}
function fmtDate(value) {
  if (!value) return 'Date not available';
  const date = new Date(`${String(value).slice(0, 10)}T12:00:00`);
  return Number.isNaN(+date) ? String(value) : new Intl.DateTimeFormat(undefined, { day: 'numeric', month: 'short', year: 'numeric' }).format(date);
}
function pct(value, digits = 0) {
  const number = Number(value);
  return value == null || value === '' || !Number.isFinite(number) ? '—' : `${(number * 100).toFixed(digits)}%`;
}
function numeric(value) { const number = Number(value); return value == null || value === '' || !Number.isFinite(number) ? null : number; }
function count(value) { const number = numeric(value); return number == null ? '—' : Math.round(number).toLocaleString(); }
function cleanError(error, fallback) {
  const message = String(error?.message || '');
  if (!message || /WinError|HTTPSConnectionPool|Traceback|TypeError|ReferenceError|SyntaxError/i.test(message)) return fallback;
  return message;
}
function setText(selector, value) { const node = $(selector); if (node) node.textContent = value == null ? '' : String(value); }
function setOption(select, value, label) { const option = document.createElement('option'); option.value = value; option.textContent = label; select.append(option); }
function seasonDataSummary(context = {}) {
  const availability = context.data_availability || {};
  const historical = availability.historical_data ? 'AVAILABLE' : 'UNAVAILABLE';
  const observations = count(availability.current_season_observations ?? context.observed_races ?? 0);
  const futureContext = availability.future_season_pre_race_context ? 'FUTURE-SEASON PRE-RACE CONTEXT' : 'SEASON CONTEXT';
  return `CALENDAR DATA · HISTORICAL DATA ${historical} · CURRENT-SEASON OBSERVATIONS ${observations} · ${futureContext}`;
}
function setError(message) {
  let current = message || '';
  if (state.modelBundleStatus === 'READY' && /No pre-trained cutoff-safe model bundle is ready/i.test(String(current))) current = '';
  $('#error-message').textContent = current;
  $('#error-message').hidden = !current;
}

function setModelBundleStatus(bundle = {}) {
  const declared = String(bundle.status || bundle.bundle_status || bundle.state || '').trim().toUpperCase();
  const next = declared.includes('BUILDING') ? 'BUILDING'
    : declared.includes('FAILED') ? 'FAILED'
      : declared.includes('NOT_REQUIRED') ? 'NOT_REQUIRED'
        : declared.includes('STALE') ? 'STALE'
    : declared.includes('READY') ? 'READY'
      : declared.includes('MISSING') || declared.includes('UNAVAILABLE') ? 'MISSING'
        : bundle.stale ? 'STALE' : bundle.available ? 'READY' : 'MISSING';
  if (state.modelBundleStatus !== next) setError('');
  state.modelBundleStatus = next;
  state.modelBundleAvailable = next === 'READY';
}

function setBusy(busy) {
  state.predictionBusy = Boolean(busy);
  const button = $('#predict-button');
  updatePredictionControls();
  button.querySelector('span:last-child').textContent = busy ? 'SCORING FEASIBLE STRATEGIES…' : 'PREDICT PIT STRATEGY';
  $('#analysis-status').hidden = !busy;
  $('#analysis-status').textContent = busy ? 'Loading the cutoff-safe bundle and evaluating legal tyre plans…' : '';
  if (busy) setText('#prediction-empty-note', 'The selected driver is set. The offline bundle is evaluating feasible plans against the event cutoff.');
}

function predictionLockMessage() {
  if (state.predictionState === 'IN_PROGRESS') return 'RACE IN PROGRESS — PRE-RACE PREDICTION LOCKED';
  if (state.predictionState === 'COMPLETED') return 'RACE COMPLETED — PRE-RACE MODE CLOSED';
  if (!['UPCOMING', 'PRE_RACE'].includes(state.predictionState)) return 'EVENT STATE UNAVAILABLE — REFRESH EVENT CONTEXT';
  if (state.modelBundleStatus === 'READY') return 'CUTOFF-SAFE MODEL READY · PREDICTION AVAILABLE';
  if (state.modelBundleStatus === 'BUILDING') return 'MODEL UPDATE IN PROGRESS';
  if (state.modelBundleStatus === 'FAILED') return `MODEL BUNDLE BUILD FAILED · ${state.context?.model_bundle?.reason || 'Build reason unavailable.'}`;
  if (state.modelBundleStatus === 'NOT_REQUIRED') return 'PRE-RACE PREDICTION NOT REQUIRED';
  if (state.modelBundleStatus === 'STALE') {
    const instruction = state.context?.model_bundle?.instruction || 'Run python scripts/train_model.py --asof-event <event_id>';
    return `PREDICTION MODEL BUNDLE STALE · ${instruction}`;
  }
  if (state.modelBundleStatus === 'MISSING') {
    const instruction = state.context?.model_bundle?.instruction || 'Run python scripts/train_model.py --asof-event <event_id>';
    return `PREDICTION MODEL NOT READY · ${instruction}`;
  }
  return 'CHECKING CUTOFF-SAFE MODEL STATUS';
}

function updatePredictionControls() {
  const phaseLocked = ['IN_PROGRESS', 'COMPLETED'].includes(state.predictionState);
  const validDriver = state.drivers.some((row) => String(row.driver_code).toUpperCase() === String(state.driver || '').toUpperCase());
  const stateKnown = ['UPCOMING', 'PRE_RACE', 'IN_PROGRESS', 'COMPLETED'].includes(state.predictionState);
  const button = $('#predict-button');
  if (button) button.disabled = Boolean(state.predictionBusy || !state.event || !stateKnown || phaseLocked || state.modelBundleStatus !== 'READY' || !validDriver);

  const status = $('#focus-model-status');
  if (status) {
    const text = phaseLocked
      ? (state.predictionState === 'IN_PROGRESS' ? 'RACE IN PROGRESS' : 'RACE COMPLETED')
      : state.modelBundleStatus === 'READY' ? 'READY · OFFLINE BUNDLE'
        : state.modelBundleStatus === 'BUILDING' ? 'BUILDING · MODEL UPDATE IN PROGRESS'
          : state.modelBundleStatus === 'FAILED' ? 'MODEL BUNDLE BUILD FAILED'
            : state.modelBundleStatus === 'NOT_REQUIRED' ? 'NOT REQUIRED'
        : state.modelBundleStatus === 'STALE' ? 'STALE · OFFLINE BUNDLE'
          : state.modelBundleStatus === 'MISSING' ? 'BUNDLE UNAVAILABLE' : 'CHECKING BUNDLE';
    status.textContent = text;
    status.className = phaseLocked ? 'model-locked' : state.modelBundleStatus === 'READY' ? 'model-ready' : 'model-missing';
  }
  const message = predictionLockMessage();
  setText('#prediction-availability', message);
  const note = $('#prediction-availability');
  if (note) note.className = `console-note prediction-availability ${phaseLocked ? 'locked' : state.modelBundleStatus === 'READY' ? 'ready' : 'missing'}`;
  const lockedState = $('.locked-state');
  if (lockedState) lockedState.textContent = phaseLocked ? message : state.modelBundleStatus === 'READY' ? (validDriver ? 'READY TO RUN · PREDICTION IS MANUAL' : 'DRIVER SELECTION REQUIRED') : message;
  if (!state.predictionBusy) {
    const emptyNote = phaseLocked ? message
      : state.modelBundleStatus !== 'READY' ? message
        : validDriver ? 'Driver context is loaded. Start a prediction only when you press the strategy button.'
          : 'Circuit analysis is ready. Select a valid driver profile to enable pre-race prediction.';
    setText('#prediction-empty-note', emptyNote);
  }
  const bundle = state.context?.model_bundle || {};
  const bundleStatus = state.modelBundleStatus === 'READY' ? `READY · ${bundle.cutoff_utc || 'CUTOFF-SAFE'}`
    : state.modelBundleStatus === 'BUILDING' ? 'BUILDING · MODEL UPDATE IN PROGRESS'
      : state.modelBundleStatus === 'FAILED' ? `FAILED · ${bundle.reason || 'Build reason unavailable.'}`
        : state.modelBundleStatus === 'NOT_REQUIRED' ? 'NOT REQUIRED · PRE-RACE MODE CLOSED'
    : state.modelBundleStatus === 'STALE' ? `STALE · ${bundle.instruction || 'offline rebuild required'}`
      : state.modelBundleStatus === 'MISSING' ? `BUNDLE UNAVAILABLE · ${bundle.instruction || 'offline build command unavailable'}`
        : 'Checking bundle status';
  setText('#bundle-status', bundleStatus);
}

function stopEventStatePolling() {
  if (state.eventStateTimer) window.clearTimeout(state.eventStateTimer);
  state.eventStateTimer = null;
  state.eventStatePending = false;
}

function monitorSelectedEventState() {
  stopEventStatePolling();
  const eventId = state.event?.event_id;
  if (!eventId) return;
  const contextRequest = state.eventRequest;
  const isCurrentContext = () => state.eventRequest === contextRequest && state.event?.event_id === eventId;
  const poll = async () => {
    if (state.eventStatePending || !isCurrentContext()) return;
    state.eventStatePending = true;
    try {
      const [current, bundle, sync] = await Promise.all([
        api(`/api/event-state/${encodeURIComponent(eventId)}`),
        api(`/api/bundle-status/${encodeURIComponent(eventId)}`),
        api('/api/sync-status'),
      ]);
      if (isCurrentContext()) {
        const previous = state.predictionState;
        state.predictionState = String(current.state || 'UNKNOWN').toUpperCase();
        if (state.context) state.context.model_bundle = bundle;
        setModelBundleStatus(bundle);
        if (current.race_start_time_utc) state.event.race_start_time_utc = current.race_start_time_utc;
        if (sync.sync_status === 'running') {
          setText('#data-updated', `DATA SYNC RUNNING · ${count(sync.bundles_building)} BUNDLES BUILDING`);
        } else if (sync.sync_status === 'partial') {
          setText('#data-updated', `DATA SYNC PARTIAL · ${count(sync.bundles_failed)} BUNDLES FAILED`);
        } else if (sync.sync_status === 'complete' && sync.last_successful_sync) {
          setText('#data-updated', `LOCAL DATA SYNCED · ${fmtDate(sync.last_successful_sync)}`);
        }
        updatePredictionControls();
        if (state.predictionState === 'IN_PROGRESS') {
          setText('#focus-label', 'RACE IN PROGRESS');
          setText('#focus-status', 'RACE IN PROGRESS');
          setText('#focus-weekend', `RACE IN PROGRESS · ROUND ${state.event.round ?? '—'}`);
        } else if (state.predictionState === 'PRE_RACE') {
          setText('#season-context', `${state.event.season} CURRENT RACE`);
          setText('#focus-label', 'CURRENT RACE');
          setText('#focus-status', 'CURRENT RACE');
          setText('#focus-weekend', `CURRENT RACE · ROUND ${state.event.round ?? '—'}`);
        } else if (state.predictionState === 'COMPLETED' && previous !== 'COMPLETED') {
          setText('#focus-label', 'RACE COMPLETED');
          setText('#focus-status', 'RACE COMPLETED');
          await loadHome();
        }
      }
    } catch { /* Retain the last known safe state if refresh is temporarily unavailable. */ }
    finally {
      state.eventStatePending = false;
      if (isCurrentContext()) {
        const delay = state.modelBundleStatus === 'BUILDING' ? 3_000
          : ['MISSING', 'STALE'].includes(state.modelBundleStatus) ? 10_000 : 60_000;
        state.eventStateTimer = window.setTimeout(poll, delay);
      }
    }
  };
  const delay = state.modelBundleStatus === 'BUILDING' ? 3_000
    : ['MISSING', 'STALE'].includes(state.modelBundleStatus) ? 10_000 : 60_000;
  state.eventStateTimer = window.setTimeout(poll, delay);
}

async function loadHome() {
  const endpoints = ['/api/health', '/api/focus-event', '/api/seasons', '/api/model-metrics', '/api/quality'];
  const settled = await Promise.allSettled(endpoints.map((path) => api(path)));
  const value = (index) => settled[index].status === 'fulfilled' ? settled[index].value : null;
  const health = value(0), focus = value(1), seasonsData = value(2);
  state.metrics = value(3); state.quality = value(4);
  if (!health || !focus || !seasonsData) {
    const reason = settled.find((item, index) => index < 3 && item.status === 'rejected')?.reason;
    setText('#focus-title', 'Race schedule unavailable');
    setText('#track-analysis-status', 'CALENDAR UNAVAILABLE');
    setText('#track-data-quality', cleanError(reason, 'The local race calendar could not be loaded.'));
    setText('#race-selector-note', 'Calendar data could not be loaded from the local service.');
    if (state.metrics) renderModelMetrics(state.metrics);
    if (state.quality) renderQuality(state.quality);
    return;
  }
  setText('#database-stats', `${count(health.lap_rows)} MODEL LAPS · SQLITE`);
  state.seasons = seasonsData.seasons || [];
  state.seasonStates = seasonsData.season_states || [];
  state.activeSeason = seasonsData.active_season ?? null;
  renderModelMetrics(state.metrics || {});
  renderQuality(state.quality || {});
  await loadRaceSeasons(Number(focus.event?.season || state.activeSeason || state.seasons[0] || 0), focus.event?.event_id);
  await loadHistoricalSeasons();
}

async function loadRaceSeasons(preferredSeason, preferredEventId = null) {
  const select = $('#race-season');
  select.innerHTML = '';
  state.seasons.forEach((year) => {
    const seasonState = state.seasonStates.find((item) => Number(item.season) === Number(year));
    setOption(select, year, seasonState ? `${year} · ${seasonState.state}` : String(year));
  });
  select.disabled = !state.seasons.length;
  const chosen = state.seasons.includes(Number(preferredSeason)) ? Number(preferredSeason) : (state.seasons[0] || '');
  if (chosen) select.value = String(chosen);
  const chosenState = state.seasonStates.find((item) => Number(item.season) === Number(chosen));
  state.seasonContext = chosenState || null;
  setText('#season-context', chosenState?.state === 'PRE-SEASON' ? `${chosen} PRE-SEASON PREVIEW`
    : chosenState?.state || (chosen ? `${chosen} SEASON CALENDAR` : 'SEASON CALENDAR'));
  await loadRaceOptions(chosen, preferredEventId);
}

async function loadRaceOptions(season, preferredEventId = null) {
  const requestId = ++state.raceOptionsRequest;
  const select = $('#race-select');
  select.innerHTML = '';
  state.raceOptions = [];
  if (!season) { setOption(select, '', 'No calendar available'); select.disabled = true; return; }
  setOption(select, '', 'Loading current and upcoming races…');
  select.disabled = true;
  setText('#race-selector-note', `Loading the ${season} current/upcoming schedule…`);
  try {
    const data = await api(`/api/events?season=${encodeURIComponent(season)}&status=available`);
    if (requestId !== state.raceOptionsRequest) return;
    state.seasonContext = data.season_context || state.seasonContext;
    const seasonLabel = state.seasonContext?.label || state.seasonContext?.state || `${season} SEASON CALENDAR`;
    setText('#season-context', seasonLabel);
    state.raceOptions = data.events || [];
    select.innerHTML = '';
    if (!state.raceOptions.length) {
      setOption(select, '', 'No current or upcoming races recorded');
      setText('#race-selector-note', state.seasonContext?.state === 'SEASON COMPLETE'
        ? `${season} SEASON COMPLETE · No later race is listed in this calendar.`
        : `${seasonDataSummary(state.seasonContext)}. ${state.seasonContext?.message || `The local calendar has no current or upcoming race in ${season}. No event entries are fabricated.`}`);
      select.disabled = true;
      clearSelectedRace(season, state.seasonContext);
      updatePresetButtons();
      return;
    }
    setOption(select, '', 'Choose a current / upcoming race');
    state.raceOptions.forEach((event) => setOption(
      select, event.event_id,
      `R${String(event.round || '—').padStart(2, '0')} · ${event.race} · ${event.circuit || 'circuit unavailable'} · ${fmtDate(event.event_date)} · ${event.data_kind || 'CALENDAR DATA'}`,
    ));
    select.disabled = false;
    const dataStateNote = state.seasonContext?.message ? ` ${state.seasonContext.message}` : '';
    setText('#race-selector-note', `${state.raceOptions.length} calendar events from the ${season} schedule. ${seasonDataSummary(state.seasonContext)}.${dataStateNote}`);
    const selected = state.raceOptions.find((event) => event.event_id === preferredEventId)
      || state.raceOptions.find((event) => event.event_status === 'CURRENT')
      || state.raceOptions.find((event) => event.event_status === 'UPCOMING');
    if (selected) {
      select.value = selected.event_id;
      await selectRace(selected.event_id, { preset: selected.event_status === 'CURRENT' ? 'current' : 'next' });
    }
    updatePresetButtons();
  } catch (error) {
    if (requestId !== state.raceOptionsRequest) return;
    select.innerHTML = '';
    setOption(select, '', 'Calendar unavailable');
    select.disabled = true;
    setText('#race-selector-note', cleanError(error, 'Could not load current/upcoming race options.'));
  }
}

function clearSelectedRace(season, seasonContext = null) {
  state.eventRequest += 1;
  state.event = null; state.context = null; state.driver = null; state.drivers = []; state.lastPrediction = null;
  state.modelBundleAvailable = false; state.modelBundleStatus = 'UNKNOWN'; state.driverSelectionConfirmed = false; state.predictionState = 'UNKNOWN'; state.activePreset = 'choose';
  stopEventStatePolling();
  $('#results').hidden = true; $('#prediction-empty').hidden = false;
  setError(''); setBusy(false);
  const complete = seasonContext?.state === 'SEASON COMPLETE';
  setText('#focus-label', seasonContext?.label || (complete ? 'SEASON COMPLETE' : 'SELECT A RACE'));
  setText('#focus-title', complete ? `${season} SEASON COMPLETE` : seasonContext?.label || `No current or upcoming race in ${season}`);
  setText('#focus-track', seasonContext?.message || (complete ? 'The published calendar has no later race.' : 'Choose a season that has a published event in the local calendar.'));
  setText('#focus-official-name', ''); setText('#focus-round', `${season} SEASON`);
  setText('#focus-status', complete ? 'SEASON COMPLETE' : seasonContext?.state || 'NO EVENT SELECTED'); $('#focus-status').className = 'status-pill';
  setText('#focus-date', '—'); setText('#focus-weekend', 'Event context unavailable');
  setText('#focus-race-laps', '—'); setText('#focus-model-status', 'NOT APPLICABLE');
  setText('#data-updated', 'No event selected');
  const map = $('#track-map'); map.removeAttribute('src'); map.hidden = true; $('#track-map-fallback').hidden = false;
  setText('#track-map-title', 'CIRCUIT'); setText('#track-map-meta', 'No event selected');
  $('#driver-select').innerHTML = ''; setOption($('#driver-select'), '', 'Select a race first'); $('#driver-select').disabled = true;
  setText('#driver-meta', 'Choose an event to load source-backed driver context.');
  setText('#driver-entry-status', 'ENTRY NOT SELECTED'); setText('#focus-driver-note', ''); resetDriverCard();
  setText('#prediction-availability', 'Select an event to check whether pre-race prediction is available.');
  $('#predict-button').disabled = true;
  setText('#track-analysis-status', 'NO EVENT SELECTED'); $('#track-analysis-status').className = 'status-pill';
  setText('#track-analysis-subtitle', 'Select a current/upcoming event to load the cutoff-safe circuit sample.');
  renderCircuitAnalysis({});
  renderWeather({ current: { available: false }, forecast: { available: false, reason: 'Select a race to check its event-date forecast.' } });
  setText('#bundle-status', 'Waiting for race context');
  setText('#training-cutoff', 'Set per selected event');
  setText('#cache-status', 'Ready when requested');
}

async function selectRace(eventId, { preset = 'choose' } = {}) {
  if (!eventId) return;
  const requestId = ++state.eventRequest;
  stopEventStatePolling();
  $('#race-select').value = eventId;
  state.driver = null; state.drivers = []; state.lastPrediction = null; state.modelBundleAvailable = false; state.modelBundleStatus = 'UNKNOWN';
  $('#results').hidden = true; $('#prediction-empty').hidden = false;
  setError(''); setBusy(false);
  setText('#focus-title', 'Loading event context…');
  setText('#track-analysis-status', 'LOADING CIRCUIT DATA…'); $('#track-analysis-status').className = 'status-pill loading';
  setText('#track-data-quality', 'Selecting completed circuit races strictly before this event cutoff…');
  setText('#weather-updated', 'LOADING WEATHER…');
  try {
    const context = await api(`/api/event-context/${encodeURIComponent(eventId)}`);
    if (requestId !== state.eventRequest) return;
    state.context = context;
    state.activePreset = preset;
    renderFocus(context);
    monitorSelectedEventState();
    renderCircuitAnalysis(context.circuit_analysis || {});
    renderWeather({ ...context.cached_weather, historical: state.weather?.historical }, context.cached_weather?.retrieved_at_utc || context.forecast?.retrieved_at);
    updatePresetButtons();
    void loadWeather(false);
  } catch (error) {
    if (requestId !== state.eventRequest) return;
    setText('#focus-title', 'Race context unavailable');
    setText('#track-analysis-status', 'TRACK DATA UNAVAILABLE'); $('#track-analysis-status').className = 'status-pill';
    setText('#track-data-quality', cleanError(error, 'Circuit analysis could not be loaded for this event.'));
    $('#driver-select').disabled = true; $('#predict-button').disabled = true;
    stopEventStatePolling();
  }
}

function renderFocus(data) {
  setError('');
  const event = data.event || {};
  const seasonContext = data.season_context || event.season_context || state.seasonContext || {};
  state.seasonContext = seasonContext;
  state.event = event; state.driver = null; state.drivers = data.drivers || [];
  state.driverSelectionConfirmed = Boolean(data.driver_selection?.confirmed);
  setModelBundleStatus(data.model_bundle || {});
  state.predictionState = String(data.prediction_state?.state || 'UNKNOWN').toUpperCase();
  const status = String(event.event_status || data.race_status || '').toUpperCase();
  const current = status === 'CURRENT';
  const seasonState = String(seasonContext.state || '').toUpperCase();
  const statusLabel = state.predictionState === 'IN_PROGRESS' ? 'RACE IN PROGRESS'
    : state.predictionState === 'COMPLETED' ? 'RACE COMPLETED'
      : current ? 'CURRENT RACE' : status === 'COMPLETED' ? 'RACE COMPLETED' : seasonState === 'PRE-SEASON' ? 'PRE-SEASON' : 'UPCOMING';
  setText('#season-context', seasonContext.label || (seasonState === 'PRE-SEASON' ? `${event.season} PRE-SEASON PREVIEW` : `${event.season} ${seasonState}`));
  setText('#focus-round', `ROUND ${String(event.round || '—').padStart(2, '0')} · ${event.season || '—'}`);
  setText('#focus-status', statusLabel);
  $('#focus-status').className = `status-pill ${current ? 'current' : status === 'COMPLETED' ? 'completed' : 'upcoming'}`;
  setText('#focus-label', seasonState === 'PRE-SEASON' ? `${event.season} PRE-SEASON PREVIEW`
    : state.predictionState === 'IN_PROGRESS' ? 'RACE IN PROGRESS'
      : state.predictionState === 'COMPLETED' ? 'RACE COMPLETED'
        : current ? 'CURRENT RACE' : (state.activePreset === 'next' ? 'NEXT RACE' : 'SELECTED UPCOMING RACE'));
  setText('#focus-title', event.race || event.official_event_name || 'Grand Prix');
  setText('#focus-track', [event.circuit, event.location, event.country].filter(Boolean).join(' · ') || 'Circuit metadata unavailable');
  setText('#focus-official-name', event.official_event_name || '');
  setText('#focus-date', fmtDate(event.event_date));
  setText('#focus-weekend', `${seasonState === 'PRE-SEASON' ? 'PRE-SEASON' : statusLabel} · ROUND ${event.round ?? '—'}`);
  setText('#focus-race-laps', event.race_laps ? `${count(event.race_laps)} LAPS` : 'NOT IN SOURCE');
  setText('#data-updated', `LOCAL DATA UPDATED · ${data.historical_data_updated_at_utc ? fmtDate(data.historical_data_updated_at_utc) : 'TIMESTAMP UNAVAILABLE'}`);
  const track = data.track || {};
  setText('#track-map-title', String(track.circuit || event.circuit || 'CIRCUIT').toUpperCase());
  const details = [track.country, numeric(track.length_km) != null ? `${numeric(track.length_km)} km` : null,
    numeric(track.turns) != null ? `${count(track.turns)} turns` : null,
    event.race_laps ? `${count(event.race_laps)} laps` : null].filter(Boolean);
  setText('#track-map-meta', details.join(' · ') || 'Track length and turns not supplied');
  const map = $('#track-map'), fallback = $('#track-map-fallback');
  map.onload = () => { map.hidden = false; fallback.hidden = true; };
  map.onerror = () => { map.hidden = true; fallback.hidden = false; };
  if (track.url) { map.src = track.url; } else { map.removeAttribute('src'); map.hidden = true; fallback.hidden = false; }
  $('#driver-select').innerHTML = '';
  setOption($('#driver-select'), '', state.drivers.length ? 'Choose driver profile' : 'No eligible prior driver history');
  state.drivers.forEach((driver) => setOption($('#driver-select'), driver.driver_code,
    `${driver.driver_name || driver.driver_code} · ${driver.driver_code}${driver.team ? ` · ${driver.team}` : ''}`));
  $('#driver-select').disabled = !state.drivers.length;
  setText('#driver-meta', state.driverSelectionConfirmed ? `${state.drivers.length} observed event participants` : `${state.drivers.length} historical profiles · entry unconfirmed`);
  setText('#driver-entry-status', state.driverSelectionConfirmed ? 'OBSERVED EVENT ENTRY' : 'CURRENT EVENT ENTRY UNCONFIRMED');
  setText('#focus-driver-note', state.driverSelectionConfirmed ? 'Source-backed event participation' : 'Historical driver context only; no future entry is inferred.');
  $('#predict-button').disabled = true;
  resetDriverCard();
  setText('#training-cutoff', data.model_bundle?.cutoff_utc || 'Set when prediction runs');
  setText('#cache-status', 'No prediction requested');
  setText('#prediction-empty-note', seasonContext.message || 'Circuit history, tyre observations, pit windows and weather load before driver selection. Choose a driver to unlock the pre-race strategy plan.');
  updatePredictionControls();
  updateDataQualityForEvent(data.circuit_analysis || {});
}

function updatePresetButtons() {
  const current = state.raceOptions.some((event) => event.event_status === 'CURRENT');
  const next = state.raceOptions.some((event) => event.event_status === 'UPCOMING');
  $('#current-race-button').disabled = !current;
  $('#next-race-button').disabled = !next;
  $('#choose-race-button').disabled = !next;
  [['current-race-button', 'current'], ['next-race-button', 'next'], ['choose-race-button', 'choose']].forEach(([id, key]) => {
    $(`#${id}`).classList.toggle('active', state.activePreset === key);
  });
}

function resetDriverCard() {
  setText('#driver-number', '—');
  setText('#driver-name', 'Select a driver');
  setText('#driver-team', 'Track analysis is ready without a driver');
  setText('#driver-entry-status', state.driverSelectionConfirmed ? 'OBSERVED EVENT ENTRY' : 'CURRENT EVENT ENTRY UNCONFIRMED');
}

function updateDriverCard() {
  const driver = state.drivers.find((row) => String(row.driver_code).toUpperCase() === String(state.driver || '').toUpperCase());
  if (!driver) {
    resetDriverCard();
    updatePredictionControls();
    return;
  }
  const rawNumber = String(driver.driver_number || '').trim();
  const number = rawNumber && !['nan', 'none', 'null'].includes(rawNumber.toLowerCase()) ? `#${rawNumber.replace(/\.0$/, '')}` : '—';
  setText('#driver-number', number);
  setText('#driver-name', String(driver.driver_name || driver.driver_code).toUpperCase());
  setText('#driver-team', String(driver.team || 'Team not recorded').toUpperCase());
  const confirmed = Boolean(state.driverSelectionConfirmed || state.event?.event_status === 'COMPLETED' || driver.event_id === state.event?.event_id);
  setText('#driver-meta', confirmed ? `Observed event entry · ${driver.driver_code}` : `Historical driver profile · ${driver.driver_code}`);
  setText('#driver-entry-status', confirmed ? 'OBSERVED EVENT ENTRY' : 'CURRENT EVENT ENTRY UNCONFIRMED');
  setText('#focus-driver-note', confirmed ? 'Participation is source-backed for this event.' : 'Historical driver context only; current event participation is unconfirmed.');
  updatePredictionControls();
}

function renderCircuitAnalysis(data) {
  const coverage = data.coverage || {};
  const raceCount = numeric(coverage.race_count);
  const countUsed = raceCount == null ? 0 : raceCount;
  setText('#track-analysis-status', raceCount == null ? 'WAITING FOR EVENT' : (countUsed ? `${countUsed} RACE${countUsed === 1 ? '' : 'S'} IN SAMPLE` : 'NO LOCAL CIRCUIT RACES'));
  $('#track-analysis-status').className = `status-pill ${countUsed ? 'ready' : ''}`;
  setText('#track-analysis-subtitle', `${data.circuit || state.event?.circuit || 'Selected circuit'} · completed races are selected before the event cutoff`);
  setText('#circuit-history-count', raceCount == null ? '— / 6' : `${countUsed} / 6`);
  setText('#circuit-history-message', raceCount == null ? 'No event selected' : countUsed < 6 ? `ONLY ${countUsed} CIRCUIT RACE${countUsed === 1 ? '' : 'S'} AVAILABLE` : 'SIX MOST RECENT COMPLETED RACES');
  const years = coverage.race_years || [];
  const yearRoot = $('#circuit-history-years'); yearRoot.innerHTML = '';
  years.forEach((year) => { const chip = document.createElement('span'); chip.className = 'history-year'; chip.textContent = String(year); yearRoot.append(chip); });
  if (!years.length) yearRoot.textContent = raceCount == null ? '—' : 'NO PRE-CUTOFF RACE YEARS';
  const metrics = [
    [coverage.race_count, 'RACES USED'], [coverage.lap_count, 'LAP OBSERVATIONS'],
    [coverage.stint_count, 'STINTS'], [coverage.pit_transition_count, 'PIT TRANSITIONS'],
    [coverage.tyre_observation_count, 'TYRE OBSERVATIONS'],
    [coverage.earliest_race_date ? fmtDate(coverage.earliest_race_date) : null, 'EARLIEST RACE'],
    [coverage.latest_race_date ? fmtDate(coverage.latest_race_date) : null, 'LATEST RACE'],
  ];
  $('#circuit-coverage-metrics').innerHTML = metrics.map(([value, label]) => `<div class="coverage-metric"><b>${escapeHtml(value == null ? '—' : (typeof value === 'number' ? count(value) : value))}</b><span>${escapeHtml(label)}</span></div>`).join('');
  const races = data.races || [];
  $('#circuit-race-list').innerHTML = races.map((race) => `<span class="circuit-race-chip"><b>${escapeHtml(race.season)}</b> · ${escapeHtml(race.race)} · ${escapeHtml(fmtDate(race.event_date))}</span>`).join('')
    || (raceCount == null ? '' : '<span class="chart-note">No completed circuit event is present before this cutoff.</span>');
  const cutoff = data.cutoff_date ? fmtDate(data.cutoff_date) : null;
  const noTyres = countUsed > 0 && Number(coverage.tyre_observation_count || 0) === 0;
  const strategySource = data.strategy_distribution_source ? ` Strategy source: ${data.strategy_distribution_source}.` : '';
  const tyreNote = noTyres ? ' These race-lap records do not include tyre compound or tyre-age observations; no local tyre values are inferred.' : '';
  setText('#track-data-quality', raceCount == null
    ? 'Select a current/upcoming event to load the actual pre-cutoff sample.'
    : `${coverage.message || `Only ${countUsed} completed circuit races available`}${cutoff ? ` · cutoff ${cutoff}` : ''}.${tyreNote}${strategySource}`);
  setText('#tyre-track-label', raceCount == null ? 'PRE-CUTOFF' : `${countUsed} CIRCUIT RACES · PRE-CUTOFF`);
  renderTyres(data.tyre_performance || []);
  renderDegradation(data.tyre_degradation || []);
  renderPitHistory(data);
  renderStrategyDistribution(data.strategy_distribution || [], data.strategy_distribution_source || 'No source-backed strategy observations.');
  renderCommonNextCompounds(data.common_next_compounds || {});
  updateDataQualityForEvent(data);
}

function renderTyres(rows) {
  const root = $('#tyre-chart');
  const values = (rows || []).filter((row) => TYRE[row.compound] && numeric(row.mean_lap_time) != null);
  $('#tyre-fallback').hidden = true;
  root.innerHTML = '';
  if (!values.length) {
    const empty = document.createElement('div'); empty.className = 'empty-state';
    const title = document.createElement('b'); title.textContent = 'CIRCUIT TYRE DATA UNAVAILABLE';
    const reason = document.createElement('span'); reason.textContent = 'Historical circuit lap/pit records contain no compound or tyre-age observations.';
    empty.append(title, reason); root.append(empty);
    const fallback = $('#tyre-fallback'); fallback.hidden = false;
    fallback.innerHTML = '<b>FALLBACK DATA · RECENT ALL-CIRCUIT F1 PROFILE</b>The current API response contains no global compound-level values to plot. No Sepang-specific pace values are inferred.';
    setText('#tyre-legend', 'The fitted prediction bundle may use its global profile internally; numerical global fallback values are not exposed by the current API response.');
    return;
  }
  const fastest = Math.min(...values.map((row) => Number(row.mean_lap_time)));
  const slowest = Math.max(...values.map((row) => Number(row.mean_lap_time)));
  values.forEach((row) => {
    const compound = row.compound;
    const wrap = document.createElement('div'); wrap.className = 'tyre-row'; wrap.style.setProperty('--compound-color', TYRE[compound]);
    const name = document.createElement('span'); name.className = 'tyre-row-name';
    const dot = document.createElement('i'); dot.className = 'compound-dot'; name.append(dot, document.createTextNode(compound));
    const track = document.createElement('div'); track.className = 'tyre-track';
    const bar = document.createElement('i');
    const relative = slowest === fastest ? 1 : .35 + .65 * (1 - (Number(row.mean_lap_time) - fastest) / (slowest - fastest));
    bar.style.width = `${Math.max(8, relative * 100)}%`; track.append(bar);
    const value = document.createElement('span'); value.className = 'tyre-value'; value.textContent = `${Number(row.mean_lap_time).toFixed(2)}s`;
    const detail = document.createElement('small'); detail.className = 'tyre-meta';
    detail.textContent = `${count(row.sample_laps)} laps · ${count(row.event_count)} events · mean tyre age ${numeric(row.mean_tyre_age) == null ? 'not supplied' : `${Number(row.mean_tyre_age).toFixed(1)} laps`}`;
    wrap.append(name, track, value, detail); root.append(wrap);
  });
  setText('#tyre-legend', 'Recorded circuit averages before the selected event. Raw lap averages are not normalized for driver, tyre age, or weather.');
}

function renderDegradation(rows) {
  const root = $('#degradation-chart'), legend = $('#degradation-legend');
  root.innerHTML = ''; legend.textContent = '';
  const values = (rows || []).filter((row) => TYRE[row.compound] && numeric(row.tyre_age) != null && numeric(row.delta_seconds) != null);
  const groups = new Map();
  values.forEach((row) => { if (!groups.has(row.compound)) groups.set(row.compound, []); groups.get(row.compound).push(row); });
  const valid = [...groups.entries()].filter(([, points]) => points.length >= 3);
  if (!valid.length) {
    const empty = document.createElement('div'); empty.className = 'degradation-empty';
    const title = document.createElement('b'); title.textContent = 'CIRCUIT-SPECIFIC DEGRADATION UNAVAILABLE';
    const reason = document.createElement('span'); reason.textContent = 'No historical tyre-age observations are available for this circuit sample. Global degradation values are not included in this API response.';
    empty.append(title, reason); root.append(empty); return;
  }
  const ages = values.map((row) => Number(row.tyre_age)), deltas = values.map((row) => Number(row.delta_seconds));
  const minAge = Math.min(...ages), maxAge = Math.max(...ages), minY = Math.min(...deltas), maxY = Math.max(...deltas);
  const width = 640, height = 150, left = 35, right = 12, top = 12, bottom = 25, plotW = width - left - right, plotH = height - top - bottom;
  const svg = document.createElementNS(SVG_NS, 'svg'); svg.setAttribute('viewBox', `0 0 ${width} ${height}`); svg.setAttribute('class', 'degradation-svg'); svg.setAttribute('role', 'img'); svg.setAttribute('aria-label', 'Tyre age against normalized lap-time delta by compound');
  for (let index = 0; index < 4; index += 1) {
    const y = top + plotH * index / 3, line = document.createElementNS(SVG_NS, 'line');
    line.setAttribute('x1', left); line.setAttribute('x2', width - right); line.setAttribute('y1', y); line.setAttribute('y2', y); line.setAttribute('stroke', '#313740'); svg.append(line);
  }
  valid.forEach(([compound, points]) => {
    points.sort((a, b) => Number(a.tyre_age) - Number(b.tyre_age));
    const path = document.createElementNS(SVG_NS, 'polyline');
    const coords = points.map((row) => {
      const x = left + (Number(row.tyre_age) - minAge) / Math.max(1, maxAge - minAge) * plotW;
      const y = top + (maxY - Number(row.delta_seconds)) / Math.max(.05, maxY - minY) * plotH;
      return `${x},${y}`;
    }).join(' ');
    path.setAttribute('points', coords); path.setAttribute('fill', 'none'); path.setAttribute('stroke', TYRE[compound]); path.setAttribute('stroke-width', '2'); path.setAttribute('stroke-linejoin', 'round'); svg.append(path);
  });
  const axis = document.createElementNS(SVG_NS, 'text'); axis.setAttribute('x', left); axis.setAttribute('y', height - 5); axis.setAttribute('fill', '#7f8993'); axis.setAttribute('font-size', '9'); axis.textContent = `TYRE AGE ${minAge}–${maxAge} LAPS`; svg.append(axis);
  root.append(svg);
  valid.forEach(([compound]) => {
    const item = document.createElement('span'); item.className = 'legend-item'; item.style.setProperty('--compound-color', TYRE[compound]);
    const line = document.createElement('i'); item.append(line, document.createTextNode(compound)); legend.append(item);
  });
  const samples = values.reduce((total, row) => total + Number(row.sample_laps || 0), 0);
  legend.append(document.createTextNode(` · ${count(samples)} tyre-age lap observations`));
}

function makePitLane(label, distribution, markers = []) {
  const row = document.createElement('div'); row.className = 'pit-lane-row';
  const name = document.createElement('span'); name.className = 'pit-lane-label'; name.textContent = label;
  const track = document.createElement('div'); track.className = 'pit-lane-track';
  (distribution || []).forEach((item) => {
    const start = Number(String(item.band || '').match(/\d+/)?.[0]);
    if (!Number.isFinite(start)) return;
    const band = document.createElement('i'); band.className = 'pit-lane-band'; band.style.left = `${Math.max(0, Math.min(100, start))}%`;
    band.style.width = `${start >= 100 ? 0 : 10}%`;
    const share = numeric(item.share) || 0; band.style.setProperty('--band-opacity', String(.2 + Math.min(.72, share * 2.4)));
    band.title = `${item.band}: ${count(item.count)} observed stops · ${pct(item.share)}`; track.append(band);
  });
  markers.forEach((marker) => {
    const position = numeric(marker.lap), distance = numeric(state.event?.race_laps);
    if (position == null || distance == null || distance <= 1) return;
    const pin = document.createElement('i'); pin.className = 'pit-marker'; pin.style.left = `${Math.max(0, Math.min(100, ((position - 1) / (distance - 1)) * 100))}%`;
    pin.style.setProperty('--marker-color', marker.color); pin.dataset.label = marker.label; pin.title = `${marker.label} · lap ${position}`; track.append(pin);
  });
  row.append(name, track); return row;
}

function renderPitHistory(data) {
  const coverage = data.coverage || {}, stats = $('#history-stats'); stats.innerHTML = '';
  const values = [
    [coverage.pit_transition_count, 'PIT TRANSITIONS'],
    [data.median_pit_lap == null ? null : `LAP ${Math.round(Number(data.median_pit_lap))}`, 'MEDIAN PIT LAP'],
    [data.median_pit_progress == null ? null : pct(data.median_pit_progress), 'MEDIAN RACE PROGRESS'],
  ];
  values.forEach(([value, label]) => {
    const node = document.createElement('div'); node.className = 'history-stat';
    const bold = document.createElement('b'); bold.textContent = value == null ? '—' : (typeof value === 'number' ? count(value) : value);
    const span = document.createElement('span'); span.textContent = label; node.append(bold, span); stats.append(node);
  });
  const lane = $('#pit-lane-chart'); lane.innerHTML = '';
  const historyRows = data.pit_window_distribution || [];
  if (historyRows.length) lane.append(makePitLane('HISTORICAL STOPS', historyRows));
  else { const empty = document.createElement('p'); empty.className = 'distribution-empty'; empty.textContent = 'No historical pit-window observations are available.'; lane.append(empty); }
  const axis = document.createElement('div'); axis.className = 'pit-lane-axis';
  axis.innerHTML = `<span>LAP 1</span><span>${escapeHtml(state.event?.race_laps ? `LAP ${count(state.event.race_laps)}` : 'RACE DISTANCE NOT IN SOURCE')}</span>`;
  lane.append(axis);
  const distribution = $('#pit-window-distribution'); distribution.innerHTML = '';
  if (historyRows.length) {
    historyRows.forEach((item) => {
      const bin = document.createElement('div'); bin.className = 'window-bin';
      const band = document.createElement('b'); band.textContent = item.band || 'Band not named';
      const amount = document.createElement('span'); amount.textContent = `${count(item.count)} stops`;
      const share = document.createElement('small'); share.textContent = `${pct(item.share)} of observed stops`;
      bin.append(band, amount, share); distribution.append(bin);
    });
  } else distribution.innerHTML = '<div class="distribution-empty">Pit-window distribution unavailable for this sample.</div>';
  const pitBands = $('#pit-lap-bands'); pitBands.innerHTML = '';
  (data.common_pit_lap_bands || []).forEach((item) => {
    const chip = document.createElement('span'); chip.className = 'pit-band-chip';
    chip.textContent = `${item.band} · ${count(item.count)}`; pitBands.append(chip);
  });
  if (!pitBands.children.length) pitBands.textContent = 'No lap-band observations';
}

function renderCommonNextCompounds(values) {
  const root = $('#next-compound-support'); root.innerHTML = '';
  const entries = Object.entries(values || {}).filter(([, amount]) => numeric(amount) != null);
  if (!entries.length) { root.textContent = 'No compound transitions recorded'; return; }
  entries.forEach(([compound, amount]) => {
    const chip = document.createElement('span'); chip.className = 'compound-chip';
    const color = TYRE[compound] || '#9ba3ac'; chip.style.setProperty('--compound-color', color);
    chip.textContent = `${compound} · ${count(amount)}`; root.append(chip);
  });
}

function renderStrategyDistribution(rows, source) {
  setText('#strategy-source', source && source !== 'No source-backed strategy observations.' ? 'OBSERVED STRATEGIES' : 'NO OBSERVATIONS');
  const root = $('#strategy-distribution'); root.innerHTML = '';
  const values = rows || [];
  if (!values.length) { root.innerHTML = '<div class="distribution-empty">No source-backed strategy observations are available before this event cutoff.</div>'; return; }
  const max = Math.max(...values.map((row) => Number(row.count || 0)), 1);
  values.forEach((row) => {
    const item = document.createElement('div'); item.className = 'distribution-row';
    const label = document.createElement('span'); label.className = 'distribution-label'; label.textContent = row.strategy || 'Strategy not named';
    const track = document.createElement('div'); track.className = 'distribution-track';
    const bar = document.createElement('i'); bar.style.width = `${Math.max(2, Number(row.count || 0) / max * 100)}%`; track.append(bar);
    const value = document.createElement('span'); value.className = 'distribution-value'; value.textContent = `${count(row.count)} · ${pct(row.share)}`;
    item.append(label, track, value); root.append(item);
  });
}

function addModelMetric(root, value, label) {
  const item = document.createElement('div'); item.className = 'model-metric';
  const bold = document.createElement('b'); bold.textContent = value;
  const span = document.createElement('span'); span.textContent = label;
  item.append(bold, span); root.append(item);
}

function renderModelMetrics(metrics) {
  const lap = metrics.lap_time || {}, pit = metrics.pit_stop || {};
  const lapRoot = $('#model-lap-metrics'), tyreRoot = $('#model-tyre-metrics');
  lapRoot.innerHTML = ''; tyreRoot.innerHTML = '';
  addModelMetric(lapRoot, numeric(lap.MAE) == null ? '—' : `${Number(lap.MAE).toFixed(2)}s`, 'MAE');
  addModelMetric(lapRoot, numeric(lap.RMSE) == null ? '—' : `${Number(lap.RMSE).toFixed(2)}s`, 'RMSE');
  addModelMetric(lapRoot, numeric(lap.R2) == null ? '—' : Number(lap.R2).toFixed(3), 'R²');
  addModelMetric(tyreRoot, pct(pit.accuracy, 1), 'ACCURACY');
  addModelMetric(tyreRoot, pct(pit.baseline_accuracy, 1), 'BASELINE');
  addModelMetric(tyreRoot, count(pit.test_rows), 'TEST ROWS');
  const season = lap.held_out_season ?? pit.held_out_season;
  const lapRows = lap.test_rows, pitRows = pit.test_rows;
  setText('#model-holdout', season == null
    ? 'Historical validation metrics · not a future-performance guarantee.'
    : `Temporal holdout ${season} · ${count(lapRows)} lap test rows · ${count(pitRows)} next-tyre test rows. Historical validation metrics, not a future-performance guarantee.`);
}

function renderQuality(report) {
  const tables = report.tables || {};
  setText('#database-rows', `${count(tables.laps)} laps · ${count(tables.events)} events`);
  const status = report.status || 'not reported';
  setText('#data-quality-status', status === 'clean' ? 'NO REPORTED ISSUES' : status.toUpperCase());
  const root = $('#data-quality-grid'); root.innerHTML = '';
  const values = [
    [tables.laps, 'MODEL LAPS'], [tables.events, 'EVENT ROWS'],
    [tables.circuit_history_laps, 'CIRCUIT LAP ROWS'], [tables.pit_transitions, 'PIT TRANSITIONS'],
    [tables.circuit_history_pit_stops, 'LEGACY PIT STOPS'], [tables.weather_observations, 'WEATHER ROWS'],
    [report.duplicate_event_driver_lap_keys, 'DUPLICATE LAP KEYS'],
    [report.missing_weather_events, 'EVENTS WITHOUT WEATHER'],
  ];
  values.forEach(([value, label]) => {
    const node = document.createElement('div'); node.className = 'quality-metric';
    const name = document.createElement('span'); name.textContent = label;
    const amount = document.createElement('b'); amount.textContent = count(value);
    node.append(name, amount); root.append(node);
  });
  const missing = numeric(report.missing_weather_events);
  setText('#data-quality-note', `${missing == null ? 'Weather coverage not reported.' : `${count(missing)} events have no historical weather observations.`} Missing weather remains an explicit fallback; absent source data is not displayed as measured.`);
}

function updateDataQualityForEvent(analysis) {
  if (!state.quality) return;
  const coverage = analysis.coverage || {};
  const local = numeric(coverage.race_count);
  const localTyres = numeric(coverage.tyre_observation_count);
  if (local == null) return;
  const baseNote = Number(state.quality.missing_weather_events || 0);
  setText('#data-quality-note', `${count(state.quality.tables?.laps)} model-input laps · ${count(local)} selected-circuit races · ${count(coverage.pit_transition_count)} selected-circuit pit transitions · ${localTyres == null ? 'Tyre count unknown' : `${count(localTyres)} tyre observations`}. Historical weather is reported separately below; ${count(baseNote)} events lack stored weather.`);
}

async function loadWeather(refresh) {
  if (!state.event?.event_id) return;
  const eventId = state.event.event_id, button = $('#refresh-weather');
  button.disabled = true; button.textContent = refresh ? 'REFRESHING…' : 'LOADING WEATHER…';
  setText('#weather-updated', 'LOADING WEATHER…');
  try {
    const response = await api(`/api/weather/${encodeURIComponent(eventId)}${refresh ? '?refresh=1' : ''}`);
    if (state.event?.event_id !== eventId) return;
    state.weather = response;
    renderWeather(response, response.retrieved_at_utc);
    updateDataQualityForEvent(state.context?.circuit_analysis || {});
  } catch {
    if (state.event?.event_id !== eventId) return;
    const unavailable = { live: { current: { available: false }, forecast: { available: false } }, historical: { available: false } };
    state.weather = unavailable; renderWeather(unavailable);
  } finally {
    if (state.event?.event_id === eventId) { button.disabled = false; button.textContent = 'REFRESH FORECAST ↻'; }
  }
}

function renderWeather(packet, retrievedAt) {
  const live = packet?.live || packet || {}, current = live.current || {}, forecast = live.forecast || {};
  const historical = packet?.historical || state.context?.cached_weather?.historical || {};
  const currentReady = Boolean(current.available), forecastReady = Boolean(forecast.available);
  setText('#weather-current', currentReady && numeric(current.temperature) != null ? `${Number(current.temperature).toFixed(0)}°C` : (currentReady ? 'Available' : 'Unavailable'));
  setText('#weather-current-detail', currentReady ? (current.weather_name || current.condition || current.time || 'Current conditions available') : 'No current reading supplied');
  setText('#weather-race', forecastReady ? (forecast.condition || forecast.dominant_weather || 'Available') : 'Unavailable');
  setText('#weather-race-detail', forecastReady && numeric(forecast.temperature) != null ? `${Number(forecast.temperature).toFixed(0)}°C${numeric(forecast.apparent_temperature) == null ? '' : ` · feels ${Number(forecast.apparent_temperature).toFixed(0)}°C`}` : (forecastReady ? 'Race-window data available' : 'Race date outside forecast / source unavailable'));
  setText('#weather-rain', forecastReady && numeric(forecast.precipitation_probability) != null ? `${Number(forecast.precipitation_probability).toFixed(0)}%` : '—');
  setText('#weather-wind', forecastReady && numeric(forecast.wind_speed) != null ? `${Number(forecast.wind_speed).toFixed(0)} km/h` : '—');
  const hourly = forecast.hourly || [];
  let coverage = 'Not available';
  if (forecastReady && hourly.length) {
    const first = hourly[0]?.time, last = hourly[hourly.length - 1]?.time;
    coverage = `${hourly.length} hourly points${first && last ? ` · ${String(first).slice(11, 16)}–${String(last).slice(11, 16)}` : ''}`;
  } else if (!forecastReady && /outside.*horizon/i.test(String(forecast.reason || ''))) coverage = 'Outside provider horizon';
  else if (forecast.window) coverage = forecast.window;
  setText('#forecast-coverage', coverage);
  const strip = $('#forecast-strip'); strip.innerHTML = '';
  if (forecastReady && hourly.length) {
    hourly.slice(0, 16).forEach((hour) => {
      const bar = document.createElement('i'); bar.className = 'forecast-bar';
      const rain = numeric(hour.precipitation_probability) ?? 0;
      bar.style.setProperty('--forecast-height', `${Math.max(5, Math.min(26, 4 + rain * .22))}px`);
      bar.dataset.time = hour.time ? String(hour.time).slice(11, 16) : '';
      bar.title = `${hour.time || 'Forecast hour'} · ${numeric(hour.temperature_2m) == null ? 'temperature unavailable' : `${Number(hour.temperature_2m).toFixed(0)}°C`} · rain ${Number(rain).toFixed(0)}%`;
      strip.append(bar);
    });
  } else {
    const note = document.createElement('span'); note.className = 'forecast-placeholder';
    note.textContent = /outside.*horizon/i.test(String(forecast.reason || '')) ? 'Race date is outside the current forecast horizon.' : 'Forecast window is not available for this event.';
    strip.append(note);
  }
  setText('#weather-historical', historical.available
    ? `${count(historical.observations)} observations · ${numeric(historical.mean_temperature_c) == null ? 'temperature not supplied' : `${Number(historical.mean_temperature_c).toFixed(1)}°C mean`}${numeric(historical.total_rain_mm) == null ? '' : ` · ${Number(historical.total_rain_mm).toFixed(1)} mm rain`}`
    : 'No stored observations for this event');
  const cache = packet?.cache_status || live.cache_status;
  setText('#weather-updated', retrievedAt ? `UPDATED ${new Date(retrievedAt).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}${cache ? ` · ${String(cache).toUpperCase()}` : ''}` : (currentReady || forecastReady ? 'SOURCE DATA READY' : 'DATA UNAVAILABLE'));
  const unavailable = !currentReady && !forecastReady;
  $('#weather-fallback').hidden = !unavailable;
  setText('#weather-fallback', unavailable
    ? 'WEATHER DATA UNAVAILABLE · Strategy analysis continues with the documented fallback assumptions.'
    : (forecastReady ? (forecast.window || '') : 'Current conditions available; the race forecast is outside the provider window.'));
}

function selectRacePreset(preset) {
  const selected = preset === 'current'
    ? state.raceOptions.find((event) => event.event_status === 'CURRENT')
    : state.raceOptions.find((event) => event.event_status === 'UPCOMING');
  if (!selected) { toast(preset === 'current' ? 'There is no current event in this season.' : 'There are no upcoming events in this season.'); return; }
  void selectRace(selected.event_id, { preset });
}

async function runPrediction() {
  const validDriver = state.drivers.some((row) => String(row.driver_code).toUpperCase() === String(state.driver || '').toUpperCase());
  if (!state.event || !validDriver || !state.modelBundleAvailable || !['UPCOMING', 'PRE_RACE'].includes(state.predictionState)) {
    updatePredictionControls();
    return;
  }
  setError(''); setBusy(true); $('#prediction-empty').hidden = true; $('#results').hidden = true;
  const started = performance.now(), eventId = state.event.event_id;
  try {
    const payload = await api('/api/predict', { method: 'POST', body: JSON.stringify({ event_id: eventId, driver: state.driver, fetch_weather: true }) });
    if (state.event?.event_id !== eventId) return;
    payload.client_elapsed_seconds = (performance.now() - started) / 1000;
    state.lastPrediction = payload;
    renderPrediction(payload);
  } catch (error) {
    $('#prediction-empty').hidden = false;
    setError(cleanError(error, 'The strategy request could not be completed. Retry when the local bundle is available.'));
  } finally { setBusy(false); }
}

function renderPrediction(payload) {
  const event = payload.event || {}, result = payload.result || {}, plan = result.expected_strategy || {};
  const primary = result.primary_strategy || plan;
  const pitLaps = primary.pit_laps || plan.pit_laps || payload.pit_laps || [];
  const window = plan.pit_window || primary.pit_window || payload.pit_window || {};
  $('#results').hidden = false; $('#prediction-empty').hidden = true;
  setText('#result-title', `${event.race || result.race || 'Grand Prix'} · ${event.season || result.season || ''}`);
  setText('#result-context', `${event.circuit || result.circuit || 'Circuit'} · ${result.driver || state.driver} · ${payload.driver_context || result.driver_context || 'driver context unavailable'} · ${result.event_status || event.event_status || ''}`);
  setText('#mode-pill', result.analysis_mode || 'PRE-RACE');
  setText('#strategy-value', primary.strategy || plan.strategy || 'Not returned');
  const stintLengths = primary.stint_lengths || plan.stint_lengths || [];
  setText('#strategy-subtitle', `${stintLengths.length} tyre stints · ${stintLengths.length ? `${stintLengths.join(' / ')} laps` : 'stint lengths unavailable'} · ${plan.strategy_type || primary.strategy_type || 'model-ranked plan'}`);
  const start = numeric(window.start_lap), end = numeric(window.end_lap), central = numeric(window.central_lap);
  setText('#pit-window-value', start != null && end != null ? `L${count(start)}–${count(end)}` : (pitLaps.length ? `L${pitLaps.map(count).join(' · ')}` : 'No stop'));
  setText('#pit-window-detail', central != null ? `Central estimate · lap ${count(central)}` : (window.basis || 'No pit window returned'));
  setText('#primary-pit-lap', pitLaps.length ? `L${pitLaps.map(count).join(' · ')}` : 'NO STOP');
  setText('#stops-value', `${pitLaps.length} stop${pitLaps.length === 1 ? '' : 's'} · ${stintLengths.length || '—'} stints`);
  const support = payload.confidence?.score ?? result.confidence_score;
  setText('#confidence-value', numeric(support) == null ? '—' : `${Number(support).toFixed(0)} / 100`);
  setText('#confidence-detail', `${result.confidence_level || 'MODEL'} · data support, not a win probability`);
  const distance = numeric(result.race_distance) ?? numeric(event.race_laps);
  setText('#distance-label', distance == null ? 'DISTANCE NOT IN SOURCE' : `${count(distance)} LAPS`);
  renderTimeline(payload.timeline || [], distance);
  renderPitPrediction(state.context?.circuit_analysis || {}, payload);
  renderStrategySupport(result.scenarios || []);
  renderComparison(plan, primary);
  renderScoreBreakdown(primary.score_breakdown || result.strategy_score_breakdown || plan.score_breakdown || {});
  renderReasons(primary, result, payload);
  const cutoff = result.training_cutoff_time_utc || result.selected_race_start_time_utc || 'Cutoff not returned';
  const trainingRows = result.training_rows ?? result.model_training_rows;
  const eventCount = result.training_event_count ?? result.model_training_event_count;
  setText('#data-cutoff', `${cutoff} · ${count(trainingRows)} pre-cutoff lap rows across ${count(eventCount)} events · selected-event laps included: ${result.selected_event_laps_used_for_training == null ? 'not reported' : (result.selected_event_laps_used_for_training ? 'yes' : 'no')}. ${result.training_cutoff_policy || ''}`);
  const inventory = result.inventory || {};
  const inventoryText = Object.entries(inventory).map(([compound, sets]) => `${compound} ${sets}`).join(' · ');
  setText('#inventory-note', `${result.nomination ? nominationLabel(result.nomination) : 'Official tyre nomination unavailable'} · ${inventoryText || 'Inventory quantities unavailable'} · ${result.inventory_source || 'Inventory source not returned'}`);
  const visibleCoverage = state.context?.circuit_analysis?.coverage || {};
  const bundledLearning = result.track_specific_learning || {};
  const bundledRaceCount = numeric(bundledLearning.circuit_races_used ?? bundledLearning.circuit_history?.race_count);
  const liveRaceCount = numeric(visibleCoverage.race_count);
  let modelNote = `${result.training_cutoff_policy || 'The event bundle uses the conservative event-date cutoff.'} Prediction models are loaded from offline artifacts; the selected event was not used for training.`;
  if (liveRaceCount != null && bundledRaceCount != null && liveRaceCount !== bundledRaceCount) {
    modelNote += ` The live circuit panel contains ${count(liveRaceCount)} races; this event bundle reports ${count(bundledRaceCount)} embedded circuit races. The API does not expose the bundle's global fallback profile values.`;
  }
  setText('#prediction-model-note', modelNote);
  updatePredictionModelState(payload);
  const timing = payload.prediction_timings || {}, runtime = payload.performance || {}, cache = payload.cache_hits || {};
  const cacheState = cache.prediction ? 'PREDICTION CACHE HIT' : cache.model_artifact ? 'MODEL MEMORY CACHE' : 'MODEL ARTIFACT LOADED';
  setText('#prediction-runtime', `WALL ${Number(runtime.seconds ?? payload.client_elapsed_seconds ?? 0).toFixed(2)}s · DATABASE ${numeric(timing.database_load_ms) == null ? '—' : `${Math.round(timing.database_load_ms)}ms`} · MODEL ${numeric(timing.model_load_ms) == null ? '—' : `${Math.round(timing.model_load_ms)}ms`} · SIMULATION ${numeric(timing.strategy_simulation_ms) == null ? '—' : `${Math.round(timing.strategy_simulation_ms)}ms`}`);
  setText('#prediction-cache-status', `${cacheState} · FEATURES ${cache.precomputed_features ? 'PRECOMPUTED' : 'NOT REPORTED'} · WEATHER ${cache.weather ? 'CACHE HIT' : (payload.weather?.source || 'SOURCE NOT REPORTED')}`);
  $('#results').scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function updatePredictionModelState(payload) {
  const result = payload.result || {};
  setText('#cache-status', payload.cache_hits?.prediction ? 'HIT · CACHED PREDICTION' : payload.cache_hits?.model_artifact ? 'MODEL ARTIFACT MEMORY HIT' : 'MISS · FIRST STRATEGY RUN');
  setText('#training-cutoff', result.training_cutoff_time_utc || 'Cutoff not returned');
  const bundle = state.context?.model_bundle;
  setText('#bundle-status', bundle?.available ? `READY · ${bundle.cutoff_utc || 'CUTOFF-SAFE'}` : 'BUNDLE STATUS UNAVAILABLE');
}

function renderTimeline(stints, distance) {
  const bar = $('#timeline'); bar.innerHTML = '';
  (stints || []).forEach((stint) => {
    const segment = document.createElement('div');
    const compound = String(stint.compound || '').toUpperCase();
    segment.className = `stint-segment tyre-${compound.toLowerCase()}`;
    segment.style.flex = `${Math.max(1, Number(stint.width_pct || 0))} 1 0`;
    segment.title = `${compound || 'Compound not named'} · laps ${stint.start_lap ?? '—'}–${stint.end_lap ?? '—'}`;
    const name = document.createElement('span'); name.textContent = compound || 'UNKNOWN'; segment.append(name); bar.append(segment);
  });
  if (!stints?.length) bar.innerHTML = '<div class="distribution-empty">The strategy API did not return a stint timeline.</div>';
  const pitText = (stints || []).map((item) => item.pit_after_lap).filter((value) => value != null).map((value) => `PIT ${count(value)}`).join(' · ') || 'PIT LAP NOT RETURNED';
  $('#timeline-axis').innerHTML = `<span>LAP 1</span><span>${escapeHtml(pitText)}</span><span>${distance == null ? 'RACE DISTANCE NOT IN SOURCE' : `LAP ${escapeHtml(count(distance))}`}</span>`;
}

function renderPitPrediction(history, payload) {
  const root = $('#prediction-pit-lane'); root.innerHTML = '';
  const historical = history.pit_window_distribution || [];
  if (historical.length) root.append(makePitLane('HISTORY', historical));
  const result = payload.result || {}, plan = result.expected_strategy || {};
  const primary = result.primary_strategy || plan;
  const primaryLaps = primary.pit_laps || plan.pit_laps || payload.pit_laps || [];
  if (primaryLaps.length) root.append(makePitLane('RECOMMENDED', [], primaryLaps.map((lap) => ({ lap, label: `REC L${lap}`, color: '#fff' }))));
  const alternatives = (plan.top_alternatives || []).slice(0, 2);
  const altMarkers = alternatives.flatMap((row) => (row.pit_laps || []).map((lap) => ({ lap, label: `ALT L${lap}`, color: '#e4bd59' })));
  if (altMarkers.length) root.append(makePitLane('ALTERNATIVES', [], altMarkers));
  if (!root.children.length) root.textContent = 'Historical and planned pit windows are unavailable.';
  const axis = document.createElement('div'); axis.className = 'pit-lane-axis';
  axis.innerHTML = `<span>LAP 1</span><span>${escapeHtml(state.event?.race_laps ? `LAP ${count(state.event.race_laps)}` : 'RACE DISTANCE NOT IN SOURCE')}</span>`;
  root.append(axis);
}

function renderStrategySupport(rows) {
  const root = $('#strategy-support'); root.innerHTML = '';
  (rows || []).slice(0, 4).forEach((row) => {
    const item = document.createElement('div'); item.className = 'support-row';
    const label = document.createElement('span'); label.textContent = `${row.condition || 'SCENARIO'} · ${row.strategy || 'Plan not named'}`;
    const track = document.createElement('div'); track.className = 'support-track';
    const fill = document.createElement('i'); fill.className = ['DAMP', 'WET'].includes(String(row.condition).toUpperCase()) ? 'damp' : 'dry';
    const weight = numeric(row.weight); if (weight != null) fill.style.width = `${Math.max(0, Math.min(100, weight * 100))}%`; track.append(fill);
    const value = document.createElement('span'); value.textContent = pct(row.weight);
    item.append(label, track, value); root.append(item);
  });
  if (!root.children.length) root.innerHTML = '<p class="chart-note">No weather scenario weights were returned.</p>';
}

function renderComparison(plan, primary) {
  const root = $('#comparison-list'); root.innerHTML = '';
  const alternatives = plan.top_alternatives || [];
  const selected = {
    strategy: primary.strategy || plan.strategy,
    pit_laps: primary.pit_laps || plan.pit_laps || [],
    score: primary.strategy_score ?? plan.strategy_score,
    estimated_race_time: plan.estimated_race_time,
    score_breakdown: primary.score_breakdown || plan.score_breakdown,
  };
  const rows = [selected, ...alternatives].slice(0, 4);
  if (!rows.length || !selected.strategy) { root.innerHTML = '<div class="distribution-empty">No feasible alternative plans were returned for this event.</div>'; return; }
  rows.forEach((row, index) => {
    const item = document.createElement('div'); item.className = `comparison-row${index === 0 ? ' recommended' : ''}`;
    const name = document.createElement('b'); name.className = 'comparison-name'; name.textContent = `${index === 0 ? 'PLAN 01 · ' : `PLAN ${String(index + 1).padStart(2, '0')} · `}${row.strategy || 'Strategy not named'}`;
    const score = numeric(row.score ?? row.score_breakdown?.total_score_seconds);
    const scoreNode = document.createElement('strong'); scoreNode.className = 'comparison-score'; scoreNode.textContent = score == null ? 'Score not supplied' : `${score.toFixed(2)}s`;
    const details = document.createElement('span'); details.className = 'comparison-detail';
    const pits = row.pit_laps || [];
    const estimated = numeric(row.estimated_race_time);
    details.textContent = `${pits.length} stop${pits.length === 1 ? '' : 's'}${pits.length ? ` · pit lap${pits.length === 1 ? '' : 's'} ${pits.map(count).join(', ')}` : ''}${estimated == null ? '' : ` · estimated race time ${(estimated / 60).toFixed(2)} min`}`;
    item.append(name, scoreNode, details);
    const breakdown = row.score_breakdown || {};
    const components = [
      breakdown.estimated_tyre_time_seconds != null ? `TYRE ${Number(breakdown.estimated_tyre_time_seconds).toFixed(1)}s` : null,
      breakdown.pit_loss_seconds != null ? `PIT LOSS ${Number(breakdown.pit_loss_seconds).toFixed(1)}s` : null,
      breakdown.ml_support_penalty_seconds != null ? `ML ${Number(breakdown.ml_support_penalty_seconds).toFixed(2)}s` : null,
    ].filter(Boolean);
    if (components.length) { const detail = document.createElement('small'); detail.className = 'comparison-components'; detail.textContent = components.join(' · '); item.append(detail); }
    root.append(item);
  });
}

function renderScoreBreakdown(breakdown) {
  const root = $('#score-breakdown'); root.innerHTML = '';
  const fields = [
    ['ESTIMATED TYRE TIME', breakdown.estimated_tyre_time_seconds, 'seconds'],
    ['DEGRADATION', breakdown.degradation_contribution_seconds, 'diagnostic · included in tyre time'],
    ['PIT LOSS', breakdown.pit_loss_seconds, 'seconds'],
    ['WEATHER', breakdown.weather_adjustment_seconds, 'diagnostic · included in tyre time'],
    ['HISTORICAL STRATEGY', breakdown.historical_strategy_penalty_seconds, 'seconds'],
    ['PIT-WINDOW PRIOR', breakdown.historical_pit_window_penalty_seconds, 'seconds'],
    ['ML SUPPORT', breakdown.ml_support_penalty_seconds, 'seconds'],
    ['TYRE AGE / INVENTORY', breakdown.age_inventory_penalty_seconds, 'seconds'],
    ['TOTAL STRATEGY SCORE', breakdown.total_score_seconds, 'lower score ranks first'],
  ].filter(([, value]) => value != null && numeric(value) != null);
  if (!fields.length) { root.innerHTML = '<p class="distribution-empty">The prediction API did not return score components.</p>'; return; }
  fields.forEach(([label, value, note], index) => {
    const row = document.createElement('div'); row.className = `score-row${index === fields.length - 1 ? ' total' : ''}`;
    const name = document.createElement('span'); name.textContent = label;
    const amount = document.createElement('b'); amount.textContent = `${Number(value).toFixed(2)}s`;
    const hint = document.createElement('small'); hint.textContent = note;
    row.append(name, hint, amount); root.append(row);
  });
  if (breakdown.interpretation) { const note = document.createElement('p'); note.className = 'score-note'; note.textContent = breakdown.interpretation; root.append(note); }
}

function nominationLabel(value) {
  if (!value) return 'No official compound nomination in the local register';
  if (typeof value === 'object' && !Array.isArray(value)) {
    const hard = value.HardCompound, medium = value.MediumCompound, soft = value.SoftCompound;
    if (hard && medium && soft) return `Hard ${hard} · Medium ${medium} · Soft ${soft}${String(value.role_mapping_source || '').startsWith('derived') ? ' · mapped from official event range' : ''}`;
    if (value.NominatedRange) return `Official range ${value.NominatedRange} · roles not mapped`;
    return value.available ? 'Official selection recorded' : 'No official compound selection recorded';
  }
  if (Array.isArray(value)) return value.map((row) => typeof row === 'string' ? row : `${row?.compound || ''} ${row?.tyre || row?.code || ''}`.trim()).filter(Boolean).join(', ');
  return String(value);
}

function renderReasons(primary, result, payload) {
  const root = $('#why-list'); root.innerHTML = '';
  const reasons = [];
  if (primary.reason) reasons.push(primary.reason);
  const track = payload.historical_track || {}, coverage = state.context?.circuit_analysis?.coverage || {};
  const localRaceCount = numeric(coverage.race_count);
  if (localRaceCount != null && localRaceCount > 0) {
    reasons.push(`${count(coverage.pit_transition_count)} source-backed pit transitions across ${count(localRaceCount)} circuit races before the event cutoff.`);
  }
  if (numeric(coverage.tyre_observation_count) === 0 && localRaceCount > 0) {
    reasons.push('This circuit sample contains no tyre compound or tyre-age observations. The API does not return numeric all-circuit fallback profiles for display.');
  }
  if (track.median_pit_lap != null) reasons.push(`The selected sample's median observed pit lap is ${Math.round(Number(track.median_pit_lap))}.`);
  const forecast = result.weather?.forecast || payload.forecast;
  if (forecast?.available) reasons.push(`Event-window weather: ${forecast.condition || forecast.dominant_weather || 'conditions available'}${numeric(forecast.precipitation_probability) == null ? '' : ` · ${Number(forecast.precipitation_probability).toFixed(0)}% mean rain chance`}.`);
  else reasons.push('No event-date forecast is available; the strategy run used the explicit weather fallback.');
  if (result.nomination) reasons.push(`Tyre nomination source: ${nominationLabel(result.nomination)}.`);
  if (result.robust_strategy?.rules?.length) reasons.push(...result.robust_strategy.rules.slice(0, 2));
  reasons.slice(0, 6).forEach((text) => { const item = document.createElement('li'); item.textContent = text; root.append(item); });
}

async function loadHistoricalSeasons() {
  const select = $('#historical-season'); select.innerHTML = ''; setOption(select, '', 'Choose season');
  state.seasons.forEach((year) => setOption(select, year, String(year)));
  select.disabled = !state.seasons.length;
}

async function loadHistoricalEvents(season) {
  const eventSelect = $('#historical-event'); eventSelect.innerHTML = ''; setOption(eventSelect, '', 'Loading completed races…'); eventSelect.disabled = true;
  $('#historical-driver').innerHTML = ''; setOption($('#historical-driver'), '', 'Select a completed race first'); $('#historical-driver').disabled = true; $('#historical-run').disabled = true;
  if (!season) return;
  try {
    const data = await api(`/api/historical-events?season=${encodeURIComponent(season)}`);
    eventSelect.innerHTML = ''; setOption(eventSelect, '', 'Choose a completed race');
    (data.events || []).forEach((event) => setOption(eventSelect, event.event_id, `${event.race} · ${event.circuit} · ${fmtDate(event.event_date)}`));
    if (!(data.events || []).length) setOption(eventSelect, '', 'No completed race rows in this season');
    eventSelect.disabled = false;
  } catch (error) { eventSelect.innerHTML = ''; setOption(eventSelect, '', cleanError(error, 'Completed races could not be loaded.')); }
}

async function loadHistoricalDrivers(eventId) {
  const select = $('#historical-driver'); select.innerHTML = ''; setOption(select, '', 'Loading event drivers…'); select.disabled = true;
  if (!eventId) return;
  try {
    const data = await api(`/api/drivers?event_id=${encodeURIComponent(eventId)}`);
    select.innerHTML = ''; setOption(select, '', 'Choose observed participant');
    (data.drivers || []).forEach((driver) => setOption(select, driver.driver_code, `${driver.driver_name || driver.driver_code} · ${driver.driver_code}`));
    if (!(data.drivers || []).length) setOption(select, '', 'No source-backed driver entries');
    select.disabled = false; $('#historical-run').disabled = true;
  } catch (error) { select.innerHTML = ''; setOption(select, '', cleanError(error, 'Event participants could not be loaded.')); }
}

async function runBacktest() {
  const eventId = $('#historical-event').value, driver = $('#historical-driver').value, output = $('#historical-result');
  if (!eventId || !driver) return;
  output.hidden = false; output.innerHTML = '<p class="empty-state"><b>BACKTEST RUNNING</b>Scoring from the pre-race cutoff, then comparing against the completed event.</p>';
  $('#historical-run').disabled = true;
  try {
    const result = await api('/api/backtest', { method: 'POST', body: JSON.stringify({ event_id: eventId, driver }) });
    output.innerHTML = '';
    const rows = [
      ['PREDICTED STRATEGY', result.predicted_strategy], ['ACTUAL STRATEGY', result.actual_strategy],
      ['PREDICTED PIT LAPS', Array.isArray(result.predicted_pit_laps) ? result.predicted_pit_laps.map(count).join(', ') || 'No stop' : 'Not returned'],
      ['ACTUAL PIT LAPS', Array.isArray(result.actual_pit_laps) ? result.actual_pit_laps.map(count).join(', ') || 'No stop' : 'Not returned'],
      ['SEQUENCE SIMILARITY', pct(result.strategy_sequence_similarity)], ['TRAINING CUTOFF', result.training_cutoff_time_utc],
      ['EVALUATION LAPS USED FOR TRAINING', result.selected_event_laps_used_for_training == null ? 'Not reported' : (result.selected_event_laps_used_for_training ? 'Yes' : 'No')],
    ];
    rows.forEach(([label, value]) => {
      const item = document.createElement('div'); item.className = 'backtest-metric';
      const name = document.createElement('span'); name.textContent = label;
      const amount = document.createElement('b'); amount.textContent = value == null || value === '' ? 'Not reported' : String(value);
      item.append(name, amount); output.append(item);
    });
    if (result.note) { const note = document.createElement('p'); note.className = 'chart-note'; note.textContent = result.note; output.append(note); }
  } catch (error) {
    output.textContent = cleanError(error, 'Backtest could not be completed for this race and driver.');
  } finally { $('#historical-run').disabled = false; }
}

async function showQuality() {
  try {
    if (!state.quality) state.quality = await api('/api/quality');
    const report = state.quality;
    const root = $('#quality-details'); root.innerHTML = '';
    const rows = [
      ...Object.entries(report.tables || {}).map(([name, amount]) => [name.replaceAll('_', ' '), count(amount)]),
      ['Duplicate event / driver / lap keys', count(report.duplicate_event_driver_lap_keys)],
      ['Events without historical weather', count(report.missing_weather_events)],
      ['Overall status', report.status || 'Not reported'],
    ];
    rows.forEach(([label, value]) => {
      const item = document.createElement('div'); item.className = 'quality-row';
      const name = document.createElement('span'); name.textContent = label;
      const amount = document.createElement('b'); amount.textContent = value;
      item.append(name, amount); root.append(item);
    });
    $('#quality-dialog').showModal();
  } catch (error) { toast(cleanError(error, 'Data-quality details could not be loaded.')); }
}

function toast(message) {
  const node = $('#toast'); node.textContent = message; node.hidden = false;
  window.setTimeout(() => { node.hidden = true; }, 4200);
}

$('#driver-select').addEventListener('change', (event) => { state.driver = event.target.value || null; updateDriverCard(); });
$('#predict-button').addEventListener('click', runPrediction);
$('#refresh-weather').addEventListener('click', () => loadWeather(true));
$('#refresh-data').addEventListener('click', async () => {
  const button = $('#refresh-data'); button.disabled = true;
  try {
    const data = await api('/api/update', { method: 'POST', body: JSON.stringify({ force: true }) });
    toast(`SQLite refreshed · ${count(data.laps)} laps · ${count(data.events)} events`);
    await loadHome();
  } catch (error) { toast(cleanError(error, 'The local database refresh failed.')); }
  finally { button.disabled = false; }
});
$('#historical-season').addEventListener('change', (event) => loadHistoricalEvents(event.target.value));
$('#historical-event').addEventListener('change', (event) => loadHistoricalDrivers(event.target.value));
$('#historical-driver').addEventListener('change', (event) => { $('#historical-run').disabled = !event.target.value; });
$('#historical-run').addEventListener('click', runBacktest);
$('#quality-link').addEventListener('click', showQuality);
$('#quality-dialog').addEventListener('click', (event) => { if (event.target === $('#quality-dialog')) $('#quality-dialog').close(); });
$('#race-season').addEventListener('change', (event) => { loadRaceOptions(event.target.value); });
$('#race-select').addEventListener('change', (event) => { if (event.target.value) selectRace(event.target.value, { preset: 'choose' }); });
$('#current-race-button').addEventListener('click', () => selectRacePreset('current'));
$('#next-race-button').addEventListener('click', () => selectRacePreset('next'));
$('#choose-race-button').addEventListener('click', () => { state.activePreset = 'choose'; updatePresetButtons(); $('#race-select').focus(); });
loadHome();
