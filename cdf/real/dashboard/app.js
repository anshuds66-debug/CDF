/* ================================================================
   CDF Real-LLM Dashboard — app.js

   Connects to FastAPI backend; drives all UI sections.
   ================================================================ */

const API = '/api';


// ── State ────────────────────────────────────────────────────────────────

let providersData = [];

let currentExpId = null;
let eventSource = null;

let expTotal = 0;
let expCompleted = 0;
let expFailed = 0;

// Per-horizon live data
let relData = {};     // {h: {n, s}}
let latData = {};     // {h: {count, total}}
let failData = {};    // {step: count}

// Chart references
let chartRel;
let chartLat;
let chartFail;

let chartAnRel;
let chartAnDrift;
let chartAnFF;

let currentExperimentTasks = [];


// ── SSE deduplication ───────────────────────────────────────────────────

// When the dashboard switches from the general stream to the
// experiment-specific stream, buffered events can overlap with
// events already received from the general stream.
//
// Keep a bounded set of event fingerprints so the same event is
// processed only once.
const seenEventKeys = new Set();
const MAX_SEEN_EVENTS = 5000;


// ── Navigation ──────────────────────────────────────────────────────────

function showSection(id) {

    document
        .querySelectorAll('section')
        .forEach(s => s.classList.remove('active'));

    document
        .querySelectorAll('nav button')
        .forEach(b => b.classList.remove('active'));

    const section =
        document.getElementById('sec-' + id);

    if (section) {
        section.classList.add('active');
    }

    const navButton =
        document.getElementById('nav-' + id);

    if (navButton) {
        navButton.classList.add('active');
    }

    if (id === 'providers') {
        loadProviders();
    }

    if (id === 'history') {
        loadHistory();
    }

    if (id === 'analysis') {
        populateAnalysisSelect();
    }
}


// ── Providers ───────────────────────────────────────────────────────────

async function loadProviders() {

    try {

        const res =
            await fetch(`${API}/providers`);

        if (!res.ok) {
            throw new Error(
                `HTTP ${res.status}`
            );
        }

        providersData =
            await res.json();

        const list =
            document.getElementById(
                'provider-list'
            );

        const sel =
            document.getElementById(
                'exp-provider'
            );

        if (!list || !sel) {
            return;
        }

        list.innerHTML = '';
        sel.innerHTML = '';

        providersData.forEach(p => {

            // Provider card
            const card =
                document.createElement('div');

            card.className = 'card';

            card.innerHTML = `
                <h3>
                    ${escapeHtml(p.name)}
                    ${
                        p.is_fake
                            ? ' <span style="color:#f39c12;font-size:0.75rem">[Mock]</span>'
                            : ''
                    }
                </h3>

                <p>
                    ID:
                    <code>${escapeHtml(p.id)}</code>
                </p>

                <p>
                    Models:
                    ${p.models
                        .map(escapeHtml)
                        .join(', ')}
                </p>

                ${
                    p.base_url
                        ? `<p>URL: ${escapeHtml(p.base_url)}</p>`
                        : ''
                }

                <p>
                    API Key:
                    ${
                        p.has_key
                            ? `<code>${escapeHtml(p.masked_key || '****')}</code>`
                            : '<em>Not set</em>'
                    }
                </p>

                <div class="provider-actions">

                    <button
                        class="btn-sm btn-test"
                        onclick="testProvider('${escapeJs(p.id)}')"
                    >
                        Test
                    </button>

                    ${
                        !p.is_fake
                            ? `
                                <button
                                    class="btn-sm btn-danger"
                                    onclick="deleteProvider('${escapeJs(p.id)}')"
                                >
                                    Delete
                                </button>
                              `
                            : ''
                    }

                </div>

                <p
                    id="test-result-${escapeHtml(p.id)}"
                    style="font-size:0.78rem;margin-top:0.4rem"
                ></p>
            `;

            list.appendChild(card);


            // Dropdown option
            const opt =
                document.createElement('option');

            opt.value = p.id;
            opt.textContent = p.name;

            sel.appendChild(opt);
        });

        updateModels();

    } catch (err) {

        console.error(
            'Failed to load providers:',
            err
        );
    }
}


// ── Provider save ────────────────────────────────────────────────────────

async function saveProvider(e) {

    e.preventDefault();

    const id =
        document
            .getElementById('prov-id')
            .value
            .trim();

    const body = {
        name:
            document.getElementById(
                'prov-name'
            ).value,

        models:
            document.getElementById(
                'prov-models'
            ).value,

        base_url:
            document.getElementById(
                'prov-url'
            ).value,

        api_key:
            document.getElementById(
                'prov-key'
            ).value,

        is_openai_compatible:
            document.getElementById(
                'prov-compat'
            ).checked,
    };

    try {

        const res =
            await fetch(
                `${API}/providers/${encodeURIComponent(id)}`,
                {
                    method: 'POST',

                    headers: {
                        'Content-Type':
                            'application/json',
                    },

                    body:
                        JSON.stringify(body),
                }
            );

        if (!res.ok) {
            throw new Error(
                `HTTP ${res.status}`
            );
        }

        e.target.reset();

        await loadProviders();

    } catch (err) {

        alert(
            `Failed to save provider: ${err}`
        );
    }
}


// ── Provider delete ─────────────────────────────────────────────────────

async function deleteProvider(id) {

    if (
        !confirm(
            `Delete provider "${id}"?`
        )
    ) {
        return;
    }

    try {

        const res =
            await fetch(
                `${API}/providers/${encodeURIComponent(id)}`,
                {
                    method: 'DELETE',
                }
            );

        if (!res.ok) {
            throw new Error(
                `HTTP ${res.status}`
            );
        }

        await loadProviders();

    } catch (err) {

        alert(
            `Failed to delete provider: ${err}`
        );
    }
}


// ── Provider test ───────────────────────────────────────────────────────

async function testProvider(id) {

    const el =
        document.getElementById(
            `test-result-${id}`
        );

    if (!el) {
        return;
    }

    el.textContent = 'Testing…';
    el.style.color = '#7f8c8d';

    try {

        const res =
            await fetch(
                `${API}/providers/${encodeURIComponent(id)}/test`
            );

        const data =
            await res.json();

        if (data.ok) {

            el.textContent =
                `✓ ${data.message}`;

            el.style.color =
                '#27ae60';

        } else {

            el.textContent =
                `✗ ${data.message}`;

            el.style.color =
                '#e74c3c';
        }

    } catch (err) {

        el.textContent =
            `✗ ${err}`;

        el.style.color =
            '#e74c3c';
    }
}


// ── Model dropdown ──────────────────────────────────────────────────────

function updateModels() {

    const providerSelect =
        document.getElementById(
            'exp-provider'
        );

    const modelSelect =
        document.getElementById(
            'exp-model'
        );

    if (!providerSelect || !modelSelect) {
        return;
    }

    const pId =
        providerSelect.value;

    const p =
        providersData.find(
            x => x.id === pId
        );

    modelSelect.innerHTML = '';

    (p ? p.models : []).forEach(m => {

        const opt =
            document.createElement(
                'option'
            );

        opt.value = m;
        opt.textContent = m;

        modelSelect.appendChild(opt);
    });
}


// ── Experiment start ────────────────────────────────────────────────────

async function startExperiment(e) {

    e.preventDefault();

    const rawH =
        document.getElementById(
            'exp-horizons'
        ).value;

    const horizons =
        rawH
            .split(',')
            .map(
                s => parseInt(
                    s.trim(),
                    10
                )
            )
            .filter(
                n => !isNaN(n)
            );

    if (!horizons.length) {

        alert(
            'Please enter at least one valid horizon.'
        );

        return;
    }

    const payload = {

        provider_id:
            document.getElementById(
                'exp-provider'
            ).value,

        model:
            document.getElementById(
                'exp-model'
            ).value,

        horizons,

        samples_per_horizon:
            parseInt(
                document.getElementById(
                    'exp-samples'
                ).value,
                10
            ),

        rollouts:
            parseInt(
                document.getElementById(
                    'exp-rollouts'
                ).value,
                10
            ),

        concurrency:
            parseInt(
                document.getElementById(
                    'exp-concurrency'
                ).value,
                10
            ),

        temperature:
            parseFloat(
                document.getElementById(
                    'exp-temperature'
                )?.value || 0
            ),
    };


    try {

        resetLiveView(horizons);

        showSection('live');

        const res =
            await fetch(
                `${API}/start`,
                {
                    method: 'POST',

                    headers: {
                        'Content-Type':
                            'application/json',
                    },

                    body:
                        JSON.stringify(payload),
                }
            );

        if (!res.ok) {

            const errorBody =
                await safeJson(res);

            throw new Error(
                errorBody.error ||
                errorBody.message ||
                `HTTP ${res.status}`
            );
        }

        const data =
            await res.json();

        currentExpId =
            data.experiment_id;

        try {
            sessionStorage.setItem(
                'cdf_active_exp_id',
                currentExpId
            );
        } catch (e) {}

        document.getElementById(
            'live-exp-id'
        ).textContent =
            currentExpId;

        document.getElementById(
            'btn-stop'
        ).style.display =
            'inline-block';

        setStatus('running');

        // Clear old deduplication state for a fresh experiment.
        // We still retain the global bound so reconnects within this
        // experiment remain protected.
        seenEventKeys.clear();

        // Connect to the experiment-specific stream.
        // The backend will replay buffered events.
        connectSSE(currentExpId);

    } catch (err) {

        setStatus('failed');

        const feed =
            document.getElementById(
                'live-feed'
            );

        if (feed) {
            logFeed(
                `Failed to start experiment: ${err.message || err}`,
                'error'
            );
        }

        alert(
            `Failed to start experiment: ${err.message || err}`
        );
    }
}


// ── Stop experiment ─────────────────────────────────────────────────────

async function stopExperiment() {

    if (!currentExpId) {
        return;
    }

    setStatus('stopping');

    try {

        await fetch(
            `${API}/stop/${encodeURIComponent(currentExpId)}`,
            {
                method: 'DELETE',
            }
        );

    } catch (err) {

        logFeed(
            `Failed to stop experiment: ${err}`,
            'error'
        );
    }
}


// ── SSE ─────────────────────────────────────────────────────────────────

function connectSSE(expId = null) {

    if (eventSource) {

        eventSource.close();
        eventSource = null;
    }

    const url =
        expId
            ? `${API}/stream?experiment_id=${encodeURIComponent(expId)}`
            : `${API}/stream`;

    eventSource =
        new EventSource(url);

    eventSource.onmessage = e => {

        try {

            const ev =
                JSON.parse(e.data);

            // Create a stable event fingerprint.
            //
            // This prevents duplicate processing when:
            // 1. the general stream already delivered an event
            // 2. the experiment stream replays the same event
            const eventKey =
                JSON.stringify(ev);

            if (
                seenEventKeys.has(eventKey)
            ) {
                return;
            }

            seenEventKeys.add(
                eventKey
            );

            // Keep memory bounded.
            if (
                seenEventKeys.size >
                MAX_SEEN_EVENTS
            ) {

                const firstKey =
                    seenEventKeys
                        .values()
                        .next()
                        .value;

                if (firstKey) {
                    seenEventKeys.delete(
                        firstKey
                    );
                }
            }

            handleEvent(ev);

        } catch (err) {

            console.error(
                'Invalid SSE event:',
                err
            );
        }
    };


    eventSource.onerror = () => {

        if (eventSource) {

            eventSource.close();
            eventSource = null;
        }

        // Reconnect only when we know the active experiment.
        setTimeout(() => {

            if (currentExpId) {
                connectSSE(
                    currentExpId
                );
            }

        }, 3000);
    };
}


// ── Page load ───────────────────────────────────────────────────────────

window.addEventListener(
    'load',
    () => {

        loadProviders();

        let savedExpId = null;
        try {
            savedExpId = sessionStorage.getItem('cdf_active_exp_id');
        } catch (e) {}

        if (savedExpId) {
            currentExpId = savedExpId;
            const liveExpEl = document.getElementById('live-exp-id');
            if (liveExpEl) liveExpEl.textContent = currentExpId;
            const stopBtn = document.getElementById('btn-stop');
            if (stopBtn) stopBtn.style.display = 'inline-block';
            setStatus('running');
            connectSSE(currentExpId);
            showSection('live');
        } else {
            connectSSE();
        }
    }
);


// ── Event handler ───────────────────────────────────────────────────────

function handleEvent(ev) {

    switch (ev.event_type) {


        // ------------------------------------------------------------
        // Experiment started
        // ------------------------------------------------------------

        case 'ExperimentStarted':

            const isReplayForCurrent =
                (currentExpId === ev.experiment_id && (expCompleted > 0 || expFailed > 0));

            currentExpId =
                ev.experiment_id;

            try {
                sessionStorage.setItem('cdf_active_exp_id', currentExpId);
            } catch (e) {}

            expTotal =
                ev.total_tasks;

            document.getElementById(
                'live-exp-id'
            ).textContent =
                currentExpId;

            document.getElementById(
                'btn-stop'
            ).style.display =
                'inline-block';

            setStatus('running');

            if (!isReplayForCurrent) {
                expCompleted = 0;
                expFailed = 0;

                document.getElementById(
                    'live-progress'
                ).textContent =
                    `0 / ${expTotal}`;

                ev.horizons.forEach(
                    h => {

                        relData[h] = {
                            n: 0,
                            s: 0,
                        };

                        latData[h] = {
                            count: 0,
                            total: 0,
                        };
                    }
                );

                failData = {};

                initLiveCharts();

                updateCharts();

                logFeed(
                    `Experiment started — provider: ${ev.provider}, model: ${ev.model}, tasks: ${ev.total_tasks}`,
                    'info'
                );
            }

            break;


        // ------------------------------------------------------------
        // Normal completed/graded task
        // ------------------------------------------------------------

        case 'TaskCompleted':

            expCompleted++;

            if (
                relData[ev.horizon] !==
                undefined
            ) {

                // Every TaskCompleted is an attempt.
                relData[
                    ev.horizon
                ].n++;

                if (ev.success) {

                    relData[
                        ev.horizon
                    ].s++;
                }
            }

            if (
                latData[ev.horizon] !==
                undefined
            ) {

                latData[
                    ev.horizon
                ].count++;

                latData[
                    ev.horizon
                ].total +=
                    Number(
                        ev.latency || 0
                    );
            }

            if (
                !ev.success &&
                ev.first_failure_step != null
            ) {

                failData[
                    ev.first_failure_step
                ] =
                    (
                        failData[
                            ev.first_failure_step
                        ] || 0
                    ) + 1;
            }

            updateCharts();

            logFeed(
                `Task ${ev.task_id} h=${ev.horizon} — ${
                    ev.success
                        ? '✓ pass'
                        : `✗ fail @ step ${ev.first_failure_step}`
                }`,
                ev.success
                    ? 'success'
                    : 'error'
            );

            break;


        // ------------------------------------------------------------
        // API / transport / worker failure
        // ------------------------------------------------------------

        case 'TaskFailed':

            expFailed++;

            // IMPORTANT:
            // API-error tasks are attempted trajectories and
            // therefore belong in the reliability denominator.
            if (
                relData[ev.horizon] !==
                undefined
            ) {

                relData[
                    ev.horizon
                ].n++;
            }

            if (
                latData[ev.horizon] !==
                    undefined &&
                ev.latency != null
            ) {

                latData[
                    ev.horizon
                ].count++;

                latData[
                    ev.horizon
                ].total +=
                    Number(
                        ev.latency || 0
                    );
            }

            const errorLabel =
                ev.is_api_error
                    ? 'API error'
                    : 'Task error';

            logFeed(
                `${errorLabel} — ${
                    ev.task_id ||
                    `h=${ev.horizon}`
                }: ${
                    ev.error_message ||
                    'Unknown error'
                }`,
                'error'
            );

            updateCharts();

            break;


        // ------------------------------------------------------------
        // Progress
        // ------------------------------------------------------------

        case 'ExperimentProgress':

            document.getElementById(
                'live-progress'
            ).textContent =
                `${ev.completed} / ${ev.total}`;

            const pct =
                ev.total > 0
                    ? (
                        ev.completed /
                        ev.total *
                        100
                    ).toFixed(0)
                    : 0;

            document.getElementById(
                'progress-fill'
            ).style.width =
                pct + '%';

            document.getElementById(
                'live-success'
            ).textContent =
                `${(
                    ev.success_rate *
                    100
                ).toFixed(1)}%`;

            document.getElementById(
                'live-latency'
            ).textContent =
                `${Number(
                    ev.average_latency || 0
                ).toFixed(2)}s`;

            document.getElementById(
                'live-tokens'
            ).textContent =
                Number(
                    ev.total_tokens || 0
                ).toLocaleString();

            document.getElementById(
                'live-failures'
            ).textContent =
                expFailed;

            break;


        // ------------------------------------------------------------
        // Experiment completed
        // ------------------------------------------------------------

        case 'ExperimentCompleted':

            try {
                sessionStorage.removeItem('cdf_active_exp_id');
            } catch (e) {}

            setStatus('completed');

            document.getElementById(
                'btn-stop'
            ).style.display =
                'none';

            logFeed(
                'Experiment completed. Results saved.',
                'info'
            );

            break;


        // ------------------------------------------------------------
        // Experiment stopped
        // ------------------------------------------------------------

        case 'ExperimentStopped':

            try {
                sessionStorage.removeItem('cdf_active_exp_id');
            } catch (e) {}

            setStatus('stopped');

            document.getElementById(
                'btn-stop'
            ).style.display =
                'none';

            logFeed(
                'Experiment stopped by user.',
                'info'
            );

            break;


        // ------------------------------------------------------------
        // Experiment failed
        // ------------------------------------------------------------

        case 'ExperimentFailed':

            try {
                sessionStorage.removeItem('cdf_active_exp_id');
            } catch (e) {}

            setStatus('failed');

            document.getElementById(
                'btn-stop'
            ).style.display =
                'none';

            logFeed(
                `Experiment FAILED: ${
                    ev.error ||
                    'Unknown error'
                }`,
                'error'
            );

            break;
    }
}


// ── Status badge ────────────────────────────────────────────────────────

function setStatus(s) {

    const el =
        document.getElementById(
            'live-status-badge'
        );

    if (!el) {
        return;
    }

    el.className =
        `badge ${s}`;

    el.textContent =
        s.charAt(0).toUpperCase() +
        s.slice(1);
}


// ── Live feed helpers ───────────────────────────────────────────────────

function logFeed(
    msg,
    type = 'info'
) {

    const feed =
        document.getElementById(
            'live-feed'
        );

    if (!feed) {
        return;
    }

    const d =
        document.createElement('div');

    d.className =
        `log-item ${type}`;

    d.textContent =
        `[${new Date().toLocaleTimeString()}] ${msg}`;

    feed.prepend(d);

    // Keep feed bounded.
    while (
        feed.children.length >
        200
    ) {

        feed.removeChild(
            feed.lastChild
        );
    }
}


function clearFeed() {

    const feed =
        document.getElementById(
            'live-feed'
        );

    if (feed) {
        feed.innerHTML = '';
    }
}


// ── Chart setup & update ────────────────────────────────────────────────

function resetLiveView(horizons) {

    relData = {};
    latData = {};
    failData = {};

    document.getElementById(
        'live-feed'
    ).innerHTML = '';

    document.getElementById(
        'live-progress'
    ).textContent =
        '0 / 0';

    document.getElementById(
        'live-success'
    ).textContent =
        '—';

    document.getElementById(
        'live-failures'
    ).textContent =
        '0';

    document.getElementById(
        'live-latency'
    ).textContent =
        '—';

    document.getElementById(
        'live-tokens'
    ).textContent =
        '0';

    document.getElementById(
        'progress-fill'
    ).style.width =
        '0%';

    horizons.forEach(
        h => {

            relData[h] = {
                n: 0,
                s: 0,
            };

            latData[h] = {
                count: 0,
                total: 0,
            };
        }
    );

    initLiveCharts();
}


function initLiveCharts() {

    if (chartRel) {
        chartRel.destroy();
    }

    if (chartLat) {
        chartLat.destroy();
    }

    if (chartFail) {
        chartFail.destroy();
    }


    chartRel =
        new Chart(
            document.getElementById(
                'chart-reliability'
            ),
            {
                type: 'line',

                data: {
                    labels: [],

                    datasets: [
                        {
                            label:
                                'Reliability S(h)',

                            data: [],

                            borderColor:
                                '#2980b9',

                            backgroundColor:
                                'rgba(41,128,185,0.1)',

                            fill: true,

                            tension: 0.3,

                            pointRadius: 5,
                        },
                    ],
                },

                options: {

                    responsive: true,

                    plugins: {
                        legend: {
                            position: 'top'
                        },
                    },

                    scales: {

                        x: {
                            title: {
                                display: true,
                                text: 'Horizon h'
                            },
                        },

                        y: {

                            min: 0,
                            max: 1,

                            title: {
                                display: true,
                                text: 'S(h)'
                            },
                        },
                    },
                },
            }
        );


    chartLat =
        new Chart(
            document.getElementById(
                'chart-latency'
            ),
            {
                type: 'bar',

                data: {

                    labels: [],

                    datasets: [
                        {
                            label:
                                'Avg Latency (s)',

                            data: [],

                            backgroundColor:
                                '#e67e22',
                        },
                    ],
                },

                options: {

                    responsive: true,

                    scales: {

                        x: {
                            title: {
                                display: true,
                                text: 'Horizon h'
                            },
                        },

                        y: {

                            min: 0,

                            title: {
                                display: true,
                                text: 'Latency (s)'
                            },
                        },
                    },
                },
            }
        );


    chartFail =
        new Chart(
            document.getElementById(
                'chart-failure'
            ),
            {
                type: 'bar',

                data: {

                    labels: [],

                    datasets: [
                        {
                            label:
                                'First Failure Step',

                            data: [],

                            backgroundColor:
                                '#e74c3c',
                        },
                    ],
                },

                options: {

                    responsive: true,

                    scales: {

                        x: {
                            title: {
                                display: true,
                                text: 'Step'
                            },
                        },

                        y: {

                            beginAtZero: true,

                            title: {
                                display: true,
                                text: 'Count'
                            },
                        },
                    },
                },
            }
        );
}


function updateCharts() {

    if (
        !chartRel ||
        !chartLat ||
        !chartFail
    ) {
        return;
    }

    const horizons =
        Object.keys(relData)
            .map(Number)
            .sort(
                (a, b) => a - b
            );

    const labels =
        horizons.map(String);


    chartRel.data.labels =
        labels;

    chartRel.data.datasets[0].data =
        horizons.map(
            h =>
                relData[h].n > 0
                    ? relData[h].s /
                      relData[h].n
                    : null
        );

    chartRel.update('none');


    chartLat.data.labels =
        labels;

    chartLat.data.datasets[0].data =
        horizons.map(
            h =>
                latData[h].count > 0
                    ? latData[h].total /
                      latData[h].count
                    : null
        );

    chartLat.update('none');


    const steps =
        Object.keys(failData)
            .map(Number)
            .sort(
                (a, b) => a - b
            );

    chartFail.data.labels =
        steps.map(String);

    chartFail.data.datasets[0].data =
        steps.map(
            s => failData[s]
        );

    chartFail.update('none');
}


// ── CDF Analysis ────────────────────────────────────────────────────────

async function populateAnalysisSelect() {

    try {

        const res =
            await fetch(
                `${API}/history`
            );

        if (!res.ok) {
            throw new Error(
                `HTTP ${res.status}`
            );
        }

        const history =
            await res.json();

        const sel =
            document.getElementById(
                'analysis-exp-select'
            );

        if (!sel) {
            return;
        }

        sel.innerHTML =
            '<option value="">— choose experiment —</option>';

        history.forEach(h => {

            const opt =
                document.createElement(
                    'option'
                );

            opt.value =
                h.experiment_id;

            opt.textContent =
                `${h.experiment_id} (${h.model}, ${new Date(
                    h.timestamp * 1000
                ).toLocaleString()})`;

            sel.appendChild(opt);
        });

    } catch (err) {

        console.error(
            'Failed to populate analysis:',
            err
        );
    }
}


// ── Run analysis ────────────────────────────────────────────────────────

async function runAnalysis() {

    const expId =
        document.getElementById(
            'analysis-exp-select'
        ).value;

    if (!expId) {

        alert(
            'Please select an experiment.'
        );

        return;
    }

    const loading =
        document.getElementById(
            'analysis-loading'
        );

    const results =
        document.getElementById(
            'analysis-results'
        );

    const error =
        document.getElementById(
            'analysis-error'
        );

    loading.style.display =
        'block';

    results.style.display =
        'none';

    error.style.display =
        'none';


    try {

        const res =
            await fetch(
                `${API}/metrics/${encodeURIComponent(expId)}`
            );

        const m =
            await res.json();

        loading.style.display =
            'none';

        if (!res.ok) {

            error.textContent =
                m.error ||
                'Unable to compute metrics.';

            error.style.display =
                'block';

            return;
        }

        // Render all returned data, including
        // insufficient-data responses.
        renderAnalysis(m);

        results.style.display =
            'block';

    } catch (err) {

        loading.style.display =
            'none';

        error.textContent =
            `Analysis request failed: ${err}`;

        error.style.display =
            'block';
    }
}


// ── Formatting helper ──────────────────────────────────────────────────

function fmt(
    v,
    decimals = 4
) {

    if (
        v === null ||
        v === undefined ||
        (
            typeof v === 'number' &&
            !Number.isFinite(v)
        )
    ) {
        return '—';
    }

    if (
        typeof v === 'number'
    ) {
        return v.toFixed(
            decimals
        );
    }

    return String(v);
}


// ── Render CDF analysis ─────────────────────────────────────────────────

function renderAnalysis(m) {

    const analysisError =
        document.getElementById(
            'analysis-error'
        );


    // ------------------------------------------------------------
    // Overall status
    // ------------------------------------------------------------

    if (
        m.status ===
        'insufficient_data'
    ) {

        analysisError.textContent =
            `Insufficient data for exponential CDF fitting: ${
                m.reason ||
                'No successful trajectories were available.'
            }`;

        analysisError.style.display =
            'block';

    } else if (
        m.status ===
        'partial'
    ) {

        analysisError.textContent =
            m.reason ||
            'Analysis completed with limited data.';

        analysisError.style.display =
            'block';

    } else {

        analysisError.style.display =
            'none';
    }


    // ------------------------------------------------------------
    // Main metrics
    // ------------------------------------------------------------

    document.getElementById(
        'an-lambda'
    ).textContent =
        fmt(m.lambda);

    const ci =
        m.lambda_ci;

    document.getElementById(
        'an-lambda-ci'
    ).textContent =
        (
            ci &&
            ci[0] != null &&
            ci[1] != null
        )
            ? `95% CI [${fmt(
                ci[0]
            )}, ${fmt(
                ci[1]
            )}]`
            : '';

    document.getElementById(
        'an-A'
    ).textContent =
        fmt(m.A, 4);

    document.getElementById(
        'an-r'
    ).textContent =
        fmt(m.r, 4);

    document.getElementById(
        'an-h12'
    ).textContent =
        fmt(m.H_1_2, 2);

    document.getElementById(
        'an-h50'
    ).textContent =
        fmt(m.H_hat_50, 2);


    // ------------------------------------------------------------
    // Horizon table
    // ------------------------------------------------------------

    const tbody =
        document.getElementById(
            'an-table-body'
        );

    tbody.innerHTML = '';

    (
        m.horizons || []
    ).forEach(
        (h, i) => {

            const delta =
                m.delta_h
                    ? m.delta_h[
                        String(h)
                    ]
                    : undefined;

            const tr =
                document.createElement(
                    'tr'
                );

            tr.innerHTML = `
                <td>${escapeHtml(h)}</td>
                <td>${escapeHtml(
                    m.n_h?.[i] ?? 0
                )}</td>
                <td>${escapeHtml(
                    m.s_h?.[i] ?? 0
                )}</td>
                <td>${escapeHtml(
                    fmt(
                        m.s_hat?.[i],
                        4
                    )
                )}</td>
                <td>${
                    delta !== undefined
                        ? escapeHtml(
                            fmt(
                                delta,
                                4
                            )
                        )
                        : '—'
                }</td>
            `;

            tbody.appendChild(
                tr
            );
        }
    );


    // ------------------------------------------------------------
    // Reliability chart
    // ------------------------------------------------------------

    const labels =
        (
            m.horizons || []
        ).map(String);

    if (chartAnRel) {
        chartAnRel.destroy();
    }

    const measuredData =
        m.s_hat || [];

    const hasFit =
        Number.isFinite(m.A) &&
        Number.isFinite(m.lambda);

    const datasets = [
        {
            label:
                'Measured S(h)',

            data:
                measuredData,

            borderColor:
                '#2980b9',

            pointRadius:
                6,

            fill:
                false,

            tension:
                0.3,
        },
    ];


    // Do NOT create a fitted curve when the
    // experiment has no valid exponential fit.
    if (hasFit) {

        datasets.push(
            {
                label:
                    'Fitted A·e^{-λh}',

                data:
                    m.horizons.map(
                        h =>
                            m.A *
                            Math.exp(
                                -m.lambda *
                                h
                            )
                    ),

                borderColor:
                    '#e74c3c',

                borderDash:
                    [6, 3],

                pointRadius:
                    0,

                fill:
                    false,
            }
        );
    }


    chartAnRel =
        new Chart(
            document.getElementById(
                'chart-an-reliability'
            ),
            {
                type: 'line',

                data: {
                    labels,
                    datasets,
                },

                options: {

                    responsive: true,

                    plugins: {

                        legend: {
                            position:
                                'top',
                        },

                        title: {

                            display:
                                true,

                            text:
                                hasFit
                                    ? 'Reliability vs Horizon'
                                    : 'Measured Reliability — No Exponential Fit',
                        },
                    },

                    scales: {

                        x: {

                            title: {
                                display:
                                    true,

                                text:
                                    'Horizon h',
                            },
                        },

                        y: {

                            min:
                                0,

                            max:
                                1,

                            title: {

                                display:
                                    true,

                                text:
                                    'S(h)',
                            },
                        },
                    },
                },
            }
        );


    // ------------------------------------------------------------
    // Excess drift chart
    // ------------------------------------------------------------

    const driftH =
        Object.keys(
            m.delta_h || {}
        )
        .map(Number)
        .sort(
            (a, b) => a - b
        );

    const driftV =
        driftH.map(
            h =>
                m.delta_h[
                    String(h)
                ]
        );

    if (chartAnDrift) {
        chartAnDrift.destroy();
    }

    chartAnDrift =
        new Chart(
            document.getElementById(
                'chart-an-drift'
            ),
            {
                type: 'bar',

                data: {

                    labels:
                        driftH.map(String),

                    datasets: [
                        {
                            label:
                                'Excess Drift δ(h)',

                            data:
                                driftV,

                            backgroundColor:
                                '#f39c12',
                        },
                    ],
                },

                options: {

                    responsive:
                        true,

                    plugins: {

                        legend: {
                            position:
                                'top',
                        },

                        title: {

                            display:
                                true,

                            text:
                                'Excess Drift',
                        },
                    },

                    scales: {

                        x: {

                            title: {

                                display:
                                    true,

                                text:
                                    'Horizon h',
                            },
                        },

                        y: {

                            title: {

                                display:
                                    true,

                                text:
                                    'δ(h)',
                            },
                        },
                    },
                },
            }
        );


    // ------------------------------------------------------------
    // First-failure histogram
    // ------------------------------------------------------------

    const ffSteps =
        Object.keys(
            m.ff_histogram || {}
        )
        .map(Number)
        .sort(
            (a, b) => a - b
        );

    const ffCounts =
        ffSteps.map(
            s =>
                m.ff_histogram[
                    String(s)
                ]
        );

    if (chartAnFF) {
        chartAnFF.destroy();
    }

    chartAnFF =
        new Chart(
            document.getElementById(
                'chart-an-ff'
            ),
            {
                type: 'bar',

                data: {

                    labels:
                        ffSteps.map(String),

                    datasets: [
                        {
                            label:
                                'First Failure Step',

                            data:
                                ffCounts,

                            backgroundColor:
                                '#e74c3c',
                        },
                    ],
                },

                options: {

                    responsive:
                        true,

                    plugins: {

                        legend: {
                            position:
                                'top',
                        },

                        title: {

                            display:
                                true,

                            text:
                                'Failure Step Distribution',
                        },
                    },

                    scales: {

                        x: {

                            title: {

                                display:
                                    true,

                                text:
                                    'Step',
                            },
                        },

                        y: {

                            beginAtZero:
                                true,

                            title: {

                                display:
                                    true,

                                text:
                                    'Count',
                            },
                        },
                    },
                },
            }
        );
}


// ── History ─────────────────────────────────────────────────────────────

async function loadHistory() {

    try {

        const res =
            await fetch(
                `${API}/history`
            );

        if (!res.ok) {
            throw new Error(
                `HTTP ${res.status}`
            );
        }

        const history =
            await res.json();

        const tbody =
            document.getElementById(
                'history-body'
            );

        if (!tbody) {
            return;
        }

        tbody.innerHTML = '';

        if (!history.length) {

            tbody.innerHTML =
                `
                <tr>
                    <td
                        colspan="7"
                        class="placeholder"
                    >
                        No experiments yet.
                    </td>
                </tr>
                `;

            return;
        }


        history.forEach(h => {

            const date =
                new Date(
                    h.timestamp * 1000
                ).toLocaleString();

            const tr =
                document.createElement(
                    'tr'
                );

            tr.innerHTML = `
                <td>
                    <code>
                        ${escapeHtml(
                            h.experiment_id
                        )}
                    </code>
                </td>

                <td>
                    ${escapeHtml(
                        h.provider
                    )}
                </td>

                <td>
                    ${escapeHtml(
                        h.model
                    )}
                </td>

                <td>
                    ${escapeHtml(
                        date
                    )}
                </td>

                <td>
                    ${escapeHtml(
                        h.status
                    )}
                </td>

                <td>
                    ${escapeHtml(
                        h.total_completed
                    )}
                    /
                    ${escapeHtml(
                        h.total_tasks
                    )}
                </td>

                <td>

                    <button
                        class="btn-sm"
                        onclick="showExperimentTasks('${escapeJs(
                            h.experiment_id
                        )}')"
                    >
                        View Tasks
                    </button>

                    <button
                        class="btn-sm btn-test"
                        onclick="loadAnalysisFor('${escapeJs(
                            h.experiment_id
                        )}')"
                    >
                        Analyse
                    </button>

                </td>
            `;

            tbody.appendChild(
                tr
            );
        });

    } catch (err) {

        console.error(
            'Failed to load history:',
            err
        );
    }
}


// ── Task detail modal ───────────────────────────────────────────────────

async function showExperimentTasks(expId) {
    try {
        const res = await fetch(`${API}/history/${encodeURIComponent(expId)}`);
        if (!res.ok) {
            const data = await safeJson(res);
            throw new Error(data.error || `HTTP ${res.status}`);
        }

        const events = await res.json();
        const tasks = events.filter(
            e => e.event_type === 'TaskCompleted' || e.event_type === 'TaskFailed'
        );

        if (!tasks.length) {
            alert('No task events found for this experiment.');
            return;
        }

        currentExperimentTasks = tasks;

        const titleEl = document.getElementById('task-modal-title');
        if (titleEl) {
            titleEl.textContent = `Tasks for ${expId} (${tasks.length} total)`;
        }

        const listTbody = document.getElementById('task-modal-list');
        if (listTbody) {
            listTbody.innerHTML = '';
            tasks.forEach((t, idx) => {
                const tr = document.createElement('tr');
                tr.id = `task-row-${idx}`;
                tr.style.cursor = 'pointer';
                const statusText = t.success ? 'PASS' : (t.is_api_error ? 'API ERROR' : 'FAIL');
                const statusColor = t.success ? '#27ae60' : '#e74c3c';
                const latText = t.latency != null ? `${Number(t.latency).toFixed(2)}s` : '—';

                tr.innerHTML = `
                    <td><code>${escapeHtml(t.task_id || `task-${idx}`)}</code></td>
                    <td>${escapeHtml(t.horizon)}</td>
                    <td>${escapeHtml(t.rollout_id)}</td>
                    <td style="color: ${statusColor}; font-weight: bold;">${escapeHtml(statusText)}</td>
                    <td>${escapeHtml(latText)}</td>
                    <td>
                        <button class="btn-sm" onclick="selectTask(${idx}); event.stopPropagation();">
                            View
                        </button>
                    </td>
                `;
                tr.onclick = () => selectTask(idx);
                listTbody.appendChild(tr);
            });
        }

        // Default to selecting the first task
        selectTask(0);

        const modal = document.getElementById('task-modal');
        if (modal) {
            modal.classList.add('open');
        }

    } catch (err) {
        alert(`Failed to load tasks: ${err.message || err}`);
    }
}

function selectTask(idx) {
    if (!currentExperimentTasks || !currentExperimentTasks[idx]) return;

    const tbody = document.getElementById('task-modal-list');
    if (tbody) {
        Array.from(tbody.children).forEach((row, i) => {
            if (i === idx) {
                row.style.backgroundColor = '#e8f4fc';
                row.style.fontWeight = 'bold';
            } else {
                row.style.backgroundColor = '';
                row.style.fontWeight = 'normal';
            }
        });
    }

    renderTaskDetail(currentExperimentTasks[idx]);
}

function renderTaskDetail(ev) {
    const content = document.getElementById('task-detail-content');
    if (!content) return;

    const fields = [
        ['Task ID', ev.task_id],
        ['Horizon', ev.horizon],
        ['Rollout ID', ev.rollout_id],
        ['Seed', ev.seed ?? '—'],
        [
            'Status',
            ev.success ? 'PASS' : (ev.is_api_error ? 'API ERROR' : 'FAIL'),
            ev.success ? 'success' : 'error'
        ],
        ['First Failure Step', ev.first_failure_step ?? '—'],
        ['Ground Truth', ev.ground_truth ? JSON.stringify(ev.ground_truth) : '—'],
        ['Parsed Output', ev.parsed_output ? JSON.stringify(ev.parsed_output) : '—'],
        ['Latency', ev.latency != null ? `${Number(ev.latency).toFixed(3)}s` : '—'],
        ['Input Tokens', ev.input_tokens ?? '—'],
        ['Output Tokens', ev.output_tokens ?? '—'],
        ['Total Tokens', ev.total_tokens ?? '—'],
        ['API Error', ev.is_api_error ? 'Yes' : 'No'],
        ['Error', ev.error_message ?? '—'],
        ['Raw Output', ev.raw_output ?? '—'],
    ];

    content.innerHTML = fields
        .map(([label, val, cls = '']) => `
            <div class="field-label">${escapeHtml(label)}</div>
            <div class="field-value ${escapeHtml(cls)}">${escapeHtml(val)}</div>
        `)
        .join('');
}

function openTaskModal(ev) {
    renderTaskDetail(ev);
    const modal = document.getElementById('task-modal');
    if (modal) {
        modal.classList.add('open');
    }
}


function closeTaskModal() {

    const modal =
        document.getElementById(
            'task-modal'
        );

    if (modal) {
        modal.classList.remove(
            'open'
        );
    }
}


function closeModal(e) {

    if (
        e.target.id ===
        'task-modal'
    ) {
        closeTaskModal();
    }
}


// ── Quick-jump from history to analysis ─────────────────────────────────

async function loadAnalysisFor(
    expId
) {

    showSection(
        'analysis'
    );

    await populateAnalysisSelect();

    const select =
        document.getElementById(
            'analysis-exp-select'
        );

    if (select) {

        select.value =
            expId;
    }

    runAnalysis();
}


// ── Utility helpers ─────────────────────────────────────────────────────

function escapeHtml(value) {

    if (
        value === null ||
        value === undefined
    ) {
        return '';
    }

    return String(value)
        .replace(
            /&/g,
            '&amp;'
        )
        .replace(
            /</g,
            '&lt;'
        )
        .replace(
            />/g,
            '&gt;'
        )
        .replace(
            /"/g,
            '&quot;'
        )
        .replace(
            /'/g,
            '&#039;'
        );
}


function escapeJs(value) {

    if (
        value === null ||
        value === undefined
    ) {
        return '';
    }

    return String(value)
        .replace(
            /\\/g,
            '\\\\'
        )
        .replace(
            /'/g,
            "\\'"
        )
        .replace(
            /"/g,
            '\\"'
        )
        .replace(
            /\n/g,
            '\\n'
        )
        .replace(
            /\r/g,
            '\\r'
        );
}


async function safeJson(
    response
) {

    try {
        return await response.json();
    } catch {
        return {};
    }
}