/**
 * Predict-to-Protect Front-End Client Application
 * Handles instant weather fetching, Chart.js visualizations, and real-time ensemble inference.
 */

document.addEventListener('DOMContentLoaded', () => {
    let currentWeatherData = null;
    let pestChartInstance = null;
    let shapChartInstance = null;

    const locationSelect = document.getElementById('location-select');
    const manualLocationInput = document.getElementById('manual-location-input');
    const presetLocGroup = document.getElementById('preset-location-group');
    const manualLocGroup = document.getElementById('manual-location-group');
    const btnSyncManualLoc = document.getElementById('btn-sync-manual-loc');
    const locModeRadios = document.getElementsByName('loc_mode');

    const varietySelect = document.getElementById('variety-select');
    const stageSelect = document.getElementById('stage-select');
    const soilSelect = document.getElementById('soil-select');
    const prevPestRadios = document.getElementsByName('prev_pest');
    const prevPestType = document.getElementById('prev-pest-type');
    const prevPestTypeGroup = document.getElementById('prev-pest-type-group');
    const btnPredict = document.getElementById('btn-predict');

    // Helper to get active location string
    function getActiveLocation() {
        const mode = document.querySelector('input[name="loc_mode"]:checked').value;
        if (mode === 'manual') {
            return manualLocationInput.value.trim() || 'Cuttack, Odisha';
        }
        return locationSelect.value;
    }

    // Toggle location mode
    locModeRadios.forEach(radio => {
        radio.addEventListener('change', (e) => {
            if (e.target.value === 'manual') {
                presetLocGroup.style.display = 'none';
                manualLocGroup.style.display = 'block';
                updateWeather(getActiveLocation());
            } else {
                presetLocGroup.style.display = 'block';
                manualLocGroup.style.display = 'none';
                updateWeather(getActiveLocation());
            }
        });
    });

    // Manual location sync handlers
    btnSyncManualLoc.addEventListener('click', () => {
        updateWeather(getActiveLocation());
    });

    manualLocationInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter') {
            e.preventDefault();
            updateWeather(getActiveLocation());
        }
    });

    // Radio toggle listener for previous pest type dropdown
    prevPestRadios.forEach(radio => {
        radio.addEventListener('change', (e) => {
            if (e.target.value === '1') {
                prevPestTypeGroup.style.opacity = '1';
                prevPestType.disabled = false;
            } else {
                prevPestTypeGroup.style.opacity = '0.5';
                prevPestType.disabled = true;
            }
        });
    });

    // 1. Function to Fetch Live Weather
    async function updateWeather(location) {
        try {
            document.getElementById('weather-source-badge').innerText = 'Fetching Live Weather...';
            const res = await fetch(`/api/weather?location=${encodeURIComponent(location)}`);
            if (!res.ok) throw new Error('Weather fetch failed');
            const data = await res.json();
            currentWeatherData = data;

            // Update UI elements
            document.getElementById('resolved-location-text').innerText = 
                `📍 Resolved: ${data.location_name} (${data.latitude.toFixed(2)}°N, ${data.longitude.toFixed(2)}°E)`;
            document.getElementById('weather-source-badge').innerText = data.source;

            document.getElementById('w-temp').innerHTML = `${data.current_temp.toFixed(1)} <span class="stat-unit">°C</span>`;
            document.getElementById('w-temp-sub').innerText = `Max: ${data.temp_max.toFixed(1)}° | Min: ${data.temp_min.toFixed(1)}°`;

            document.getElementById('w-humidity').innerHTML = `${data.humidity.toFixed(0)} <span class="stat-unit">%</span>`;
            document.getElementById('w-rh-sub').innerText = `Morning RH: ${data.humidity_morning.toFixed(0)}%`;

            document.getElementById('w-rain').innerHTML = `${data.rainfall_7d.toFixed(1)} <span class="stat-unit">mm</span>`;
            document.getElementById('w-rain-sub').innerText = `14-Day Sum: ${data.rainfall_14d.toFixed(1)} mm`;

            const trendBadge = document.getElementById('w-trend-badge');
            trendBadge.className = `trend-badge ${data.rainfall_trend_badge}`;
            trendBadge.innerText = `${data.rainfall_trend_icon} ${Math.abs(data.rainfall_trend_pct).toFixed(1)}%`;
            document.getElementById('w-trend-sub').innerText = `${data.rainfall_trend_label} vs prior 7d`;

            document.getElementById('w-wind').innerText = `${data.wind_speed.toFixed(1)} km/h`;
            document.getElementById('w-sun').innerText = `${data.sunshine_hours.toFixed(1)} hrs`;
            document.getElementById('w-evap').innerText = `${data.evaporation.toFixed(1)} mm`;

        } catch (err) {
            console.error('Weather error:', err);
            document.getElementById('weather-source-badge').innerText = 'Agro-climatic Baseline';
        }
    }

    // Trigger weather update on preset location change
    locationSelect.addEventListener('change', (e) => {
        updateWeather(e.target.value);
    });

    // 2. Initialize Charts
    function initCharts() {
        const ctxPest = document.getElementById('pestChart').getContext('2d');
        pestChartInstance = new Chart(ctxPest, {
            type: 'bar',
            data: {
                labels: ['Brownplanthopper', 'Yellowstemborer', 'LeafFolder', 'Gallmidge', 'Greenleafhopper', 'LeafBlast'],
                datasets: [{
                    label: 'Outbreak Probability (%)',
                    data: [82.4, 11.2, 3.4, 1.8, 0.8, 0.4],
                    backgroundColor: ['#EF4444', '#F59E0B', '#94A3B8', '#CBD5E1', '#E2E8F0', '#F1F5F9'],
                    borderRadius: 6
                }]
            },
            options: {
                indexAxis: 'y',
                responsive: true,
                maintainAspectRatio: false,
                plugins: { legend: { display: false } },
                scales: {
                    x: { beginAtZero: true, max: 100, grid: { color: '#F1F5F9' } },
                    y: { grid: { display: false }, ticks: { font: { family: 'Plus Jakarta Sans', size: 11, weight: '600' } } }
                }
            }
        });

        const ctxShap = document.getElementById('shapChart').getContext('2d');
        shapChartInstance = new Chart(ctxShap, {
            type: 'bar',
            data: {
                labels: ['Relative Humidity', 'Rainfall Trend', 'Prior Pest Infestation', 'Temperature', 'Growth Stage'],
                datasets: [{
                    label: 'Feature Contribution (%)',
                    data: [32, 26, 21, 14, 7],
                    backgroundColor: ['#EF4444', '#EF4444', '#EF4444', '#EF4444', '#10B981'],
                    borderRadius: 6
                }]
            },
            options: {
                indexAxis: 'y',
                responsive: true,
                maintainAspectRatio: false,
                plugins: { legend: { display: false } },
                scales: {
                    x: { beginAtZero: true, max: 40, grid: { color: '#F1F5F9' } },
                    y: { grid: { display: false }, ticks: { font: { family: 'Plus Jakarta Sans', size: 11, weight: '600' } } }
                }
            }
        });
    }

    // 3. Execute Outbreak Prediction
    async function runPrediction() {
        const prevPestVal = document.querySelector('input[name="prev_pest"]:checked').value;
        const activeLocation = getActiveLocation();
        const payload = {
            location: activeLocation,
            rice_variety: varietySelect.value,
            growth_stage: stageSelect.value,
            soil_type: soilSelect.value,
            previous_pest_occurrence: parseInt(prevPestVal),
            previous_pest_type: prevPestType.value,
            weather: currentWeatherData
        };

        btnPredict.disabled = true;
        btnPredict.innerHTML = '<span class="btn-icon">⏳</span> Computing Ensemble ML & SHAP...';

        try {
            const res = await fetch('/api/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });

            if (!res.ok) throw new Error('Prediction API failed');
            const data = await res.json();

            // 1. Update Result Card Header & Risk Badge
            const resultCard = document.getElementById('result-card');
            resultCard.style.borderLeft = `6px solid ${data.risk_color}`;

            document.getElementById('res-alert-title').innerText = data.advisory.alert_title;
            
            const riskBadge = document.getElementById('res-risk-badge');
            riskBadge.innerText = `${data.risk_icon} ${data.risk_level} RISK`;
            riskBadge.className = `risk-badge ${data.risk_level === 'HIGH' ? 'risk-badge-high' : (data.risk_level === 'MEDIUM' ? 'risk-badge-medium' : 'risk-badge-low')}`;

            // 2. Update Metrics Trio
            document.getElementById('res-pest-name').innerText = data.common_name;
            document.getElementById('res-pest-sci').innerText = `(${data.scientific_name})`;

            const probElem = document.getElementById('res-prob-val');
            probElem.innerText = `${data.outbreak_probability}%`;
            probElem.style.color = data.risk_color;

            document.getElementById('res-rf-prob').innerText = `${data.rf_probability}%`;
            document.getElementById('res-xgb-prob').innerText = `${data.xgb_probability}%`;

            document.getElementById('res-summary-text').innerHTML = data.advisory.summary;

            // 3. Update Pest Spectrum Chart
            if (pestChartInstance && data.pest_spectrum) {
                pestChartInstance.data.labels = data.pest_spectrum.map(p => p.pest_name);
                pestChartInstance.data.datasets[0].data = data.pest_spectrum.map(p => p.probability);
                pestChartInstance.data.datasets[0].backgroundColor = data.pest_spectrum.map((p, i) => 
                    i === 0 ? data.risk_color : (i === 1 ? '#F59E0B' : '#94A3B8')
                );
                pestChartInstance.update();
            }

            // 4. Update SHAP Chart & Explanation
            if (shapChartInstance && data.xai) {
                document.getElementById('xai-summary-note').innerHTML = `💡 <strong>Key Rationale:</strong> ${data.xai.summary}`;
                
                const topFactors = data.xai.top_factors;
                shapChartInstance.data.labels = topFactors.map(f => f.name);
                shapChartInstance.data.datasets[0].data = topFactors.map(f => f.percentage);
                shapChartInstance.data.datasets[0].backgroundColor = topFactors.map(f => 
                    f.impact === 'Increases Risk' ? '#EF4444' : '#10B981'
                );
                shapChartInstance.update();
            }

            // 5. Update Advisory Section
            document.getElementById('adv-vuln-stage').innerText = data.advisory.vulnerable_stage;
            document.getElementById('adv-fav-clim').innerText = data.advisory.favorable_conditions;

            const stepsContainer = document.getElementById('advisory-steps-list');
            stepsContainer.innerHTML = '';
            data.advisory.action_list.forEach((step, idx) => {
                const item = document.createElement('div');
                item.className = 'advisory-item';
                item.innerHTML = `<strong>Step ${idx + 1}:</strong> ${step}`;
                stepsContainer.appendChild(item);
            });

            const contextBox = document.getElementById('contextual-notes-box');
            if (data.advisory.contextual_notes && data.advisory.contextual_notes.length > 0) {
                contextBox.style.display = 'block';
                contextBox.innerHTML = `<strong>📌 Micro-Climate & Field Context:</strong><br>` + 
                    data.advisory.contextual_notes.map(n => `• ${n}`).join('<br>');
            } else {
                contextBox.style.display = 'none';
            }

            // Smooth scroll to results
            resultCard.scrollIntoView({ behavior: 'smooth', block: 'start' });

        } catch (err) {
            console.error('Prediction failed:', err);
            alert('Prediction could not be completed. Please ensure server is running.');
        } finally {
            btnPredict.disabled = false;
            btnPredict.innerHTML = '<span class="btn-icon">🔮</span> PREDICT PEST OUTBREAK (NEXT 2–3 WEEKS)';
        }
    }

    btnPredict.addEventListener('click', runPrediction);

    // Initial Startup
    initCharts();
    updateWeather(locationSelect.value).then(() => {
        runPrediction(); // Auto-run initial prediction for instant interactive presentation
    });
});
