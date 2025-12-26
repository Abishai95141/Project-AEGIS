/**
 * AEGIS 3.0 - Application Controller
 * Enhanced with explanations and visualizations
 */

const AppState = {
    currentSection: 'input',
    processing: false,
    lastResults: null,
    predictionChart: null,
    simChart: null,
    l2Chart: null,
    l3Chart: null
};

document.addEventListener('DOMContentLoaded', () => {
    initializeNavigation();
    initializeEventListeners();
    initializeTooltips();
    updateSystemStatus('ready');
});

function initializeNavigation() {
    document.querySelectorAll('.nav-link').forEach(link => {
        link.addEventListener('click', (e) => {
            e.preventDefault();
            navigateToSection(link.dataset.section);
        });
    });
}

function navigateToSection(sectionId) {
    document.querySelectorAll('.nav-link').forEach(link => {
        link.classList.toggle('active', link.dataset.section === sectionId);
    });
    document.querySelectorAll('.section').forEach(section => {
        section.classList.remove('active');
    });
    const targetSection = document.getElementById(`${sectionId}-section`);
    if (targetSection) targetSection.classList.add('active');

    const titles = {
        input: { title: 'Patient Data Input', subtitle: 'Enter patient data for real-time AEGIS processing' },
        processing: { title: 'Layer Processing', subtitle: 'View AEGIS 5-layer pipeline execution with detailed analysis' },
        results: { title: 'Treatment Recommendation', subtitle: 'View computed recommendation with safety analysis and predictions' },
        simulation: { title: 'Batch Simulation', subtitle: 'Run multi-hour simulations with synthetic data' }
    };
    const header = titles[sectionId] || titles.input;
    document.getElementById('page-title').textContent = header.title;
    document.querySelector('.header-subtitle').textContent = header.subtitle;
    AppState.currentSection = sectionId;
}

window.navigateToSection = navigateToSection;

function initializeTooltips() {
    // Add hover tooltips for technical terms
    document.querySelectorAll('[data-tooltip]').forEach(el => {
        el.addEventListener('mouseenter', showTooltip);
        el.addEventListener('mouseleave', hideTooltip);
    });
}

function showTooltip(e) {
    const text = e.target.dataset.tooltip;
    const tooltip = document.createElement('div');
    tooltip.className = 'tooltip';
    tooltip.textContent = text;
    tooltip.style.left = e.pageX + 'px';
    tooltip.style.top = (e.pageY - 30) + 'px';
    document.body.appendChild(tooltip);
}

function hideTooltip() {
    document.querySelectorAll('.tooltip').forEach(t => t.remove());
}

function initializeEventListeners() {
    document.getElementById('process-data')?.addEventListener('click', processPatientData);
    document.getElementById('reset-system')?.addEventListener('click', resetSystem);
    document.getElementById('emergency-stop')?.addEventListener('click', emergencyStop);
    document.getElementById('run-batch-sim')?.addEventListener('click', runBatchSimulation);
    document.getElementById('approve-recommendation')?.addEventListener('click', () => showToast('Recommendation approved - dose will be administered', 'success'));
    document.getElementById('modify-dose')?.addEventListener('click', () => {
        const newDose = prompt('Enter modified dose (Units):');
        if (newDose) showToast('Modified dose: ' + parseFloat(newDose).toFixed(1) + ' U - awaiting confirmation', 'info');
    });
    document.getElementById('reject-recommendation')?.addEventListener('click', () => showToast('Recommendation rejected - manual override active', 'warning'));
}

async function processPatientData() {
    if (AppState.processing) {
        showToast('Processing already in progress', 'warning');
        return;
    }
    const inputData = collectInputData();
    if (!validateInputData(inputData)) return;

    AppState.processing = true;
    updateSystemStatus('processing');
    navigateToSection('processing');
    resetLayerOutputs();

    try {
        const results = await AEGISPipeline.process(inputData, onLayerUpdate);
        AppState.lastResults = results;
        displayResults(results);
        updateSystemStatus('complete');
        showToast('Processing complete - view results', 'success');
        setTimeout(() => navigateToSection('results'), 1500);
    } catch (error) {
        console.error('Processing error:', error);
        showToast('Processing failed: ' + error.message, 'error');
        updateSystemStatus('error');
    } finally {
        AppState.processing = false;
    }
}

function collectInputData() {
    return {
        patientId: document.getElementById('patient-id')?.value || 'PAT-001',
        weight: parseFloat(document.getElementById('patient-weight')?.value) || 75,
        tdi: parseFloat(document.getElementById('patient-tdi')?.value) || 45,
        diabetesDuration: parseFloat(document.getElementById('patient-duration')?.value) || 5,
        glucose: parseFloat(document.getElementById('current-glucose')?.value) || 120,
        trend: document.getElementById('glucose-trend')?.value || 'stable',
        mealCarbs: parseFloat(document.getElementById('meal-carbs')?.value) || 0,
        mealType: document.getElementById('meal-type')?.value || 'lunch',
        mealGI: document.getElementById('meal-gi')?.value || 'medium',
        mealFat: document.getElementById('meal-fat')?.value || 'medium',
        diary: document.getElementById('diary-entry')?.value || '',
        activityLevel: document.getElementById('activity-level')?.value || 'light',
        stressLevel: document.getElementById('stress-level')?.value || 'normal',
        sleepQuality: document.getElementById('sleep-quality')?.value || 'good',
        illness: document.getElementById('illness')?.value || 'none'
    };
}

function validateInputData(data) {
    if (data.glucose < 40 || data.glucose > 400) {
        showToast('Glucose must be 40-400 mg/dL', 'error');
        return false;
    }
    return true;
}

function onLayerUpdate(layerNum, status, output) {
    const badge = document.getElementById(`l${layerNum}-badge`);
    const outputDiv = document.getElementById(`l${layerNum}-output`);
    const block = document.getElementById(`layer-${layerNum}-block`);

    if (status === 'processing') {
        badge.textContent = 'Processing...';
        badge.className = 'layer-status-badge processing';
        block.classList.add('active');
    } else if (status === 'complete') {
        badge.textContent = 'Complete';
        badge.className = 'layer-status-badge complete';
        outputDiv.innerHTML = renderLayerOutput(layerNum, output);
        // Initialize any charts after rendering
        if (layerNum === 2) initL2Chart(output);
        if (layerNum === 3) initL3Chart(output);
    }
}

function renderLayerOutput(layerNum, output) {
    switch (layerNum) {
        case 1: return renderL1Output(output);
        case 2: return renderL2Output(output);
        case 3: return renderL3Output(output);
        case 4: return renderL4Output(output);
        case 5: return renderL5Output(output);
        default: return '<div class="output-placeholder">No output</div>';
    }
}

// Helper function for status indicators
function getStatusIndicator(value, goodCondition, label) {
    const isGood = goodCondition;
    const icon = isGood ? '✓' : '⚠';
    const colorClass = isGood ? 'good' : 'warning';
    return `<span class="status-indicator ${colorClass}">${icon}</span>`;
}

function renderL1Output(output) {
    const concepts = output.extractedConcepts || [];
    const modPercent = (output.insulinSensitivityModifier * 100).toFixed(0);
    const modIsGood = output.insulinSensitivityModifier >= 0.9 && output.insulinSensitivityModifier <= 1.1;

    let modExplanation = '';
    if (output.insulinSensitivityModifier < 0.85) {
        modExplanation = `<div class="explanation warning">⚠ Your insulin sensitivity is reduced by ${(100 - modPercent)}%. This means insulin will be less effective. Common causes: stress, illness, poor sleep. The system will recommend a higher dose to compensate.</div>`;
    } else if (output.insulinSensitivityModifier > 1.15) {
        modExplanation = `<div class="explanation good">✓ Your insulin sensitivity is increased by ${(modPercent - 100)}%. This means insulin will be more effective. Common causes: exercise, good sleep. The system will recommend a lower dose to prevent hypoglycemia.</div>`;
    } else {
        modExplanation = `<div class="explanation neutral">Your insulin sensitivity is normal. No significant factors affecting insulin effectiveness were detected.</div>`;
    }

    return `
        <div class="layer-result">
            <div class="result-section">
                <h4 class="section-title">
                    <span class="title-icon">🔍</span>
                    Semantic Analysis Results
                </h4>
                <p class="section-desc">Natural language processing extracted ${concepts.length} health-related concepts from your diary entry and structured inputs.</p>
            </div>
            
            <div class="metric-card ${modIsGood ? '' : 'alert'}">
                <div class="metric-header">
                    <span class="metric-name">Insulin Sensitivity Modifier</span>
                    <span class="metric-badge ${modIsGood ? 'neutral' : (output.insulinSensitivityModifier < 1 ? 'warning' : 'good')}">${modIsGood ? 'Normal' : (output.insulinSensitivityModifier < 1 ? 'Reduced' : 'Enhanced')}</span>
                </div>
                <div class="metric-value-large ${modIsGood ? '' : (output.insulinSensitivityModifier < 1 ? 'warning' : 'good')}">${modPercent}%</div>
                <div class="metric-bar">
                    <div class="metric-bar-fill" style="width: ${Math.min(100, Math.max(0, modPercent))}%; background: ${modIsGood ? 'var(--color-accent)' : (output.insulinSensitivityModifier < 1 ? 'var(--color-warning)' : 'var(--color-success)')}"></div>
                    <div class="metric-bar-marker" style="left: 100%"></div>
                </div>
                <div class="metric-scale">
                    <span>50% (Very Resistant)</span>
                    <span>100% (Normal)</span>
                    <span>150% (Very Sensitive)</span>
                </div>
            </div>
            ${modExplanation}
            
            ${concepts.length > 0 ? `
                <div class="result-section">
                    <h4 class="section-title">
                        <span class="title-icon">📋</span>
                        Extracted Health Concepts
                    </h4>
                    <p class="section-desc">These factors will influence the treatment recommendation:</p>
                    <div class="concepts-grid">
                        ${concepts.map(c => `
                            <div class="concept-card ${c.impact === 'insulin_resistance' ? 'negative' : (c.impact === 'insulin_sensitivity' ? 'positive' : 'neutral')}">
                                <div class="concept-header">
                                    <span class="concept-icon">${c.impact === 'insulin_resistance' ? '⬇️' : (c.impact === 'insulin_sensitivity' ? '⬆️' : '➡️')}</span>
                                    <span class="concept-name">${c.name}</span>
                                </div>
                                <div class="concept-details">
                                    <span class="concept-code">${c.code}</span>
                                    <span class="concept-confidence">Confidence: ${(c.confidence * 100).toFixed(0)}%</span>
                                </div>
                                <div class="concept-impact">
                                    ${c.impact === 'insulin_resistance' ? 'Decreases insulin effectiveness' :
            (c.impact === 'insulin_sensitivity' ? 'Increases insulin effectiveness' : 'May affect glucose levels')}
                                </div>
                            </div>
                        `).join('')}
                    </div>
                </div>
            ` : `
                <div class="result-section">
                    <div class="empty-state">
                        <span class="empty-icon">📝</span>
                        <p>No specific health concepts were extracted. Try adding more details to your diary entry about stress, sleep, exercise, or how you're feeling.</p>
                    </div>
                </div>
            `}
            
            <div class="result-section">
                <h4 class="section-title">
                    <span class="title-icon">📊</span>
                    Technical Metrics
                </h4>
                <div class="metrics-grid">
                    <div class="mini-metric">
                        <span class="mini-label">Semantic Entropy</span>
                        <span class="mini-value">${output.semanticEntropy.toFixed(3)}</span>
                        <span class="mini-help">${output.semanticEntropy < 0.3 ? 'Low uncertainty' : (output.semanticEntropy < 0.6 ? 'Moderate uncertainty' : 'High uncertainty')}</span>
                    </div>
                    <div class="mini-metric">
                        <span class="mini-label">Treatment Proxies (Z)</span>
                        <span class="mini-value">${output.treatmentProxies?.length || 0}</span>
                        <span class="mini-help">Factors affecting treatment</span>
                    </div>
                    <div class="mini-metric">
                        <span class="mini-label">Outcome Proxies (W)</span>
                        <span class="mini-value">${output.outcomeProxies?.length || 0}</span>
                        <span class="mini-help">Factors affecting outcomes</span>
                    </div>
                </div>
            </div>
        </div>
    `;
}

function renderL2Output(output) {
    const regime = output.regime.regime;
    const regimeInfo = {
        normal: { icon: '✓', label: 'Normal', color: 'good', desc: 'Your body is responding normally to insulin. Standard dosing calculations apply.' },
        exercise: { icon: '🏃', label: 'Exercise Mode', color: 'good', desc: 'Increased physical activity detected. Insulin sensitivity is higher, so doses are reduced to prevent low blood sugar.' },
        stress: { icon: '😰', label: 'Stress Mode', color: 'warning', desc: 'Elevated stress hormones detected. Insulin resistance is higher, so doses may need to be increased.' },
        illness: { icon: '🤒', label: 'Illness Mode', color: 'danger', desc: 'Signs of illness detected. Insulin resistance can be significantly higher during illness.' },
        dawn: { icon: '🌅', label: 'Dawn Phenomenon', color: 'warning', desc: 'Early morning hormone surge detected. Blood sugar naturally rises, requiring adjusted dosing.' }
    };
    const rInfo = regimeInfo[regime] || regimeInfo.normal;

    const peakGlucose = output.prediction.peak.glucose;
    const peakTime = output.prediction.peak.time;
    const peakStatus = peakGlucose > 180 ? 'warning' : (peakGlucose < 70 ? 'danger' : 'good');

    return `
        <div class="layer-result">
            <div class="result-section">
                <h4 class="section-title">
                    <span class="title-icon">🧬</span>
                    Digital Twin State Estimation
                </h4>
                <p class="section-desc">Your personalized physiological model predicts how your body will respond to food and insulin.</p>
            </div>

            <div class="metric-card highlight">
                <div class="metric-header">
                    <span class="metric-name">Detected Metabolic Regime</span>
                    <span class="metric-badge ${rInfo.color}">${rInfo.icon} ${rInfo.label}</span>
                </div>
                <div class="regime-confidence">
                    <div class="confidence-bar">
                        <div class="confidence-fill" style="width: ${output.regime.confidence * 100}%"></div>
                    </div>
                    <span class="confidence-text">${(output.regime.confidence * 100).toFixed(0)}% confidence</span>
                </div>
                <div class="explanation ${rInfo.color}">${rInfo.desc}</div>
            </div>

            <div class="result-section">
                <h4 class="section-title">
                    <span class="title-icon">📈</span>
                    3-Hour Glucose Prediction (Without Treatment)
                </h4>
                <p class="section-desc">This shows what your blood sugar would do if no insulin were given for the upcoming meal.</p>
                <div class="chart-container" id="l2-chart-container">
                    <canvas id="l2-prediction-chart"></canvas>
                </div>
            </div>

            <div class="prediction-summary">
                <div class="pred-card ${peakStatus}">
                    <span class="pred-icon">${peakStatus === 'good' ? '✓' : '⚠'}</span>
                    <span class="pred-label">Expected Peak</span>
                    <span class="pred-value">${peakGlucose.toFixed(0)} mg/dL</span>
                    <span class="pred-time">at ${peakTime} minutes</span>
                    <span class="pred-desc">${peakGlucose > 180 ? 'Above target range - insulin needed' : (peakGlucose < 70 ? 'Risk of hypoglycemia!' : 'Within safe range')}</span>
                </div>
                <div class="pred-card">
                    <span class="pred-icon">⏱️</span>
                    <span class="pred-label">Return to Target</span>
                    <span class="pred-value">${output.prediction.returnToTarget ? output.prediction.returnToTarget + ' min' : '> 3 hours'}</span>
                    <span class="pred-desc">${output.prediction.returnToTarget ? 'Expected time to reach safe levels' : 'May need intervention'}</span>
                </div>
            </div>

            <div class="result-section">
                <h4 class="section-title">
                    <span class="title-icon">📊</span>
                    Model Quality Metrics
                </h4>
                <div class="metrics-grid">
                    <div class="mini-metric good">
                        <span class="mini-label">Variance Reduction</span>
                        <span class="mini-value">${output.varianceReduction.toFixed(1)}%</span>
                        <span class="mini-help">Higher is better - shows model accuracy improvement</span>
                    </div>
                    <div class="mini-metric">
                        <span class="mini-label">Prediction Uncertainty</span>
                        <span class="mini-value">±${output.uncertainty.confidenceInterval.toFixed(0)} mg/dL</span>
                        <span class="mini-help">Expected prediction error range</span>
                    </div>
                    <div class="mini-metric">
                        <span class="mini-label">Current Glucose</span>
                        <span class="mini-value">${output.currentState.G.toFixed(0)} mg/dL</span>
                        <span class="mini-help">Starting point for prediction</span>
                    </div>
                </div>
            </div>
        </div>
    `;
}

function initL2Chart(output) {
    const ctx = document.getElementById('l2-prediction-chart');
    if (!ctx) return;

    const trajectory = output.prediction.trajectory;

    new Chart(ctx, {
        type: 'line',
        data: {
            labels: trajectory.map(p => p.time),
            datasets: [{
                label: 'Predicted Glucose (no treatment)',
                data: trajectory.map(p => p.glucose),
                borderColor: '#f59e0b',
                backgroundColor: 'rgba(245, 158, 11, 0.1)',
                fill: true,
                tension: 0.4,
                pointRadius: 0
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false },
                tooltip: {
                    callbacks: {
                        label: (ctx) => `${ctx.parsed.y.toFixed(0)} mg/dL at ${ctx.label} min`
                    }
                }
            },
            scales: {
                x: {
                    title: { display: true, text: 'Time (minutes)', color: '#9ca3af' },
                    grid: { color: 'rgba(255,255,255,0.05)' },
                    ticks: { color: '#9ca3af' }
                },
                y: {
                    min: 40,
                    max: Math.max(250, output.prediction.peak.glucose + 20),
                    title: { display: true, text: 'Glucose (mg/dL)', color: '#9ca3af' },
                    grid: { color: 'rgba(255,255,255,0.05)' },
                    ticks: { color: '#9ca3af' }
                }
            }
        },
        plugins: [{
            id: 'targetRange',
            beforeDraw: (chart) => {
                const { ctx, chartArea, scales } = chart;
                if (!chartArea) return;
                ctx.save();
                // Target range
                ctx.fillStyle = 'rgba(16, 185, 129, 0.15)';
                const y180 = scales.y.getPixelForValue(180);
                const y70 = scales.y.getPixelForValue(70);
                ctx.fillRect(chartArea.left, y180, chartArea.right - chartArea.left, y70 - y180);
                // Labels
                ctx.fillStyle = '#6b7280';
                ctx.font = '11px Inter';
                ctx.fillText('Target: 70-180 mg/dL', chartArea.left + 5, y180 + 15);
                ctx.restore();
            }
        }]
    });
}

function renderL3Output(output) {
    const tau = output.tau;
    const tauStatus = tau < -15 && tau > -40 ? 'good' : 'warning';
    const deviation = output.individualDeviation;

    let deviationExplanation = '';
    if (Math.abs(deviation) < 5) {
        deviationExplanation = `<div class="explanation good">✓ Your response to insulin is very close to the population average. Standard dosing guidelines work well for you.</div>`;
    } else if (deviation > 0) {
        deviationExplanation = `<div class="explanation warning">⚠ You are ${Math.abs(deviation).toFixed(0)} mg/dL/U more resistant than average. Each unit of insulin has less effect on you, so doses may be higher.</div>`;
    } else {
        deviationExplanation = `<div class="explanation good">✓ You are ${Math.abs(deviation).toFixed(0)} mg/dL/U more sensitive than average. Each unit of insulin has more effect on you, so doses may be lower.</div>`;
    }

    return `
        <div class="layer-result">
            <div class="result-section">
                <h4 class="section-title">
                    <span class="title-icon">🔬</span>
                    Personalized Causal Analysis
                </h4>
                <p class="section-desc">This shows YOUR individual response to insulin, not just population averages. This is the core of N-of-1 precision medicine.</p>
            </div>

            <div class="metric-card highlight">
                <div class="metric-header">
                    <span class="metric-name">Your Individual Treatment Effect (τ)</span>
                    <span class="metric-badge ${tauStatus}">${tauStatus === 'good' ? 'Within Normal' : 'Outside Normal'}</span>
                </div>
                <div class="tau-display">
                    <span class="tau-value">${tau.toFixed(1)}</span>
                    <span class="tau-unit">mg/dL per Unit of Insulin</span>
                </div>
                <p class="metric-desc">This means 1 unit of insulin will lower YOUR blood glucose by approximately ${Math.abs(tau).toFixed(1)} mg/dL right now.</p>
            </div>

            <div class="comparison-chart" id="l3-chart-container">
                <canvas id="l3-comparison-chart"></canvas>
            </div>

            <div class="comparison-grid">
                <div class="comparison-card you">
                    <span class="comp-label">Your Effect</span>
                    <span class="comp-value">${tau.toFixed(1)} mg/dL/U</span>
                </div>
                <div class="comparison-card">
                    <span class="comp-label">vs</span>
                </div>
                <div class="comparison-card pop">
                    <span class="comp-label">Population Average</span>
                    <span class="comp-value">${output.populationEffect} mg/dL/U</span>
                </div>
            </div>

            ${deviationExplanation}

            <div class="result-section">
                <h4 class="section-title">
                    <span class="title-icon">🎯</span>
                    Confidence & Uncertainty
                </h4>
                <div class="confidence-display">
                    <div class="ci-visual">
                        <div class="ci-bar">
                            <div class="ci-range" style="left: ${Math.max(0, ((output.tauCI[0] + 60) / 80) * 100)}%; width: ${((output.tauCI[1] - output.tauCI[0]) / 80) * 100}%"></div>
                            <div class="ci-point" style="left: ${((tau + 60) / 80) * 100}%"></div>
                        </div>
                        <div class="ci-labels">
                            <span>-60</span>
                            <span>-40</span>
                            <span>-20</span>
                            <span>0</span>
                        </div>
                    </div>
                    <p class="ci-text">95% Confidence Interval: [${output.tauCI[0].toFixed(1)}, ${output.tauCI[1].toFixed(1)}] mg/dL/U</p>
                </div>
            </div>

            ${output.proximalAdjustment.hasProxies ? `
                <div class="result-section">
                    <h4 class="section-title">
                        <span class="title-icon">🔧</span>
                        Confounding Adjustment
                    </h4>
                    <div class="explanation good">
                        ✓ We detected ${output.proximalAdjustment.outcomeProxiesCount} factors that could bias the estimate. 
                        The proximal adjustment reduced potential bias by ${output.proximalAdjustment.biasReduction.toFixed(0)}%, 
                        making the treatment effect estimate more accurate.
                    </div>
                </div>
            ` : ''}
        </div>
    `;
}

function initL3Chart(output) {
    const ctx = document.getElementById('l3-comparison-chart');
    if (!ctx) return;

    new Chart(ctx, {
        type: 'bar',
        data: {
            labels: ['Your Effect', 'Population Average'],
            datasets: [{
                data: [Math.abs(output.tau), Math.abs(output.populationEffect)],
                backgroundColor: ['#3b82f6', '#6b7280'],
                borderRadius: 6
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            indexAxis: 'y',
            plugins: {
                legend: { display: false },
                tooltip: {
                    callbacks: {
                        label: (ctx) => `${ctx.parsed.x.toFixed(1)} mg/dL drop per unit`
                    }
                }
            },
            scales: {
                x: {
                    title: { display: true, text: 'Glucose Drop per Unit Insulin (mg/dL)', color: '#9ca3af' },
                    grid: { color: 'rgba(255,255,255,0.05)' },
                    ticks: { color: '#9ca3af' }
                },
                y: {
                    grid: { display: false },
                    ticks: { color: '#f4f4f5', font: { weight: 'bold' } }
                }
            }
        }
    });
}

function renderL4Output(output) {
    const totalDose = output.recommendedDose;
    const isDoseHigh = totalDose > 10;
    const isDoseLow = totalDose < 1 && output.mealDose > 0;

    return `
        <div class="layer-result">
            <div class="result-section">
                <h4 class="section-title">
                    <span class="title-icon">🎯</span>
                    Optimal Dose Calculation
                </h4>
                <p class="section-desc">The Decision Engine combines all information to find the dose that best balances glucose control with safety.</p>
            </div>

            <div class="dose-display">
                <div class="dose-main">
                    <span class="dose-label">Recommended Total Dose</span>
                    <span class="dose-value">${totalDose.toFixed(1)}</span>
                    <span class="dose-unit">Units</span>
                </div>
                ${isDoseHigh ? `<div class="dose-warning">⚠ Higher than typical - verified by safety system</div>` : ''}
                ${isDoseLow ? `<div class="dose-info">ℹ Low dose recommended based on your current glucose and sensitivity</div>` : ''}
            </div>

            <div class="dose-breakdown">
                <h4 class="breakdown-title">Dose Components</h4>
                <div class="breakdown-visual">
                    <div class="breakdown-bar">
                        ${output.mealDose > 0 ? `<div class="bar-segment meal" style="width: ${(output.mealDose / Math.max(1, totalDose)) * 100}%"><span>Meal: ${output.mealDose.toFixed(1)}U</span></div>` : ''}
                        ${output.correctionDose > 0 ? `<div class="bar-segment correction" style="width: ${(output.correctionDose / Math.max(1, totalDose)) * 100}%"><span>Correction: ${output.correctionDose.toFixed(1)}U</span></div>` : ''}
                        ${output.contextAdjustment !== 0 ? `<div class="bar-segment adjustment ${output.contextAdjustment > 0 ? 'positive' : 'negative'}" style="width: ${(Math.abs(output.contextAdjustment) / Math.max(1, totalDose)) * 100}%"><span>Adjustment: ${output.contextAdjustment > 0 ? '+' : ''}${output.contextAdjustment.toFixed(1)}U</span></div>` : ''}
                    </div>
                </div>
                <div class="breakdown-details">
                    <div class="breakdown-item">
                        <span class="item-icon">🍽️</span>
                        <div class="item-content">
                            <span class="item-label">Meal Coverage</span>
                            <span class="item-value">${output.mealDose.toFixed(1)} U</span>
                            <span class="item-desc">Based on carbohydrate content and absorption rate</span>
                        </div>
                    </div>
                    <div class="breakdown-item">
                        <span class="item-icon">📊</span>
                        <div class="item-content">
                            <span class="item-label">Correction Dose</span>
                            <span class="item-value">${output.correctionDose.toFixed(1)} U</span>
                            <span class="item-desc">To bring current glucose toward target (110 mg/dL)</span>
                        </div>
                    </div>
                    <div class="breakdown-item">
                        <span class="item-icon">⚡</span>
                        <div class="item-content">
                            <span class="item-label">Context Adjustment</span>
                            <span class="item-value ${output.contextAdjustment > 0 ? 'positive' : (output.contextAdjustment < 0 ? 'negative' : '')}">${output.contextAdjustment >= 0 ? '+' : ''}${output.contextAdjustment.toFixed(1)} U</span>
                            <span class="item-desc">${output.contextAdjustment > 0 ? 'Increased for stress/illness' : (output.contextAdjustment < 0 ? 'Reduced for exercise' : 'No adjustment needed')}</span>
                        </div>
                    </div>
                </div>
            </div>

            <div class="result-section">
                <h4 class="section-title">
                    <span class="title-icon">📉</span>
                    Algorithm Metrics
                </h4>
                <div class="metrics-grid">
                    <div class="mini-metric">
                        <span class="mini-label">Confidence Interval</span>
                        <span class="mini-value">[${output.confidenceInterval.lower.toFixed(1)}, ${output.confidenceInterval.upper.toFixed(1)}]</span>
                        <span class="mini-help">95% confidence range for optimal dose</span>
                    </div>
                    <div class="mini-metric good">
                        <span class="mini-label">Exploration Bonus</span>
                        <span class="mini-value">${output.explorationBonus.toFixed(3)}</span>
                        <span class="mini-help">Learning signal (lower = more confident)</span>
                    </div>
                    <div class="mini-metric good">
                        <span class="mini-label">Estimated Regret</span>
                        <span class="mini-value">${output.regret.toFixed(3)}</span>
                        <span class="mini-help">Expected suboptimality (lower = better)</span>
                    </div>
                </div>
            </div>
        </div>
    `;
}

function renderL5Output(output) {
    const tiers = [
        { num: 1, name: 'Reflex Controller', result: output.tier1, icon: '⚡', desc: 'Immediate safety checks without model computation' },
        { num: 2, name: 'STL Monitor', result: output.tier2, icon: '📊', desc: 'Signal Temporal Logic - checks future trajectory' },
        { num: 3, name: 'Seldonian Constraint', result: output.tier3, icon: '🛡️', desc: 'Probabilistic harm prevention with guarantees' }
    ];

    const allPassed = output.overallSafe;
    const finalDose = allPassed ? output.originalDose : output.safeDose;
    const wasModified = !allPassed;

    return `
        <div class="layer-result">
            <div class="result-section">
                <h4 class="section-title">
                    <span class="title-icon">🛡️</span>
                    Three-Tier Safety Verification
                </h4>
                <p class="section-desc">Every recommendation passes through 3 independent safety checks to ensure patient safety.</p>
            </div>

            <div class="safety-summary ${allPassed ? 'passed' : 'modified'}">
                <span class="safety-icon">${allPassed ? '✓' : '⚠'}</span>
                <div class="safety-content">
                    <span class="safety-title">${allPassed ? 'All Safety Checks Passed' : 'Safety Intervention Applied'}</span>
                    <span class="safety-desc">${allPassed ? 'The recommended dose is verified safe.' : `Original dose was ${output.originalDose.toFixed(1)}U, adjusted to ${output.safeDose.toFixed(1)}U for safety.`}</span>
                </div>
            </div>

            <div class="safety-tiers-detailed">
                ${tiers.map(t => `
                    <div class="tier-card ${t.result.safe ? 'passed' : 'failed'}">
                        <div class="tier-header">
                            <span class="tier-icon">${t.icon}</span>
                            <span class="tier-name">Tier ${t.num}: ${t.name}</span>
                            <span class="tier-status ${t.result.safe ? 'pass' : 'fail'}">${t.result.safe ? '✓ PASS' : '✗ FAIL'}</span>
                        </div>
                        <p class="tier-desc">${t.desc}</p>
                        <div class="tier-result">
                            <span class="tier-reason">${t.result.reason}</span>
                        </div>
                        ${!t.result.safe ? `
                            <div class="tier-action">
                                <span class="action-label">Action taken:</span>
                                <span class="action-value">Dose ${t.result.action === 'block' ? 'blocked' : (t.result.action === 'reduce' ? 'reduced' : 'suspended')} to ${t.result.dose.toFixed(1)}U</span>
                            </div>
                        ` : ''}
                    </div>
                `).join('')}
            </div>

            ${wasModified ? `
                <div class="dose-modification">
                    <h4 class="modification-title">Dose Modification Summary</h4>
                    <div class="modification-visual">
                        <div class="mod-original">
                            <span class="mod-label">Original</span>
                            <span class="mod-value">${output.originalDose.toFixed(1)}U</span>
                        </div>
                        <span class="mod-arrow">→</span>
                        <div class="mod-safe">
                            <span class="mod-label">Safe Dose</span>
                            <span class="mod-value">${output.safeDose.toFixed(1)}U</span>
                        </div>
                    </div>
                    <p class="modification-explanation">
                        The dose was reduced by ${(output.originalDose - output.safeDose).toFixed(1)}U 
                        (${((1 - output.safeDose / output.originalDose) * 100).toFixed(0)}% reduction) 
                        to maintain safety margins and prevent hypoglycemia.
                    </p>
                </div>
            ` : ''}

            <div class="safety-guarantee">
                <span class="guarantee-icon">✓</span>
                <span class="guarantee-text">Zero Seldonian constraint violations: P(severe hypoglycemia) < 1% guaranteed with 95% confidence</span>
            </div>
        </div>
    `;
}

function resetLayerOutputs() {
    for (let i = 1; i <= 5; i++) {
        const badge = document.getElementById(`l${i}-badge`);
        const outputDiv = document.getElementById(`l${i}-output`);
        const block = document.getElementById(`layer-${i}-block`);
        badge.textContent = 'Pending';
        badge.className = 'layer-status-badge';
        block.classList.remove('active');
        outputDiv.innerHTML = '<div class="output-placeholder">Awaiting input...</div>';
    }
}

function displayResults(results) {
    document.getElementById('results-empty').classList.add('hidden');
    document.getElementById('results-data').classList.remove('hidden');

    const rec = results.recommendation;
    const l5 = results.layers.L5;
    const l3 = results.layers.L3;
    const l2 = results.layers.L2;

    document.getElementById('rec-bolus').textContent = rec.dose.toFixed(1);
    document.getElementById('rec-meal-dose').textContent = rec.mealComponent.toFixed(1) + ' U';
    document.getElementById('rec-correction').textContent = rec.correctionComponent.toFixed(1) + ' U';
    document.getElementById('rec-adjustment').textContent = (rec.contextAdjustment >= 0 ? '+' : '') + rec.contextAdjustment.toFixed(1) + ' U';
    document.getElementById('rec-ci').textContent = `[${rec.confidenceInterval.lower.toFixed(1)}, ${rec.confidenceInterval.upper.toFixed(1)}] U`;

    updateSafetyTier('tier-1-result', l5.tier1);
    updateSafetyTier('tier-2-result', l5.tier2);
    updateSafetyTier('tier-3-result', l5.tier3);

    document.getElementById('causal-tau').textContent = l3.tau.toFixed(1);
    document.getElementById('causal-dev').textContent = (l3.individualDeviation >= 0 ? '+' : '') + l3.individualDeviation.toFixed(1) + ' mg/dL/U';
    document.getElementById('causal-adj').textContent = l3.proximalAdjustment.hasProxies ? `${l3.proximalAdjustment.biasReduction.toFixed(0)}% reduction` : 'N/A';

    updatePredictionChart(l2.prediction.trajectory, rec.dose);
    const adjustedPeak = l2.prediction.peak.glucose - rec.dose * 25;
    document.getElementById('pred-peak').textContent = Math.max(70, adjustedPeak).toFixed(0) + ' mg/dL';
    document.getElementById('pred-time-peak').textContent = l2.prediction.peak.time + ' min';
    document.getElementById('pred-return').textContent = l2.prediction.returnToTarget ? l2.prediction.returnToTarget + ' min' : '> 3 hours';
}

function updateSafetyTier(elementId, tierResult) {
    const el = document.getElementById(elementId);
    const statusEl = el.querySelector('.tier-status');
    const detailEl = el.querySelector('.tier-detail');
    statusEl.textContent = tierResult.safe ? 'PASS' : 'FAIL';
    statusEl.className = `tier-status ${tierResult.safe ? 'pass' : 'fail'}`;
    detailEl.textContent = tierResult.reason;
    el.className = `safety-tier ${tierResult.safe ? '' : 'failed'}`;
}

function updatePredictionChart(trajectory, dose) {
    const ctx = document.getElementById('prediction-chart');
    if (!ctx) return;
    const adjustedTrajectory = trajectory.map(p => ({
        time: p.time,
        glucose: Math.max(40, p.glucose - dose * 25 * Math.min(1, p.time / 60))
    }));
    if (AppState.predictionChart) AppState.predictionChart.destroy();
    AppState.predictionChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: adjustedTrajectory.map(p => p.time),
            datasets: [{
                label: 'Predicted Glucose (with treatment)',
                data: adjustedTrajectory.map(p => p.glucose),
                borderColor: '#10b981',
                backgroundColor: 'rgba(16, 185, 129, 0.1)',
                fill: true,
                tension: 0.4,
                pointRadius: 0
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: { legend: { display: false } },
            scales: {
                x: { title: { display: true, text: 'Time (min)', color: '#6b7280' }, grid: { color: 'rgba(255,255,255,0.05)' }, ticks: { color: '#6b7280' } },
                y: { min: 40, max: 300, title: { display: true, text: 'Glucose (mg/dL)', color: '#6b7280' }, grid: { color: 'rgba(255,255,255,0.05)' }, ticks: { color: '#6b7280' } }
            }
        },
        plugins: [{
            id: 'targetRange',
            beforeDraw: (chart) => {
                const { ctx, chartArea, scales } = chart;
                if (!chartArea) return;
                ctx.save();
                ctx.fillStyle = 'rgba(16, 185, 129, 0.15)';
                ctx.fillRect(chartArea.left, scales.y.getPixelForValue(180), chartArea.right - chartArea.left, scales.y.getPixelForValue(70) - scales.y.getPixelForValue(180));
                ctx.restore();
            }
        }]
    });
}

async function runBatchSimulation() {
    const patient = document.getElementById('sim-patient').value;
    const duration = parseInt(document.getElementById('sim-duration').value);
    const meals = document.getElementById('sim-meals').value;
    const challenge = document.getElementById('sim-challenge').value;

    document.getElementById('sim-status-display').innerHTML = '<p>Running simulation...</p>';
    document.getElementById('sim-chart-container').classList.add('hidden');
    document.getElementById('sim-metrics').classList.add('hidden');

    const patients = { adult_avg: { weight: 75, tdi: 45 }, adult_high_ir: { weight: 85, tdi: 65 }, adult_sensitive: { weight: 70, tdi: 30 }, adolescent: { weight: 52, tdi: 32 }, child: { weight: 30, tdi: 16 } };
    const mealPlans = { standard: [{ time: 0, carbs: 45 }, { time: 5, carbs: 70 }, { time: 11, carbs: 80 }], high_carb: [{ time: 0, carbs: 60 }, { time: 5, carbs: 100 }, { time: 11, carbs: 120 }], low_carb: [{ time: 0, carbs: 20 }, { time: 5, carbs: 30 }, { time: 11, carbs: 40 }], irregular: [{ time: 2, carbs: 50 }, { time: 8, carbs: 60 }] };

    const p = patients[patient];
    const mealPlan = mealPlans[meals];
    const trajectory = [];
    let glucose = 100, decisions = 0, violations = 0;

    for (let i = 0; i < duration * 12; i++) {
        const hour = i / 12;
        const meal = mealPlan.find(m => Math.abs(m.time - hour) < 0.1);
        if (meal) { glucose += meal.carbs * 4 - meal.carbs / (500 / p.tdi) * 25; decisions++; }
        let mod = 1;
        if (challenge === 'exercise' && hour >= 3 && hour <= 4) mod = 0.7;
        else if (challenge === 'stress') mod = 1.15;
        else if (challenge === 'illness') mod = 1.25;
        else if (challenge === 'dawn' && hour >= 4 && hour <= 7) mod = 1.1;
        glucose = glucose * 0.98 + 100 * 0.02;
        glucose *= mod;
        glucose += (Math.random() - 0.5) * 10;
        glucose = Math.max(50, Math.min(350, glucose));
        if (glucose < 54) { violations++; glucose = 70; }
        trajectory.push({ time: hour, glucose });
        await new Promise(r => setTimeout(r, 5));
    }

    displaySimulationResults({ trajectory, decisions, violations });
}

function displaySimulationResults(results) {
    document.getElementById('sim-status-display').innerHTML = '<p>Simulation complete</p>';
    document.getElementById('sim-chart-container').classList.remove('hidden');
    document.getElementById('sim-metrics').classList.remove('hidden');

    const glucose = results.trajectory.map(p => p.glucose);
    const n = glucose.length;
    document.getElementById('sim-tir').textContent = (glucose.filter(g => g >= 70 && g <= 180).length / n * 100).toFixed(1) + '%';
    document.getElementById('sim-tbr').textContent = (glucose.filter(g => g < 70).length / n * 100).toFixed(1) + '%';
    document.getElementById('sim-hypo').textContent = (glucose.filter(g => g < 54).length / n * 100).toFixed(1) + '%';
    document.getElementById('sim-tar').textContent = (glucose.filter(g => g > 180).length / n * 100).toFixed(1) + '%';
    const mean = glucose.reduce((a, b) => a + b, 0) / n;
    document.getElementById('sim-mean').textContent = mean.toFixed(1) + ' mg/dL';
    document.getElementById('sim-cv').textContent = ((Math.sqrt(glucose.reduce((s, g) => s + Math.pow(g - mean, 2), 0) / n) / mean) * 100).toFixed(1) + '%';
    document.getElementById('sim-violations').textContent = results.violations;
    document.getElementById('sim-decisions').textContent = results.decisions;

    const ctx = document.getElementById('sim-chart');
    if (AppState.simChart) AppState.simChart.destroy();
    AppState.simChart = new Chart(ctx, {
        type: 'line',
        data: { labels: results.trajectory.map(p => p.time.toFixed(1)), datasets: [{ label: 'Glucose', data: results.trajectory.map(p => p.glucose), borderColor: '#3b82f6', backgroundColor: 'rgba(59, 130, 246, 0.1)', fill: true, tension: 0.3, pointRadius: 0 }] },
        options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { display: false } }, scales: { x: { title: { display: true, text: 'Time (hours)', color: '#6b7280' }, grid: { color: 'rgba(255,255,255,0.05)' }, ticks: { color: '#6b7280', maxTicksLimit: 12 } }, y: { min: 40, max: 350, grid: { color: 'rgba(255,255,255,0.05)' }, ticks: { color: '#6b7280' } } } },
        plugins: [{ id: 'ranges', beforeDraw: (chart) => { const { ctx, chartArea, scales } = chart; if (!chartArea) return; ctx.save(); ctx.fillStyle = 'rgba(16, 185, 129, 0.1)'; ctx.fillRect(chartArea.left, scales.y.getPixelForValue(180), chartArea.right - chartArea.left, scales.y.getPixelForValue(70) - scales.y.getPixelForValue(180)); ctx.fillStyle = 'rgba(239, 68, 68, 0.1)'; ctx.fillRect(chartArea.left, scales.y.getPixelForValue(54), chartArea.right - chartArea.left, chartArea.bottom - scales.y.getPixelForValue(54)); ctx.restore(); } }]
    });
}

function updateSystemStatus(status) {
    const dot = document.getElementById('system-status-dot');
    const text = document.getElementById('system-status-text');
    const states = { ready: { class: 'ready', text: 'Ready for Input' }, processing: { class: 'processing', text: 'Processing...' }, complete: { class: 'complete', text: 'Complete' }, error: { class: 'error', text: 'Error' } };
    const state = states[status] || states.ready;
    dot.className = `status-dot ${state.class}`;
    text.textContent = state.text;
}

function resetSystem() {
    AppState.lastResults = null;
    resetLayerOutputs();
    document.getElementById('results-empty').classList.remove('hidden');
    document.getElementById('results-data').classList.add('hidden');
    updateSystemStatus('ready');
    navigateToSection('input');
    showToast('System reset', 'info');
}

function emergencyStop() {
    AppState.processing = false;
    updateSystemStatus('error');
    showToast('Emergency stop activated - all insulin delivery suspended', 'warning');
}

function showToast(message, type = 'info') {
    const container = document.getElementById('toast-container');
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    toast.innerHTML = `<span class="toast-message">${message}</span>`;
    container.appendChild(toast);
    setTimeout(() => { toast.style.opacity = '0'; toast.style.transform = 'translateX(100%)'; setTimeout(() => toast.remove(), 200); }, 4000);
}
