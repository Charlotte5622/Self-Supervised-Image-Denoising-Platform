/* ═══════════════════════════════════════════════════════════
   自监督图像去噪平台 —— 赛博朋克数据大屏 前端 JS
   ═══════════════════════════════════════════════════════════ */
'use strict';

const state = {
    mode: 'inference',
    uploadedFile: null,
    gtFile: null,
    presetDatasets: [],
    modelsData: [],
    lossChart: null,
    compareChart: null,
    distributionChart: null,
    gaugePsnr: null,
    gaugeSsim: null,
    trainingTaskId: null,
    pollingTimer: null,
    distributionRequestId: 0,
    distributionSignature: '',
    elapsedTimerFrame: null,
    elapsedStartTime: 0,
    compareData: null,
    compareViewMode: 'bars',
    compareDistributionMode: 'active',
    compareScopeMode: 'full',
    compareActiveModel: '',
    compareImageStore: null,
    compareAssetRequestId: 0,
    comparePreviewRequestId: 0,
    compareAnalysisTimer: null,
    compareDistributionRenderTimer: null,
    compareProgressFrame: null,
    compareProgressValue: 0,
    compareRoi: {
        x: 0.32,
        y: 0.32,
        size: 0.36,
    },
};

const CATEGORY_LABELS = {
    pretrained:       { text: '预训练',   cls: 'cat-badge--pretrained' },
    self_supervised:  { text: '自监督',   cls: 'cat-badge--self_supervised' },
    traditional:      { text: '传统算法', cls: 'cat-badge--traditional' },
};

const UI_TEXT_SECONDARY = '#9eb4d6';
const UI_TEXT_MUTED = '#6a81a5';
const UI_GRID_LINE = 'rgba(0,210,255,0.08)';

function floorTo(value, step) {
    return Math.floor(value / step) * step;
}

function ceilTo(value, step) {
    return Math.ceil(value / step) * step;
}

function clamp(value, min, max) {
    return Math.min(max, Math.max(min, value));
}

function computeHistogramFocusWindow(seriesList, options = {}) {
    const arrays = (seriesList || []).filter(arr => Array.isArray(arr) && arr.length > 0);
    if (arrays.length === 0) {
        return { start: 0, end: 255, yMax: 0.1 };
    }

    let globalPeak = 0;
    arrays.forEach(arr => {
        arr.forEach(value => {
            if (Number.isFinite(value)) globalPeak = Math.max(globalPeak, value);
        });
    });

    if (globalPeak <= 0) {
        return { start: 0, end: 255, yMax: 0.1 };
    }

    const threshold = Math.max(globalPeak * 0.015, 0.0002);
    let start = 255;
    let end = 0;

    for (let i = 0; i < 256; i += 1) {
        const active = arrays.some(arr => (arr[i] || 0) >= threshold);
        if (active) {
            start = Math.min(start, i);
            end = Math.max(end, i);
        }
    }

    if (start > end) {
        start = 0;
        end = 255;
    } else {
        start = Math.max(0, start - 3);
        end = Math.min(255, end + 3);
    }

    let yPeak = 0;
    const activeValues = [];
    arrays.forEach(arr => {
        for (let i = start; i <= end; i += 1) {
            const value = arr[i] || 0;
            yPeak = Math.max(yPeak, value);
            if (value > 0) activeValues.push(value);
        }
    });

    activeValues.sort((a, b) => a - b);
    const percentile = clamp(options.percentile ?? 0.98, 0.5, 1);
    const percentileIndex = activeValues.length > 0
        ? Math.min(activeValues.length - 1, Math.floor(activeValues.length * percentile))
        : 0;
    const percentilePeak = activeValues.length > 0 ? activeValues[percentileIndex] : yPeak;
    const usePercentileCap = Boolean(options.usePercentileCap);
    const referencePeak = usePercentileCap ? Math.max(percentilePeak, yPeak * 0.9) : yPeak;

    return {
        start,
        end,
        yMax: Math.max(0.02, Math.max(yPeak * 1.08, referencePeak * 1.16)),
    };
}

function smoothHistogramBins(bins, radius = 1) {
    if (!Array.isArray(bins) || bins.length === 0 || radius <= 0) return bins;
    const result = new Array(bins.length).fill(0);
    for (let i = 0; i < bins.length; i += 1) {
        let sum = 0;
        let weightSum = 0;
        for (let offset = -radius; offset <= radius; offset += 1) {
            const idx = i + offset;
            if (idx < 0 || idx >= bins.length) continue;
            const weight = radius + 1 - Math.abs(offset);
            sum += (bins[idx] || 0) * weight;
            weightSum += weight;
        }
        result[i] = weightSum > 0 ? sum / weightSum : 0;
    }
    return result;
}

function withAlpha(color, alpha) {
    if (typeof color !== 'string') return color;
    if (color.startsWith('#')) {
        if (color.length === 7) {
            const r = parseInt(color.slice(1, 3), 16);
            const g = parseInt(color.slice(3, 5), 16);
            const b = parseInt(color.slice(5, 7), 16);
            return `rgba(${r}, ${g}, ${b}, ${alpha})`;
        }
        if (color.length === 4) {
            const r = parseInt(color[1] + color[1], 16);
            const g = parseInt(color[2] + color[2], 16);
            const b = parseInt(color[3] + color[3], 16);
            return `rgba(${r}, ${g}, ${b}, ${alpha})`;
        }
    }
    if (color.startsWith('rgb(')) {
        return color.replace('rgb(', 'rgba(').replace(')', `, ${alpha})`);
    }
    if (color.startsWith('rgba(')) {
        return color.replace(/rgba\((.+),\s*[\d.]+\)/, `rgba($1, ${alpha})`);
    }
    if (color.startsWith('hsl(')) {
        return color.replace('hsl(', 'hsla(').replace(')', `, ${alpha})`);
    }
    if (color.startsWith('hsla(')) {
        return color.replace(/hsla\((.+),\s*[\d.]+\)/, `hsla($1, ${alpha})`);
    }
    return color;
}

function computeAxisBounds(values, { step, paddingRatio, minPadding, clampMin = -Infinity, clampMax = Infinity }) {
    const finiteValues = values.filter(v => Number.isFinite(v));
    if (finiteValues.length === 0) {
        return { min: clampMin, max: clampMax };
    }
    const vMin = Math.min(...finiteValues);
    const vMax = Math.max(...finiteValues);
    const span = Math.max(vMax - vMin, 0);
    const padding = Math.max(span * paddingRatio, minPadding);
    let min = floorTo(vMin - padding, step);
    let max = ceilTo(vMax + padding, step);
    if (min === max) {
        min -= step;
        max += step;
    }
    min = Math.max(clampMin, min);
    max = Math.min(clampMax, max);
    return { min, max };
}

function initAppModal() {
    const close = () => hideAppModal();
    document.getElementById('appModalClose')?.addEventListener('click', close);
    document.getElementById('appModalBackdrop')?.addEventListener('click', close);
    document.getElementById('appModalConfirm')?.addEventListener('click', close);
    document.addEventListener('keydown', event => {
        if (event.key === 'Escape') hideAppModal();
    });
}

function showAppModal(message, title = '提示') {
    const modal = document.getElementById('appModal');
    const titleEl = document.getElementById('appModalTitle');
    const bodyEl = document.getElementById('appModalBody');
    if (!modal || !titleEl || !bodyEl) return;
    titleEl.textContent = title;
    bodyEl.textContent = message;
    modal.style.display = 'block';
}

function hideAppModal() {
    const modal = document.getElementById('appModal');
    if (modal) modal.style.display = 'none';
}

/* ═══════════════════════════════════════════════════════════
   1. 粒子星空 Canvas 背景
   ═══════════════════════════════════════════════════════════ */
function initStarfield() {
    const canvas = document.getElementById('bgCanvas');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    let W, H;
    const particles = [];
    const STAR_COUNT = 200;
    const LINK_DIST = 150;

    function resize() {
        W = canvas.width = window.innerWidth;
        H = canvas.height = window.innerHeight;
    }
    resize();
    window.addEventListener('resize', resize);

    for (let i = 0; i < STAR_COUNT; i++) {
        const bright = Math.random() < 0.2;
        particles.push({
            x: Math.random() * W, y: Math.random() * H,
            vx: (Math.random() - 0.5) * 0.3,
            vy: (Math.random() - 0.5) * 0.2,
            r: bright ? 2 + Math.random() * 2 : 0.8 + Math.random() * 1.2,
            bright,
            phase: Math.random() * Math.PI * 2,
        });
    }

    function draw(time) {
        ctx.clearRect(0, 0, W, H);
        for (let i = 0; i < particles.length; i++) {
            const p = particles[i];
            p.x += p.vx; p.y += p.vy;
            if (p.x < 0) p.x = W; if (p.x > W) p.x = 0;
            if (p.y < 0) p.y = H; if (p.y > H) p.y = 0;

            let alpha = p.bright ? 0.6 + 0.4 * Math.sin(time * 0.002 + p.phase) : 0.55;
            ctx.beginPath();
            ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
            if (p.bright) {
                ctx.fillStyle = `rgba(0, 210, 255, ${alpha})`;
                ctx.shadowBlur = 18; ctx.shadowColor = 'rgba(0, 210, 255, 0.8)';
            } else {
                ctx.fillStyle = `rgba(160, 200, 240, ${alpha})`;
                ctx.shadowBlur = 4; ctx.shadowColor = 'rgba(100, 180, 240, 0.3)';
            }
            ctx.fill();
            ctx.shadowBlur = 0;

            for (let j = i + 1; j < particles.length; j++) {
                const q = particles[j];
                const dx = p.x - q.x, dy = p.y - q.y;
                const dist = Math.sqrt(dx * dx + dy * dy);
                if (dist < LINK_DIST) {
                    ctx.beginPath(); ctx.moveTo(p.x, p.y); ctx.lineTo(q.x, q.y);
                    ctx.strokeStyle = `rgba(0, 210, 255, ${0.12 * (1 - dist / LINK_DIST)})`;
                    ctx.lineWidth = 0.6; ctx.stroke();
                }
            }
        }
        requestAnimationFrame(draw);
    }
    requestAnimationFrame(draw);
}

/* ═══════════════════════════════════════════════════════════
   2. 时钟
   ═══════════════════════════════════════════════════════════ */
function initClock() {
    const el = document.getElementById('clockDisplay');
    if (!el) return;
    function tick() {
        const now = new Date();
        el.textContent = [now.getHours(), now.getMinutes(), now.getSeconds()]
            .map(n => String(n).padStart(2, '0')).join(':');
    }
    tick(); setInterval(tick, 1000);
}

/* ═══════════════════════════════════════════════════════════
   3. 模式切换
   ═══════════════════════════════════════════════════════════ */
function initModeSwitch() {
    document.querySelectorAll('.mode-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            document.querySelectorAll('.mode-btn').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            state.mode = btn.dataset.mode;
            applyModeUI();
        });
    });
    applyModeUI();
}

function applyModeUI() {
    const m = state.mode;
    const singleGrp = document.getElementById('singleModelGroup');
    const multiGrp  = document.getElementById('multiModelGroup');
    const trainP    = document.getElementById('trainParams');
    const gtHint    = document.getElementById('gtHint');
    const btnText   = document.querySelector('#btnExecute .btn-text');
    const centerT   = document.getElementById('centerTitle');

    singleGrp.style.display = m === 'compare' ? 'none' : '';
    multiGrp.style.display  = m === 'compare' ? '' : 'none';
    trainP.style.display    = m === 'training' ? '' : 'none';

    if (m === 'compare') {
        gtHint.textContent = '（必选）'; gtHint.className = 'gt-hint required';
    } else {
        gtHint.textContent = '（可选）'; gtHint.className = 'gt-hint';
    }

    const labels = { inference: '开始去噪', training: '开始训练', compare: '开始对比' };
    const titles = { inference: '图像对比', training: '自监督训练', compare: '多算法对比' };
    if (btnText) btnText.textContent = labels[m];
    if (centerT) centerT.textContent = titles[m];

    updateModelSelect();
    if (m === 'compare' && state.compareData) {
        renderCompareTable();
        renderCompareChart();
    } else {
        hideCompareUI();
    }
    updateAnalysisSectionsByMode();
    syncCompareToolbarButtons();
    syncCompareScopeButtons();
}

/* ═══════════════════════════════════════════════════════════
   4. 模型加载 & 选择
   ═══════════════════════════════════════════════════════════ */
async function loadModels() {
    try {
        const res = await fetch('/api/models/');
        const data = await res.json();
        state.modelsData = data.models || [];
        const devEl = document.getElementById('deviceInfo');
        if (devEl) devEl.textContent = (data.device || 'cpu').toUpperCase();
        const cntEl = document.getElementById('modelCount');
        if (cntEl) cntEl.textContent = state.modelsData.length;
        const gpuStatus = document.getElementById('gpuStatus');
        if (gpuStatus) {
            const isGpu = (data.device || '').includes('cuda');
            gpuStatus.querySelector('.label').textContent = isGpu ? 'GPU ONLINE' : 'CPU MODE';
            if (isGpu) gpuStatus.classList.add('online');
        }
        updateModelSelect();
        buildCheckboxList();
    } catch (e) {
        console.error('模型加载失败', e);
    }
}

function updateModelSelect() {
    const sel = document.getElementById('modelSelect');
    if (!sel) return;
    const allowedCats = state.mode === 'training'
        ? ['self_supervised']
        : ['pretrained', 'traditional'];

    const filtered = state.modelsData.filter(m => allowedCats.includes(m.category));
    sel.innerHTML = filtered.length === 0
        ? '<option value="">无可用模型</option>'
        : filtered.map(m => `<option value="${m.name}">${m.display_name}</option>`).join('');
    showCategoryTag(sel.value);
    sel.onchange = () => showCategoryTag(sel.value);
}

function showCategoryTag(modelName) {
    const el = document.getElementById('modelCategoryTag');
    if (!el) return;
    const m = state.modelsData.find(x => x.name === modelName);
    if (!m) { el.innerHTML = ''; return; }
    const info = CATEGORY_LABELS[m.category] || { text: m.category, cls: '' };
    el.innerHTML = `<span class="cat-badge ${info.cls}">${info.text}</span>`;
}

function buildCheckboxList() {
    const wrap = document.getElementById('modelCheckboxList');
    if (!wrap) return;
    const list = state.modelsData.filter(m => m.category !== 'self_supervised');
    wrap.innerHTML = list.map(m => {
        const catInfo = CATEGORY_LABELS[m.category] || { text: m.category };
        return `<label class="checkbox-item">
            <input type="checkbox" value="${m.name}" checked>
            <span>${m.display_name}</span>
            <span class="cb-category">${catInfo.text}</span>
        </label>`;
    }).join('');

    document.getElementById('btnSelectAll')?.addEventListener('click', () =>
        wrap.querySelectorAll('input').forEach(i => i.checked = true));
    document.getElementById('btnSelectNone')?.addEventListener('click', () =>
        wrap.querySelectorAll('input').forEach(i => i.checked = false));
}

function getSelectedCompareModels() {
    const wrap = document.getElementById('modelCheckboxList');
    return [...wrap.querySelectorAll('input:checked')].map(i => i.value);
}

function isTemporaryUpload(path) {
    return typeof path === 'string' && path.startsWith('uploads/');
}

function getActiveTemporaryUploads() {
    return [...new Set([state.uploadedFile, state.gtFile].filter(isTemporaryUpload))];
}

function cleanupTemporaryUploads(files, useBeacon = false) {
    const uniqueFiles = [...new Set((files || []).filter(isTemporaryUpload))];
    if (uniqueFiles.length === 0) return Promise.resolve();

    const payload = JSON.stringify({ filenames: uniqueFiles });
    if (useBeacon && navigator.sendBeacon) {
        const blob = new Blob([payload], { type: 'application/json' });
        navigator.sendBeacon('/api/upload/cleanup/', blob);
        return Promise.resolve();
    }

    return fetch('/api/upload/cleanup/', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: payload,
        keepalive: true,
    }).catch(() => {});
}

function setNoisySelection(file, url, name) {
    state.uploadedFile = file;
    showPreview('uploadPreview', 'previewImg', 'previewName', url, name);
}

function clearGroundTruthSelection() {
    state.gtFile = null;
    const preview = document.getElementById('gtPreview');
    if (preview) preview.style.display = 'none';
}

function setGroundTruthSelection(file, url, name) {
    if (!file || !url) {
        clearGroundTruthSelection();
        return;
    }
    state.gtFile = file;
    showPreview('gtPreview', 'gtPreviewImg', 'gtPreviewName', url, name);
}

/* ═══════════════════════════════════════════════════════════
   5. 文件上传
   ═══════════════════════════════════════════════════════════ */
function initUpload() {
    setupUploadZone('uploadZone', 'fileInput', async file => {
        const fd = new FormData();
        const previous = isTemporaryUpload(state.uploadedFile) ? state.uploadedFile : '';
        fd.append('image', file);
        if (previous) fd.append('replace', previous);
        try {
            const res = await fetch('/api/upload/', { method: 'POST', body: fd });
            const data = await res.json();
            if (data.success) {
                setNoisySelection(data.filename, data.url, file.name);
            } else { showAppModal(data.error || '上传失败'); }
        } catch { showAppModal('上传请求异常'); }
    });
    setupUploadZone('gtUploadZone', 'gtFileInput', async file => {
        const fd = new FormData();
        const previous = isTemporaryUpload(state.gtFile) ? state.gtFile : '';
        fd.append('image', file);
        if (previous) fd.append('replace', previous);
        try {
            const res = await fetch('/api/upload/', { method: 'POST', body: fd });
            const data = await res.json();
            if (data.success) {
                setGroundTruthSelection(data.filename, data.url, file.name);
            }
        } catch { showAppModal('上传请求异常'); }
    });

    const releaseActiveUploads = () => cleanupTemporaryUploads(getActiveTemporaryUploads(), true);
    window.addEventListener('pagehide', releaseActiveUploads);
    window.addEventListener('beforeunload', releaseActiveUploads);
}

function setupUploadZone(zoneId, inputId, handler) {
    const zone = document.getElementById(zoneId);
    const input = document.getElementById(inputId);
    if (!zone || !input) return;
    zone.addEventListener('click', () => input.click());
    zone.addEventListener('dragover', e => { e.preventDefault(); zone.classList.add('dragover'); });
    zone.addEventListener('dragleave', () => zone.classList.remove('dragover'));
    zone.addEventListener('drop', e => {
        e.preventDefault(); zone.classList.remove('dragover');
        if (e.dataTransfer.files[0]) handler(e.dataTransfer.files[0]);
    });
    input.addEventListener('change', () => { if (input.files[0]) handler(input.files[0]); });
}

function showPreview(wrapId, imgId, nameId, url, name) {
    const w = document.getElementById(wrapId);
    const img = document.getElementById(imgId);
    const n = document.getElementById(nameId);
    if (w) w.style.display = 'flex';
    if (img) img.src = url;
    if (n) n.textContent = name;
}

/* ═══════════════════════════════════════════════════════════
   6. 预置图像
   ═══════════════════════════════════════════════════════════ */
async function loadPresetImages() {
    try {
        const res = await fetch('/api/preset-images/');
        const data = await res.json();
        state.presetDatasets = Array.isArray(data.datasets) ? data.datasets.slice(0, 5) : [];
        renderPresetDatasetButtons();
    } catch (e) { console.error('加载预置图失败', e); }
}

function renderPresetDatasetButtons() {
    const grid = document.getElementById('presetDatasetGrid');
    if (!grid) return;

    if (state.presetDatasets.length === 0) {
        grid.innerHTML = '<p class="placeholder-text">暂无预置数据集</p>';
        return;
    }

    const slots = [];
    for (let i = 0; i < 5; i++) {
        const dataset = state.presetDatasets[i];
        if (!dataset) {
            slots.push(`
                <button class="preset-dataset-btn" type="button" disabled>
                    <span class="preset-dataset-btn__name">数据集 ${i + 1}</span>
                    <span class="preset-dataset-btn__meta">EMPTY</span>
                </button>
            `);
            continue;
        }
        slots.push(`
            <button class="preset-dataset-btn" type="button" data-dataset-idx="${i}">
                <span class="preset-dataset-btn__name">${dataset.name}</span>
                <span class="preset-dataset-btn__meta">${dataset.count} IMG</span>
            </button>
        `);
    }
    grid.innerHTML = slots.join('');

    grid.querySelectorAll('[data-dataset-idx]').forEach(btn => {
        btn.addEventListener('click', () => openPresetModal(parseInt(btn.dataset.datasetIdx, 10)));
    });
}

function initPresetModal() {
    document.getElementById('presetModalClose')?.addEventListener('click', closePresetModal);
    document.getElementById('presetModalBackdrop')?.addEventListener('click', closePresetModal);
    document.addEventListener('keydown', e => {
        if (e.key === 'Escape') closePresetModal();
    });
}

function openPresetModal(datasetIdx) {
    const dataset = state.presetDatasets[datasetIdx];
    const modal = document.getElementById('presetModal');
    const title = document.getElementById('presetModalTitle');
    const body = document.getElementById('presetModalBody');
    if (!dataset || !modal || !title || !body) return;

    title.textContent = `选择预置图像 · ${dataset.name}`;
    if (!dataset.presets || dataset.presets.length === 0) {
        body.innerHTML = '<p class="placeholder-text">当前数据集暂无图片</p>';
    } else {
        body.innerHTML = `<div class="preset-thumb-grid">${dataset.presets.map((preset, idx) => `
            <button class="preset-thumb ${state.uploadedFile === preset.noisy_file ? 'selected' : ''}" type="button" data-preset-idx="${idx}">
                <img src="${preset.noisy_url}" alt="${preset.name}" loading="lazy">
                ${preset.has_gt ? '<span class="preset-gt-dot"></span>' : ''}
                <span class="preset-thumb__caption">${preset.name}</span>
            </button>
        `).join('')}</div>`;

        body.querySelectorAll('[data-preset-idx]').forEach(node => {
            node.addEventListener('click', async () => {
                const preset = dataset.presets[parseInt(node.dataset.presetIdx, 10)];
                if (!preset) return;
                const filesToCleanup = getActiveTemporaryUploads();
                setNoisySelection(preset.noisy_file, preset.noisy_url, preset.name);
                if (preset.has_gt) {
                    setGroundTruthSelection(preset.clean_file, preset.clean_url, `${preset.name} GT`);
                } else {
                    clearGroundTruthSelection();
                }
                closePresetModal();
                await cleanupTemporaryUploads(filesToCleanup);
            });
        });
    }

    modal.style.display = 'block';
    document.body.style.overflow = 'hidden';
}

function closePresetModal() {
    const modal = document.getElementById('presetModal');
    if (!modal || modal.style.display === 'none') return;
    modal.style.display = 'none';
    document.body.style.overflow = '';
}

/* ═══════════════════════════════════════════════════════════
   7. 执行按钮分发
   ═══════════════════════════════════════════════════════════ */
function initExecuteButton() {
    document.getElementById('btnExecute')?.addEventListener('click', () => {
        if (!state.uploadedFile) { showAppModal('请先上传或选择图像'); return; }
        if (state.mode === 'compare' && !state.gtFile) { showAppModal('多算法对比需要提供 Ground Truth'); return; }
        const dispatch = { inference: runInference, training: runTraining, compare: runCompare };
        dispatch[state.mode]?.();
    });
}

/* ═══════════════════════════════════════════════════════════
   8. 推理去噪
   ═══════════════════════════════════════════════════════════ */
async function runInference() {
    const model = document.getElementById('modelSelect')?.value;
    if (!model) { showAppModal('请选择模型'); return; }

    let finalElapsedMs = null;
    resetMetrics(); setLoading(true); showScanOverlay(true); hideCompareUI();

    try {
        const res = await fetch('/api/denoise/', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ filename: state.uploadedFile, model, ground_truth: state.gtFile || '' }),
        });
        const data = await res.json();
        if (data.success || data.result_url) {
            finalElapsedMs = data.elapsed_ms ?? null;
            const noisyUrl = getNoisyImageUrl(data.input_url);
            showSlider(noisyUrl, data.result_url);
            updateDistributionChart(noisyUrl, data.result_url, data.gt_url || getGroundTruthImageUrl());
            animateMetric('metricTime', data.elapsed_ms);
            if (data.psnr_before !== undefined) {
                animateMetric('metricPsnrBefore', data.psnr_before);
                animateMetric('metricSsimBefore', data.ssim_before);
                animateMetric('metricPsnrAfter', data.psnr_after);
                animateMetric('metricSsimAfter', data.ssim_after);
                updateGauges(data.psnr_after, data.ssim_after);
            }
        } else {
            showAppModal(data.error || '去噪失败');
        }
    } catch (e) { showAppModal('请求失败: ' + e.message); }
    finally { setLoading(false, finalElapsedMs); showScanOverlay(false); }
}

/* ═══════════════════════════════════════════════════════════
   9. 自监督训练 + 轮询
   ═══════════════════════════════════════════════════════════ */
async function runTraining() {
    const arch = document.getElementById('modelSelect')?.value;
    if (!arch) { showAppModal('请选择训练架构'); return; }

    resetMetrics(); setLoading(true); showScanOverlay(true); hideCompareUI();
    showTrainingUI(true);

    const epochs = parseInt(document.getElementById('inputEpochs')?.value) || 200;
    const lr = parseFloat(document.getElementById('inputLR')?.value) || 0.001;

    try {
        const res = await fetch('/api/train/', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ filename: state.uploadedFile, arch, epochs, lr, ground_truth: state.gtFile || '' }),
        });
        const data = await res.json();
        if (data.success) {
            state.trainingTaskId = data.task_id;
            pollTrainingStatus(data.task_id);
        } else {
            showAppModal(data.error || '启动训练失败');
            setLoading(false); showScanOverlay(false); showTrainingUI(false);
        }
    } catch (e) {
        showAppModal('请求失败: ' + e.message);
        setLoading(false); showScanOverlay(false); showTrainingUI(false);
    }
}

function pollTrainingStatus(taskId) {
    if (state.pollingTimer) clearInterval(state.pollingTimer);
    state.pollingTimer = setInterval(async () => {
        try {
            const res = await fetch(`/api/train/status/${taskId}/`);
            const d = await res.json();
            const pct = d.total_epochs > 0 ? Math.round(d.epoch / d.total_epochs * 100) : 0;

            document.getElementById('progressFill').style.width = pct + '%';
            document.getElementById('progressText').textContent = pct + '%';
            document.getElementById('epochDisplay').textContent = `${d.epoch}/${d.total_epochs}`;
            document.getElementById('lrDisplay').textContent = d.current_lr ? d.current_lr.toExponential(2) : '--';

            if (d.all_losses && d.all_losses.length > 0) updateLossChart(d.all_losses);

            if (d.intermediate_url) {
                const noisyUrl = getNoisyImageUrl(`/media/${state.uploadedFile}`);
                showSlider(noisyUrl, d.intermediate_url);
                updateDistributionChart(noisyUrl, d.intermediate_url, getGroundTruthImageUrl());
            }

            if (d.psnr_before != null) animateMetric('metricPsnrBefore', d.psnr_before);
            if (d.ssim_before != null) animateMetric('metricSsimBefore', d.ssim_before);

            if (d.status === 'completed') {
                clearInterval(state.pollingTimer); state.pollingTimer = null;
                setLoading(false); showScanOverlay(false);
                const noisyUrlFinal = getNoisyImageUrl(`/media/${state.uploadedFile}`);
                if (d.result_url) {
                    showSlider(noisyUrlFinal, d.result_url);
                    updateDistributionChart(noisyUrlFinal, d.result_url, getGroundTruthImageUrl());
                }
                if (d.psnr != null) { animateMetric('metricPsnrAfter', d.psnr); updateGauges(d.psnr, d.ssim); }
                if (d.ssim != null) animateMetric('metricSsimAfter', d.ssim);
            } else if (d.status === 'error') {
                clearInterval(state.pollingTimer); state.pollingTimer = null;
                setLoading(false); showScanOverlay(false);
                showAppModal('训练出错: ' + (d.error || '未知错误'));
            }
        } catch (e) { console.error('轮询失败', e); }
    }, 1500);
}

/* ═══════════════════════════════════════════════════════════
   10. 多算法对比
   ═══════════════════════════════════════════════════════════ */
async function runCompare() {
    const models = getSelectedCompareModels();
    if (models.length === 0) { showAppModal('请至少选择一个算法'); return; }

    resetMetrics(); setLoading(true); showScanOverlay(true);
    hideSlider();
    state.compareData = null;
    state.compareImageStore = null;
    state.compareActiveModel = '';
    updateCompareScopeNote('正在准备多算法对比...');
    updateCompareRoiPreview('');
    startCompareProgress(models.length);

    try {
        const res = await fetch('/api/compare/', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ filename: state.uploadedFile, ground_truth: state.gtFile, models }),
        });
        const data = await res.json();
        if (data.success) {
            const firstResult = data.results.find(r => r.success);
            state.compareData = data;
            state.compareActiveModel = firstResult?.model || '';
            updateCompareRoiPreview(getNoisyImageUrl(data.input_url));
            renderCompareTable();
            renderCompareChart();
            stopCompareProgress(true, `完成 ${data.results.filter(r => r.success).length}/${models.length} 个算法`);
            prepareCompareAnalysisAssets(data);
        } else { showAppModal(data.error || '对比失败'); }
    } catch (e) { showAppModal('请求失败: ' + e.message); }
    finally {
        if (!state.compareData) {
            stopCompareProgress(false, '对比失败');
        }
        setLoading(false); showScanOverlay(false);
    }
}

function renderCompareChart() {
    if (!state.compareData) return;
    if (state.compareViewMode === 'distribution') {
        renderCompareDistributionView();
    } else {
        renderCompareBarChart();
    }
}

function ensureCompareChartReady() {
    const dom = document.getElementById('compareChart');
    const wrapper = document.getElementById('compareChartWrapper');
    if (!dom || !wrapper) return null;

    document.getElementById('emptyState').style.display = 'none';
    document.getElementById('imageSlider').style.display = 'none';
    wrapper.style.display = 'flex';

    if (!state.compareChart) {
        state.compareChart = echarts.init(dom, null, { renderer: 'canvas' });
        window.addEventListener('resize', () => state.compareChart?.resize());
    }
    return state.compareChart;
}

function renderComparePlaceholder(message) {
    const chart = ensureCompareChartReady();
    if (!chart) return;
    chart.clear();
    chart.setOption({
        backgroundColor: 'transparent',
        title: {
            text: message,
            left: 'center',
            top: 'middle',
            textStyle: {
                color: UI_TEXT_MUTED,
                fontFamily: 'Rajdhani',
                fontSize: 14,
                fontWeight: 500,
            },
        },
        xAxis: { show: false },
        yAxis: { show: false },
        series: [],
    }, true);
}

function renderCompareBarChart() {
    const chart = ensureCompareChartReady();
    const metrics = getCompareMetricsData();
    if (!chart || !metrics || metrics.results.length === 0) {
        renderComparePlaceholder('当前没有可展示的对比结果');
        return;
    }

    const names = metrics.results.map(r => r.display_name);
    const psnrVals = metrics.results.map(r => r.psnr);
    const ssimVals = metrics.results.map(r => r.ssim);
    const psnrSpan = Math.max(...psnrVals) - Math.min(...psnrVals);
    const ssimSpan = Math.max(...ssimVals) - Math.min(...ssimVals);
    const psnrBounds = computeAxisBounds(psnrVals, {
        step: psnrSpan <= 0.8 ? 0.1 : 0.5,
        paddingRatio: 0.16,
        minPadding: psnrSpan <= 0.8 ? 0.12 : 0.4,
        clampMin: 0,
        clampMax: 60,
    });
    const ssimBounds = computeAxisBounds(ssimVals, {
        step: ssimSpan <= 0.02 ? 0.001 : (ssimSpan <= 0.08 ? 0.002 : 0.005),
        paddingRatio: 0.16,
        minPadding: ssimSpan <= 0.02 ? 0.003 : 0.01,
        clampMin: 0,
        clampMax: 1,
    });
    const showBaselineLine = Number.isFinite(metrics?.baseline?.psnr)
        && metrics.baseline.psnr >= psnrBounds.min
        && metrics.baseline.psnr <= psnrBounds.max;
    const activeModel = state.compareActiveModel;

    chart.off('click');
    chart.on('click', params => {
        const hit = metrics.results.find(item => item.display_name === params?.name);
        if (hit) setCompareActiveModel(hit.model);
    });

    chart.setOption({
        backgroundColor: 'transparent',
        tooltip: {
            trigger: 'axis',
            backgroundColor: 'rgba(8, 16, 32, 0.9)',
            borderColor: 'rgba(0, 210, 255, 0.2)',
            textStyle: { color: '#dce6f5', fontFamily: 'Rajdhani' },
        },
        title: {
            text: metrics.fromRegion ? '局部区域指标对比' : '整图指标对比',
            left: 14,
            top: 10,
            textStyle: { color: UI_TEXT_SECONDARY, fontFamily: 'Rajdhani', fontSize: 12, fontWeight: 600 },
            subtext: metrics.fromRegion && metrics.bounds
                ? `ROI ${metrics.bounds.width}x${metrics.bounds.height} · 相对坐标 (${Math.round(metrics.bounds.leftRatio * 100)}%, ${Math.round(metrics.bounds.topRatio * 100)}%)`
                : '点击柱体可切换下方分布图的结果图像',
            subtextStyle: { color: UI_TEXT_MUTED, fontFamily: 'Rajdhani', fontSize: 10 },
        },
        legend: {
            top: 52,
            data: ['PSNR', 'SSIM'],
            textStyle: { color: UI_TEXT_SECONDARY, fontFamily: 'Rajdhani' },
        },
        grid: { top: 82, bottom: 76, left: 68, right: 64 },
        xAxis: {
            type: 'category', data: names,
            axisLabel: {
                color: UI_TEXT_SECONDARY,
                fontFamily: 'Rajdhani',
                fontSize: 11,
                interval: 0,
                hideOverlap: false,
                rotate: names.length > 5 ? 18 : 0,
                margin: 14,
                width: names.length > 5 ? 82 : null,
                overflow: names.length > 5 ? 'break' : null,
            },
            axisLine: { lineStyle: { color: 'rgba(0,210,255,0.1)' } },
            splitLine: { show: false },
        },
        yAxis: [
            {
                type: 'value', name: 'PSNR (dB)',
                nameTextStyle: { color: '#00d2ff', fontFamily: 'Orbitron', fontSize: 10 },
                axisLabel: {
                    color: UI_TEXT_SECONDARY,
                    fontFamily: 'Rajdhani',
                    formatter: value => Number(value).toFixed(psnrSpan <= 0.8 ? 1 : 0),
                },
                axisLine: { lineStyle: { color: 'rgba(0,210,255,0.1)' } },
                splitLine: { lineStyle: { color: UI_GRID_LINE } },
                min: psnrBounds.min,
                max: psnrBounds.max,
            },
            {
                type: 'value', name: 'SSIM',
                nameTextStyle: { color: '#ff007f', fontFamily: 'Orbitron', fontSize: 10 },
                axisLabel: {
                    color: UI_TEXT_SECONDARY,
                    fontFamily: 'Rajdhani',
                    formatter: value => Number(value).toFixed(ssimSpan <= 0.02 ? 3 : 2),
                },
                axisLine: { lineStyle: { color: 'rgba(255,0,127,0.1)' } },
                splitLine: { show: false },
                min: ssimBounds.min,
                max: ssimBounds.max,
            },
        ],
        series: [
            {
                name: 'PSNR', type: 'bar', yAxisIndex: 0, data: psnrVals, barMaxWidth: 32, color: '#00d2ff',
                itemStyle: {
                    color: params => metrics.results[params.dataIndex]?.model === activeModel
                        ? new echarts.graphic.LinearGradient(0, 0, 0, 1, [
                            { offset: 0, color: 'rgba(0, 255, 220, 0.98)' },
                            { offset: 1, color: 'rgba(0, 85, 255, 0.45)' },
                        ])
                        : new echarts.graphic.LinearGradient(0, 0, 0, 1, [
                            { offset: 0, color: 'rgba(0, 210, 255, 0.9)' },
                            { offset: 1, color: 'rgba(0, 85, 255, 0.3)' },
                        ]),
                    borderRadius: [4, 4, 0, 0],
                    borderWidth: 1,
                    borderColor: params => metrics.results[params.dataIndex]?.model === activeModel
                        ? 'rgba(255,255,255,0.72)'
                        : 'rgba(0,210,255,0.08)',
                },
                emphasis: { itemStyle: { shadowBlur: 20, shadowColor: 'rgba(0,210,255,0.5)' } },
                markLine: {
                    silent: true, symbol: 'none',
                    lineStyle: { color: 'rgba(0,210,255,0.3)', type: 'dashed' },
                    data: showBaselineLine ? [{
                        yAxis: metrics.baseline.psnr,
                        label: {
                            formatter: 'Baseline',
                            color: UI_TEXT_SECONDARY,
                            fontFamily: 'Rajdhani',
                            fontSize: 10,
                        },
                    }] : [],
                },
            },
            {
                name: 'SSIM', type: 'bar', yAxisIndex: 1, data: ssimVals, barMaxWidth: 32, color: '#ff007f',
                itemStyle: {
                    color: params => metrics.results[params.dataIndex]?.model === activeModel
                        ? new echarts.graphic.LinearGradient(0, 0, 0, 1, [
                            { offset: 0, color: 'rgba(255, 108, 196, 0.98)' },
                            { offset: 1, color: 'rgba(139, 92, 246, 0.45)' },
                        ])
                        : new echarts.graphic.LinearGradient(0, 0, 0, 1, [
                            { offset: 0, color: 'rgba(255, 0, 127, 0.9)' },
                            { offset: 1, color: 'rgba(139, 92, 246, 0.3)' },
                        ]),
                    borderRadius: [4, 4, 0, 0],
                    borderWidth: 1,
                    borderColor: params => metrics.results[params.dataIndex]?.model === activeModel
                        ? 'rgba(255,255,255,0.72)'
                        : 'rgba(255,0,127,0.08)',
                },
                emphasis: { itemStyle: { shadowBlur: 20, shadowColor: 'rgba(255,0,127,0.5)' } },
            },
        ],
        animationDuration: 1200, animationEasing: 'cubicOut',
    }, true);
}

function renderCompareDistributionView() {
    const chart = ensureCompareChartReady();
    const metrics = getCompareMetricsData();
    const activeResult = getActiveCompareResult();
    if (!chart || !metrics || !activeResult) {
        renderComparePlaceholder('请先运行多算法对比');
        return;
    }
    if (!state.compareImageStore) {
        renderComparePlaceholder('正在加载图像分布分析资源...');
        return;
    }

    const bounds = metrics.bounds || getCompareBounds(state.compareImageStore);
    const showAll = state.compareDistributionMode === 'all';
    const histogramOptions = showAll ? { maxSamples: 45000 } : { maxSamples: 120000 };
    const smoothRadius = showAll ? 2 : 1;
    const noisyBins = smoothHistogramBins(computeHistogramBins(state.compareImageStore.noisy, bounds, histogramOptions), smoothRadius);
    const gtBins = smoothHistogramBins(computeHistogramBins(state.compareImageStore.gt, bounds, histogramOptions), smoothRadius);
    const activeBins = smoothHistogramBins(computeHistogramBins(activeResult, bounds, histogramOptions), smoothRadius);
    const palette = ['#ff007f', '#8b5cf6', '#ffaa33', '#22d3ee', '#f97316', '#e879f9', '#a3e635', '#38bdf8'];
    const compareSeries = [
        { name: '含噪原图', bins: noisyBins, color: '#00d2ff' },
        ...(showAll
            ? state.compareImageStore.results.map((item, index) => ({
                name: item.display_name || item.model || `结果 ${index + 1}`,
                bins: smoothHistogramBins(computeHistogramBins(item, bounds, histogramOptions), smoothRadius),
                color: palette[index % palette.length],
                fill: index === 0,
            }))
            : [{ name: activeResult.display_name, bins: activeBins, color: '#ff007f', fill: true }]),
        { name: '真值图像', bins: gtBins, color: '#00ff88' },
    ];
    const labels = Array.from({ length: 256 }, (_, idx) => idx);
    const focus = computeHistogramFocusWindow(compareSeries.map(item => item.bins), {
        usePercentileCap: showAll,
        percentile: showAll ? 0.97 : 1,
    });
    const focusedLabels = labels.slice(focus.start, focus.end + 1);

    chart.off('click');
    chart.setOption({
        backgroundColor: 'transparent',
        title: {
            text: showAll
                ? (metrics.fromRegion ? '局部区域分布 · 全部结果' : '整图分布 · 全部结果')
                : (metrics.fromRegion ? `局部区域分布 · ${activeResult.display_name}` : `整图分布 · ${activeResult.display_name}`),
            left: 14,
            top: 10,
            textStyle: { color: UI_TEXT_SECONDARY, fontFamily: 'Rajdhani', fontSize: 12, fontWeight: 600 },
            subtext: metrics.fromRegion && bounds
                ? `ROI ${bounds.width}x${bounds.height} · 对比含噪 / 去噪 / 真值亮度分布`
                : (showAll ? '当前展示含噪、全部去噪结果与真值图像的亮度分布' : '点击右侧表格算法可切换当前分布图'),
            subtextStyle: { color: UI_TEXT_MUTED, fontFamily: 'Rajdhani', fontSize: 10 },
        },
        tooltip: {
            trigger: 'axis',
            backgroundColor: 'rgba(8,16,32,0.92)',
            borderColor: 'rgba(0,210,255,0.18)',
            textStyle: { color: '#dce6f5', fontFamily: 'Rajdhani' },
            formatter: params => {
                const lines = [`亮度值 ${params[0]?.axisValue ?? '--'}`];
                params.forEach(item => lines.push(`${item.seriesName}: ${(item.value * 100).toFixed(2)}%`));
                return lines.join('<br>');
            },
        },
        legend: {
            type: 'scroll',
            top: 52,
            itemWidth: 12,
            itemHeight: 8,
            pageTextStyle: { color: UI_TEXT_SECONDARY },
            textStyle: { color: UI_TEXT_SECONDARY, fontFamily: 'Rajdhani', fontSize: 11 },
        },
        grid: { top: 82, bottom: 28, left: 44, right: 14 },
        xAxis: {
            type: 'category',
            data: focusedLabels,
            boundaryGap: false,
            axisLabel: {
                color: UI_TEXT_SECONDARY,
                fontFamily: 'Rajdhani',
                fontSize: 10,
                interval: Math.max(0, Math.floor((focusedLabels.length - 1) / 6)),
            },
            axisLine: { lineStyle: { color: 'rgba(0,210,255,0.1)' } },
            splitLine: { show: false },
        },
        yAxis: {
            type: 'value',
            name: '占比',
            max: focus.yMax,
            nameTextStyle: { color: UI_TEXT_SECONDARY, fontFamily: 'Rajdhani', fontSize: 10 },
            axisLabel: {
                color: UI_TEXT_SECONDARY,
                fontFamily: 'Rajdhani',
                fontSize: 10,
                formatter: value => `${(value * 100).toFixed(0)}%`,
            },
            axisLine: { lineStyle: { color: 'rgba(0,210,255,0.1)' } },
            splitLine: { lineStyle: { color: UI_GRID_LINE } },
        },
        series: compareSeries.map(item =>
            buildCompareDistributionSeries(
                item.name,
                item.bins.slice(focus.start, focus.end + 1),
                item.color,
                item.fill !== false
            )
        ),
        animationDuration: 800,
    }, true);
}

function buildCompareDistributionSeries(name, bins, color, withArea = true) {
    return {
        name,
        type: 'line',
        smooth: true,
        showSymbol: false,
        data: bins,
        lineStyle: {
            width: 2,
            color,
            shadowBlur: 10,
            shadowColor: color,
        },
        areaStyle: withArea ? {
            color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [
                { offset: 0, color: withAlpha(color, 0.25) },
                { offset: 1, color: withAlpha(color, 0.03) },
            ]),
        } : undefined,
    };
}

function renderCompareTable() {
    const section = document.getElementById('compareTableSection');
    const tbody = document.querySelector('#compareTable tbody');
    const metrics = getCompareMetricsData();
    if (!section || !tbody || !metrics) return;
    section.style.display = '';

    const okResults = metrics.results.filter(r => Number.isFinite(r.psnr) && Number.isFinite(r.ssim));
    const bestPsnr = okResults.length ? Math.max(...okResults.map(r => r.psnr)) : null;
    const bestSsim = okResults.length ? Math.max(...okResults.map(r => r.ssim)) : null;

    let html = `<tr class="baseline-row"><td>Baseline (含噪)</td><td>${metrics.baseline.psnr}</td><td>${metrics.baseline.ssim}</td><td>-</td></tr>`;
    okResults.forEach(r => {
        const p = r.psnr === bestPsnr ? 'best-val' : '';
        const s = r.ssim === bestSsim ? 'best-val' : '';
        const activeCls = r.model === state.compareActiveModel ? 'is-active' : '';
        html += `<tr class="${activeCls}" data-model="${r.model}" style="cursor:pointer"><td>${r.display_name}</td><td class="${p}">${r.psnr}</td><td class="${s}">${r.ssim}</td><td>${r.elapsed_ms}ms</td></tr>`;
    });
    tbody.innerHTML = html;

    tbody.querySelectorAll('tr[data-model]').forEach(row => {
        row.addEventListener('click', () => {
            setCompareActiveModel(row.dataset.model);
        });
    });
}

function getCompareMetricsData() {
    if (!state.compareData) return null;
    const baseResults = state.compareData.results.filter(r => r.success);
    if (!state.compareImageStore) {
        return {
            baseline: {
                psnr: roundMetric(state.compareData.baseline.psnr, 2),
                ssim: roundMetric(state.compareData.baseline.ssim, 4),
            },
            results: baseResults.map(r => ({
                ...r,
                psnr: roundMetric(r.psnr, 2),
                ssim: roundMetric(r.ssim, 4),
            })),
            fromRegion: false,
            bounds: null,
        };
    }

    const bounds = getCompareBounds(state.compareImageStore);
    const baselineMetrics = computeRegionMetrics(state.compareImageStore.noisy, state.compareImageStore.gt, bounds);
    const results = baseResults.map(serverResult => {
        const loaded = state.compareImageStore.results.find(item => item.model === serverResult.model);
        if (!loaded) return null;
        return {
            ...serverResult,
            ...computeRegionMetrics(loaded, state.compareImageStore.gt, bounds),
        };
    }).filter(Boolean);

    return {
        baseline: baselineMetrics,
        results,
        fromRegion: state.compareScopeMode === 'roi',
        bounds,
    };
}

function roundMetric(value, decimals) {
    if (!Number.isFinite(value)) return '--';
    return Number(value.toFixed(decimals));
}

function getActiveCompareResult() {
    if (!state.compareImageStore) return null;
    return state.compareImageStore.results.find(item => item.model === state.compareActiveModel)
        || state.compareImageStore.results[0]
        || null;
}

function setCompareActiveModel(model) {
    if (!model || state.compareActiveModel === model) return;
    state.compareActiveModel = model;
    renderCompareTable();
    renderCompareChart();
    updateCompareScopeNote();
}

async function prepareCompareAnalysisAssets(data) {
    const successful = data.results.filter(r => r.success);
    const noisyUrl = getNoisyImageUrl(data.input_url);
    const gtUrl = data.gt_url || getGroundTruthImageUrl();
    updateCompareRoiPreview(noisyUrl);
    if (!successful.length || !noisyUrl || !gtUrl) {
        updateCompareScopeNote('当前缺少局部分析所需图像，仍使用整图指标。');
        return;
    }

    const requestId = ++state.compareAssetRequestId;
    updateCompareScopeNote('正在加载区域分析图像...');

    try {
        const imageEntries = await Promise.all([
            loadImageElement(noisyUrl),
            loadImageElement(gtUrl),
            ...successful.map(item => loadImageElement(item.result_url)),
        ]);
        if (requestId !== state.compareAssetRequestId) return;

        const baseWidth = Math.max(1, Math.min(...imageEntries.map(img => img.naturalWidth || img.width || 1)));
        const baseHeight = Math.max(1, Math.min(...imageEntries.map(img => img.naturalHeight || img.height || 1)));
        const [noisyImg, gtImg, ...resultImgs] = imageEntries;

        state.compareImageStore = {
            width: baseWidth,
            height: baseHeight,
            noisy: rasterizeImageForAnalysis(noisyImg, baseWidth, baseHeight),
            gt: rasterizeImageForAnalysis(gtImg, baseWidth, baseHeight),
            results: successful.map((item, index) => ({
                ...item,
                ...rasterizeImageForAnalysis(resultImgs[index], baseWidth, baseHeight),
            })),
        };

        updateCompareRoiWindow();
        updateCompareScopeNote();
        scheduleCompareAnalysis(true);
    } catch (error) {
        if (requestId !== state.compareAssetRequestId) return;
        state.compareImageStore = null;
        updateCompareScopeNote('区域分析资源加载失败，当前保留整图统计。');
        console.error('加载多算法对比分析图像失败', error);
    }
}

function loadImageElement(url) {
    return new Promise((resolve, reject) => {
        const img = new Image();
        img.crossOrigin = 'anonymous';
        img.onload = () => resolve(img);
        img.onerror = () => reject(new Error(`图像加载失败: ${url}`));
        img.src = url;
    });
}

function rasterizeImageForAnalysis(img, width, height) {
    const canvas = document.createElement('canvas');
    canvas.width = width;
    canvas.height = height;
    const ctx = canvas.getContext('2d', { willReadFrequently: true });
    ctx.drawImage(img, 0, 0, width, height);
    const rgba = ctx.getImageData(0, 0, width, height).data;
    const gray = new Float32Array(width * height);
    for (let i = 0, j = 0; i < rgba.length; i += 4, j += 1) {
        gray[j] = 0.299 * rgba[i] + 0.587 * rgba[i + 1] + 0.114 * rgba[i + 2];
    }
    return { width, height, rgba, gray, histogramCache: new Map() };
}

function getCompareBounds(store) {
    const width = store.width;
    const height = store.height;
    if (state.compareScopeMode !== 'roi') {
        return {
            x: 0,
            y: 0,
            width,
            height,
            leftRatio: 0,
            topRatio: 0,
        };
    }

    const roiWidth = Math.max(16, Math.round(width * state.compareRoi.size));
    const roiHeight = Math.max(16, Math.round(height * state.compareRoi.size));
    const maxX = Math.max(0, width - roiWidth);
    const maxY = Math.max(0, height - roiHeight);
    const x = clamp(Math.round(width * state.compareRoi.x), 0, maxX);
    const y = clamp(Math.round(height * state.compareRoi.y), 0, maxY);

    return {
        x,
        y,
        width: roiWidth,
        height: roiHeight,
        leftRatio: width > 0 ? x / width : 0,
        topRatio: height > 0 ? y / height : 0,
    };
}

function computeRegionMetrics(source, target, bounds) {
    const width = Math.min(source.width, target.width);
    const height = Math.min(source.height, target.height);
    const x0 = clamp(bounds.x, 0, Math.max(0, width - 1));
    const y0 = clamp(bounds.y, 0, Math.max(0, height - 1));
    const roiWidth = clamp(bounds.width, 1, width - x0);
    const roiHeight = clamp(bounds.height, 1, height - y0);
    let rgbError = 0;
    let rgbCount = 0;
    let sumX = 0;
    let sumY = 0;
    let sumXX = 0;
    let sumYY = 0;
    let sumXY = 0;
    let count = 0;

    for (let row = 0; row < roiHeight; row += 1) {
        let pixelIndex = (y0 + row) * width + x0;
        for (let col = 0; col < roiWidth; col += 1, pixelIndex += 1) {
            const rgbaIndex = pixelIndex * 4;
            const dr = source.rgba[rgbaIndex] - target.rgba[rgbaIndex];
            const dg = source.rgba[rgbaIndex + 1] - target.rgba[rgbaIndex + 1];
            const db = source.rgba[rgbaIndex + 2] - target.rgba[rgbaIndex + 2];
            rgbError += dr * dr + dg * dg + db * db;
            rgbCount += 3;

            const gx = source.gray[pixelIndex];
            const gy = target.gray[pixelIndex];
            sumX += gx;
            sumY += gy;
            sumXX += gx * gx;
            sumYY += gy * gy;
            sumXY += gx * gy;
            count += 1;
        }
    }

    const mse = rgbCount > 0 ? rgbError / rgbCount : 0;
    const psnr = mse <= 1e-12 ? 99.99 : 10 * Math.log10((255 * 255) / mse);
    const meanX = count > 0 ? sumX / count : 0;
    const meanY = count > 0 ? sumY / count : 0;
    const varX = count > 0 ? Math.max(0, sumXX / count - meanX * meanX) : 0;
    const varY = count > 0 ? Math.max(0, sumYY / count - meanY * meanY) : 0;
    const covXY = count > 0 ? sumXY / count - meanX * meanY : 0;
    const c1 = Math.pow(0.01 * 255, 2);
    const c2 = Math.pow(0.03 * 255, 2);
    const ssimNumerator = (2 * meanX * meanY + c1) * (2 * covXY + c2);
    const ssimDenominator = (meanX * meanX + meanY * meanY + c1) * (varX + varY + c2);
    const ssim = ssimDenominator > 0 ? clamp(ssimNumerator / ssimDenominator, -1, 1) : 1;

    return {
        psnr: roundMetric(psnr, 2),
        ssim: roundMetric(ssim, 4),
    };
}

function getHistogramCacheKey(bounds, maxSamples) {
    return [bounds.x, bounds.y, bounds.width, bounds.height, maxSamples || 0].join(':');
}

function computeHistogramBins(image, bounds, options = {}) {
    const width = image.width;
    const x0 = clamp(bounds.x, 0, Math.max(0, image.width - 1));
    const y0 = clamp(bounds.y, 0, Math.max(0, image.height - 1));
    const roiWidth = clamp(bounds.width, 1, image.width - x0);
    const roiHeight = clamp(bounds.height, 1, image.height - y0);
    const maxSamples = options.maxSamples || 0;
    const cacheKey = getHistogramCacheKey({ x: x0, y: y0, width: roiWidth, height: roiHeight }, maxSamples);
    if (image.histogramCache?.has(cacheKey)) {
        return image.histogramCache.get(cacheKey);
    }

    const bins = new Array(256).fill(0);
    let count = 0;
    const sampleCount = roiWidth * roiHeight;
    const stride = maxSamples > 0 && sampleCount > maxSamples
        ? Math.max(1, Math.ceil(Math.sqrt(sampleCount / maxSamples)))
        : 1;

    for (let row = 0; row < roiHeight; row += stride) {
        let pixelIndex = (y0 + row) * width + x0;
        for (let col = 0; col < roiWidth; col += stride, pixelIndex += stride) {
            bins[Math.max(0, Math.min(255, Math.round(image.gray[pixelIndex])))] += 1;
            count += 1;
        }
    }

    const normalized = bins.map(value => count > 0 ? value / count : 0);
    image.histogramCache?.set(cacheKey, normalized);
    return normalized;
}

function scheduleCompareAnalysis(immediate = false) {
    if (state.compareAnalysisTimer) {
        clearTimeout(state.compareAnalysisTimer);
        state.compareAnalysisTimer = null;
    }
    const delay = immediate ? 0 : 80;
    state.compareAnalysisTimer = setTimeout(() => {
        state.compareAnalysisTimer = null;
        renderCompareTable();
        renderCompareChart();
        updateCompareScopeNote();
    }, delay);
}

function setCompareProgress(percent, status, modelCount = null) {
    const fill = document.getElementById('compareProgressFill');
    const text = document.getElementById('compareProgressText');
    const statusEl = document.getElementById('compareProgressStatus');
    const countEl = document.getElementById('compareModelCount');
    const safePercent = clamp(Math.round(percent), 0, 100);
    state.compareProgressValue = safePercent;
    if (fill) fill.style.width = `${safePercent}%`;
    if (text) text.textContent = `${safePercent}%`;
    if (statusEl && status) statusEl.textContent = status;
    if (countEl && modelCount != null) countEl.textContent = String(modelCount);
}

function startCompareProgress(modelCount) {
    if (state.compareProgressFrame) {
        cancelAnimationFrame(state.compareProgressFrame);
        state.compareProgressFrame = null;
    }
    const start = performance.now();
    setCompareProgress(6, '正在推理与汇总对比结果...', modelCount);

    const tick = now => {
        const elapsed = now - start;
        const pct = Math.min(92, 6 + (1 - Math.exp(-elapsed / 1800)) * 86);
        state.compareProgressValue = pct;
        setCompareProgress(pct, '正在推理与汇总对比结果...', modelCount);
        state.compareProgressFrame = requestAnimationFrame(tick);
    };
    state.compareProgressFrame = requestAnimationFrame(tick);
}

function stopCompareProgress(success, status) {
    if (state.compareProgressFrame) {
        cancelAnimationFrame(state.compareProgressFrame);
        state.compareProgressFrame = null;
    }
    setCompareProgress(success ? 100 : Math.max(0, state.compareProgressValue || 0), status, getSelectedCompareModels().length);
}

function syncCompareToolbarButtons() {
    document.querySelectorAll('[data-compare-view]').forEach(btn => {
        btn.classList.toggle('active', btn.dataset.compareView === state.compareViewMode);
    });
    const sub = document.getElementById('compareDistributionToolbar');
    if (sub) sub.style.display = state.compareViewMode === 'distribution' ? 'flex' : 'none';
    syncCompareDistributionButtons();
}

function syncCompareDistributionButtons() {
    document.querySelectorAll('[data-compare-distribution]').forEach(btn => {
        btn.classList.toggle('active', btn.dataset.compareDistribution === state.compareDistributionMode);
    });
}

function syncCompareScopeButtons() {
    document.querySelectorAll('[data-compare-scope]').forEach(btn => {
        btn.classList.toggle('active', btn.dataset.compareScope === state.compareScopeMode);
    });
    const panel = document.getElementById('compareRoiPanel');
    if (panel) panel.style.display = state.compareScopeMode === 'roi' ? '' : 'none';
    updateCompareRoiWindow();
    updateCompareScopeNote();
}

function updateCompareRoiPreview(url) {
    const preview = document.getElementById('compareRoiPreview');
    const img = document.getElementById('compareRoiImage');
    if (!img || !preview) return;
    if (!url) {
        state.comparePreviewRequestId += 1;
        img.removeAttribute('src');
        preview.classList.add('is-empty');
        return;
    }

    const requestId = ++state.comparePreviewRequestId;
    preview.classList.add('is-empty');
    const probe = new Image();
    probe.onload = () => {
        if (requestId !== state.comparePreviewRequestId) return;
        img.src = url;
        preview.classList.remove('is-empty');
    };
    probe.onerror = () => {
        if (requestId !== state.comparePreviewRequestId) return;
        img.removeAttribute('src');
        preview.classList.add('is-empty');
        updateCompareScopeNote('区域预览图加载失败，但局部指标计算仍可继续。');
    };
    probe.src = url;
}

function updateCompareRoiWindow() {
    const windowEl = document.getElementById('compareRoiWindow');
    const sizeEl = document.getElementById('compareRoiSizeValue');
    const coordsEl = document.getElementById('compareRoiCoords');
    const pixelsEl = document.getElementById('compareRoiPixels');
    const sizePercent = Math.round(state.compareRoi.size * 100);
    if (sizeEl) sizeEl.textContent = `${sizePercent}%`;
    if (windowEl) {
        windowEl.style.left = `${state.compareRoi.x * 100}%`;
        windowEl.style.top = `${state.compareRoi.y * 100}%`;
        windowEl.style.width = `${state.compareRoi.size * 100}%`;
        windowEl.style.height = `${state.compareRoi.size * 100}%`;
    }
    if (coordsEl) {
        coordsEl.textContent = `x ${Math.round(state.compareRoi.x * 100)}% · y ${Math.round(state.compareRoi.y * 100)}%`;
    }
    if (pixelsEl) {
        if (state.compareImageStore) {
            const bounds = getCompareBounds(state.compareImageStore);
            pixelsEl.textContent = `ROI ${bounds.width}x${bounds.height}`;
        } else {
            pixelsEl.textContent = 'ROI --';
        }
    }
}

function updateCompareScopeNote(message = '') {
    const note = document.getElementById('compareScopeNote');
    if (!note) return;
    if (message) {
        note.textContent = message;
        return;
    }
    const active = getActiveCompareResult();
    const distributionModeText = state.compareDistributionMode === 'all' ? '全部结果' : (active ? active.display_name : '当前算法');
    if (state.compareScopeMode !== 'roi') {
        note.textContent = `当前使用整张图像统计多算法指标，分布图聚焦 ${distributionModeText}。`;
        return;
    }
    if (!state.compareImageStore) {
        note.textContent = '正在准备区域分析资源，加载完成后会自动切换为局部统计。';
        return;
    }
    const bounds = getCompareBounds(state.compareImageStore);
    note.textContent = `已切换到局部区域分析，拖动窗口后会实时刷新 ${distributionModeText} 的局部分布与全部算法局部指标。当前 ROI 为 ${bounds.width}x${bounds.height}。`;
}

function initCompareControls() {
    document.querySelectorAll('[data-compare-view]').forEach(btn => {
        btn.addEventListener('click', () => {
            state.compareViewMode = btn.dataset.compareView;
            syncCompareToolbarButtons();
            if (state.mode === 'compare' && state.compareData) renderCompareChart();
        });
    });

    document.querySelectorAll('[data-compare-distribution]').forEach(btn => {
        btn.addEventListener('click', () => {
            state.compareDistributionMode = btn.dataset.compareDistribution;
            syncCompareDistributionButtons();
            updateCompareScopeNote();
            if (state.mode === 'compare' && state.compareData && state.compareViewMode === 'distribution') {
                if (state.compareDistributionRenderTimer) {
                    clearTimeout(state.compareDistributionRenderTimer);
                    state.compareDistributionRenderTimer = null;
                }
                if (state.compareDistributionMode === 'all') {
                    renderComparePlaceholder('正在汇总全部结果分布...');
                }
                state.compareDistributionRenderTimer = setTimeout(() => {
                    state.compareDistributionRenderTimer = null;
                    renderCompareChart();
                }, 0);
            }
        });
    });

    document.querySelectorAll('[data-compare-scope]').forEach(btn => {
        btn.addEventListener('click', () => {
            state.compareScopeMode = btn.dataset.compareScope;
            syncCompareScopeButtons();
            if (state.mode === 'compare' && state.compareData) scheduleCompareAnalysis(true);
        });
    });

    const sizeInput = document.getElementById('compareRoiSize');
    if (sizeInput) {
        sizeInput.addEventListener('input', () => {
            const nextSize = clamp((parseInt(sizeInput.value, 10) || 36) / 100, 0.2, 0.8);
            state.compareRoi.size = nextSize;
            state.compareRoi.x = clamp(state.compareRoi.x, 0, 1 - nextSize);
            state.compareRoi.y = clamp(state.compareRoi.y, 0, 1 - nextSize);
            updateCompareRoiWindow();
            if (state.mode === 'compare' && state.compareScopeMode === 'roi' && state.compareData) {
                scheduleCompareAnalysis();
            }
        });
    }

    const preview = document.getElementById('compareRoiPreview');
    if (!preview) return;
    const previewImg = document.getElementById('compareRoiImage');

    let dragging = false;
    let pointerOffsetX = 0;
    let pointerOffsetY = 0;

    preview.addEventListener('dragstart', event => event.preventDefault());
    previewImg?.addEventListener('dragstart', event => event.preventDefault());

    function updateRoiFromEvent(event, keepOffset) {
        const rect = preview.getBoundingClientRect();
        if (rect.width <= 0 || rect.height <= 0) return;
        const pointerX = (event.clientX - rect.left) / rect.width;
        const pointerY = (event.clientY - rect.top) / rect.height;
        const nextX = keepOffset ? pointerX - pointerOffsetX : pointerX - state.compareRoi.size / 2;
        const nextY = keepOffset ? pointerY - pointerOffsetY : pointerY - state.compareRoi.size / 2;
        state.compareRoi.x = clamp(nextX, 0, 1 - state.compareRoi.size);
        state.compareRoi.y = clamp(nextY, 0, 1 - state.compareRoi.size);
        updateCompareRoiWindow();
        if (state.mode === 'compare' && state.compareScopeMode === 'roi' && state.compareData) {
            scheduleCompareAnalysis();
        }
    }

    preview.addEventListener('pointerdown', event => {
        event.preventDefault();
        dragging = true;
        preview.classList.add('dragging');
        const rect = preview.getBoundingClientRect();
        const pointerX = (event.clientX - rect.left) / rect.width;
        const pointerY = (event.clientY - rect.top) / rect.height;
        const insideX = pointerX >= state.compareRoi.x && pointerX <= state.compareRoi.x + state.compareRoi.size;
        const insideY = pointerY >= state.compareRoi.y && pointerY <= state.compareRoi.y + state.compareRoi.size;
        if (insideX && insideY) {
            pointerOffsetX = pointerX - state.compareRoi.x;
            pointerOffsetY = pointerY - state.compareRoi.y;
            updateRoiFromEvent(event, true);
        } else {
            pointerOffsetX = state.compareRoi.size / 2;
            pointerOffsetY = state.compareRoi.size / 2;
            updateRoiFromEvent(event, false);
        }
        preview.setPointerCapture(event.pointerId);
    });

    preview.addEventListener('pointermove', event => {
        if (!dragging) return;
        updateRoiFromEvent(event, true);
    });

    const release = event => {
        if (!dragging) return;
        dragging = false;
        preview.classList.remove('dragging');
        try {
            preview.releasePointerCapture(event.pointerId);
        } catch (_) {}
    };

    preview.addEventListener('pointerup', release);
    preview.addEventListener('pointercancel', release);
    preview.addEventListener('pointerleave', event => {
        if (dragging && event.buttons === 0) release(event);
    });

    updateCompareRoiWindow();
}

/* ═══════════════════════════════════════════════════════════
   11. 图像对比滑块
   ═══════════════════════════════════════════════════════════ */
function showSlider(beforeUrl, afterUrl) {
    const slider = document.getElementById('imageSlider');
    const empty = document.getElementById('emptyState');
    if (!slider || !empty) return;
    empty.style.display = 'none';
    slider.style.display = 'flex';

    const imgB = document.getElementById('imgBefore');
    const imgA = document.getElementById('imgAfter');
    imgB.src = beforeUrl;
    imgA.src = afterUrl;
    resetSliderPosition();
}

function getNoisyImageUrl(fallbackUrl) {
    if (fallbackUrl) {
        return fallbackUrl;
    }
    if (state.uploadedFile) {
        return `/media/${state.uploadedFile}`;
    }
    return '';
}

function getGroundTruthImageUrl() {
    if (state.gtFile) {
        return `/media/${state.gtFile}`;
    }
    return '';
}

function hideSlider() {
    const slider = document.getElementById('imageSlider');
    if (slider) slider.style.display = 'none';
}

function resetSliderPosition() {
    const overlay = document.getElementById('sliderOverlay');
    const handle = document.getElementById('sliderHandle');
    if (overlay) overlay.style.clipPath = 'inset(0 50% 0 0)';
    if (handle) handle.style.left = '50%';
}

function initSlider() {
    const container = document.querySelector('.slider-container');
    if (!container) return;
    let dragging = false;

    function updatePos(clientX) {
        const rect = container.getBoundingClientRect();
        let pct = Math.max(0, Math.min(100, (clientX - rect.left) / rect.width * 100));
        document.getElementById('sliderOverlay').style.clipPath = `inset(0 ${100 - pct}% 0 0)`;
        document.getElementById('sliderHandle').style.left = pct + '%';
    }

    container.addEventListener('mousedown', e => { dragging = true; updatePos(e.clientX); });
    document.addEventListener('mousemove', e => { if (dragging) updatePos(e.clientX); });
    document.addEventListener('mouseup', () => { dragging = false; });
    container.addEventListener('touchstart', e => { dragging = true; updatePos(e.touches[0].clientX); }, { passive: true });
    document.addEventListener('touchmove', e => { if (dragging) updatePos(e.touches[0].clientX); }, { passive: true });
    document.addEventListener('touchend', () => { dragging = false; });
}

/* ═══════════════════════════════════════════════════════════
   12. 图像亮度分布
   ═══════════════════════════════════════════════════════════ */
function initDistributionChart() {
    const dom = document.getElementById('distributionChart');
    if (!dom) return;
    state.distributionChart = echarts.init(dom, null, { renderer: 'canvas' });
    renderDistributionPlaceholder();
    window.addEventListener('resize', () => state.distributionChart?.resize());
}

function renderDistributionPlaceholder(message = '等待去噪结果') {
    const note = document.getElementById('distributionNote');
    if (note) note.textContent = message;
    if (!state.distributionChart) return;
    state.distributionChart.setOption({
        backgroundColor: 'transparent',
        title: {
            text: message,
            left: 'center',
            top: 'middle',
            textStyle: {
                color: UI_TEXT_MUTED,
                fontFamily: 'Rajdhani',
                fontSize: 12,
                fontWeight: 500,
            },
        },
        xAxis: { show: false },
        yAxis: { show: false },
        series: [],
    }, true);
}

function loadImageHistogram(url) {
    return new Promise((resolve, reject) => {
        const img = new Image();
        img.crossOrigin = 'anonymous';
        img.onload = () => {
            const canvas = document.createElement('canvas');
            const maxSide = 320;
            const scale = Math.min(1, maxSide / Math.max(img.naturalWidth || 1, img.naturalHeight || 1));
            canvas.width = Math.max(1, Math.round((img.naturalWidth || 1) * scale));
            canvas.height = Math.max(1, Math.round((img.naturalHeight || 1) * scale));
            const ctx = canvas.getContext('2d', { willReadFrequently: true });
            ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
            const pixels = ctx.getImageData(0, 0, canvas.width, canvas.height).data;
            const bins = new Array(256).fill(0);
            let mean = 0;
            let sq = 0;
            let count = 0;
            for (let i = 0; i < pixels.length; i += 4) {
                const lum = Math.max(0, Math.min(255, Math.round(
                    0.299 * pixels[i] + 0.587 * pixels[i + 1] + 0.114 * pixels[i + 2]
                )));
                bins[lum] += 1;
                mean += lum;
                sq += lum * lum;
                count += 1;
            }
            const norm = bins.map(v => count > 0 ? v / count : 0);
            mean = count > 0 ? mean / count : 0;
            const variance = count > 0 ? Math.max(0, sq / count - mean * mean) : 0;
            resolve({
                bins: norm,
                mean,
                std: Math.sqrt(variance),
            });
        };
        img.onerror = () => reject(new Error(`图像加载失败: ${url}`));
        img.src = url;
    });
}

async function updateDistributionChart(noisyUrl, resultUrl, gtUrl = '') {
    if (!state.distributionChart || state.mode !== 'inference') return;
    const signature = [noisyUrl || '', resultUrl || '', gtUrl || ''].join('|');
    if (!resultUrl || signature === state.distributionSignature) return;
    state.distributionSignature = signature;
    const requestId = ++state.distributionRequestId;
    const note = document.getElementById('distributionNote');
    if (note) note.textContent = '正在分析图像亮度分布...';

    const entries = [
        { key: 'noisy', label: '含噪', url: noisyUrl, color: '#00d2ff' },
        { key: 'result', label: '去噪结果', url: resultUrl, color: '#ff007f' },
    ];
    if (gtUrl) {
        entries.push({ key: 'gt', label: '真值', url: gtUrl, color: '#00ff88' });
    }

    try {
        const histograms = await Promise.all(entries.map(async entry => ({
            ...entry,
            ...(await loadImageHistogram(entry.url)),
        })));
        if (requestId !== state.distributionRequestId) return;

        const labels = Array.from({ length: 256 }, (_, idx) => idx);
        const summary = histograms
            .map(item => `${item.label}: mean ${item.mean.toFixed(1)}, std ${item.std.toFixed(1)}`)
            .join(' | ');
        if (note) note.textContent = summary;

        const focus = computeHistogramFocusWindow(histograms.map(item => item.bins), {
            usePercentileCap: false,
            percentile: 1,
        });
        const focusedLabels = labels.slice(focus.start, focus.end + 1);

        state.distributionChart.setOption({
            backgroundColor: 'transparent',
            tooltip: {
                trigger: 'axis',
                backgroundColor: 'rgba(8,16,32,0.92)',
                borderColor: 'rgba(0,210,255,0.18)',
                textStyle: { color: '#dce6f5', fontFamily: 'Rajdhani' },
                formatter: params => {
                    const lines = [`亮度值 ${params[0]?.axisValue ?? '--'}`];
                    params.forEach(item => {
                        lines.push(`${item.seriesName}: ${(item.value * 100).toFixed(2)}%`);
                    });
                    return lines.join('<br>');
                },
            },
            legend: {
                top: 8,
                itemWidth: 12,
                itemHeight: 8,
                textStyle: { color: UI_TEXT_SECONDARY, fontFamily: 'Rajdhani', fontSize: 11 },
            },
            grid: { top: 38, bottom: 28, left: 42, right: 14 },
            xAxis: {
                type: 'category',
                data: focusedLabels,
                boundaryGap: false,
                axisLabel: {
                    color: UI_TEXT_SECONDARY,
                    fontFamily: 'Rajdhani',
                    fontSize: 10,
                    interval: Math.max(0, Math.floor((focusedLabels.length - 1) / 6)),
                },
                axisLine: { lineStyle: { color: 'rgba(0,210,255,0.1)' } },
                splitLine: { show: false },
            },
            yAxis: {
                type: 'value',
                name: '占比',
                max: focus.yMax,
                nameTextStyle: { color: UI_TEXT_SECONDARY, fontFamily: 'Rajdhani', fontSize: 10 },
                axisLabel: {
                    color: UI_TEXT_SECONDARY,
                    fontFamily: 'Rajdhani',
                    fontSize: 10,
                    formatter: value => `${(value * 100).toFixed(0)}%`,
                },
                axisLine: { lineStyle: { color: 'rgba(0,210,255,0.1)' } },
                splitLine: { lineStyle: { color: UI_GRID_LINE } },
            },
            series: histograms.map(item => ({
                name: item.label,
                type: 'line',
                smooth: true,
                showSymbol: false,
                data: item.bins.slice(focus.start, focus.end + 1),
                lineStyle: {
                    width: 2,
                    color: item.color,
                    shadowBlur: 10,
                    shadowColor: item.color,
                },
                areaStyle: {
                    color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [
                        { offset: 0, color: `${item.color}40` },
                        { offset: 1, color: `${item.color}05` },
                    ]),
                },
            })),
            animationDuration: 800,
        }, true);
    } catch (error) {
        if (requestId !== state.distributionRequestId) return;
        renderDistributionPlaceholder('图像分布分析暂不可用');
        console.error('图像分布分析失败', error);
    }
}

/* ═══════════════════════════════════════════════════════════
   13. ECharts：Loss 曲线（渐变填充 + 发光数据点）
   ═══════════════════════════════════════════════════════════ */
function initLossChart() {
    const dom = document.getElementById('lossChart');
    if (!dom) return;
    if (!state.lossChart) {
        state.lossChart = echarts.init(dom, null, { renderer: 'canvas' });
    }
    renderLossPlaceholder();
    window.addEventListener('resize', () => state.lossChart?.resize());
}

function resizeLossChartSoon() {
    if (!state.lossChart) return;
    requestAnimationFrame(() => {
        requestAnimationFrame(() => {
            state.lossChart?.resize();
        });
    });
}

function renderLossPlaceholder(message = '等待开始训练') {
    if (!state.lossChart) return;
    state.lossChart.setOption({
        backgroundColor: 'transparent',
        title: {
            text: message,
            left: 'center',
            top: 'middle',
            textStyle: {
                color: UI_TEXT_MUTED,
                fontFamily: 'Rajdhani',
                fontSize: 12,
                fontWeight: 500,
            },
        },
        xAxis: { show: false },
        yAxis: { show: false },
        series: [],
    }, true);
}

function updateLossChart(losses) {
    if (!state.lossChart) return;
    state.lossChart.resize();

    state.lossChart.setOption({
        backgroundColor: 'transparent',
        title: { show: false },
        tooltip: {
            trigger: 'axis',
            backgroundColor: 'rgba(8,16,32,0.9)', borderColor: 'rgba(0,210,255,0.2)',
            textStyle: { color: '#dce6f5', fontFamily: 'Rajdhani' },
        },
        grid: { top: 16, bottom: 28, left: 50, right: 12 },
        xAxis: {
            type: 'category', data: losses.map((_, i) => i + 1), boundaryGap: false,
            axisLabel: { color: UI_TEXT_SECONDARY, fontFamily: 'Rajdhani', fontSize: 10 },
            axisLine: { lineStyle: { color: 'rgba(0,210,255,0.1)' } },
            splitLine: { show: false },
        },
        yAxis: {
            type: 'value',
            axisLabel: { color: UI_TEXT_SECONDARY, fontFamily: 'Rajdhani', fontSize: 10 },
            axisLine: { lineStyle: { color: 'rgba(0,210,255,0.1)' } },
            splitLine: { lineStyle: { color: UI_GRID_LINE } },
        },
        series: [{
            type: 'line', data: losses, smooth: true, symbol: 'circle', symbolSize: 3,
            showSymbol: false,
            lineStyle: { width: 2, color: '#00d2ff', shadowBlur: 8, shadowColor: 'rgba(0,210,255,0.4)' },
            itemStyle: { color: '#00d2ff', shadowBlur: 10, shadowColor: 'rgba(0,210,255,0.6)' },
            areaStyle: {
                color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [
                    { offset: 0, color: 'rgba(0, 210, 255, 0.25)' },
                    { offset: 1, color: 'rgba(0, 210, 255, 0)' },
                ]),
            },
        }],
        animation: true,
    }, true);
}

/* ═══════════════════════════════════════════════════════════
   14. ECharts：仪表盘 Gauge
   ═══════════════════════════════════════════════════════════ */
function initGauges() {
    const psnrDom = document.getElementById('gaugePsnr');
    const ssimDom = document.getElementById('gaugeSsim');
    if (!psnrDom || !ssimDom) return;

    state.gaugePsnr = echarts.init(psnrDom, null, { renderer: 'canvas' });
    state.gaugeSsim = echarts.init(ssimDom, null, { renderer: 'canvas' });

    function gaugeOpt(title, max, val, color1, color2) {
        return {
            backgroundColor: 'transparent',
            series: [{
                type: 'gauge', startAngle: 220, endAngle: -40,
                min: 0, max,
                radius: '100%', center: ['50%', '58%'],
                progress: { show: true, width: 8, roundCap: true,
                    itemStyle: {
                        color: new echarts.graphic.LinearGradient(0, 0, 1, 0, [
                            { offset: 0, color: color1 },
                            { offset: 1, color: color2 },
                        ]),
                        shadowBlur: 10, shadowColor: color1,
                    },
                },
                axisLine: { lineStyle: { width: 8, color: [[1, 'rgba(255,255,255,0.04)']] } },
                axisTick: { show: false },
                splitLine: { show: false },
                axisLabel: { show: false },
                pointer: { show: false },
                anchor: { show: false },
                title: {
                    offsetCenter: [0, '65%'], fontSize: 10,
                    fontFamily: 'Orbitron', color: UI_TEXT_SECONDARY, fontWeight: 600,
                },
                detail: {
                    offsetCenter: [0, '15%'], fontSize: 20, fontWeight: 900,
                    fontFamily: 'Orbitron',
                    color: color2,
                    valueAnimation: true,
                    formatter: v => v === 0 ? '--' : v.toFixed(max > 1 ? 1 : 3),
                    textShadowBlur: 10, textShadowColor: color1,
                },
                data: [{ value: val, name: title }],
                animationDuration: 1200,
            }],
        };
    }

    state.gaugePsnr.setOption(gaugeOpt('PSNR', 50, 0, 'rgba(0,85,255,0.8)', '#00d2ff'));
    state.gaugeSsim.setOption(gaugeOpt('SSIM', 1, 0, 'rgba(139,92,246,0.8)', '#ff007f'));

    window.addEventListener('resize', () => { state.gaugePsnr?.resize(); state.gaugeSsim?.resize(); });
}

function updateGauges(psnr, ssim) {
    if (state.gaugePsnr && psnr != null) {
        state.gaugePsnr.setOption({ series: [{ data: [{ value: psnr, name: 'PSNR' }] }] });
    }
    if (state.gaugeSsim && ssim != null) {
        state.gaugeSsim.setOption({ series: [{ data: [{ value: ssim, name: 'SSIM' }] }] });
    }
}

/* ═══════════════════════════════════════════════════════════
   15. 数字跳动动画
   ═══════════════════════════════════════════════════════════ */
function animateMetric(elementId, targetValue) {
    const el = document.getElementById(elementId);
    if (!el) return;
    const isFloat = !Number.isInteger(targetValue) && targetValue < 100;
    const decimals = isFloat ? (targetValue < 1 ? 4 : 2) : 0;
    const start = parseFloat(el.textContent) || 0;
    const diff = targetValue - start;
    const duration = 600;
    const startTime = performance.now();

    function step(now) {
        const t = Math.min((now - startTime) / duration, 1);
        const ease = 1 - Math.pow(1 - t, 3);
        el.textContent = (start + diff * ease).toFixed(decimals);
        if (t < 1) requestAnimationFrame(step);
        else {
            el.textContent = targetValue.toFixed(decimals);
            el.classList.add('updated');
            setTimeout(() => el.classList.remove('updated'), 500);
        }
    }
    requestAnimationFrame(step);
}

function setMetricValue(elementId, value, decimals = 0) {
    const el = document.getElementById(elementId);
    if (!el) return;
    el.textContent = Number.isFinite(value) ? value.toFixed(decimals) : '--';
}

function stopElapsedTimer(finalElapsed = null) {
    if (state.elapsedTimerFrame) {
        cancelAnimationFrame(state.elapsedTimerFrame);
        state.elapsedTimerFrame = null;
    }
    if (finalElapsed != null) {
        setMetricValue('metricTime', finalElapsed, 0);
    }
    state.elapsedStartTime = 0;
}

function startElapsedTimer() {
    stopElapsedTimer();
    state.elapsedStartTime = performance.now();

    function tick(now) {
        const elapsed = Math.max(0, Math.round(now - state.elapsedStartTime));
        setMetricValue('metricTime', elapsed, 0);
        state.elapsedTimerFrame = requestAnimationFrame(tick);
    }

    state.elapsedTimerFrame = requestAnimationFrame(tick);
}

/* ═══════════════════════════════════════════════════════════
   16. UI 辅助
   ═══════════════════════════════════════════════════════════ */
function setLoading(on, finalElapsedMs = null) {
    const btn = document.getElementById('btnExecute');
    if (!btn) return;
    btn.disabled = on;
    btn.querySelector('.btn-text').style.display = on ? 'none' : '';
    btn.querySelector('.btn-loader').style.display = on ? 'inline-block' : 'none';
    if (on) startElapsedTimer();
    else if (state.elapsedStartTime) {
        stopElapsedTimer(finalElapsedMs != null ? finalElapsedMs : Math.round(performance.now() - state.elapsedStartTime));
    }
}

function showScanOverlay(on) {
    const el = document.getElementById('scanOverlay');
    if (el) el.style.display = on ? 'flex' : 'none';
}

function showTrainingChartSection(on) {
    const el = document.getElementById('chartSection');
    if (el) el.style.display = on ? '' : 'none';
    if (on) resizeLossChartSoon();
}

function showTrainingUI(on) {
    const progress = document.getElementById('progressSection');
    if (progress) progress.style.display = on ? '' : 'none';
    if (on) {
        showTrainingChartSection(true);
        resizeLossChartSoon();
    }
}

function showDistributionSection(on) {
    const el = document.getElementById('distributionSection');
    if (el) el.style.display = on ? '' : 'none';
}

function updateAnalysisSectionsByMode() {
    const isTraining = state.mode === 'training';
    const isCompare = state.mode === 'compare';
    const gaugeRow = document.getElementById('gaugeRow');
    const metricsGrid = document.getElementById('metricsGrid');
    const comparePanel = document.getElementById('compareAnalysisPanel');
    const compareToolbar = document.getElementById('compareCenterToolbar');

    showDistributionSection(state.mode === 'inference');
    showTrainingChartSection(isTraining);
    if (gaugeRow) gaugeRow.style.display = isCompare ? 'none' : '';
    if (metricsGrid) metricsGrid.style.display = isCompare ? 'none' : 'grid';
    if (comparePanel) comparePanel.style.display = isCompare ? '' : 'none';
    if (compareToolbar) compareToolbar.style.display = isCompare ? 'flex' : 'none';

    if (isTraining) {
        renderLossPlaceholder();
        showTrainingUI(false);
    } else {
        showTrainingChartSection(false);
        showTrainingUI(false);
    }

    if (!isCompare) {
        const tableSection = document.getElementById('compareTableSection');
        if (tableSection) tableSection.style.display = 'none';
    } else if (state.compareData) {
        renderCompareTable();
        renderCompareChart();
    }
}

function resetMetrics() {
    stopElapsedTimer();
    stopCompareProgress(false, '等待开始');
    setCompareProgress(0, '等待开始', getSelectedCompareModels().length);
    if (state.compareAnalysisTimer) {
        clearTimeout(state.compareAnalysisTimer);
        state.compareAnalysisTimer = null;
    }
    if (state.compareDistributionRenderTimer) {
        clearTimeout(state.compareDistributionRenderTimer);
        state.compareDistributionRenderTimer = null;
    }
    ['metricPsnrBefore', 'metricPsnrAfter', 'metricSsimBefore', 'metricSsimAfter', 'metricTime']
        .forEach(id => { const el = document.getElementById(id); if (el) el.textContent = '--'; });
    updateGauges(0, 0);
    state.distributionSignature = '';
    renderDistributionPlaceholder();
    renderLossPlaceholder();
    updateCompareRoiWindow();
    updateCompareScopeNote('当前使用整张图像统计多算法指标。');
}

function hideCompareUI() {
    const cw = document.getElementById('compareChartWrapper');
    const cs = document.getElementById('compareTableSection');
    if (cw) cw.style.display = 'none';
    if (cs) cs.style.display = 'none';
}

/* ═══════════════════════════════════════════════════════════
   INIT
   ═══════════════════════════════════════════════════════════ */
document.addEventListener('DOMContentLoaded', () => {
    initStarfield();
    initClock();
    initModeSwitch();
    initUpload();
    initPresetModal();
    initAppModal();
    initSlider();
    initCompareControls();
    initDistributionChart();
    initLossChart();
    initGauges();
    loadModels();
    loadPresetImages();
    initExecuteButton();
});
