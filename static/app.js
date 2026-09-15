(() => {
  "use strict";

  const state = {
    mode: "image",
    file: null,
    currentFilter: "all",
    lastResult: null,
  };

  // ---------- DOM refs ----------
  const modeButtons = document.querySelectorAll(".mode-btn");
  const dropzone = document.getElementById("dropzone");
  const fileInput = document.getElementById("file-input");
  const dzTitle = document.getElementById("dz-title");
  const filenameEl = document.getElementById("filename");
  const analyzeBtn = document.getElementById("analyze-btn");
  const form = document.getElementById("upload-form");

  const placeholder = document.getElementById("placeholder");
  const resultEl = document.getElementById("result");
  const loadingEl = document.getElementById("loading");
  const loadingText = document.getElementById("loading-text");
  const errorBox = document.getElementById("error-box");

  const minAreaInput = document.getElementById("min-area");
  const minAreaValue = document.getElementById("min-area-value");
  const minCircInput = document.getElementById("min-circularity");
  const minCircValue = document.getElementById("min-circularity-value");

  const resultImage = document.getElementById("result-image");
  const statStrip = document.getElementById("stat-strip");
  const colorBreakdown = document.getElementById("color-breakdown");
  const videoTimeline = document.getElementById("video-timeline");
  const downloadRow = document.getElementById("download-row");
  const downloadLink = document.getElementById("download-link");
  const engineBadge = document.getElementById("engine-badge");
  const fieldLabelInput = document.getElementById("field-label");
  const comparePanel = document.getElementById("compare-panel");
  const intelGrid = document.getElementById("intel-grid");
  const harvestCard = document.getElementById("harvest-card");
  const healthCard = document.getElementById("health-card");
  const riskCard = document.getElementById("risk-card");
  const historyPanel = document.getElementById("history-panel");
  const historyToggle = document.getElementById("history-toggle");
  const historyBody = document.getElementById("history-body");
  const priorityGroups = document.getElementById("priority-groups");
  const detectionsFilters = document.getElementById("detections-filters");

  const heroReadiness = document.getElementById("hero-readiness");
  const heroRingFg = document.getElementById("hero-ring-fg");
  const heroEstimateLabel = document.getElementById("hero-estimate-label");
  const heroRecommendation = document.getElementById("hero-recommendation");
  const heroDelta = document.getElementById("hero-delta");

  // ---------- Slider displays ----------
  minAreaInput.addEventListener("input", () => {
    minAreaValue.textContent = `${minAreaInput.value} px²`;
  });
  minCircInput.addEventListener("input", () => {
    minCircValue.textContent = minCircInput.value;
  });

  // ---------- Mode switching ----------
  function setMode(mode) {
    state.mode = mode;
    state.file = null;
    fileInput.value = "";
    filenameEl.hidden = true;
    analyzeBtn.disabled = true;
    modeButtons.forEach(b =>
      b.classList.toggle("active", b.dataset.mode === mode)
    );
    fileInput.accept = mode === "image" ? "image/*" : "video/*";
    dzTitle.textContent =
      mode === "image" ? "Drop a photo here" : "Drop a video here";
    resetAll();
    placeholder.hidden = false;
  }

  modeButtons.forEach(btn => {
    btn.addEventListener("click", () => setMode(btn.dataset.mode));
  });

  // ---------- File handling ----------
  dropzone.addEventListener("dragover", e => {
    e.preventDefault();
    dropzone.classList.add("drag-over");
  });
  dropzone.addEventListener("dragleave", () =>
    dropzone.classList.remove("drag-over")
  );
  dropzone.addEventListener("drop", e => {
    e.preventDefault();
    dropzone.classList.remove("drag-over");
    const f = e.dataTransfer.files[0];
    if (f) handleFile(f);
  });
  fileInput.addEventListener("change", () => {
    if (fileInput.files[0]) handleFile(fileInput.files[0]);
  });

  function handleFile(file) {
    state.file = file;
    filenameEl.hidden = false;
    filenameEl.textContent = file.name;
    analyzeBtn.disabled = false;
  }

  // ---------- Reset ----------
  function resetAll() {
    resultEl.hidden = true;
    errorBox.hidden = true;
    loadingEl.hidden = true;
    placeholder.hidden = true;

    resultImage.src = "";
    statStrip.innerHTML = "";
    colorBreakdown.innerHTML = "";
    videoTimeline.innerHTML = "";
    videoTimeline.hidden = true;
    downloadRow.hidden = true;
    engineBadge.hidden = true;
    comparePanel.hidden = true;
    comparePanel.innerHTML = "";
    intelGrid.hidden = true;
    harvestCard.innerHTML = "";
    healthCard.innerHTML = "";
    riskCard.innerHTML = "";
    priorityGroups.innerHTML = "";
    historyPanel.hidden = true;
    historyBody.hidden = true;
    historyBody.innerHTML = "";

    heroReadiness.textContent = "–";
    heroRingFg.style.strokeDashoffset = 326.7;
    heroRingFg.style.stroke = "var(--moss)";
    heroEstimateLabel.textContent = "Waiting for scan…";
    heroRecommendation.textContent = "—";
    heroDelta.hidden = true;
    heroDelta.textContent = "";

    state.currentFilter = "all";
    document.querySelectorAll(".filter-btn").forEach(b =>
      b.classList.toggle("active", b.dataset.filter === "all")
    );
  }

  // ---------- Form submit ----------
  form.addEventListener("submit", async e => {
    e.preventDefault();
    if (!state.file) return;

    resetAll();
    placeholder.hidden = true;
    loadingEl.hidden = false;
    loadingText.textContent =
      state.mode === "image"
        ? "Segmenting colors and checking shapes…"
        : "Processing every frame and rendering annotated video…";
    analyzeBtn.disabled = true;

    try {
      const endpoint =
        state.mode === "image" ? "/api/analyze/image" : "/api/analyze/video";
      const fd = new FormData();
      fd.append("file", state.file);
      fd.append("min_area", minAreaInput.value);
      fd.append("min_circularity", minCircInput.value);
      fd.append(
        "field_label",
        (fieldLabelInput.value || "").trim() || "default"
      );

      const res = await fetch(endpoint, { method: "POST", body: fd });
      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        throw new Error(body.detail || `Request failed (${res.status})`);
      }
      const data = await res.json();

      state.lastResult = data;

      if (state.mode === "image") renderImageResult(data);
      else renderVideoResult(data);
    } catch (err) {
      errorBox.hidden = false;
      errorBox.textContent =
        err.message || "Something went wrong analyzing that file.";
      placeholder.hidden = false;
      resultEl.hidden = true;
    } finally {
      loadingEl.hidden = true;
      analyzeBtn.disabled = false;
    }
  });

  // ---------- Render helpers ----------
  function kpi(value, label) {
    const div = document.createElement("div");
    div.className = "kpi";
    div.innerHTML = `<div class="kpi-value">${value}</div><div class="kpi-label">${label}</div>`;
    return div;
  }

  function renderEngineBadge(engine) {
    if (!engine) {
      engineBadge.hidden = true;
      return;
    }
    const isYolo = engine === "yolo";
    engineBadge.hidden = false;
    engineBadge.className = `engine-badge ${isYolo ? "yolo" : "classical"}`;
    engineBadge.textContent = isYolo
      ? "YOLO — trained model"
      : "Classical CV";
  }

  function renderColorChips(byColor) {
    colorBreakdown.innerHTML = "";
    const entries = Object.entries(byColor || {});
    if (entries.length === 0) {
      colorBreakdown.innerHTML = `<span style="font-size:13px;color:var(--ink-soft)">No fruit-colored blobs matched the detection rules.</span>`;
      return;
    }
    for (const [color, count] of entries) {
      const chip = document.createElement("span");
      chip.className = `chip ${color}`;
      chip.textContent = `${count} ${color}`;
      colorBreakdown.appendChild(chip);
    }
  }

  // ---------- Hero score ring ----------
  const RING_CIRCUMFERENCE = 2 * Math.PI * 52; // r=52 → ≈326.7
  function setHeroRing(score) {
    const clamped = Math.max(0, Math.min(100, score || 0));
    const offset = RING_CIRCUMFERENCE * (1 - clamped / 100);
    heroRingFg.style.strokeDashoffset = offset;
    let color = "var(--moss)";
    if (clamped < 40) color = "var(--clay)";
    else if (clamped < 65) color = "var(--amber-deep)";
    heroRingFg.style.stroke = color;
    heroReadiness.textContent = clamped;
  }

  // ---------- Priority grouping ----------
  const PRIORITY_ORDER = ["harvest", "attention", "monitor", "developing"];
  const PRIORITY_LABELS = {
    harvest: "Ready to harvest",
    monitor: "Monitor",
    attention: "Needs attention",
    developing: "Developing",
  };

  function renderPriorityGroups(detections) {
    priorityGroups.innerHTML = "";

    if (!detections || detections.length === 0) {
      priorityGroups.innerHTML = `<div class="det-empty">No detections matched the current settings.</div>`;
      return;
    }

    // Bucket by priority
    const buckets = { harvest: [], monitor: [], attention: [], developing: [] };
    for (const d of detections) {
      const p = d.priority || "developing";
      (buckets[p] || buckets.developing).push(d);
    }

    // Order: harvest, attention, monitor, developing
    for (const key of PRIORITY_ORDER) {
      const list = buckets[key];
      if (!list || list.length === 0) continue;

      const group = document.createElement("div");
      group.className = `priority-group ${key}`;
      // Open "harvest" and "attention" by default; keep others collapsed
      const openByDefault = key === "harvest" || key === "attention";
      if (openByDefault) group.classList.add("open");
      group.dataset.priority = key;

      const head = document.createElement("button");
      head.type = "button";
      head.className = "priority-group-head";
      head.innerHTML = `
        <span class="priority-chevron">▸</span>
        <span class="priority-dot"></span>
        <span class="priority-group-name">${PRIORITY_LABELS[key]}</span>
        <span class="priority-group-count">${list.length}</span>
      `;
      head.addEventListener("click", () => group.classList.toggle("open"));

      const body = document.createElement("div");
      body.className = "priority-group-body";

      // Sort: by confidence desc, then id
      list
        .slice()
        .sort((a, b) => b.confidence - a.confidence || a.id - b.id)
        .forEach(d => body.appendChild(detectionRow(d)));

      group.appendChild(head);
      group.appendChild(body);
      priorityGroups.appendChild(group);
    }

    applyFilter(state.currentFilter);
  }

  function detectionRow(d) {
    const row = document.createElement("div");
    row.className = "det-row";

    const healthClass = d.health_score >= 70 ? "ok" : "warn";
    const healthShort =
      d.health_score >= 70 ? "clean" : `${d.health_score}/100`;

    row.innerHTML = `
      <span class="det-id">#${d.id}</span>
      <span class="det-color">
        <span class="det-swatch ${d.color_profile}"></span>
        ${d.color_profile}
      </span>
      <span class="det-maturity">${d.maturity_label}</span>
      <span class="det-health ${healthClass}">${healthShort}</span>
      <span class="det-conf">${Math.round(d.confidence * 100)}%</span>
    `;
    return row;
  }

  function applyFilter(filter) {
    state.currentFilter = filter;
    document.querySelectorAll(".filter-btn").forEach(b =>
      b.classList.toggle("active", b.dataset.filter === filter)
    );
    document.querySelectorAll(".priority-group").forEach(g => {
      const show = filter === "all" || g.dataset.priority === filter;
      g.style.display = show ? "" : "none";
      if (show && filter !== "all") g.classList.add("open");
    });
  }

  detectionsFilters.addEventListener("click", e => {
    const btn = e.target.closest(".filter-btn");
    if (!btn) return;
    applyFilter(btn.dataset.filter);
  });

  // ---------- Intel cards ----------
  function renderHarvestCard(hr) {
    harvestCard.innerHTML = `
      <h4>Harvest readiness</h4>
      <div class="intel-score">${hr.score}/100</div>
      <div class="intel-score-label">Estimated: ${hr.estimated_days_label}</div>
      <p class="rec">${hr.recommendation}</p>
    `;
  }

  function renderHealthCard(h) {
    const positives = (h.positive_factors || [])
      .map(x => `<li>${x}</li>`)
      .join("");
    const risks = (h.risk_factors || []).map(x => `<li>${x}</li>`).join("");
    healthCard.innerHTML = `
      <h4>Crop health</h4>
      <div class="intel-score">${h.score}/100</div>
      <div class="intel-score-label">${h.assessment}</div>
      ${
        positives
          ? `<div class="intel-subheading">Positive</div><ul class="intel-list positive">${positives}</ul>`
          : ""
      }
      ${
        risks
          ? `<div class="intel-subheading">Risks</div><ul class="intel-list risk">${risks}</ul>`
          : ""
      }
    `;
  }

  function renderRiskCard(r) {
    riskCard.innerHTML = `
      <h4>Crop risk</h4>
      <span class="risk-badge ${r.overall_risk}">${r.overall_risk_label}</span>
      <div class="risk-row"><span>Disease / anomaly</span><strong>${r.disease_risk_pct}%</strong></div>
      <div class="risk-row"><span>Over-ripening</span><strong>${r.over_ripening_risk_pct}%</strong></div>
      <div class="risk-row"><span>Under-ripening</span><strong>${r.under_ripening_risk_pct}%</strong></div>
      <div class="risk-row"><span>Damage</span><strong>${r.damage_risk_pct}%</strong></div>
    `;
  }

  function renderIntelligence(intel) {
    if (!intel) {
      intelGrid.hidden = true;
      return;
    }
    renderHarvestCard(intel.harvest_readiness);
    renderHealthCard(intel.health);
    renderRiskCard(intel.risk);
    intelGrid.hidden = false;

    // Hero score + meta
    setHeroRing(intel.harvest_readiness.score);
    heroEstimateLabel.textContent = `Estimated harvest: ${intel.harvest_readiness.estimated_days_label}`;
    heroRecommendation.textContent = intel.harvest_readiness.recommendation;
  }

  // ---------- Compare ----------
  function deltaClass(delta) {
    if (delta > 0) return "up";
    if (delta < 0) return "down";
    return "flat";
  }
  function deltaSign(delta) {
    return delta > 0 ? `+${delta}` : `${delta}`;
  }

  function renderCompare(compare) {
    if (!compare) {
      comparePanel.hidden = true;
      heroDelta.hidden = true;
      return;
    }
    const rows = [
      ["Fruit count", compare.fruit_count],
      ["Avg. maturity", compare.avg_maturity_score],
      ["Avg. health", compare.avg_health_score],
      ["Harvest readiness", compare.harvest_readiness_score],
      ["Ready to harvest", compare.ready_to_harvest],
      ["Flagged for attention", compare.attention_needed],
    ];
    const items = rows
      .map(
        ([label, d]) => `
      <div class="compare-item">
        <span class="compare-label">${label}</span>
        <span class="compare-value ${deltaClass(d.delta)}">${deltaSign(d.delta)}</span>
      </div>
    `
      )
      .join("");
    comparePanel.innerHTML = `<h3>What changed since your last scan of this field</h3><div class="compare-grid">${items}</div>`;
    comparePanel.hidden = false;

    // Hero delta pill: show readiness delta
    const readiness = compare.harvest_readiness_score;
    if (readiness && readiness.delta !== 0) {
      heroDelta.hidden = false;
      heroDelta.classList.toggle("negative", readiness.delta < 0);
      heroDelta.textContent = `Readiness ${deltaSign(readiness.delta)} since last scan`;
    } else {
      heroDelta.hidden = true;
    }
  }

  // ---------- History ----------
  function formatScanDate(ts) {
    if (!ts) return "";
    return new Date(ts * 1000).toLocaleString();
  }

  async function loadHistory(field) {
    historyBody.innerHTML = `<div class="history-empty">Loading…</div>`;
    historyBody.hidden = false;
    try {
      const res = await fetch(
        `/api/history?field=${encodeURIComponent(field)}`
      );
      const data = await res.json();
      if (!data.scans || data.scans.length === 0) {
        historyBody.innerHTML = `<div class="history-empty">No saved scans yet for this field.</div>`;
        return;
      }
      const rows = data.scans
        .slice()
        .reverse()
        .map(
          s => `
        <div class="history-row">
          <span class="h-date">${formatScanDate(s.timestamp)}</span>
          <span>${s.summary.total_detected} fruit</span>
          <span>readiness ${s.intelligence.harvest_readiness.score}/100</span>
          <span>${s.source_type}</span>
        </div>
      `
        )
        .join("");
      historyBody.innerHTML = rows;
    } catch (err) {
      historyBody.innerHTML = `<div class="history-empty" style="color:var(--clay)">Couldn't load history.</div>`;
    }
  }

  historyToggle.addEventListener("click", () => {
    const field = (fieldLabelInput.value || "").trim() || "default";
    if (!historyBody.hidden) {
      historyBody.hidden = true;
      return;
    }
    loadHistory(field);
  });

  // ---------- Image result ----------
  function renderImageResult(data) {
    resultImage.src = data.annotated_image;
    renderEngineBadge(data.engine);

    statStrip.innerHTML = "";
    statStrip.appendChild(
      kpi(data.summary.total_detected, "fruit-like blobs detected")
    );
    statStrip.appendChild(
      kpi(
        data.summary.avg_maturity_score + "/100",
        "avg. maturity color score"
      )
    );
    statStrip.appendChild(
      kpi(
        data.summary.avg_health_score + "/100",
        "avg. surface health score"
      )
    );
    statStrip.appendChild(
      kpi(data.summary.attention_needed, "flagged for a closer look")
    );

    renderColorChips(data.summary.by_color);
    renderPriorityGroups(data.detections);
    renderIntelligence(data.intelligence);
    renderCompare(data.compare_to_previous);
    historyPanel.hidden = false;

    resultEl.hidden = false;
  }

  // ---------- Video result ----------
  function renderVideoResult(data) {
    resultImage.src = "";
    renderEngineBadge(data.engine);

    statStrip.innerHTML = "";
    statStrip.appendChild(
      kpi(data.summary.total_detected, "unique fruit tracked")
    );
    statStrip.appendChild(
      kpi(data.video_meta.frames_processed, "frames processed")
    );
    statStrip.appendChild(
      kpi(
        data.summary.avg_maturity_score + "/100",
        "avg. maturity color score"
      )
    );
    statStrip.appendChild(
      kpi(data.summary.attention_needed, "flagged for a closer look")
    );

    renderColorChips(data.summary.by_color);

    if (data.annotated_video_url) {
      downloadRow.hidden = false;
      downloadLink.href = data.annotated_video_url;
    } else {
      downloadRow.hidden = true;
    }

    // Timeline
    if (data.timeline && data.timeline.length > 0) {
      const maxDet = Math.max(
        1,
        ...data.timeline.map(t => t.detections_in_frame)
      );
      const bars = data.timeline
        .map(t => {
          const h = Math.max(
            4,
            Math.round((t.detections_in_frame / maxDet) * 60)
          );
          return `<div class="bar" style="height:${h}px" title="frame ${t.frame_index}: ${t.detections_in_frame} detected"></div>`;
        })
        .join("");
      videoTimeline.innerHTML = `<h3>Detections over time</h3><div class="timeline-bars">${bars}</div>`;
      videoTimeline.hidden = false;
    } else {
      videoTimeline.hidden = true;
    }

    renderPriorityGroups(data.unique_fruit_detections);
    renderIntelligence(data.intelligence);
    renderCompare(data.compare_to_previous);
    historyPanel.hidden = false;

    resultEl.hidden = false;
  }

  // ---------- Init ----------
  setMode("image");
})();