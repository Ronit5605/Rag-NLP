const state = {
  mode: "text",
  samples: [],
  selectedTimedItems: null,
  lastAnalysis: null,
  activeEntityFilter: "ALL",
};

const els = {
  healthBadge: document.getElementById("healthBadge"),
  modeTextBtn: document.getElementById("modeTextBtn"),
  modeYoutubeBtn: document.getElementById("modeYoutubeBtn"),
  textInputGroup: document.getElementById("textInputGroup"),
  youtubeInputGroup: document.getElementById("youtubeInputGroup"),
  sampleSelect: document.getElementById("sampleSelect"),
  segmentCount: document.getElementById("segmentCount"),
  transcriptText: document.getElementById("transcriptText"),
  videoIdInput: document.getElementById("videoIdInput"),
  analyzeBtn: document.getElementById("analyzeBtn"),
  statusMessage: document.getElementById("statusMessage"),
  kpiGrid: document.getElementById("kpiGrid"),
  kpiWords: document.getElementById("kpiWords"),
  kpiReadTime: document.getElementById("kpiReadTime"),
  kpiTtr: document.getElementById("kpiTtr"),
  kpiUnique: document.getElementById("kpiUnique"),
  kpiFlesch: document.getElementById("kpiFlesch"),
  kpiGrade: document.getElementById("kpiGrade"),
  kpiSentiment: document.getElementById("kpiSentiment"),
  kpiPolarity: document.getElementById("kpiPolarity"),
  kpiEmotion: document.getElementById("kpiEmotion"),
  kpiEngine: document.getElementById("kpiEngine"),
  kpiEntities: document.getElementById("kpiEntities"),
  kpiEntityTypes: document.getElementById("kpiEntityTypes"),
  summaryList: document.getElementById("summaryList"),
  keywordsChips: document.getElementById("keywordsChips"),
  nounPhrasesChips: document.getElementById("nounPhrasesChips"),
  topTokensList: document.getElementById("topTokensList"),
  bigramsList: document.getElementById("bigramsList"),
  trigramsList: document.getElementById("trigramsList"),
  entityFilterBar: document.getElementById("entityFilterBar"),
  highlightedTranscript: document.getElementById("highlightedTranscript"),
  posChart: document.getElementById("posChart"),
  entitiesGrouped: document.getElementById("entitiesGrouped"),
  sentimentChartContainer: document.getElementById("sentimentChartContainer"),
  emotionBars: document.getElementById("emotionBars"),
  segmentsGrid: document.getElementById("segmentsGrid"),
  cleanedTextBox: document.getElementById("cleanedTextBox"),
  tokenSearch: document.getElementById("tokenSearch"),
  tokenTableBody: document.getElementById("tokenTableBody"),
  ragVideoId: document.getElementById("ragVideoId"),
  chatMessages: document.getElementById("chatMessages"),
  chatForm: document.getElementById("chatForm"),
  ragQuestion: document.getElementById("ragQuestion"),
  askBtn: document.getElementById("askBtn"),
};

function escapeHtml(str) {
  return String(str)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function switchMode(mode) {
  state.mode = mode;
  els.modeTextBtn.classList.toggle("active", mode === "text");
  els.modeYoutubeBtn.classList.toggle("active", mode === "youtube");
  els.textInputGroup.classList.toggle("hidden", mode !== "text");
  els.youtubeInputGroup.classList.toggle("hidden", mode !== "youtube");
}

function setupTabs() {
  document.querySelectorAll(".tab-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      const targetId = btn.dataset.tab;
      document.querySelectorAll(".tab-btn").forEach((b) => b.classList.remove("active"));
      document.querySelectorAll(".tab-panel").forEach((p) => p.classList.remove("active"));
      btn.classList.add("active");
      document.getElementById(targetId).classList.add("active");
    });
  });
}

async function loadHealthAndSamples() {
  try {
    const res = await fetch("/health");
    const data = await res.json();
    els.healthBadge.textContent = `API Ready • spaCy (${data.spacy_model})`;
    els.healthBadge.className = "status-pill status-ok";
  } catch {
    els.healthBadge.textContent = "API Offline";
  }

  try {
    const res = await fetch("/nlp/samples");
    const data = await res.json();
    state.samples = data.samples || [];
    state.samples.forEach((sample) => {
      const opt = document.createElement("option");
      opt.value = sample.id;
      opt.textContent = sample.title;
      els.sampleSelect.appendChild(opt);
    });
  } catch {
    // Ignore if unavailable
  }
}

function renderMiniBars(container, items, labelKey, valueKey, suffix = "") {
  if (!items || !items.length) {
    container.innerHTML = '<div class="empty-state">No items found.</div>';
    return;
  }
  const maxVal = Math.max(...items.map((i) => Number(i[valueKey]) || 0), 1);
  container.innerHTML = items
    .map((item) => {
      const val = Number(item[valueKey]) || 0;
      const pct = Math.max(4, Math.round((val / maxVal) * 100));
      return `
        <div class="bar-row">
          <span class="bar-label" title="${escapeHtml(item[labelKey])}">${escapeHtml(item[labelKey])}</span>
          <div class="bar-track"><div class="bar-fill" style="width:${pct}%"></div></div>
          <span class="bar-val">${escapeHtml(val)}${suffix}</span>
        </div>
      `;
    })
    .join("");
}

function renderChips(container, items, textKey, scoreKey) {
  if (!items || !items.length) {
    container.innerHTML = '<span class="empty-state">None detected.</span>';
    return;
  }
  container.innerHTML = items
    .map(
      (item) => `
      <span class="chip">
        <span>${escapeHtml(item[textKey])}</span>
        <span class="chip-score">${escapeHtml(item[scoreKey])}</span>
      </span>
    `
    )
    .join("");
}

function renderHighlightedEntities(cleanedText, entities, filterLabel) {
  if (!cleanedText) {
    els.highlightedTranscript.textContent = "No text available.";
    return;
  }

  const filtered = (entities || [])
    .filter((e) => filterLabel === "ALL" || e.label === filterLabel)
    .sort((a, b) => a.start_char - b.start_char);

  if (!filtered.length) {
    els.highlightedTranscript.textContent = cleanedText;
    return;
  }

  let html = "";
  let cursor = 0;
  for (const ent of filtered) {
    if (ent.start_char < cursor) continue;
    html += escapeHtml(cleanedText.slice(cursor, ent.start_char));
    const cls = ["PERSON", "ORG", "GPE", "LOC", "DATE", "TIME"].includes(ent.label)
      ? `ent-${ent.label}`
      : "ent-DEFAULT";
    html += `<mark class="ent-mark ${cls}" title="${escapeHtml(ent.description)}">${escapeHtml(
      cleanedText.slice(ent.start_char, ent.end_char)
    )}<span class="ent-tag">${escapeHtml(ent.label)}</span></mark>`;
    cursor = ent.end_char;
  }
  html += escapeHtml(cleanedText.slice(cursor));
  els.highlightedTranscript.innerHTML = html;
}

function renderSentimentChart(segments) {
  if (!segments || !segments.length) {
    els.sentimentChartContainer.innerHTML = '<div class="empty-state">No segments to plot.</div>';
    return;
  }

  const width = 540;
  const height = 220;
  const padL = 44;
  const padR = 20;
  const padT = 20;
  const padB = 34;
  const plotW = width - padL - padR;
  const plotH = height - padT - padB;

  const yFor = (pol) => padT + ((1 - pol) / 2) * plotH;
  const xFor = (idx) =>
    segments.length === 1 ? padL + plotW / 2 : padL + (idx / (segments.length - 1)) * plotW;

  const zeroY = yFor(0);
  const points = segments.map((s, idx) => ({
    x: xFor(idx),
    y: yFor(s.polarity),
    seg: s,
  }));

  const linePath = points.map((p, i) => `${i === 0 ? "M" : "L"} ${p.x.toFixed(1)} ${p.y.toFixed(1)}`).join(" ");
  const areaPath =
    `${linePath} L ${points[points.length - 1].x.toFixed(1)} ${zeroY.toFixed(1)} ` +
    `L ${points[0].x.toFixed(1)} ${zeroY.toFixed(1)} Z`;

  const dots = points
    .map((p) => {
      const color = p.seg.polarity >= 0.08 ? "#10b981" : p.seg.polarity <= -0.08 ? "#ef4444" : "#f59e0b";
      return `
        <g>
          <circle cx="${p.x.toFixed(1)}" cy="${p.y.toFixed(1)}" r="5.5" fill="${color}" stroke="#0b1020" stroke-width="2">
            <title>${escapeHtml(p.seg.label)}: polarity ${p.seg.polarity} (${p.seg.sentiment_label}, ${p.seg.dominant_emotion})</title>
          </circle>
          <text x="${p.x.toFixed(1)}" y="${height - 10}" text-anchor="middle" fill="#94a3c8" font-size="10">
            S${p.seg.segment_index}
          </text>
        </g>
      `;
    })
    .join("");

  els.sentimentChartContainer.innerHTML = `
    <svg viewBox="0 0 ${width} ${height}" width="100%" height="220" role="img" aria-label="Sentiment polarity chart">
      <line x1="${padL}" y1="${padT}" x2="${width - padR}" y2="${padT}" stroke="#283558" stroke-dasharray="3,3" />
      <line x1="${padL}" y1="${zeroY.toFixed(1)}" x2="${width - padR}" y2="${zeroY.toFixed(1)}" stroke="#475569" stroke-width="1.2" />
      <line x1="${padL}" y1="${height - padB}" x2="${width - padR}" y2="${height - padB}" stroke="#283558" stroke-dasharray="3,3" />
      <text x="${padL - 8}" y="${padT + 4}" text-anchor="end" fill="#6ee7b7" font-size="10">+1.0</text>
      <text x="${padL - 8}" y="${zeroY + 4}" text-anchor="end" fill="#94a3c8" font-size="10">0.0</text>
      <text x="${padL - 8}" y="${height - padB + 4}" text-anchor="end" fill="#fca5a5" font-size="10">-1.0</text>
      <path d="${areaPath}" fill="rgba(99, 102, 241, 0.16)" />
      <path d="${linePath}" fill="none" stroke="#818cf8" stroke-width="2.6" stroke-linecap="round" stroke-linejoin="round" />
      ${dots}
    </svg>
  `;
}

function renderTokenTable(tokens, filterText = "") {
  const q = filterText.trim().toLowerCase();
  const filtered = (tokens || []).filter(
    (t) =>
      !q ||
      t.text.toLowerCase().includes(q) ||
      t.lemma.toLowerCase().includes(q) ||
      t.pos.toLowerCase().includes(q)
  );

  if (!filtered.length) {
    els.tokenTableBody.innerHTML = '<tr><td colspan="7" class="empty-cell">No matching tokens.</td></tr>';
    return;
  }

  els.tokenTableBody.innerHTML = filtered
    .map(
      (t, idx) => `
      <tr>
        <td>${idx + 1}</td>
        <td><strong>${escapeHtml(t.text)}</strong></td>
        <td>${escapeHtml(t.lemma)}</td>
        <td>${escapeHtml(t.stem)}</td>
        <td><code>${escapeHtml(t.pos)}</code></td>
        <td>${escapeHtml(t.tag)}</td>
        <td>${t.is_stop ? "Yes" : "No"}</td>
      </tr>
    `
    )
    .join("");
}

function renderAnalysis(data) {
  state.lastAnalysis = data;
  state.activeEntityFilter = "ALL";
  els.kpiGrid.classList.remove("hidden");

  // KPIs
  const m = data.metrics;
  const s = data.sentiment.overall;
  els.kpiWords.textContent = `${m.word_count} / ${m.sentence_count}`;
  els.kpiReadTime.textContent = `~${m.estimated_reading_minutes} min read`;
  els.kpiTtr.textContent = `${(m.lexical_diversity_ttr * 100).toFixed(1)}%`;
  els.kpiUnique.textContent = `${m.unique_content_lemmas} unique lemmas`;
  els.kpiFlesch.textContent = m.flesch_reading_ease;
  els.kpiGrade.textContent = `FK Grade ${m.flesch_kincaid_grade}`;
  els.kpiSentiment.textContent = s.label;
  els.kpiPolarity.textContent = `Polarity: ${s.polarity >= 0 ? "+" : ""}${s.polarity}`;
  els.kpiEmotion.textContent = s.dominant_emotion.toUpperCase();
  els.kpiEngine.textContent = `Engine: ${s.engine}`;
  els.kpiEntities.textContent = data.linguistics.total_entities;
  els.kpiEntityTypes.textContent = `${data.linguistics.entities_by_type.length} entity types`;

  // Summary & Keywords
  if (data.summary && data.summary.length) {
    els.summaryList.innerHTML = data.summary
      .map(
        (item, idx) => `
        <div class="summary-item">
          <div class="summary-meta">Key Sentence #${idx + 1} • Salience ${item.salience_score}</div>
          <div>${escapeHtml(item.text)}</div>
        </div>
      `
      )
      .join("");
  } else {
    els.summaryList.innerHTML = '<div class="empty-state">Not enough sentences to summarize.</div>';
  }

  renderChips(els.keywordsChips, data.linguistics.tfidf_keywords, "keyword", "score");
  renderChips(els.nounPhrasesChips, data.linguistics.noun_phrases, "phrase", "count");
  renderMiniBars(els.topTokensList, data.preprocessing.top_tokens.slice(0, 10), "token", "count");
  renderMiniBars(els.bigramsList, data.preprocessing.bigrams.slice(0, 10), "ngram", "count");
  renderMiniBars(els.trigramsList, data.preprocessing.trigrams.slice(0, 10), "ngram", "count");

  // Linguistics & NER
  const types = ["ALL", ...data.linguistics.entities_by_type.map((g) => g.label)];
  els.entityFilterBar.innerHTML = types
    .map(
      (lbl) =>
        `<button type="button" class="chip chip-btn ${lbl === "ALL" ? "active" : ""}" data-ent="${escapeHtml(lbl)}">${escapeHtml(lbl)}</button>`
    )
    .join("");
  els.entityFilterBar.querySelectorAll(".chip-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      state.activeEntityFilter = btn.dataset.ent;
      els.entityFilterBar.querySelectorAll(".chip-btn").forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      renderHighlightedEntities(data.cleaned_text, data.linguistics.entities, state.activeEntityFilter);
    });
  });

  renderHighlightedEntities(data.cleaned_text, data.linguistics.entities, "ALL");
  renderMiniBars(els.posChart, data.linguistics.pos_distribution.slice(0, 12), "pos", "percentage", "%");

  if (data.linguistics.entities_by_type.length) {
    els.entitiesGrouped.innerHTML = data.linguistics.entities_by_type
      .map(
        (grp) => `
        <div class="summary-item">
          <div class="summary-meta"><strong>${escapeHtml(grp.label)}</strong> — ${escapeHtml(grp.description)} (${grp.total_mentions})</div>
          <div class="chip-wrap">
            ${grp.unique_entities
              .map((u) => `<span class="chip">${escapeHtml(u.text)} <span class="chip-score">${u.count}</span></span>`)
              .join("")}
          </div>
        </div>
      `
      )
      .join("");
  } else {
    els.entitiesGrouped.innerHTML = '<div class="empty-state">No named entities found.</div>';
  }

  // Sentiment & Emotions
  renderSentimentChart(data.sentiment.segments);
  const emotionItems = Object.entries(s.emotion_scores)
    .map(([emotion, score]) => ({
      emotion: emotion.charAt(0).toUpperCase() + emotion.slice(1),
      pct: Math.round(score * 100),
    }))
    .sort((a, b) => b.pct - a.pct);
  renderMiniBars(els.emotionBars, emotionItems, "emotion", "pct", "%");

  els.segmentsGrid.innerHTML = data.sentiment.segments
    .map(
      (seg) => `
      <div class="seg-card">
        <div class="seg-top">
          <span class="seg-title">${escapeHtml(seg.label)}</span>
          <div>
            <span class="badge badge-${escapeHtml(seg.sentiment_label)}">${escapeHtml(seg.sentiment_label)} (${seg.polarity})</span>
            <span class="badge badge-emotion">${escapeHtml(seg.dominant_emotion)}</span>
          </div>
        </div>
        <p class="seg-preview">${escapeHtml(seg.text_preview)}</p>
      </div>
    `
    )
    .join("");

  // Preprocessing Tab
  els.cleanedTextBox.textContent = data.cleaned_text;
  renderTokenTable(data.preprocessing.token_samples, els.tokenSearch.value);
}

async function runAnalysis() {
  const numSegments = Number(els.segmentCount.value) || 6;
  const payload = { num_segments: numSegments, summary_sentences: 3, max_token_samples: 120 };

  if (state.mode === "text") {
    const txt = els.transcriptText.value.trim();
    if (!txt) {
      els.statusMessage.textContent = "Please enter some transcript text or select a sample.";
      return;
    }
    payload.text = txt;
    if (state.selectedTimedItems) {
      payload.timed_items = state.selectedTimedItems;
    }
  } else {
    const vid = els.videoIdInput.value.trim();
    if (!vid) {
      els.statusMessage.textContent = "Please enter a YouTube Video ID or URL.";
      return;
    }
    payload.video_id = vid;
    els.ragVideoId.value = vid;
  }

  els.analyzeBtn.disabled = true;
  els.statusMessage.textContent = "Running NLP pipeline...";

  try {
    const res = await fetch("/nlp/analyze", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const data = await res.json();
    if (!res.ok) {
      throw new Error(data.detail || "Analysis failed.");
    }
    renderAnalysis(data);
    els.statusMessage.textContent = `Analysis complete (${data.source}).`;
  } catch (err) {
    els.statusMessage.textContent = `Error: ${err.message}`;
  } finally {
    els.analyzeBtn.disabled = false;
  }
}

function appendChatMessage(role, text) {
  const div = document.createElement("div");
  div.className = `chat-msg chat-${role}`;
  div.textContent = text;
  els.chatMessages.appendChild(div);
  els.chatMessages.scrollTop = els.chatMessages.scrollHeight;
}

async function handleAskSubmit(e) {
  e.preventDefault();
  const videoId = els.ragVideoId.value.trim() || els.videoIdInput.value.trim();
  const question = els.ragQuestion.value.trim();

  if (!videoId) {
    appendChatMessage("system", "Please enter a YouTube Video ID or URL first.");
    return;
  }
  if (!question) return;

  appendChatMessage("user", question);
  els.ragQuestion.value = "";
  els.askBtn.disabled = true;

  try {
    const res = await fetch("/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ video_id: videoId, question }),
    });
    const data = await res.json();
    if (!res.ok) {
      throw new Error(data.detail || "RAG request failed.");
    }
    appendChatMessage("bot", data.answer);
  } catch (err) {
    appendChatMessage("system", `Error: ${err.message}`);
  } finally {
    els.askBtn.disabled = false;
  }
}

// Event Listeners
els.modeTextBtn.addEventListener("click", () => switchMode("text"));
els.modeYoutubeBtn.addEventListener("click", () => switchMode("youtube"));
els.transcriptText.addEventListener("input", () => {
  state.selectedTimedItems = null;
});
els.sampleSelect.addEventListener("change", () => {
  const selected = state.samples.find((s) => s.id === els.sampleSelect.value);
  if (!selected) return;
  switchMode("text");
  els.transcriptText.value = selected.text;
  state.selectedTimedItems = selected.timed_items;
  els.statusMessage.textContent = `Loaded sample: ${selected.title}`;
});
els.tokenSearch.addEventListener("input", () => {
  if (state.lastAnalysis) {
    renderTokenTable(state.lastAnalysis.preprocessing.token_samples, els.tokenSearch.value);
  }
});
els.analyzeBtn.addEventListener("click", runAnalysis);
els.chatForm.addEventListener("submit", handleAskSubmit);

setupTabs();
loadHealthAndSamples();
