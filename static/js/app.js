"use strict";

// ── Navbar scroll shadow ──
window.addEventListener("scroll", () => {
  document.getElementById("navbar")?.classList.toggle("scrolled", window.scrollY > 10);
});

// ── Tab switching ──
const navBtns = document.querySelectorAll(".nav-btn");
const tabPanes = document.querySelectorAll(".tab-pane");

function switchTab(name) {
  navBtns.forEach(b => b.classList.toggle("active", b.dataset.tab === name));
  tabPanes.forEach(p => p.classList.toggle("active", p.dataset.tabContent === name));
  const target = document.getElementById(`section-${name}`);
  if (target) {
    const top = target.getBoundingClientRect().top + window.scrollY - 70;
    window.scrollTo({ top, behavior: "smooth" });
  }
}
window.switchTab = switchTab;

navBtns.forEach(b => b.addEventListener("click", () => switchTab(b.dataset.tab)));

// ── Form & Prediction ──
const form = document.getElementById("predict-form");
const btnPredict = document.getElementById("btn-predict");

if (form) form.addEventListener("submit", async e => {
  e.preventDefault();
  if (!validateForm()) return;
  await runPrediction();
});

function validateForm() {
  let ok = true;
  let firstInvalid = null;
  form.querySelectorAll("[required]").forEach(el => {
    const valid = el.checkValidity();
    el.style.borderColor = valid ? "" : "var(--red)";
    if (!valid) {
      ok = false;
      if (!firstInvalid) firstInvalid = el;
    }
  });
  if (!ok) {
    form.style.animation = "none";
    requestAnimationFrame(() => form.style.animation = "shake .35s ease");
    if (firstInvalid) firstInvalid.reportValidity();
  }
  return ok;
}

document.querySelectorAll(".fi-inp,.fi-sel").forEach(el =>
  el.addEventListener("input", () => el.style.borderColor = "")
);

const shakeStyle = document.createElement("style");
shakeStyle.textContent = `@keyframes shake{0%,100%{transform:translateX(0)}25%{transform:translateX(-6px)}75%{transform:translateX(6px)}}`;
document.head.appendChild(shakeStyle);

function collectData() {
  return {
    Map: document.getElementById("input-map").value,
    AverageRank: document.getElementById("input-rank").value,
    EconRating: parseFloat(document.getElementById("input-econ").value),
    KD_Ratio: parseFloat(document.getElementById("input-kd").value),
    Kills: parseInt(document.getElementById("input-kills").value),
    Deaths: parseInt(document.getElementById("input-deaths").value),
    FirstBloods: parseInt(document.getElementById("input-fb").value),
    SpikePlants: parseInt(document.getElementById("input-spike").value),
  };
}

async function runPrediction() {
  setLoading(true);
  hideAll();
  try {
    const res = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(collectData()),
    });
    const data = await res.json();
    if (!res.ok || data.error) { showError(data.error || `Error ${res.status}`); return; }
    showResult(data);
  } catch {
    showError("Tidak dapat terhubung ke server Flask.");
  } finally {
    setLoading(false);
  }
}

function setLoading(on) {
  btnPredict.disabled = on;
  btnPredict.querySelector(".btn-text").style.display = on ? "none" : "flex";
  btnPredict.querySelector(".btn-loading").style.display = on ? "flex" : "none";
}

function showResult(data) {
  const { label, probability, confidence, input_stats, feature_stats, feature_importance } = data;
  const isWin = label === "VICTORY";

  const verdict = document.getElementById("rc-verdict");
  verdict.textContent = label;
  verdict.className = `rc-verdict ${isWin ? "victory" : "defeat"}`;

  document.getElementById("rc-sublabel").textContent = isWin
    ? "Tim diprediksi MENANG pada pertandingan ini"
    : "Tim diprediksi KALAH pada pertandingan ini";

  const fill = document.getElementById("rcp-fill");
  const pct = document.getElementById("rcp-pct");
  fill.style.width = "0%";
  fill.className = `rcp-fill ${isWin ? "victory" : "defeat"}`;
  requestAnimationFrame(() => requestAnimationFrame(() => {
    fill.style.width = `${probability}%`;
    pct.textContent = `${probability}%`;
  }));

  const badge = document.getElementById("rcr-badge");
  badge.textContent = confidence;
  badge.className = `rcr-badge ${confidence}`;

  // ── Tactical Analysis ──
  renderTactical(input_stats, feature_stats, isWin);

  // ── Feature Impact Bars ──
  renderFeatureImpact(feature_importance);

  // ── What-If Simulator init ──
  initWhatIf(input_stats);

  // ── Save to Prediction History ──
  saveToHistory({
    map: document.getElementById("input-map").value,
    rank: document.getElementById("input-rank").value,
    kd: (input_stats?.KD_Ratio || 0).toFixed(2),
    label, probability
  });

  document.getElementById("result-content").style.display = "flex";
}

// ── Tactical Analysis Renderer ──
function renderTactical(stats, avgStats, isWin) {
  const body = document.getElementById("tactical-body");
  if (!body || !stats) return;
  body.innerHTML = "";

  const tips = [];
  const kd = stats.KD_Ratio || 0;
  const fb = stats.FirstBloods || 0;
  const sp = stats.SpikePlants || 0;
  const econ = stats.EconRating || 0;
  const avgKd = avgStats?.KD_Ratio?.mean || 1.03;
  const avgFb = avgStats?.FirstBloods?.mean || 12;
  const avgSp = avgStats?.SpikePlants?.mean || 7;
  const avgEcon = avgStats?.EconRating?.mean || 55;

  // K/D Ratio analysis
  if (kd >= 1.5) {
    tips.push({ type: "positive", icon: "▲", text: `<strong>K/D Ratio ${kd.toFixed(2)}</strong> sangat dominan (avg: ${avgKd.toFixed(2)}). Fragging power tim sangat tinggi — pertahankan agresivitas ini.` });
  } else if (kd < 1.0) {
    tips.push({ type: "negative", icon: "▼", text: `<strong>K/D Ratio ${kd.toFixed(2)}</strong> di bawah rata-rata (avg: ${avgKd.toFixed(2)}). Fokus pada crosshair placement dan trade kills untuk meningkatkan survival rate.` });
  } else {
    tips.push({ type: "neutral", icon: "●", text: `<strong>K/D Ratio ${kd.toFixed(2)}</strong> berada di zona kompetitif (avg: ${avgKd.toFixed(2)}). Tingkatkan ke >1.5 untuk dominasi yang lebih konsisten.` });
  }

  // First Bloods analysis
  if (fb >= avgFb * 1.3) {
    tips.push({ type: "positive", icon: "▲", text: `<strong>First Bloods ${fb}</strong> sangat tinggi. Opening duel tim sangat efektif — ini memberi keuntungan besar di setiap round.` });
  } else if (fb < avgFb * 0.7) {
    tips.push({ type: "negative", icon: "▼", text: `<strong>First Bloods ${fb}</strong> rendah (avg: ${Math.round(avgFb)}). Gunakan utility untuk mengamankan opening pick — agent Duelist perlu lebih agresif.` });
  }

  // Spike Plants analysis
  if (sp >= avgSp * 1.2) {
    tips.push({ type: "positive", icon: "▲", text: `<strong>Spike Plants ${sp}</strong> menunjukkan kontrol site yang konsisten. Tim memiliki eksekusi strategi yang solid.` });
  } else if (sp < avgSp * 0.6) {
    tips.push({ type: "negative", icon: "▼", text: `<strong>Spike Plants ${sp}</strong> sangat minim (avg: ${Math.round(avgSp)}). Prioritaskan planting bahkan dalam situasi tertekan — post-plant advantage sangat krusial.` });
  }

  // Econ Rating analysis
  if (econ >= 70) {
    tips.push({ type: "positive", icon: "▲", text: `<strong>Econ Rating ${econ}</strong> menunjukkan manajemen ekonomi yang sangat efisien. Tim memanfaatkan buy rounds dengan optimal.` });
  } else if (econ < 40) {
    tips.push({ type: "negative", icon: "▼", text: `<strong>Econ Rating ${econ}</strong> terlalu rendah (avg: ${Math.round(avgEcon)}). Koordinasikan full buy/eco rounds dengan lebih disiplin.` });
  }

  // Overall verdict tip
  if (isWin) {
    tips.push({ type: "positive", icon: "★", text: "Secara keseluruhan, statistik tim cukup kuat untuk meraih kemenangan. Pertahankan konsistensi dan momentum." });
  } else {
    tips.push({ type: "negative", icon: "!", text: "Model memprediksi kekalahan berdasarkan kombinasi statistik saat ini. Perbaiki area merah di atas untuk meningkatkan peluang menang." });
  }

  tips.forEach((t, i) => {
    const el = document.createElement("div");
    el.className = `ta-item ${t.type}`;
    el.style.animationDelay = `${i * 0.08}s`;
    el.innerHTML = `<div class="ta-icon ${t.type}">${t.icon}</div><div class="ta-text">${t.text}</div>`;
    body.appendChild(el);
  });
}

// ── Feature Impact Renderer ──
function renderFeatureImpact(importance) {
  const body = document.getElementById("impact-body");
  if (!body || !importance || !importance.length) return;
  body.innerHTML = "";

  const colorClasses = ["c1","c2","c3","c4","c5","c6"];
  const nameMap = {
    KD_Ratio: "K/D RATIO", FirstBloods: "1ST BLOOD", SpikePlants: "PLANTS",
    EconRating: "ECON", Kills: "KILLS", Deaths: "DEATHS",
    Map_enc: "MAP", Rank_enc: "RANK"
  };
  const maxImp = importance[0]?.importance || 1;
  const top6 = importance.slice(0, 6);

  top6.forEach((f, i) => {
    const pctVal = (f.importance / maxImp * 100).toFixed(1);
    const rawPct = (f.importance * 100).toFixed(1);
    const row = document.createElement("div");
    row.className = "fi-bar-row";
    row.innerHTML = `
      <span class="fi-bar-name">${nameMap[f.feature] || f.feature}</span>
      <div class="fi-bar-track"><div class="fi-bar-fill ${colorClasses[i % 6]}" style="width:0%"></div></div>
      <span class="fi-bar-pct">${rawPct}%</span>`;
    body.appendChild(row);
    // Animate fill
    requestAnimationFrame(() => requestAnimationFrame(() => {
      row.querySelector(".fi-bar-fill").style.width = `${pctVal}%`;
    }));
  });
}

// ── What-If Simulator ──
let _wifData = null;
let _wifTimer = null;

function initWhatIf(inputStats) {
  _wifData = { ...collectData() }; // snapshot current form state
  const kdSlider = document.getElementById("wif-kd");
  const fbSlider = document.getElementById("wif-fb");
  const kdVal = document.getElementById("wif-kd-val");
  const fbVal = document.getElementById("wif-fb-val");
  const pctEl = document.getElementById("wif-pct");

  if (!kdSlider || !fbSlider) return;

  kdSlider.value = inputStats.KD_Ratio || 1.03;
  fbSlider.value = inputStats.FirstBloods || 12;
  kdVal.textContent = parseFloat(kdSlider.value).toFixed(2);
  fbVal.textContent = fbSlider.value;
  pctEl.textContent = document.getElementById("rcp-pct")?.textContent || "-";

  const handler = () => {
    kdVal.textContent = parseFloat(kdSlider.value).toFixed(2);
    fbVal.textContent = fbSlider.value;
    clearTimeout(_wifTimer);
    _wifTimer = setTimeout(() => runWhatIf(kdSlider.value, fbSlider.value), 250);
  };
  kdSlider.oninput = handler;
  fbSlider.oninput = handler;
}

async function runWhatIf(kd, fb) {
  if (!_wifData) return;
  const payload = { ..._wifData, KD_Ratio: parseFloat(kd), FirstBloods: parseInt(fb) };
  try {
    const res = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const d = await res.json();
    if (d.probability !== undefined) {
      const el = document.getElementById("wif-pct");
      el.textContent = `${d.probability}%`;
      el.style.color = d.probability >= 50 ? "#22c55e" : "#ff4655";
    }
  } catch { /* silent */ }
}

function showError(msg) {
  document.getElementById("err-msg").textContent = msg;
  document.getElementById("result-error").style.display = "flex";
}

function hideAll() {
  document.getElementById("result-idle").style.display = "none";
  document.getElementById("result-content").style.display = "none";
  document.getElementById("result-error").style.display = "none";
}

// ── Map Coordinates Lookup ──
const mapCoords = {
  ascent: "45°26'15\" N, 12°20'09\" E",
  bind: "34°02'00\" N, 6°51'00\" W",
  breeze: "25°00'00\" N, 71°00'00\" W",
  haven: "27°29'00\" N, 89°38'00\" E",
  icebox: "76°44'00\" N, 149°30'00\" E",
  lotus: "14°00'00\" N, 74°00'00\" E",
  pearl: "38°43'00\" N, 09°08'00\" W",
  split: "35°41'22\" N, 139°41'30\" E",
  sunset: "34°03'00\" N, 118°15'00\" W"
};

function updateMapPreview(mapName) {
  const previewDiv = document.getElementById("result-map-preview");
  const resultCard = document.getElementById("result-card");
  const imgEl = document.getElementById("rmp-img");
  const nameEl = document.getElementById("rmp-name");
  const coordsEl = document.getElementById("rmp-coords");
  
  if (!mapName) {
    if (previewDiv) previewDiv.style.display = "none";
    if (resultCard) resultCard.classList.remove("has-preview");
    return;
  }
  
  const slug = mapName.toLowerCase();
  if (imgEl) imgEl.src = `/static/img/valomap/${slug}.png`;
  if (nameEl) nameEl.textContent = mapName.toUpperCase();
  if (coordsEl) coordsEl.textContent = mapCoords[slug] || "0.0° N, 0.0° E";
  
  if (previewDiv) previewDiv.style.display = "flex";
  if (resultCard) resultCard.classList.add("has-preview");
}

const mapSelect = document.getElementById("input-map");
if (mapSelect) {
  mapSelect.addEventListener("change", () => {
    updateMapPreview(mapSelect.value);
  });
}

// ── Rank Icons Lookup (Online Embeds) ──
const rankIcons = {
  silver: "https://wiki.playvalorant.com/en-us/images/Silver_1_Rank.png?ca291",
  gold: "https://wiki.playvalorant.com/en-us/images/Gold_1_Rank.png?170a9",
  platinum: "https://wiki.playvalorant.com/en-us/images/Platinum_1_Rank.png?46430",
  diamond: "https://wiki.playvalorant.com/en-us/images/Diamond_1_Rank.png?cd057",
  ascendant: "https://wiki.playvalorant.com/en-us/images/Ascendant_1_Rank.png?c818e",
  immortal: "https://wiki.playvalorant.com/en-us/images/Immortal_1_Rank.png?d43a7"
};

function updateRankPreview(rankName) {
  const formIcon = document.getElementById("form-rank-icon");
  const badge = document.getElementById("rmp-rank-badge");
  const badgeIcon = document.getElementById("rmp-rank-icon");
  
  if (!rankName) {
    if (formIcon) formIcon.style.display = "none";
    if (badge) badge.style.display = "none";
    return;
  }
  
  const slug = rankName.toLowerCase();
  const url = rankIcons[slug];
  
  if (url) {
    if (formIcon) {
      formIcon.src = url;
      formIcon.style.display = "block";
    }
    if (badgeIcon) {
      badgeIcon.src = url;
    }
    if (badge) {
      badge.style.display = "flex";
    }
  } else {
    if (formIcon) formIcon.style.display = "none";
    if (badge) badge.style.display = "none";
  }
}

const rankSelect = document.getElementById("input-rank");
if (rankSelect) {
  rankSelect.addEventListener("change", () => {
    updateRankPreview(rankSelect.value);
  });
}

function resetResult() {
  document.getElementById("result-idle").style.display = "flex";
  document.getElementById("result-content").style.display = "none";
  document.getElementById("result-error").style.display = "none";
  document.getElementById("rcp-fill").style.width = "0%";
  document.getElementById("rcp-pct").textContent = "0%";
  
  const previewDiv = document.getElementById("result-map-preview");
  if (previewDiv) previewDiv.style.display = "none";
  const resultCard = document.getElementById("result-card");
  if (resultCard) resultCard.classList.remove("has-preview");
  
  const formIcon = document.getElementById("form-rank-icon");
  if (formIcon) formIcon.style.display = "none";
  const badge = document.getElementById("rmp-rank-badge");
  if (badge) badge.style.display = "none";

  form.reset();
  form.querySelectorAll(".fi-inp,.fi-sel").forEach(el => el.style.borderColor = "");
}
window.resetResult = resetResult;

// ── KPI Animated Counters ──
let _kpiAnimated = false;
function animateCounters() {
  if (_kpiAnimated) return;
  _kpiAnimated = true;
  document.querySelectorAll(".kpi-val[data-count]").forEach(el => {
    const target = parseFloat(el.dataset.count);
    const suffix = el.dataset.suffix || "";
    const decimals = parseInt(el.dataset.decimals || "0");
    const duration = 1600;
    const start = performance.now();
    function tick(now) {
      const elapsed = now - start;
      const progress = Math.min(elapsed / duration, 1);
      // easeOutQuad
      const ease = 1 - (1 - progress) * (1 - progress);
      const current = target * ease;
      el.textContent = (decimals > 0 ? current.toFixed(decimals) : Math.round(current).toLocaleString("id-ID")) + suffix;
      if (progress < 1) requestAnimationFrame(tick);
    }
    requestAnimationFrame(tick);
  });
}

const kpiGrid = document.getElementById("kpi-grid");
if (kpiGrid) {
  new IntersectionObserver(entries => {
    entries.forEach(e => { if (e.isIntersecting) animateCounters(); });
  }, { threshold: 0.2 }).observe(kpiGrid);
}

// ── Prediction History (LocalStorage) ──
const HISTORY_KEY = "valo_pred_history";
const MAX_HISTORY = 5;

function loadHistory() {
  try { return JSON.parse(localStorage.getItem(HISTORY_KEY)) || []; }
  catch { return []; }
}

function saveToHistory(entry) {
  const list = loadHistory();
  list.unshift(entry);
  if (list.length > MAX_HISTORY) list.length = MAX_HISTORY;
  localStorage.setItem(HISTORY_KEY, JSON.stringify(list));
  renderHistory();
}

function renderHistory() {
  const tbody = document.getElementById("history-tbody");
  const card = document.getElementById("history-card");
  if (!tbody || !card) return;
  const list = loadHistory();
  if (list.length === 0) { card.style.display = "none"; return; }
  card.style.display = "block";
  tbody.innerHTML = "";
  list.forEach(h => {
    const tr = document.createElement("tr");
    const cls = h.label === "VICTORY" ? "ht-victory" : "ht-defeat";
    tr.innerHTML = `
      <td>${h.map}</td>
      <td>${h.rank}</td>
      <td>${h.kd}</td>
      <td class="${cls}">${h.label}</td>
      <td class="${cls}">${h.probability}%</td>`;
    tbody.appendChild(tr);
  });
}

// ── Scroll-triggered feature bar animation ──
new IntersectionObserver(entries => {
  entries.forEach(e => {
    if (e.isIntersecting) {
      e.target.querySelectorAll(".fi-fill").forEach(b => b.style.animationPlayState = "running");
    }
  });
}, { threshold: 0.2 }).observe(document.querySelector(".fi-list") || document.body);

document.querySelectorAll(".fi-fill").forEach(b => b.style.animationPlayState = "paused");

// ── Init ──
document.addEventListener("DOMContentLoaded", () => {
  switchTab("dashboard");
  renderHistory();
});
