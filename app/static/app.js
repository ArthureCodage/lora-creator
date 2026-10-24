// LoRA Creator — interface
const $ = (s, el = document) => el.querySelector(s);
const $$ = (s, el = document) => [...el.querySelectorAll(s)];

const S = {
  types: {}, type: null, quality: "equilibre", dataset: null, images: [], models: [], model: null,
  estimate: null, defaults: null, triggerTouched: false, viewRun: null, job: null, pollTimer: null,
};
const QLABEL = { rapide: "Rapide", equilibre: "Équilibré", qualite: "Qualité" };
const FAMLABEL = { sd15: "SD 1.5", sdxl: "SDXL / Pony" };

const store = {
  get(k, d) { try { const v = localStorage.getItem("lc_" + k); return v === null ? d : JSON.parse(v); } catch { return d; } },
  set(k, v) { try { localStorage.setItem("lc_" + k, JSON.stringify(v)); } catch {} },
};

async function api(path, opts = {}) {
  if (opts.json !== undefined) {
    opts.body = JSON.stringify(opts.json);
    opts.headers = { "Content-Type": "application/json" };
    opts.method = opts.method || "POST";
  }
  const r = await fetch(path, opts);
  const data = await r.json().catch(() => null);
  if (!r.ok) throw new Error((data && data.detail) || `Erreur ${r.status}`);
  return data;
}

function toast(msg, bad = false) {
  const t = $("#toast");
  t.textContent = msg;
  t.className = "toast" + (bad ? " bad" : "");
  clearTimeout(t._h);
  t._h = setTimeout(() => t.classList.add("hidden"), bad ? 6000 : 3000);
}

const debounce = (fn, ms) => { let h; return (...a) => { clearTimeout(h); h = setTimeout(() => fn(...a), ms); }; };

function fmtDur(sec) {
  if (sec == null || isNaN(sec)) return "–";
  sec = Math.round(sec);
  const h = Math.floor(sec / 3600), m = Math.floor((sec % 3600) / 60), s = sec % 60;
  if (h) return `${h} h ${String(m).padStart(2, "0")}`;
  if (m) return `${m} min${m < 10 && s ? " " + String(s).padStart(2, "0") : ""}`;
  return `${s} s`;
}
const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

// ------------------------------------------------------------------ vues
function showView(v) {
  $$(".view").forEach((el) => el.classList.toggle("hidden", el.id !== "view-" + v));
  $$(".tab").forEach((b) => b.classList.toggle("active", b.dataset.view === v));
  if (v === "history") loadHistory();
  if (v === "train") renderTrainView();
  window.scrollTo(0, 0);
}
$$(".tab").forEach((b) => (b.onclick = () => showView(b.dataset.view)));
document.addEventListener("click", (e) => {
  const g = e.target.closest("[data-goto]");
  if (g) { e.preventDefault(); showView(g.dataset.goto); }
});

// ------------------------------------------------------------------ type (popup)
function openTypeModal() { $("#type-modal").classList.remove("hidden"); }
$$("#type-modal [data-type]").forEach((b) => (b.onclick = () => { setType(b.dataset.type); $("#type-modal").classList.add("hidden"); }));
$("#f-type").onclick = openTypeModal;
function setType(t) {
  S.type = t;
  $("#f-type").textContent = S.types[t]?.label || t;
  store.set("type", t);
  loadDefaults();
}

// ------------------------------------------------------------------ nom / mot-clé
function suggestTrigger(name) {
  return name.normalize("NFD").replace(/[̀-ͯ]/g, "").toLowerCase().replace(/[^a-z0-9]+/g, "").slice(0, 24);
}
$("#f-name").oninput = () => {
  if (!S.triggerTouched) $("#f-trigger").value = suggestTrigger($("#f-name").value);
  updateSummary();
};
$("#f-trigger").oninput = () => { S.triggerTouched = $("#f-trigger").value.trim() !== ""; };

// ------------------------------------------------------------------ datasets
async function loadDatasets(select) {
  const list = await api("/api/datasets");
  const sel = $("#f-dataset");
  sel.innerHTML = `<option value="">＋ Nouveau dataset</option>` +
    list.map((d) => `<option value="${esc(d.slug)}">${esc(d.name)} — ${d.n_images} images</option>`).join("");
  sel.value = select ?? S.dataset ?? "";
}
$("#f-dataset").onchange = () => { S.dataset = $("#f-dataset").value || null; loadDataset(); };

async function ensureDataset() {
  if (S.dataset) return S.dataset;
  const name = $("#f-name").value.trim() || "dataset";
  const r = await api("/api/datasets", { json: { name } });
  S.dataset = r.slug;
  await loadDatasets(r.slug);
  return r.slug;
}

async function loadDataset() {
  if (!S.dataset) { S.images = []; renderGrid(); refreshEstimate(); return; }
  const d = await api(`/api/datasets/${encodeURIComponent(S.dataset)}`);
  S.images = d.images;
  renderGrid();
  refreshEstimate();
}

function renderGrid() {
  const n = S.images.length;
  $("#ds-tools").classList.toggle("hidden", !n);
  $("#ds-tools2").classList.toggle("hidden", !n);
  const capped = S.images.filter((i) => i.caption).length;
  const small = S.images.filter((i) => i.small).length;
  $("#ds-count").innerHTML = `<b>${n}</b> images · ${capped} avec légende` + (small ? ` · <span style="color:var(--warn)">${small} petite(s)</span>` : "");
  const ds = encodeURIComponent(S.dataset || "");
  $("#grid").innerHTML = S.images.map((im) => `
    <div class="tile ${im.caption ? "" : "nocap"}" data-file="${esc(im.file)}">
      <img loading="lazy" src="/files/dataset/${ds}/${encodeURIComponent(im.file)}?thumb=1" title="${im.w}×${im.h}">
      ${im.small ? `<span class="warn">${im.w}×${im.h}</span>` : ""}
      <button class="del" title="Retirer l'image">✕</button>
      <textarea placeholder="légende (tags séparés par des virgules)">${esc(im.caption)}</textarea>
    </div>`).join("");
}

const pendingCaps = {};
const flushCaptions = debounce(async () => {
  const body = { ...pendingCaps };
  Object.keys(pendingCaps).forEach((k) => delete pendingCaps[k]);
  if (!Object.keys(body).length) return;
  await api(`/api/datasets/${encodeURIComponent(S.dataset)}/captions`, { method: "PUT", json: body });
  $("#save-state").textContent = "Légendes enregistrées ✓";
}, 600);

$("#grid").addEventListener("input", (e) => {
  if (e.target.tagName !== "TEXTAREA") return;
  const tile = e.target.closest(".tile");
  const f = tile.dataset.file;
  pendingCaps[f] = e.target.value;
  const im = S.images.find((i) => i.file === f);
  if (im) im.caption = e.target.value;
  tile.classList.toggle("nocap", !e.target.value.trim());
  $("#save-state").textContent = "Enregistrement…";
  flushCaptions();
});
$("#grid").addEventListener("click", async (e) => {
  const tile = e.target.closest(".tile");
  if (!tile) return;
  if (e.target.classList.contains("del")) {
    await api(`/api/datasets/${encodeURIComponent(S.dataset)}/images/${encodeURIComponent(tile.dataset.file)}`, { method: "DELETE" });
    S.images = S.images.filter((i) => i.file !== tile.dataset.file);
    renderGrid(); refreshEstimate();
  } else if (e.target.tagName === "IMG") {
    lightbox(`/files/dataset/${encodeURIComponent(S.dataset)}/${encodeURIComponent(tile.dataset.file)}`);
  }
});

async function uploadFiles(files) {
  files = [...files].filter((f) => f.type.startsWith("image/") || /\.(png|jpe?g|webp|bmp)$/i.test(f.name));
  if (!files.length) return toast("Aucune image reconnue.", true);
  const slug = await ensureDataset();
  let ok = 0, dup = 0;
  for (let i = 0; i < files.length; i += 8) {
    const fd = new FormData();
    files.slice(i, i + 8).forEach((f) => fd.append("files", f, f.name));
    toast(`Import… ${Math.min(i + 8, files.length)}/${files.length}`);
    const res = await api(`/api/datasets/${encodeURIComponent(slug)}/upload`, { method: "POST", body: fd });
    ok += res.filter((r) => r.status === "ok").length;
    dup += res.filter((r) => r.status === "doublon").length;
  }
  toast(`${ok} image(s) ajoutée(s)` + (dup ? `, ${dup} doublon(s) ignoré(s)` : ""));
  await loadDatasets(slug);
  await loadDataset();
}

async function readEntries(items) {
  // permet de glisser un dossier entier
  const files = [];
  const walk = (entry) => new Promise((res) => {
    if (entry.isFile) entry.file((f) => { files.push(f); res(); }, res);
    else if (entry.isDirectory) {
      const reader = entry.createReader();
      const all = [];
      const read = () => reader.readEntries(async (ents) => {
        if (!ents.length) { await Promise.all(all.map(walk)); res(); } else { all.push(...ents); read(); }
      }, res);
      read();
    } else res();
  });
  await Promise.all(items.map((i) => i.webkitGetAsEntry?.()).filter(Boolean).map(walk));
  return files;
}

const drop = $("#drop");
["dragenter", "dragover"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add("over"); }));
["dragleave", "drop"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.remove("over"); }));
drop.addEventListener("drop", async (e) => {
  const items = [...(e.dataTransfer.items || [])];
  const files = items.length && items[0].webkitGetAsEntry ? await readEntries(items) : e.dataTransfer.files;
  uploadFiles(files).catch((err) => toast(err.message, true));
});
$("#pick-files").onclick = (e) => { e.preventDefault(); $("#file-input").click(); };
$("#file-input").onchange = () => uploadFiles($("#file-input").files).catch((err) => toast(err.message, true));
$("#import-btn").onclick = async () => {
  const folder = $("#import-path").value.trim();
  if (!folder) return;
  try {
    const slug = await ensureDataset();
    toast("Import du dossier…");
    const r = await api(`/api/datasets/${encodeURIComponent(slug)}/import`, { json: { folder } });
    toast(`${r.added} image(s) importée(s)` + (r.duplicates ? `, ${r.duplicates} doublon(s)` : ""));
    $("#import-path").value = "";
    await loadDatasets(slug);
    await loadDataset();
  } catch (err) { toast(err.message, true); }
};

$("#autotag-btn").onclick = async () => {
  try {
    await flushCaptions();
    S.job = await api(`/api/datasets/${encodeURIComponent(S.dataset)}/autotag`, {
      json: { method: $("#tag-method").value, overwrite: $("#tag-overwrite").checked },
    });
    toast("Création des légendes lancée" + ($("#tag-method").value === "wd14" ? " (le 1er lancement télécharge le modèle, ~450 Mo)" : ""));
    startPolling();
  } catch (err) { toast(err.message, true); }
};
$("#rm-tag-btn").onclick = async () => {
  const tag = $("#rm-tag").value.trim();
  if (!tag) return;
  const r = await api(`/api/datasets/${encodeURIComponent(S.dataset)}/remove-tag`, { json: { tag } });
  toast(`« ${tag} » retiré de ${r.changed} légende(s)`);
  $("#rm-tag").value = "";
  loadDataset();
};

// ------------------------------------------------------------------ modèles + estimation
async function loadModels() {
  S.models = await api("/api/models");
  const last = store.get("model");
  if (!S.model) S.model = S.models.find((m) => m.path === last)?.path || S.models.find((m) => m.family === "sd15")?.path || S.models[0]?.path;
  renderModels();
}
const selFamily = () => S.models.find((m) => m.path === S.model)?.family || "sd15";

function renderModels() {
  if (!S.models.length) {
    $("#models").innerHTML = `<div class="muted">Aucun modèle SD 1.5 / SDXL trouvé dans les dossiers configurés (config.json).</div>`;
    return;
  }
  const est = S.estimate || {};
  $("#models").innerHTML = ["sd15", "sdxl"].map((fam) => {
    const ms = S.models.filter((m) => m.family === fam);
    if (!ms.length) return "";
    const e = est[fam];
    const eta = e && S.images.length ? `≈ ${fmtDur(e.seconds)}` : "";
    const cal = e && !e.calibrated ? `<small>estimation de départ, affinée après ton 1er entraînement ${FAMLABEL[fam]}</small>` : "";
    return `<div class="model-group">
      <h3>${FAMLABEL[fam]} <span class="eta">${eta}</span> ${e && S.images.length ? `<small>${e.steps} pas</small>` : ""} ${cal}</h3>
      <div class="models">${ms.map((m) => `
        <button class="model ${m.path === S.model ? "on" : ""}" data-path="${esc(m.path)}" title="${esc(m.path)}">
          <b>${esc(m.name)}</b>
          <span class="badge ${fam === "sdxl" ? "xl" : ""}">${FAMLABEL[fam]}</span>${m.pony ? '<span class="badge xl">Pony</span>' : ""}
          <span class="badge">${m.size_gb} Go</span><span class="badge">${esc(m.folder)}</span>
        </button>`).join("")}</div>
      ${fam === "sdxl" ? (S.profile === "big"
        ? `<div class="note ok">Gros GPU : SDXL en bf16, 2 images à la fois, images de test activées.</div>`
        : `<div class="note">SDXL sur 6 Go : ça marche (~4,5 s par pas), mais c'est lent. Images de test désactivées par défaut : environ 9 min chacune.</div>`) : ""}
    </div>`;
  }).join("");
}
$("#models").addEventListener("click", (e) => {
  const b = e.target.closest(".model");
  if (!b) return;
  const before = selFamily();
  S.model = b.dataset.path;
  store.set("model", S.model);
  renderModels();
  if (before !== selFamily()) { $$("[data-adv]").forEach((i) => (i.value = "")); }
  loadDefaults();
});

$$("#quality button").forEach((b) => (b.onclick = () => {
  S.quality = b.dataset.q;
  store.set("quality", S.quality);
  $$("#quality button").forEach((x) => x.classList.toggle("on", x === b));
  loadDefaults();
}));

function advValues() {
  const o = {};
  $$("[data-adv]").forEach((i) => {
    const v = i.value.trim();
    if (v === "") return;
    o[i.dataset.adv] = ["unet_lr", "te_lr"].includes(i.dataset.adv) ? parseFloat(v) : parseInt(v, 10);
  });
  o.samples = $("#opt-samples").checked;
  return o;
}
$$("[data-adv]").forEach((i) => i.addEventListener("input", () => refreshEstimate()));
$("#opt-samples").onchange = () => refreshEstimate();

async function loadDefaults() {
  if (!S.type) return;
  S.defaults = await api("/api/defaults", { json: { type: S.type, quality: S.quality, profile: S.profile } });
  const fam = selFamily(), d = S.defaults[fam];
  if (S.defFam !== fam) { $("#opt-samples").checked = d.samples; S.defFam = fam; }
  $$("[data-adv]").forEach((i) => {
    if (i.tagName === "SELECT") return;
    const v = d[i.dataset.adv];
    i.placeholder = i.dataset.adv === "repeats" ? "auto" : v ?? "";
  });
  const rs = $("#res-select"), cur = rs.value;
  rs.innerHTML = `<option value="">${d.resolution} (conseillé)</option>` +
    d.resolutions.filter((r) => r !== d.resolution).map((r) => `<option value="${r}">${r}</option>`).join("");
  rs.value = d.resolutions.includes(+cur) && +cur !== d.resolution ? cur : "";
  $("#te-field").classList.toggle("hidden", fam !== "sd15");
  refreshEstimate();
}

const refreshEstimate = debounce(async () => {
  if (!S.type) return;
  const adv = advValues();
  S.estimate = await api("/api/estimate", {
    json: { n_images: S.images.length, type: S.type, quality: S.quality, profile: S.profile, overrides: { [selFamily()]: adv } },
  });
  renderModels();
  updateSummary();
}, 250);

function updateSummary() {
  const fam = selFamily(), e = S.estimate?.[fam], n = S.images.length;
  const missing = [];
  if (!$("#f-name").value.trim()) missing.push("un nom");
  if (!n) missing.push("des images");
  if (!S.model) missing.push("un modèle");
  if (!e || !n) {
    $("#summary").innerHTML = missing.length ? `Il manque ${missing.join(", ")}.` : "";
  } else {
    const nocap = S.images.filter((i) => !i.caption.trim()).length;
    $("#summary").innerHTML = `<b>≈ ${fmtDur(e.seconds)}</b> · ${e.steps} pas · ${n} images × ${e.repeats} répét. × ${e.epochs} epochs` +
      (nocap ? ` · <span style="color:var(--warn)">${nocap} sans légende</span>` : "") +
      (missing.length ? ` · il manque ${missing.join(", ")}` : "");
  }
  const running = S.job && S.job.status === "running";
  $("#launch-btn").disabled = !!missing.length || running;
  $("#launch-btn").title = running ? "Un travail tourne déjà sur la carte graphique" : "";
}

// ------------------------------------------------------------------ lancement
$("#launch-btn").onclick = async () => {
  try {
    await flushCaptions();
    const job = await api("/api/train", {
      json: {
        dataset: S.dataset, name: $("#f-name").value.trim(), trigger: $("#f-trigger").value.trim(),
        type: S.type, quality: S.quality, model: S.model, profile: S.profile, advanced: advValues(),
        sample_prompts: $("#opt-prompts").value, auto_forge: $("#opt-forge").checked,
      },
    });
    S.job = job;
    S.viewRun = job.run_id;
    showView("train");
    startPolling();
  } catch (err) { toast(err.message, true); }
};

// ------------------------------------------------------------------ suivi du job
function startPolling() {
  clearInterval(S.pollTimer);
  S.pollTimer = setInterval(pollJob, 1500);
  pollJob();
}
let lastSampleCount = -1;
async function pollJob() {
  const prev = S.job;
  S.job = await api("/api/job").catch(() => S.job);
  const j = S.job;
  const running = j && j.status === "running";
  $("#tab-dot").classList.toggle("hidden", !(running && j.kind === "train"));
  updateSummary();

  // progression des légendes auto
  const tp = $("#tag-progress");
  if (j && j.kind === "tag" && running) {
    tp.classList.remove("hidden");
    $("i", tp).style.width = j.total ? `${(100 * j.step) / j.total}%` : "5%";
    $("span", tp).textContent = j.total ? `${j.phase} ${j.step}/${j.total}` : j.phase;
    $("#autotag-btn").disabled = true;
  } else {
    tp.classList.add("hidden");
    $("#autotag-btn").disabled = false;
  }
  if (prev && prev.status === "running" && j && j.status !== "running") {
    if (j.kind === "tag") {
      if (j.status === "finished") toast("Légendes créées ✓"); else toast(j.error || "Les légendes n'ont pas pu être créées.", true);
      loadDataset();
    } else {
      toast(j.status === "finished" ? "Entraînement terminé ✓" : j.status === "cancelled" ? "Entraînement arrêté" : "L'entraînement a échoué", j.status !== "finished");
      refreshEstimate();
    }
  }
  if (j && j.kind === "train" && (!S.viewRun || S.viewRun === j.run_id)) {
    S.viewRun = j.run_id;
    if (!$("#view-train").classList.contains("hidden")) renderJob(j);
  }
  if (!running) { clearInterval(S.pollTimer); S.pollTimer = null; }
}

async function renderTrainView() {
  const j = S.job;
  if (!S.viewRun && j && j.kind === "train") S.viewRun = j.run_id;
  if (!S.viewRun) { $("#train-empty").classList.remove("hidden"); $("#train-panel").classList.add("hidden"); return; }
  $("#train-empty").classList.add("hidden");
  $("#train-panel").classList.remove("hidden");
  lastSampleCount = -1;
  if (j && j.kind === "train" && j.run_id === S.viewRun) renderJob(j);
  else renderRunStatic(await api(`/api/runs/${encodeURIComponent(S.viewRun)}`));
}

let runCache = null;
async function renderJob(j) {
  if (!runCache || runCache.id !== j.run_id || j.status === "running" || runCache.status === "running") {
    runCache = await api(`/api/runs/${encodeURIComponent(j.run_id)}`).catch(() => runCache);
  }
  const r = runCache || {};
  const running = j.status === "running";
  $("#t-name").textContent = r.name || "";
  $("#t-sub").textContent = `${S.types[r.type]?.label || ""} · ${r.model_name || ""} · ${QLABEL[r.quality] || ""} · ${profShort(r)} · mot-clé « ${r.trigger || ""} » · estimé ${fmtDur(r.estimate)}`;
  const phase = { finished: "Terminé ✓", cancelled: "Arrêté", error: "Échec" }[j.status] || j.phase;
  $("#t-phase").textContent = phase;
  $("#t-bar").style.width = j.status === "finished" ? "100%" : j.total ? `${(100 * j.step) / j.total}%` : "2%";
  $("#t-step").textContent = j.total ? `${j.step} / ${j.total}` : "–";
  $("#t-epoch").textContent = j.epochs ? `${j.epoch} / ${j.epochs}` : "–";
  $("#t-elapsed").textContent = fmtDur(j.elapsed);
  $("#t-eta").textContent = running ? (j.eta && j.eta !== "?" ? j.eta : "–") : "–";
  $("#t-rate").textContent = j.rate ? j.rate.toFixed(2) : "–";
  $("#t-loss").textContent = j.loss != null ? j.loss.toFixed(4) : "–";
  $("#t-cancel").classList.toggle("hidden", !running);
  $("#t-error").classList.toggle("hidden", !j.error);
  $("#t-error").textContent = j.error || "";
  const log = $("#t-log");
  const atBottom = log.scrollTop + log.clientHeight >= log.scrollHeight - 20;
  log.textContent = j.log.join("\n");
  if (atBottom) log.scrollTop = log.scrollHeight;
  renderFiles(r, running);
}

function renderRunStatic(r) {
  runCache = r;
  $("#t-name").textContent = r.name;
  $("#t-sub").textContent = `${S.types[r.type]?.label || ""} · ${r.model_name} · ${QLABEL[r.quality]} · ${profShort(r)} · mot-clé « ${r.trigger} » · ${r.started.replace("T", " ")}`;
  $("#t-phase").textContent = { finished: "Terminé ✓", cancelled: "Arrêté", error: "Échec", interrupted: "Interrompu" }[r.status] || r.status;
  $("#t-bar").style.width = r.status === "finished" ? "100%" : "0";
  $("#t-step").textContent = r.params.steps;
  $("#t-epoch").textContent = r.params.epochs;
  $("#t-elapsed").textContent = fmtDur(r.duration);
  $("#t-eta").textContent = "–";
  $("#t-rate").textContent = r.measured ? r.measured.spi.toFixed(2) : "–";
  $("#t-loss").textContent = "–";
  $("#t-cancel").classList.add("hidden");
  $("#t-error").classList.toggle("hidden", !r.error);
  $("#t-error").textContent = r.error || "";
  $("#t-log").textContent = "(journal complet : train.log dans le dossier de l'entraînement)";
  lastSampleCount = -1;
  renderFiles(r, false);
}

function renderFiles(r, running) {
  if (!r || !r.id) return;
  const samples = r.samples || [];
  if (samples.length !== lastSampleCount) {
    lastSampleCount = samples.length;
    $("#t-samples").innerHTML = samples.length ? samples.slice().reverse().map((s) => {
      const ep = (s.match(/_e(\d+)_/) || [])[1];
      return `<figure><img loading="lazy" src="/files/run/${encodeURIComponent(r.id)}/sample/${encodeURIComponent(s)}?thumb=1" data-full="/files/run/${encodeURIComponent(r.id)}/sample/${encodeURIComponent(s)}">
        <figcaption>${ep ? "epoch " + parseInt(ep, 10) : esc(s)}</figcaption></figure>`;
    }).join("") : `<span class="muted">${running ? "Elles apparaîtront à chaque sauvegarde." : "Aucune image de test."}</span>`;
  }
  const loras = r.loras || [];
  $("#t-files").innerHTML = loras.length ? loras.map((f) => {
    const final = f === `${r.name}.safetensors`;
    const ep = (f.match(/-(\d+)\.safetensors$/) || [])[1];
    return `<div class="file-row"><code>${esc(f)}</code>
      <span class="muted">${final ? "version finale" : "epoch " + parseInt(ep, 10)}</span>
      ${r.forge === f && final ? '<span class="status finished">dans Forge</span>' : ""}
      <button class="btn" data-forge="${esc(f)}">Copier vers Forge</button></div>`;
  }).join("") : `<span class="muted">${running ? "Les fichiers apparaîtront à chaque sauvegarde (plusieurs epochs pour pouvoir choisir le meilleur)." : "Aucun fichier."}</span>`;
}
$("#t-files").addEventListener("click", async (e) => {
  const b = e.target.closest("[data-forge]");
  if (!b) return;
  try {
    const r = await api(`/api/runs/${encodeURIComponent(runCache.id)}/forge`, { json: { file: b.dataset.forge } });
    toast(`Copié dans Forge : ${r.file}`);
  } catch (err) { toast(err.message, true); }
});
$("#t-samples").addEventListener("click", (e) => { if (e.target.dataset.full) lightbox(e.target.dataset.full); });
$("#t-cancel").onclick = async () => {
  if (!confirm("Arrêter l'entraînement ? Les fichiers déjà sauvegardés sont gardés.")) return;
  await api("/api/job/cancel", { method: "POST" });
};
// sur une machine sans écran (Linux distant), le dossier ne peut pas s'ouvrir : on affiche son chemin
const openFolder = (body) => api("/api/open-folder", { json: body })
  .then((r) => { if (!r.opened && r.path) toast("Dossier : " + r.path); }).catch((e) => toast(e.message, true));
$("#t-folder").onclick = () => runCache && openFolder({ run: runCache.id });

// ------------------------------------------------------------------ historique
async function loadHistory() {
  const runs = await api("/api/runs");
  if (!runs.length) { $("#history").innerHTML = `<div class="card empty">Pas encore d'entraînement.</div>`; return; }
  $("#history").innerHTML = runs.map((r) => {
    return `<div class="card hist" data-run="${esc(r.id)}">
      <img class="ph" data-thumb="${esc(r.id)}" alt="">
      <div>
        <h3>${esc(r.name)} <span class="status ${r.status}">${{ finished: "terminé", running: "en cours", error: "échec", cancelled: "arrêté", interrupted: "interrompu" }[r.status] || r.status}</span></h3>
        <div class="muted">${S.types[r.type]?.label || r.type} · ${esc(r.model_name)} · ${QLABEL[r.quality]} · ${profShort(r)} · « ${esc(r.trigger)} »</div>
        <div class="muted">${r.started.slice(0, 16).replace("T", " à ")} · ${r.n_images} images · ${r.params.steps} pas · ${r.duration ? "durée " + fmtDur(r.duration) : "estimé " + fmtDur(r.estimate)}${r.forge ? " · dans Forge" : ""}</div>
      </div>
      <div class="acts">
        <button class="btn" data-act="view">Voir</button>
        <button class="btn primary" data-act="again">Relancer</button>
        <button class="btn ghost" data-act="folder">Dossier</button>
      </div>
    </div>`;
  }).join("");
  // vignettes : dernière image de test
  for (const r of runs) {
    api(`/api/runs/${encodeURIComponent(r.id)}`).then((d) => {
      const img = $(`[data-thumb="${CSS.escape(r.id)}"]`);
      if (img && d.samples?.length) img.src = `/files/run/${encodeURIComponent(r.id)}/sample/${encodeURIComponent(d.samples.at(-1))}?thumb=1`;
    }).catch(() => {});
  }
}
$("#history").addEventListener("click", async (e) => {
  const b = e.target.closest("[data-act]");
  if (!b) return;
  const id = b.closest("[data-run]").dataset.run;
  if (b.dataset.act === "view") { S.viewRun = id; showView("train"); }
  if (b.dataset.act === "folder") openFolder({ run: id });
  if (b.dataset.act === "again") reuseRun(await api(`/api/runs/${encodeURIComponent(id)}`));
});
$$("[data-open]").forEach((b) => (b.onclick = () => openFolder({ which: b.dataset.open })));

async function reuseRun(r) {
  $("#f-name").value = r.name;
  $("#f-trigger").value = r.trigger;
  S.triggerTouched = true;
  setType(r.type);
  S.quality = r.quality;
  $$("#quality button").forEach((x) => x.classList.toggle("on", x.dataset.q === r.quality));
  S.model = r.model;
  S.dataset = r.dataset;
  await loadDatasets(r.dataset);
  await loadDataset();
  renderModels();
  await loadDefaults();
  $$("[data-adv]").forEach((i) => (i.value = r.advanced?.[i.dataset.adv] ?? ""));
  $("#opt-samples").checked = r.params.samples !== false;
  $("#opt-prompts").value = r.sample_prompts || "";
  $("#opt-forge").checked = r.auto_forge !== false;
  if (Object.keys(r.advanced || {}).some((k) => k !== "samples")) $("#advanced").open = true;
  showView("new");
  refreshEstimate();
  toast("Réglages repris — modifie ce que tu veux puis lance.");
}

// ------------------------------------------------------------------ divers
function lightbox(src) {
  const lb = $("#lightbox");
  $("img", lb).src = src;
  lb.classList.remove("hidden");
}
$("#lightbox").onclick = () => $("#lightbox").classList.add("hidden");
document.addEventListener("keydown", (e) => { if (e.key === "Escape") $("#lightbox").classList.add("hidden"); });

// ------------------------------------------------------------------ démarrage
// ------------------------------------------------------------------ carte graphique (popup au lancement)
function openGpuModal(first) {
  const info = S.info, g = info.gpus || [];
  $("#gpu-detected").textContent = g.length
    ? "Carte détectée : " + g.map((x) => `${x.name} (${x.vram_gb} Go)`).join(", ")
    : "Aucune carte NVIDIA détectée (nvidia-smi introuvable).";
  $("#gpu-cards").innerHTML = Object.entries(info.profiles).map(([k, p]) => `
    <button data-profile="${k}" class="${k === S.profile ? "on" : ""}">
      <span class="ico">${k === "big" ? "⚡" : "◐"}</span><b>${esc(p.label)}</b><small>${esc(p.desc)}</small>
      ${k === info.recommended ? '<span class="badge xl">conseillé pour cette carte</span>' : ""}
      ${k === "big" && g.length && Math.max(...g.map((x) => x.vram_gb)) < 16
        ? `<span class="badge warn">trop juste pour ${Math.max(...g.map((x) => x.vram_gb))} Go : très lent</span>` : ""}
    </button>`).join("");
  S.gpuFirst = first;
  $("#gpu-modal").classList.remove("hidden");
}
$("#gpu-cards").addEventListener("click", async (e) => {
  const b = e.target.closest("[data-profile]");
  if (!b) return;
  await api("/api/profile", { json: { profile: b.dataset.profile } });
  setProfile(b.dataset.profile);
  $("#gpu-modal").classList.add("hidden");
  if (S.gpuFirst) { S.gpuFirst = false; afterGpu(); }
});
$("#gpu-chip").onclick = () => openGpuModal(false);

function setProfile(p) {
  const changed = S.profile && S.profile !== p;
  S.profile = p;
  $("#gpu-chip").textContent = "GPU : " + S.info.profiles[p].short;
  if (changed) $$("[data-adv]").forEach((i) => (i.value = "")); // les réglages avancés dépendent de la carte
  S.defFam = null;
  loadDefaults();
}
const profShort = (r) => S.info?.profiles[r.profile || "small"]?.short || "";

function afterGpu() {
  $("#f-type").textContent = "choisir…";
  openTypeModal();
  updateSummary();
}

// ------------------------------------------------------------------ démarrage
(async function init() {
  const info = await api("/api/info");
  S.info = info;
  S.types = info.types;
  S.quality = store.get("quality", "equilibre");
  $$("#quality button").forEach((x) => x.classList.toggle("on", x.dataset.q === S.quality));
  await Promise.all([loadDatasets(""), loadModels()]);
  if (info.profile) setProfile(info.profile);
  S.job = await api("/api/job");
  if (S.job && S.job.status === "running") {
    startPolling();
    if (S.job.kind === "train") { S.viewRun = S.job.run_id; setType(store.get("type", "personnage")); showView("train"); return; }
  }
  if (!info.profile) openGpuModal(true); // 1er affichage depuis le lancement : on demande la carte
  else afterGpu();
})();
