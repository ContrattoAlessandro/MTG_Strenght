"use strict";

// ------------------------------------------------------------------ helpers
const $ = (sel, el = document) => el.querySelector(sel);
const $$ = (sel, el = document) => [...el.querySelectorAll(sel)];
const view = $("#view");
const cache = new Map();
let META = null;

async function api(path, opts) {
  if (!opts && cache.has(path)) return cache.get(path);
  const p = fetch(path, opts).then(async (r) => {
    if (!r.ok) throw new Error((await r.json().catch(() => ({}))).detail || r.statusText);
    return r.json();
  });
  if (!opts) {
    cache.set(path, p);
    p.catch(() => cache.delete(path));
  }
  return p;
}

const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const fmt = (x, d = 1) => (x == null || Number.isNaN(x) ? "–" : Number(x).toFixed(d));
const signed = (x, d = 1) => (x == null ? "–" : (x >= 0 ? "+" : "−") + Math.abs(x).toFixed(d));
const PP = 22.14; // win-rate points per logit
const pp = (theta) => (theta == null ? null : theta * PP);
const ordinal = (n) => { const s = ["th", "st", "nd", "rd"], v = n % 100; return n + (s[(v - 20) % 10] || s[v] || s[0]); };
const setName = (code) => (META?.sets.find((s) => s.code === code) || {}).name || code;
const setIcon = (code) => `https://svgs.scryfall.io/sets/${code.toLowerCase()}.svg`;
const TYPICAL = "typical";

const GRADE_VAR = { "A+": "--g-a", A: "--g-a", "A-": "--g-a2", "B+": "--g-b", B: "--g-b", "B-": "--g-b2",
  "C+": "--g-c2", C: "--g-c", "C-": "--g-c", "D+": "--g-d", D: "--g-d2", "D-": "--g-d2", F: "--g-f" };
const GRADES = ["A+", "A", "A-", "B+", "B", "B-", "C+", "C", "C-", "D+", "D", "D-", "F"];
const GRADE_CUTS = [99, 95, 90, 80, 70, 60, 50, 40, 30, 20, 10, 4, 0];
const gradeOf = (pct) => (pct == null ? null : GRADES[GRADE_CUTS.findIndex((c) => pct >= c)]);

function gradeBadge(g, size = "") {
  if (!g) return `<span class="grade none ${size}" title="No grade">–</span>`;
  const light = ["B-", "C+"].includes(g) ? " light-ink" : "";
  return `<span class="grade ${size}${light}" style="--gc: var(${GRADE_VAR[g]})" title="Grade ${g}">${g}</span>`;
}

function verdictWords(pct) {
  if (pct == null) return "Unrated";
  if (pct >= 95) return "A bomb";
  if (pct >= 85) return "Very strong";
  if (pct >= 70) return "Strong";
  if (pct >= 55) return "Above average";
  if (pct >= 40) return "Average";
  if (pct >= 25) return "Below average";
  if (pct >= 10) return "Weak";
  return "Very weak";
}

function manaSyms(cost) {
  if (!cost) return "";
  return cost.split(" // ").map((part) =>
    (part.match(/\{[^}]+\}/g) || []).map((s) => {
      const code = s.slice(1, -1).replace(/\//g, "").toUpperCase();
      return `<img class="ms" alt="${esc(s)}" src="https://svgs.scryfall.io/card-symbols/${encodeURIComponent(code)}.svg" loading="lazy">`;
    }).join("")
  ).join('<span class="muted"> // </span>');
}
const mana = (cost, big = false) => (cost ? `<span class="mana${big ? " big" : ""}">${manaSyms(cost)}</span>` : "");

function oracleHTML(text) {
  if (!text) return "";
  return text.split("\n").map((line) => {
    let h = esc(line).replace(/\([^)]*\)/g, (m) => `<i>${m}</i>`);
    h = h.replace(/\{[^}]+\}/g, (m) => manaSyms(m));
    return `<p>${h}</p>`;
  }).join("");
}

function cardImg(src, alt, cls = "") {
  return src ? `<img class="${cls}" src="${esc(src)}" alt="${esc(alt)}" loading="lazy">`
             : `<div class="${cls}" style="display:grid;place-items:center;height:100%;color:var(--text-muted);font-size:12px;padding:8px;text-align:center">${esc(alt)}</div>`;
}

function miniCard(c, opts = {}) {
  const g = opts.grade !== undefined ? opts.grade : c.grade_obs || c.lim_grade;
  return `<a class="mini" href="#/card/${c.oracle_id}" title="${esc(c.name)}">
    <div class="img">${cardImg(c.image_small || c.image, c.name)}</div>
    ${gradeBadge(g)}
    ${opts.caption === false ? "" : `<div class="cap"><b>${esc(c.name.split(" // ")[0])}</b><span>${opts.right ?? ""}</span></div>`}
  </a>`;
}

const tip = $("#tooltip");
function showTip(html, x, y) {
  tip.innerHTML = html;
  tip.hidden = false;
  const r = tip.getBoundingClientRect();
  tip.style.left = Math.min(window.innerWidth - r.width - 8, x + 14) + "px";
  tip.style.top = Math.max(8, y - r.height - 10) + "px";
}
const hideTip = () => { tip.hidden = true; };

// ------------------------------------------------------------------ distribution chart
// Histogram of a set's card strengths with the card's position: observed
// (17lands) as a white marker, the model estimate as a blue marker with its
// 80% interval as a band.
function distChart(values, { obs = null, model = null, lo = null, hi = null } = {}) {
  const W = 640, H = 150, pad = { l: 8, r: 8, t: 26, b: 26 };
  const all = values.concat([obs, model, lo, hi].filter((v) => v != null));
  let xmin = Math.min(...all), xmax = Math.max(...all);
  const span = xmax - xmin || 1;
  xmin -= span * 0.03; xmax += span * 0.03;
  const X = (v) => pad.l + ((v - xmin) / (xmax - xmin)) * (W - pad.l - pad.r);
  const nb = 34, bw = (xmax - xmin) / nb;
  const counts = new Array(nb).fill(0);
  values.forEach((v) => counts[Math.min(nb - 1, Math.floor((v - xmin) / bw))]++);
  const cmax = Math.max(...counts);
  const ih = H - pad.t - pad.b;
  const ref = obs ?? model;
  let s = `<svg class="dist" viewBox="0 0 ${W} ${H}" role="img" aria-label="Distribution of card strength in this environment">`;
  if (lo != null && hi != null) s += `<rect class="band" x="${X(lo)}" y="${pad.t - 6}" width="${Math.max(2, X(hi) - X(lo))}" height="${ih + 6}" rx="4"/>`;
  counts.forEach((c, i) => {
    if (!c) return;
    const x0 = X(xmin + i * bw) + 1, w = Math.max(1, X(xmin + (i + 1) * bw) - X(xmin + i * bw) - 2);
    const h = Math.max(2, (c / cmax) * ih);
    const hit = ref != null && ref >= xmin + i * bw && ref < xmin + (i + 1) * bw;
    s += `<rect class="bar${hit ? " hit" : ""}" x="${x0}" y="${pad.t + ih - h}" width="${w}" height="${h}" rx="2"><title>${c} cards between ${signed(pp(xmin + i * bw))} and ${signed(pp(xmin + (i + 1) * bw))} pts</title></rect>`;
  });
  s += `<line class="axis" x1="${pad.l}" x2="${W - pad.r}" y1="${pad.t + ih + 0.5}" y2="${pad.t + ih + 0.5}"/>`;
  // Ticks in win-rate points.
  const step = [1, 2, 2.5, 5, 10].find((st) => (xmax - xmin) * PP / st <= 8) || 10;
  for (let t = Math.ceil(xmin * PP / step) * step; t <= xmax * PP; t += step) {
    s += `<text x="${X(t / PP)}" y="${H - 6}" text-anchor="middle">${t > 0 ? "+" : ""}${t}</text>`;
  }
  // Labels sit above the markers; when the two markers are close, each label
  // points away from the other.
  const close = obs != null && model != null && Math.abs(X(model) - X(obs)) < 70;
  const marker = (v, cls, label, other) => {
    const x = X(v);
    let anchor = x > W - 90 ? "end" : x < 90 ? "start" : "middle", tx = x;
    if (close) { const left = v < other; anchor = left ? "end" : "start"; tx = x + (left ? -8 : 8); }
    return `<line class="m-${cls}" x1="${x}" x2="${x}" y1="${pad.t - 2}" y2="${pad.t + ih}"/>
      <circle class="m-${cls}-dot" cx="${x}" cy="${pad.t - 2}" r="5"/>
      <text class="lbl" x="${tx}" y="${pad.t - 12}" text-anchor="${anchor}">${label}</text>`;
  };
  if (model != null) s += marker(model, "model", "Model", obs);
  if (obs != null) s += marker(obs, "obs", "17lands", model);
  s += `</svg>`;
  return s;
}

function legendHTML({ obs, model }) {
  return `<div class="legend">
    ${obs ? `<span><i style="background:var(--marker-obs)"></i>Observed (17lands games)</span>` : ""}
    ${model ? `<span><i style="background:var(--accent)"></i>Model estimate</span><span><i style="background:var(--band)"></i>80% range of the estimate</span>` : ""}
    <span><i style="background:var(--hist)"></i>Other cards in the environment</span>
    <span class="muted">x axis: win-rate points vs a replacement-level common</span>
  </div>`;
}

// ------------------------------------------------------------------ search box
function attachSearch(input, { onPick, labelledOnly = false } = {}) {
  const wrap = input.parentElement;
  wrap.classList.add("search-wrap");
  let dd = null, items = [], sel = -1, timer = null, seq = 0;
  const close = () => { dd?.remove(); dd = null; sel = -1; };
  const pick = (it) => { close(); input.value = ""; input.blur(); (onPick || ((x) => go(`#/card/${x.oracle_id}`)))(it); };
  const render = () => {
    if (!dd) { dd = document.createElement("div"); dd.className = "dropdown"; wrap.appendChild(dd); }
    dd.innerHTML = items.length ? items.map((it, i) => `
      <div class="dd-item${i === sel ? " sel" : ""}" data-i="${i}">
        ${it.image_small ? `<img src="${esc(it.image_small)}" alt="">` : `<img alt="">`}
        <div class="dd-main"><div class="dd-name">${esc(it.name)} ${mana(it.mana_cost)}</div>
        <div class="dd-sub">${esc(it.type_line || "")}${it.labelled_sets ? ` · 17lands data: ${esc(it.labelled_sets.replaceAll(",", ", "))}` : ""}</div></div>
        ${gradeBadge(it.lim_grade)}
      </div>`).join("") : `<div class="dd-empty">No cards found</div>`;
    $$(".dd-item", dd).forEach((el) => el.addEventListener("mousedown", (e) => { e.preventDefault(); pick(items[+el.dataset.i]); }));
  };
  input.addEventListener("input", () => {
    clearTimeout(timer);
    const q = input.value.trim();
    if (q.length < 2) { close(); return; }
    timer = setTimeout(async () => {
      const my = ++seq;
      const res = await api(`/api/search?q=${encodeURIComponent(q)}&limit=10${labelledOnly ? "&labelled_only=true" : ""}`);
      if (my !== seq || input.value.trim() !== q) return;
      items = res; sel = res.length ? 0 : -1; render();
    }, 120);
  });
  input.addEventListener("keydown", (e) => {
    if (!dd) return;
    if (e.key === "ArrowDown") { sel = Math.min(items.length - 1, sel + 1); render(); e.preventDefault(); }
    else if (e.key === "ArrowUp") { sel = Math.max(0, sel - 1); render(); e.preventDefault(); }
    else if (e.key === "Enter" && items[sel]) { pick(items[sel]); e.preventDefault(); }
    else if (e.key === "Escape") close();
  });
  input.addEventListener("blur", () => setTimeout(close, 120));
}

// ------------------------------------------------------------------ router
function go(hash) { if (location.hash === hash) route(); else location.hash = hash; }

async function route() {
  hideTip();
  const [path, query] = location.hash.replace(/^#/, "").split("?");
  const parts = path.split("/").filter(Boolean);
  const params = new URLSearchParams(query || "");
  const page = parts[0] || "home";
  document.body.classList.toggle("home", page === "home");
  $$(".nav a").forEach((a) => a.classList.toggle("active", a.dataset.nav === (page === "card" ? "home" : page === "set" ? "sets" : page)));
  view.innerHTML = `<div class="skeleton" style="height:220px"></div>`;
  try {
    if (page === "card") await cardPage(parts[1], params);
    else if (page === "sets") await setsPage();
    else if (page === "set") await setPage(parts[1], params);
    else if (page === "lab") await labPage(params);
    else if (page === "about") await aboutPage();
    else await homePage();
  } catch (e) {
    view.innerHTML = `<div class="empty-state"><div class="big">⚠</div><p>${esc(e.message)}</p><a class="btn" href="#/">Back to search</a></div>`;
  }
  window.scrollTo(0, 0);
}

// ------------------------------------------------------------------ home
async function homePage() {
  const newest = META.sets[META.sets.length - 1];
  view.innerHTML = `
    <section class="hero">
      <h1>Is this card <span>actually good?</span></h1>
      <p>Search any Magic card to see how strong it is in Limited, measured on millions of 17lands games and predicted by a model for cards that have no data yet. Build your own card in the Card Lab and the model will grade it.</p>
      <div><input id="heroSearch" type="search" placeholder="Search a card, e.g. Lightning Bolt" autocomplete="off" autofocus aria-label="Search cards"></div>
      <div class="chips">
        <a class="chip" id="randomBtn">🎲 Random card</a>
        <a class="chip" href="#/set/${newest.code}"><span class="set-icon" style="--icon:url(${setIcon(newest.code)});width:16px;height:16px;background:currentColor"></span>${esc(newest.name)} rankings</a>
        <a class="chip" href="#/lab">✦ Build a custom card</a>
      </div>
    </section>
    <div class="section-head"><h2>Best cards of ${esc(newest.name)}</h2><a href="#/set/${newest.code}">Full ranking →</a></div>
    <div class="card-grid" id="topGrid">${'<div class="skeleton" style="aspect-ratio:488/680"></div>'.repeat(6)}</div>
    <div class="section-head"><h2>How to read a grade</h2><a href="#/about">How it works →</a></div>
    <div class="how">
      <div class="panel"><div class="icon">A</div><h4>Grades are relative to the set</h4><p>A card's grade is its percentile among the cards of its Limited set: A+ is the top 1%, C+ the middle, F the bottom 4%.</p></div>
      <div class="panel"><div class="icon">%</div><h4>Win-rate points</h4><p>Strength is the change in win probability when the card replaces a replacement-level common in a deck that can play it, from a debiased model of 17lands games.</p></div>
      <div class="panel"><div class="icon">±</div><h4>Honest uncertainty</h4><p>Model estimates come with an 80% range. Estimates for sets with data were made without seeing that set, so you can check the model against reality.</p></div>
    </div>`;
  attachSearch($("#heroSearch"));
  $("#randomBtn").addEventListener("click", async () => { const r = await fetch("/api/random").then((x) => x.json()); go(`#/card/${r.oracle_id}`); });
  const cards = await api(`/api/set/${newest.code}`);
  $("#topGrid").innerHTML = cards.slice(0, 12).map((c) => miniCard(c, { right: signed(pp(c.theta_vs_repl)) + " pts" })).join("");
}

// ------------------------------------------------------------------ card page
async function cardPage(id, params) {
  const c = await api(`/api/card/${id}`);
  const apps = c.appearances;
  const want = params.get("set");
  const a = apps.find((x) => x.set === want) || apps[0] || null;
  const isTest = a && META.sets.find((s) => s.code === a.set)?.test;
  document.title = `${c.name.split(" // ")[0]} · Card Strength Oracle`;

  let verdict, chart, tiles, extra = "";
  if (a) {
    const pct = a.pct_obs;
    verdict = `
      <div class="verdict">${gradeBadge(a.grade_obs, "lg")}
        <div><div class="headline">${verdictWords(pct)} in ${esc(a.set_name)} Limited</div>
        <div class="sub">Ranked <b>#${a.rank_obs}</b> of ${a.n_in_set} cards by 17lands results · better than ${fmt(pct, 0)}% of the set</div></div>
      </div>`;
    chart = distChart(a.dist, { obs: a.theta_vs_repl, model: a.model_pred, lo: a.model_pred - 1.2816 * a.model_sd, hi: a.model_pred + 1.2816 * a.model_sd }) + legendHTML({ obs: true, model: true });
    const src = a.model_source === "prerelease" ? "Pre-release model estimate" : "Model estimate (set held out)";
    tiles = `
      <div class="tile"><div class="k">Win-rate impact</div><div class="v">${signed(pp(a.theta_vs_repl))} pts</div><div class="s">± ${fmt(pp(a.theta_hand_sd))} pts (1 SD) vs replacement level</div></div>
      <div class="tile"><div class="k">${src}</div><div class="v" style="display:flex;gap:10px;align-items:center">${gradeBadge(a.grade_model)} ${signed(pp(a.model_pred))}</div><div class="s">Model rank #${a.rank_model} of ${a.n_in_set}</div></div>
      <div class="tile"><div class="k">Win rate when drawn</div><div class="v">${a.raw_wr_in_hand != null ? fmt(100 * a.raw_wr_in_hand, 1) + "%" : "–"}</div><div class="s">Raw 17lands GIH WR (not debiased)</div></div>
      <div class="tile"><div class="k">Games measured</div><div class="v">${a.n_games_in_hand != null ? Number(a.n_games_in_hand).toLocaleString() : "–"}</div><div class="s">Games with the card in hand</div></div>`;
    if (a.draftsim != null || a.ata_week1 != null) {
      tiles += `<div class="tile"><div class="k">What humans said</div><div class="v" style="font-size:18px">${a.draftsim != null ? `Draftsim ${fmt(a.draftsim, 1)}/5` : ""}</div><div class="s">${a.ata_week1 != null ? `Picked ${fmt(a.ata_week1, 1)} on average in week 1` : ""}</div></div>`;
    }
    const diff = a.pct_model - a.pct_obs;
    const note = Math.abs(diff) > 30
      ? (diff > 0 ? "The model expected more from this card than it delivered." : "This card overperformed what the model expected from its text.")
      : "The model's estimate is close to what the games showed.";
    extra = `<p class="note">${a.model_source === "prerelease"
      ? "This set was released after the model's training data, so the model estimate is a genuine pre-release prediction."
      : "The model estimate was made with this whole set held out of training."} ${note}</p>`;
  } else {
    verdict = `
      <div class="verdict">${gradeBadge(c.lim_grade, "lg")}
        <div><div class="headline">${verdictWords(c.lim_pct)} in a typical Limited set <span class="muted" style="font-weight:500">(model estimate)</span></div>
        <div class="sub">Would beat about ${fmt(c.lim_pct, 0)}% of the cards in a recent set · 80% range ${gradeOf(c.lim_pct_lo)} to ${gradeOf(c.lim_pct_hi)}</div></div>
      </div>`;
    chart = distChart(c.lim_dist, { model: c.lim_pred, lo: c.lim_pred - 1.2816 * c.lim_sd, hi: c.lim_pred + 1.2816 * c.lim_sd }) + legendHTML({ model: true });
    tiles = `
      <div class="tile"><div class="k">Estimated win-rate impact</div><div class="v">${signed(pp(c.lim_pred))} pts</div><div class="s">80% range ${signed(pp(c.lim_pred - 1.2816 * c.lim_sd))} to ${signed(pp(c.lim_pred + 1.2816 * c.lim_sd))}</div></div>
      <div class="tile"><div class="k">Model</div><div class="v" style="font-size:17px">${c.lim_model === "ensemble" ? "Fine-tuned Gemma + GBT" : "Gradient boosting"}</div><div class="s">${c.lim_model === "ensemble" ? "Average of the two models" : "Fine-tuned encoder not loaded"}</div></div>
      <div class="tile"><div class="k">17lands data</div><div class="v" style="font-size:18px">None</div><div class="s">This card is not in one of the ${META.sets.length} sets with game data, so only the model can grade it.</div></div>`;
  }

  const setTabs = apps.length > 1 ? `<div class="set-tabs">${apps.map((x) =>
    `<a class="chip${x.set === a.set ? " on" : ""}" href="#/card/${id}?set=${x.set}"><span class="set-icon" style="--icon:url(${setIcon(x.set)});width:15px;height:15px;background:currentColor"></span>${esc(x.set_name)}</a>`).join("")}</div>` : "";

  const synergy = a && a.synergy?.length ? `
    <div class="panel"><h3>Best partners in ${esc(a.set_name)}</h3>
      ${a.synergy.map((p) => `<a class="list-row" href="#/card/${p.oracle_id || ""}?set=${a.set}">${p.image_small ? `<img src="${esc(p.image_small)}" alt="">` : "<img alt=''>"}<span class="lr-name">${esc(p.name)}</span><span class="lr-val">${p.pair} deck</span>${gradeBadge(p.grade)}</a>`).join("")}
      <p class="note">Pairs the deck model expects to win more together than apart (top 40 cards per colour pair, test sets only).</p>
    </div>` : "";

  const neigh = a ? `
    <div class="panel"><h3>Around it in the ranking</h3>
      ${a.neighbours.map((n) => `<a class="list-row${n.oracle_id === id ? " self" : ""}" href="#/card/${n.oracle_id}?set=${a.set}"><span class="lr-val" style="width:34px">#${n.rank_obs}</span>${n.image_small ? `<img src="${esc(n.image_small)}" alt="">` : "<img alt=''>"}<span class="lr-name">${esc(n.name)}</span><span class="lr-val">${signed(pp(n.theta_vs_repl))}</span>${gradeBadge(n.grade_obs)}</a>`).join("")}
    </div>` : "";

  // Other formats.
  const cmdObs = c.edhrec_incl != null;
  const cmdVal = cmdObs ? c.edhrec_incl : c.cmd_incl_pred;
  const conObs = c.staple_pct != null;
  const formats = `
    <div class="panel"><h3>Other formats</h3>
      <div class="format-row"><div class="fname">Commander</div><div class="fval">
        <div><b>${cmdVal != null ? fmt(100 * cmdVal, cmdVal < 0.01 ? 2 : 1) + "%" : "–"}</b> of eligible decks ${cmdObs ? `<span class="muted">on EDHREC${c.edhrec_decks ? ` (${Number(c.edhrec_decks).toLocaleString()} decks)` : ""}</span>` : `<span class="tag">model estimate</span>`}</div>
        <div class="meter" title="Percentile among cards from recent Limited sets"><div style="width:${fmt(c.cmd_pct, 0)}%"></div></div>
        <div class="small muted">More played than ${fmt(c.cmd_pct, 0)}% of cards from recent sets</div></div></div>
      <div class="format-row"><div class="fname">Constructed</div><div class="fval">
        ${conObs ? `<div><b>Staple</b> in ${esc((c.staple_formats || "").replaceAll(",", ", "))}: up to ${fmt(c.staple_pct, 0)}% of top ${esc(c.staple_format)} decks <span class="muted">(MTGGoldfish top 50)</span></div>
                   <div class="small muted">The model alone would have given it a ${fmt(100 * c.con_prob, 1)}% chance.</div>`
                 : `<div><b>${fmt(100 * c.con_prob, 1)}%</b> chance of becoming a staple <span class="tag">model estimate</span></div>
        <div class="meter"><div style="width:${fmt(c.con_pct, 0)}%"></div></div>
        <div class="small muted">Higher than ${fmt(c.con_pct, 0)}% of cards from recent sets</div>`}</div></div>
      <p class="note">Limited, Commander and Constructed strength are nearly unrelated, so each has its own model. Estimates outside recent Limited sets are rough.</p>
    </div>`;

  view.innerHTML = `
    <div class="card-page">
      <div class="card-art">
        <div class="frame" id="cardFrame">${cardImg(c.image, c.name)}</div>
        <div class="actions">
          ${c.image_back ? `<button class="btn" id="flipBtn">↻ Flip</button>` : ""}
          <a class="btn" href="#/lab?from=${id}${a ? `&env=${a.set}` : ""}">✦ Test in another set</a>
          ${c.scryfall_uri ? `<a class="btn ghost" href="${esc(c.scryfall_uri)}" target="_blank" rel="noopener">Scryfall ↗</a>` : ""}
        </div>
        <div class="panel oracle" style="margin-top:14px;font-size:14px">${facesHTML(c)}</div>
      </div>
      <div>
        <div class="card-title"><h1>${esc(c.name)}</h1>${mana(c.mana_cost, true)}</div>
        <div class="typeline"><span>${esc(c.type_line)}</span>${a ? `<span class="rarity ${esc(a.rarity)}">${esc(a.rarity)}</span>` : `<span class="rarity ${esc(c.rarity)}">${esc(c.rarity)}</span>`}
          ${a ? `<a class="tag${isTest ? " test" : ""}" href="#/set/${a.set}">${esc(a.set)}${isTest ? " · new set" : ""}</a>` : ""}</div>
        ${setTabs}
        <div class="panel">${verdict}<div style="margin-top:18px">${chart}</div>${extra}</div>
        <div class="panel"><h3>${a ? "Limited numbers" : "Model estimate"}</h3><div class="tiles">${tiles}</div></div>
        ${a ? `<div class="split" style="margin-top:16px">${neigh}${synergy || formats}</div>${synergy ? formats : ""}` : formats}
      </div>
    </div>`;
  if (c.image_back) {
    let back = false;
    $("#flipBtn").addEventListener("click", () => { back = !back; $("#cardFrame").innerHTML = cardImg(back ? c.image_back : c.image, c.name); });
  }
}

// Rules text per face: double-faced cards store each field as "front // back".
function facesHTML(c) {
  const split = (v) => (v == null ? [] : String(v).split(" // "));
  const names = c.name.split(" // "), texts = split(c.oracle_text);
  const pw = split(c.power), tg = split(c.toughness), ly = split(c.loyalty);
  const n = Math.max(1, texts.length);
  return Array.from({ length: n }, (_, i) => {
    const pt = pw[i] && tg[i] ? `<p><b>${esc(pw[i])}/${esc(tg[i])}</b></p>` : "";
    const loy = ly[i] ? `<p><b>Loyalty ${esc(ly[i])}</b></p>` : "";
    const head = n > 1 ? `<p class="small muted" style="text-transform:uppercase;letter-spacing:.05em;font-weight:700;${i ? "margin-top:12px;padding-top:12px;border-top:1px solid var(--border)" : ""}">${esc(names[i] || "")}</p>` : "";
    return head + oracleHTML(texts[i] || "") + pt + loy;
  }).join("");
}

function labInputFromCard(c) {
  return { name: c.name, mana_cost: c.mana_cost || "", type_line: c.type_line || "", oracle_text: c.oracle_text || "",
    power: c.power, toughness: c.toughness, loyalty: c.loyalty, rarity: ["common", "uncommon", "rare", "mythic"].includes(c.rarity) ? c.rarity : "rare" };
}

// ------------------------------------------------------------------ sets
async function setsPage() {
  document.title = "Sets · Card Strength Oracle";
  const sets = [...META.sets].reverse();
  view.innerHTML = `
    <div class="section-head" style="margin-top:0"><h2>Limited sets with 17lands data</h2><span class="muted">${sets.length} sets · ${META.n_labelled.toLocaleString()} cards</span></div>
    <p class="secondary" style="margin-top:-6px">Pick a set to see every card ranked by measured strength, as a tier list or a table, next to what the model predicted.
      Sets marked <span class="tag test">new</span> came out after the model's training data.</p>
    <div class="set-grid">${sets.map((s) => `
      <a class="set-tile" href="#/set/${s.code}"><span class="set-icon" style="--icon:url(${setIcon(s.code)})"></span>
        <div><div class="nm">${esc(s.name)}</div><div class="dt">${s.code} · ${s.released.slice(0, 7)} · ${s.n_cards} cards ${s.test ? '<span class="tag test">new</span>' : ""}</div></div></a>`).join("")}
    </div>`;
}

const COLORS = [["W", "White"], ["U", "Blue"], ["B", "Black"], ["R", "Red"], ["G", "Green"], ["M", "Multicolor"], ["C", "Colorless"]];
const RARITIES = [["common", "C"], ["uncommon", "U"], ["rare", "R"], ["mythic", "M"]];

async function setPage(code, params) {
  code = code.toUpperCase();
  const s = META.sets.find((x) => x.code === code);
  if (!s) throw new Error("Unknown set " + code);
  document.title = `${s.name} · Card Strength Oracle`;
  const cards = await api(`/api/set/${code}`);
  const st = { mode: params.get("view") || "tiers", by: "obs", colors: new Set(), rarities: new Set(), q: "", sort: "rank_obs", asc: true };
  view.innerHTML = `
    <div class="set-head"><span class="set-icon" style="--icon:url(${setIcon(code)})"></span>
      <div><h1>${esc(s.name)}</h1><div class="muted">${code} · released ${s.released} · ${cards.length} cards ${s.test ? '<span class="tag test">new set: model estimates are pre-release predictions</span>' : '<span class="tag">model estimates made with the set held out</span>'}</div></div></div>
    <div class="toolbar">
      <div class="seg" id="modeSeg"><button data-v="tiers">Tier list</button><button data-v="table">Table</button><button data-v="scatter">Model vs games</button></div>
      <div class="seg" id="bySeg" title="What the grades are based on"><button data-v="obs">17lands data</button><button data-v="model">Model</button></div>
      <span class="sep"></span>
      <div class="grp" id="colorChips">${COLORS.map(([k, n]) => `<span class="chip" data-v="${k}" title="${n}">${k === "M" ? "Multi" : k === "C" ? "Colorless" : `<img src="https://svgs.scryfall.io/card-symbols/${k}.svg" alt="${n}">`}</span>`).join("")}</div>
      <span class="sep"></span>
      <div class="grp" id="rarChips">${RARITIES.map(([k, n]) => `<span class="chip" data-v="${k}" title="${k}">${n}</span>`).join("")}</div>
      <input type="search" id="setFilter" placeholder="Filter by name or type…" style="flex:1;min-width:180px">
    </div>
    <div id="setBody"></div>`;

  const filtered = () => cards.filter((c) => {
    const ci = c.color_identity || "";
    if (st.colors.size) {
      const tag = ci.length > 1 ? "M" : ci.length === 0 ? "C" : ci;
      if (!st.colors.has(tag)) return false;
    }
    if (st.rarities.size && !st.rarities.has(c.rarity)) return false;
    if (st.q && !(c.name + " " + c.type_line).toLowerCase().includes(st.q)) return false;
    return true;
  });
  const gradeKey = () => (st.by === "obs" ? "grade_obs" : "grade_model");

  function render() {
    $$("#modeSeg button").forEach((b) => b.classList.toggle("on", b.dataset.v === st.mode));
    $$("#bySeg button").forEach((b) => b.classList.toggle("on", b.dataset.v === st.by));
    $("#bySeg").style.display = st.mode === "scatter" ? "none" : "";
    const list = filtered();
    const body = $("#setBody");
    if (!list.length) { body.innerHTML = `<div class="empty-state">No cards match these filters.</div>`; return; }
    if (st.mode === "tiers") {
      const sorted = [...list].sort((a, b) => (st.by === "obs" ? a.rank_obs - b.rank_obs : a.rank_model - b.rank_model));
      const tiers = [["A+", ["A+"]], ["A", ["A", "A-"]], ["B", ["B+", "B", "B-"]], ["C", ["C+", "C", "C-"]], ["D", ["D+", "D", "D-"]], ["F", ["F"]]];
      body.innerHTML = tiers.map(([t, gs]) => {
        const xs = sorted.filter((c) => gs.includes(c[gradeKey()]));
        return `<div class="tier${xs.length ? "" : " empty"}"><div class="tl">${gradeBadge(t, "md")}</div>
          <div class="tc">${xs.map((c) => miniCard(c, { grade: c[gradeKey()], caption: false })).join("")}</div></div>`;
      }).join("") + `<p class="note">Grades are percentiles within the set: A+ top 1%, A 95th, A- 90th, B+ 80th, B 70th, B- 60th, C+ 50th, C 40th, C- 30th, D+ 20th, D 10th, D- 4th, F below.</p>`;
    } else if (st.mode === "table") {
      const cols = [
        ["rank_obs", "#", true], ["name", "Card", false], ["rarity", "Rarity", false], ["grade_obs", "Grade", false],
        ["theta_vs_repl", "Win-rate pts", true], ["raw_wr_in_hand", "GIH WR", true], ["n_games_in_hand", "Games", true],
        ["rank_model", "Model #", true], ["model_pred", "Model pts", true],
      ];
      if (list.some((c) => c.draftsim != null)) cols.push(["draftsim", "Draftsim", true], ["ata_week1", "ATA wk1", true]);
      const sorted = [...list].sort((a, b) => {
        const x = a[st.sort], y = b[st.sort];
        if (x == null) return 1; if (y == null) return -1;
        const r = typeof x === "string" ? x.localeCompare(y) : x - y;
        return st.asc ? r : -r;
      });
      body.innerHTML = `<div class="table-wrap"><table class="data"><thead><tr>${cols.map(([k, n, num]) => `<th data-k="${k}" class="${num ? "num " : ""}${st.sort === k ? "sorted" + (st.asc ? " asc" : "") : ""}">${n}</th>`).join("")}</tr></thead>
        <tbody>${sorted.map((c) => `<tr>
          <td class="num">${c.rank_obs}</td>
          <td class="name"><a href="#/card/${c.oracle_id}?set=${code}">${esc(c.name)}</a> ${mana(c.mana_cost)}</td>
          <td><span class="rarity ${c.rarity}">${c.rarity[0].toUpperCase()}</span></td>
          <td>${gradeBadge(c.grade_obs)}</td>
          <td class="num">${signed(pp(c.theta_vs_repl))}</td>
          <td class="num">${c.raw_wr_in_hand != null ? fmt(100 * c.raw_wr_in_hand) + "%" : "–"}</td>
          <td class="num">${c.n_games_in_hand != null ? Number(c.n_games_in_hand).toLocaleString() : "–"}</td>
          <td class="num">${c.rank_model}</td>
          <td class="num">${signed(pp(c.model_pred))}</td>
          ${cols.length > 9 ? `<td class="num">${fmt(c.draftsim)}</td><td class="num">${fmt(c.ata_week1)}</td>` : ""}
        </tr>`).join("")}</tbody></table></div>`;
      $$("th", body).forEach((th) => th.addEventListener("click", () => {
        const k = th.dataset.k;
        if (st.sort === k) st.asc = !st.asc; else { st.sort = k; st.asc = ["rank_obs", "rank_model", "name", "rarity", "grade_obs"].includes(k); }
        render();
      }));
    } else {
      body.innerHTML = scatterChart(list, code);
      bindScatter(body, list, code);
    }
  }
  $$("#modeSeg button").forEach((b) => b.addEventListener("click", () => { st.mode = b.dataset.v; render(); }));
  $$("#bySeg button").forEach((b) => b.addEventListener("click", () => { st.by = b.dataset.v; render(); }));
  const toggleChip = (set, el) => { const v = el.dataset.v; set.has(v) ? set.delete(v) : set.add(v); el.classList.toggle("on"); render(); };
  $$("#colorChips .chip").forEach((el) => el.addEventListener("click", () => toggleChip(st.colors, el)));
  $$("#rarChips .chip").forEach((el) => el.addEventListener("click", () => toggleChip(st.rarities, el)));
  $("#setFilter").addEventListener("input", (e) => { st.q = e.target.value.trim().toLowerCase(); render(); });
  render();
}

// Model estimate (x) against measured strength (y), one dot per card; a single
// series in the accent colour, rarity is in the tooltip.
function scatterChart(list) {
  const W = 760, H = 460, p = { l: 48, r: 16, t: 16, b: 42 };
  const xs = list.map((c) => c.model_pred * PP), ys = list.map((c) => c.theta_vs_repl * PP);
  const lo = Math.min(...xs, ...ys), hi = Math.max(...xs, ...ys);
  const pad = (hi - lo) * 0.05;
  const a = lo - pad, b = hi + pad;
  const X = (v) => p.l + ((v - a) / (b - a)) * (W - p.l - p.r);
  const Y = (v) => H - p.b - ((v - a) / (b - a)) * (H - p.t - p.b);
  const step = [1, 2, 2.5, 5, 10].find((s) => (b - a) / s <= 8) || 10;
  let s = `<div class="panel"><svg class="scatter" viewBox="0 0 ${W} ${H}" role="img" aria-label="Model estimate against measured strength">`;
  for (let t = Math.ceil(a / step) * step; t <= b; t += step) {
    s += `<line class="grid" x1="${X(t)}" x2="${X(t)}" y1="${p.t}" y2="${H - p.b}"/><line class="grid" x1="${p.l}" x2="${W - p.r}" y1="${Y(t)}" y2="${Y(t)}"/>`;
    s += `<text x="${X(t)}" y="${H - p.b + 16}" text-anchor="middle">${t > 0 ? "+" : ""}${t}</text><text x="${p.l - 8}" y="${Y(t) + 4}" text-anchor="end">${t > 0 ? "+" : ""}${t}</text>`;
  }
  s += `<line class="diag" x1="${X(a)}" y1="${Y(a)}" x2="${X(b)}" y2="${Y(b)}"/>`;
  s += `<text x="${(p.l + W - p.r) / 2}" y="${H - 6}" text-anchor="middle">Model estimate (win-rate pts)</text>`;
  s += `<text transform="translate(12 ${(p.t + H - p.b) / 2}) rotate(-90)" text-anchor="middle">Measured on 17lands (win-rate pts)</text>`;
  list.forEach((c, i) => {
    s += `<circle class="dot" data-i="${i}" cx="${X(xs[i])}" cy="${Y(ys[i])}" r="5" fill="var(--accent)" fill-opacity=".75"/>`;
  });
  s += `</svg><p class="note">Each dot is a card. Dots above the dashed line did better than the model predicted, dots below did worse. Click a dot to open the card.</p></div>`;
  return s;
}

function bindScatter(body, list, code) {
  $$(".dot", body).forEach((d) => {
    const c = list[+d.dataset.i];
    d.addEventListener("mousemove", (e) => showTip(`<b>${esc(c.name)}</b>${esc(c.rarity)} · grade ${c.grade_obs}<br>Measured ${signed(pp(c.theta_vs_repl))} pts, model ${signed(pp(c.model_pred))} pts`, e.clientX, e.clientY));
    d.addEventListener("mouseleave", hideTip);
    d.addEventListener("click", () => go(`#/card/${c.oracle_id}?set=${code}`));
  });
}

// ------------------------------------------------------------------ card lab
const FRAME = { W: "#e8dfc4", U: "#4f86c6", B: "#4a4545", R: "#d0583f", G: "#3f8a55", M: "#d4b04c", C: "#a8a8a8" };
const ART = { W: ["#f5ecd0", "#b9a77a"], U: ["#7fb0e8", "#1d3c6b"], B: ["#6b5a7a", "#1a1420"], R: ["#f39a6b", "#7a1f12"], G: ["#8fcf8a", "#1d4a26"], M: ["#f1d98a", "#7a5b14"], C: ["#d0d0d0", "#555"] };

async function labPage(params) {
  document.title = "Card Lab · Card Strength Oracle";
  let init = { name: "Skyhunter Initiate", mana_cost: "{2}{W}", type_line: "Creature — Bird Soldier", oracle_text: "Flying\nWhen this creature enters, scry 2.", power: "2", toughness: "2", loyalty: "", rarity: "common" };
  if (params.get("from")) {
    try { init = { ...init, ...labInputFromCard(await api(`/api/card/${params.get("from")}`)) }; } catch (e) { /* keep defaults */ }
  }
  const env0 = params.get("env") || TYPICAL;
  view.innerHTML = `
    <div class="section-head" style="margin-top:0"><h2>Card Lab</h2><span class="muted">Design a card, or load a real one, and see how the model grades it.</span></div>
    <div class="lab">
      <div>
        <div class="panel">
          <div style="margin-bottom:14px"><label for="loadCard">Start from an existing card</label><div><input type="search" id="loadCard" placeholder="Search a card to copy…" autocomplete="off" style="width:100%"></div></div>
          <form class="form-grid" id="labForm" autocomplete="off">
            <div class="full"><label for="f_name">Name</label><input type="text" id="f_name" name="name"></div>
            <div><label for="f_cost">Mana cost</label><input type="text" id="f_cost" name="mana_cost" placeholder="{2}{W}{W}"></div>
            <div><label for="f_rarity">Rarity</label><select id="f_rarity" name="rarity"><option>common</option><option>uncommon</option><option>rare</option><option>mythic</option></select></div>
            <div class="full"><label for="f_type">Type line</label><input type="text" id="f_type" name="type_line" placeholder="Creature — Elf Warrior"></div>
            <div class="full"><label for="f_text">Rules text</label><textarea id="f_text" name="oracle_text" placeholder="Flying&#10;When this creature enters, draw a card."></textarea>
              <div class="small muted" style="margin-top:4px">Use {T}, {W}, {2} for symbols. "This creature" or the card's name both work.</div></div>
            <div class="full pt">
              <div><label for="f_pow">Power</label><input type="text" id="f_pow" name="power"></div>
              <div><label for="f_tou">Toughness</label><input type="text" id="f_tou" name="toughness"></div>
              <div><label for="f_loy">Loyalty</label><input type="text" id="f_loy" name="loyalty"></div>
            </div>
            <div class="full"><label for="f_env">Limited environment</label><select id="f_env" name="env">
              <option value="${TYPICAL}">Typical recent set (average of the last ${META.typical_sets.length})</option>
              ${[...META.sets].reverse().map((s) => `<option value="${s.code}">${esc(s.name)} (${s.code})</option>`).join("")}</select></div>
            <div class="full" style="display:flex;gap:10px;align-items:center;flex-wrap:wrap">
              <button class="btn primary" type="submit" id="evalBtn">Grade this card</button>
              <span class="small muted">Ctrl+Enter</span>
            </div>
          </form>
        </div>
      </div>
      <div>
        <div style="display:grid;grid-template-columns:minmax(200px,260px) 1fr;gap:20px;align-items:start" id="labTop">
          <div id="mock"></div>
          <div id="labVerdict" class="panel"><div class="empty-state" style="padding:30px 10px"><div class="big">✦</div>Fill in the card and press <b>Grade this card</b>.</div></div>
        </div>
        <div id="labResults"></div>
      </div>
    </div>`;
  if (window.innerWidth < 700) $("#labTop").style.gridTemplateColumns = "1fr";
  const form = $("#labForm");
  const fill = (c) => {
    form.name.value = c.name || ""; form.mana_cost.value = c.mana_cost || ""; form.rarity.value = c.rarity || "common";
    form.type_line.value = c.type_line || ""; form.oracle_text.value = c.oracle_text || "";
    form.power.value = c.power ?? ""; form.toughness.value = c.toughness ?? ""; form.loyalty.value = c.loyalty ?? "";
    renderMock();
  };
  form.env.value = META.sets.some((s) => s.code === env0) ? env0 : TYPICAL;
  attachSearch($("#loadCard"), { onPick: async (it) => { fill(labInputFromCard(await api(`/api/card/${it.oracle_id}`))); evaluate(); } });

  function data() {
    const d = Object.fromEntries(new FormData(form).entries());
    for (const k of ["power", "toughness", "loyalty"]) if (!d[k]) d[k] = null;
    return d;
  }
  function renderMock() {
    const d = data();
    const syms = (d.mana_cost.match(/\{[^}]+\}/g) || []).join("");
    const cols = "WUBRG".split("").filter((c) => syms.includes(c));
    const key = d.type_line.includes("Land") && !cols.length ? "C" : cols.length > 1 ? "M" : cols[0] || "C";
    $("#mock").innerHTML = `<div class="mock" style="--frame:${FRAME[key]};--art1:${ART[key][0]};--art2:${ART[key][1]}"><div class="mock-inner">
      <div class="mock-bar"><span>${esc(d.name || "Unnamed")}</span>${mana(d.mana_cost)}</div>
      <div class="mock-art"></div>
      <div class="mock-bar"><span>${esc(d.type_line || "Type")}</span><span class="rarity ${esc(d.rarity)}" style="background:none;padding:0">${esc(d.rarity[0].toUpperCase())}</span></div>
      <div class="mock-text">${oracleHTML(d.oracle_text)}</div>
      ${d.power != null || d.toughness != null ? `<div class="mock-pt">${esc(d.power ?? "")}/${esc(d.toughness ?? "")}</div>` : d.loyalty ? `<div class="mock-pt">◆ ${esc(d.loyalty)}</div>` : ""}
    </div></div>`;
  }
  async function evaluate() {
    const btn = $("#evalBtn");
    btn.disabled = true;
    btn.innerHTML = `<span class="spinner"></span> Grading…`;
    try {
      const r = await api("/api/predict", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(data()) });
      showLab(r, data());
    } catch (e) {
      $("#labVerdict").innerHTML = `<div class="callout">Could not grade the card: ${esc(e.message)}</div>`;
    } finally {
      btn.disabled = false; btn.textContent = "Grade this card";
    }
  }
  form.addEventListener("input", renderMock);
  form.addEventListener("submit", (e) => { e.preventDefault(); evaluate(); });
  form.addEventListener("keydown", (e) => { if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) { e.preventDefault(); evaluate(); } });
  form.env.addEventListener("change", evaluate);
  fill(init);
  if (params.get("from")) evaluate();
}

function showLab(r, d) {
  const envName = r.env === TYPICAL ? "a typical recent set" : setName(r.env);
  $("#labVerdict").innerHTML = `
    <div class="verdict">${gradeBadge(r.grade, "lg")}
      <div><div class="headline">${verdictWords(r.pct)} in ${esc(envName)}</div>
      <div class="sub">${r.rank ? `Would rank about <b>#${r.rank}</b> of ${r.n_in_set} · ` : ""}better than ${fmt(r.pct, 0)}% of the cards · 80% range ${gradeOf(r.pct_lo)} to ${gradeOf(r.pct_hi)}</div></div>
    </div>
    <div class="tiles" style="margin-top:16px">
      <div class="tile"><div class="k">Win-rate impact</div><div class="v">${signed(r.pp)} pts</div><div class="s">80% range ${signed(pp(r.lo))} to ${signed(pp(r.hi))}</div></div>
      <div class="tile"><div class="k">Model</div><div class="v" style="font-size:17px">${r.model === "ensemble" ? "Fine-tuned encoder + GBT" : "Gradient boosting"}</div><div class="s">${r.model === "ensemble" ? `GBT ${signed(pp(r.gbt))}, encoder ${signed(pp(r.ft))} pts` : "Reads the card's stats, keywords and text"}</div></div>
    </div>
    <p class="note">Read as: mana value ${fmt(r.mana_value, 0)}, colours ${r.colors || "none"}${r.keywords?.length ? `, keywords ${esc(r.keywords.join(", "))}` : ", no keywords"}.</p>`;
  $("#labResults").innerHTML = `
    <div class="panel" style="margin-top:16px"><h3>Where it lands in ${esc(envName)}</h3>${distChart(r.dist, { model: r.pred, lo: r.lo, hi: r.hi })}${legendHTML({ model: true })}</div>
    <div class="split" style="margin-top:16px">
      <div class="panel"><h3>Real cards the model rates the same</h3>
        ${r.comparable.map((n) => `<a class="list-row" href="#/card/${n.oracle_id}?set=${n.set || r.env}">${n.image_small ? `<img src="${esc(n.image_small)}" alt="">` : "<img alt=''>"}<span class="lr-name">${esc(n.name)}</span><span class="lr-val">${n.set ? n.set + " · " : ""}played ${signed(pp(n.theta_vs_repl))}</span>${gradeBadge(n.grade_obs)}</a>`).join("")}
        <p class="note">Their actual 17lands result shows how much cards with the same estimate can differ.</p></div>
      <div class="panel"><h3>Other formats (rough estimates)</h3>
        <div class="format-row"><div class="fname">Commander</div><div class="fval"><b>${fmt(100 * r.cmd_incl, r.cmd_incl < 0.01 ? 2 : 1)}%</b> of eligible decks<div class="meter"><div style="width:${fmt(r.cmd_pct, 0)}%"></div></div><div class="small muted">Above ${fmt(r.cmd_pct, 0)}% of cards from recent sets</div></div></div>
        <div class="format-row"><div class="fname">Constructed</div><div class="fval"><b>${fmt(100 * r.con_prob, 1)}%</b> chance to be a staple<div class="meter"><div style="width:${fmt(r.con_pct, 0)}%"></div></div><div class="small muted">Above ${fmt(r.con_pct, 0)}% of cards from recent sets</div></div></div>
      </div>
    </div>`;
}

// ------------------------------------------------------------------ about
async function aboutPage() {
  document.title = "How it works · Card Strength Oracle";
  const m = META.metrics;
  const row = (label, x, note = "") => x ? `<tr><td>${label}${note ? ` <span class="muted small">${note}</span>` : ""}</td><td class="num">${fmt(x.spearman, 2)}</td><td class="num">${fmt(x.ndcg20, 2)}</td><td class="num">${x.mae_pp != null ? fmt(x.mae_pp, 2) : "–"}</td></tr>` : "";
  view.innerHTML = `<div class="prose">
    <h1 class="display" style="font-size:34px">How it works</h1>
    <p>The question this site answers: <b>if this card replaces an average filler card in a Limited deck that can play it, how much more often does that deck win?</b> That number, in win-rate points, is the card's strength.</p>
    <h2>Measured strength (17lands)</h2>
    <p>For ${META.sets.length} sets, from Kaldheim to ${esc(META.sets[META.sets.length - 1].name)}, strength is measured from the public 17lands game data with a hierarchical logistic regression. It controls for player skill, deck colours, play or draw and mulligans, so a card does not look good just because good players pick it. Cards with few games are pulled toward the average for their rarity and colour, and every number has a standard error.</p>
    <h2>Predicted strength (the model)</h2>
    <p>For cards without data, a model reads the card itself: mana cost, type, power and toughness, keywords, rules text and rarity, plus a few numbers about the set it lives in. The default model is gradient boosting over 167 hand-built features and the text. When the fine-tuned EmbeddingGemma-2 weights are installed, custom cards are scored by the average of both.
    ${META.ft_active ? "<b>The fine-tuned encoder is active.</b>" : `<span class="muted">Fine-tuned encoder: ${esc(META.ft_status)}.</span>`}</p>
    <p>On the 7 newest sets, which the model never saw, here is how well each predictor ranks the cards of a set (Spearman correlation with measured strength, 1 is perfect), how well it finds the top 20 (NDCG@20), and its average error:</p>
    <table><thead><tr><th>Predictor</th><th class="num">Ranking</th><th class="num">Top 20</th><th class="num">Error (pts)</th></tr></thead><tbody>
      ${row("Model ensemble", m.ensemble, "shown on new sets")}
      ${row("Gradient boosting + text", m.gbt_tfidf, "used for custom cards")}
      ${row("Fine-tuned EmbeddingGemma-2", m.ft_gemma2)}
      ${row("Draftsim pre-release ratings", m.draftsim, "human experts")}
      ${row("First-week pick order", m.ata_week1, "17lands drafters")}
      ${row("Rarity only", m.rarity_only)}
    </tbody></table>
    <p>The model beats expert pre-release ratings but not the drafting crowd after a week of play. The uncertainty ranges are calibrated: about 80% of cards land inside their 80% range.</p>
    <h2>Grades</h2>
    <div class="grade-scale">${GRADES.map((g, i) => `<span title="${g}">${gradeBadge(g)}<span class="small muted"> ${GRADE_CUTS[i]}+</span></span>`).join(" ")}</div>
    <p>A grade is a percentile within the set. Grades from 17lands data rank the measured strength; grades from the model rank the model's estimates of that set's cards, so a custom card is compared with how the model rates the real cards.</p>
    <h2>Other formats</h2>
    <p>Commander uses EDHREC inclusion rates, and Constructed uses MTGGoldfish staple lists. Both have their own models trained on cards from the Limited sets, so they are rough for older cards. Limited strength says almost nothing about either format.</p>
    <h2>Limits</h2>
    <ul>
      <li>The model only knows what a card says, not new mechanics it has never seen, and it cannot see the rest of a set beyond a few summary numbers.</li>
      <li>Measured strength is "games in hand", which slightly favours cards that show up in long games.</li>
      <li>Card text is Scryfall's current wording.</li>
    </ul>
    <p class="muted small">Data built ${esc(META.built)} · ${META.n_labelled.toLocaleString()} labelled cards · ${META.n_scryfall.toLocaleString()} searchable cards.</p>
  </div>`;
}

// ------------------------------------------------------------------ start
(async function start() {
  META = await api("/api/meta");
  $("#modelStatus").textContent = META.ft_active ? "Model: fine-tuned encoder + GBT" : "Model: gradient boosting";
  attachSearch($("#topSearch"));
  window.addEventListener("hashchange", route);
  route();
})();
