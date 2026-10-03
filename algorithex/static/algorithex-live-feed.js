/**
 * Algorithex live market feed.
 *
 * Mounts a read-only market panel into the dashboard's Live page.
 *
 * Why this is injected rather than built into the app
 * --------------------------------------------------
 * The frontend ships as a compiled Nuxt bundle: 506 hashed chunks and no
 * source, so there is no Live page component to edit. It also routes in hash
 * mode (`/#/live/overview`), which means the server never sees the route at all
 * and cannot serve a different page for it. Injecting one script into
 * index.html is the only seam that survives both facts.
 *
 * What it does and does not touch
 * ------------------------------
 * It adds a panel. It does not remove, reorder or restyle anything the dashboard
 * renders, and it lives in a shadow root so neither side's CSS can reach the
 * other -- the compiled Tailwind stylesheet only contains the utility classes
 * the original build happened to use, so relying on it for a new component
 * would silently produce unstyled output.
 *
 * Everything is wrapped so that a failure here cannot take the dashboard down
 * with it. If anything throws, the panel is removed and the app carries on.
 *
 * Read-only, as everywhere else in this project: public ticker, depth and
 * candles. No order placement, no credentials.
 */
(function () {
  'use strict';

  var HOST_ID = 'ah-live-feed-host';
  var ROUTE_PREFIX = '#/live';
  var REFRESH_MS = 10000;

  // Palette sampled from the running dashboard rather than invented: the page
  // is neutral-950 on the body, neutral-900 panels, neutral-700 borders.
  var C = {
    panel: '#171717',
    border: '#404040',
    text: '#e7e5e4',
    muted: '#a3a3a3',
    accent: '#818cf8',
    up: '#4ade80',
    down: '#f87171'
  };

  var state = {
    exchange: 'binance',
    symbol: 'BTC-USDT',
    interval: '4h',
    timer: null
  };

  // --- helpers -------------------------------------------------------------

  function token() {
    try {
      var raw = localStorage.getItem('main');
      if (raw) {
        var parsed = JSON.parse(raw);
        if (parsed && parsed.authToken) return parsed.authToken;
      }
      return localStorage.getItem('auth_token') || '';
    } catch (e) {
      return '';
    }
  }

  function fmt(n, dp) {
    if (n === null || n === undefined || !isFinite(n)) return '—';
    return Number(n).toLocaleString(undefined, {
      minimumFractionDigits: dp, maximumFractionDigits: dp
    });
  }

  /** Prices span orders of magnitude, so significant digits beat fixed ones. */
  function px(n) {
    if (!isFinite(n)) return '—';
    var a = Math.abs(n);
    return fmt(n, a >= 1000 ? 2 : a >= 1 ? 4 : 8);
  }

  /** A spread is a difference on a large price, not a price, so it is formatted
   *  by significant digits rather than by its own magnitude. */
  function sig(n) {
    if (!isFinite(n)) return '—';
    if (n === 0) return '0';
    return String(Number(Number(n).toPrecision(4)));
  }

  function esc(s) {
    return String(s).replace(/[&<>"]/g, function (c) {
      return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c];
    });
  }

  // --- styles --------------------------------------------------------------

  var CSS = [
    ':host { all: initial; display: block; margin: 0 0 20px; }',
    '*, *::before, *::after { box-sizing: border-box; }',
    '.panel { background:' + C.panel + '; border:1px solid ' + C.border + ';',
    '  border-radius:12px; padding:16px 18px; color:' + C.text + ';',
    '  font:13px/1.5 Inter, "Space Grotesk", system-ui, sans-serif; }',
    '.top { display:flex; flex-wrap:wrap; gap:10px; align-items:center; margin-bottom:14px; }',
    '.title { font-size:14px; font-weight:600; margin:0; }',
    '.chip { font-size:10px; font-weight:700; letter-spacing:.06em; color:' + C.up + ';',
    '  background:rgba(74,222,128,.13); border:1px solid rgba(74,222,128,.3);',
    '  border-radius:999px; padding:2px 8px; }',
    '.spacer { flex:1 1 auto; }',
    'select, input, button { background:#0c0c0c; color:' + C.text + ';',
    '  border:1px solid ' + C.border + '; border-radius:7px; padding:5px 9px;',
    '  font:12px ui-monospace, SFMono-Regular, Menlo, monospace; }',
    'input { width:104px; }',
    'button { cursor:pointer; font-family:inherit; }',
    'button:hover { border-color:' + C.accent + '; }',
    'button[disabled] { opacity:.5; cursor:not-allowed; }',
    '.row { display:flex; flex-wrap:wrap; gap:22px; align-items:flex-end; }',
    '.price { font:600 27px/1 ui-monospace, SFMono-Regular, Menlo, monospace;',
    '  letter-spacing:-.02em; }',
    '.chg { font:600 13px/1 ui-monospace, SFMono-Regular, Menlo, monospace;',
    '  margin-left:8px; }',
    '.stat { min-width:74px; }',
    '.k { font-size:9.5px; color:' + C.muted + '; text-transform:uppercase;',
    '  letter-spacing:.07em; }',
    '.v { font:13px ui-monospace, SFMono-Regular, Menlo, monospace; margin-top:2px; }',
    '.chartbox { position:relative; margin:14px 0 4px; }',
    'svg { width:100%; height:150px; display:block; background:#0c0c0c;',
    '  border:1px solid ' + C.border + '; border-radius:8px; }',
    '.axis { position:absolute; left:9px; font:11px ui-monospace, monospace;',
    '  color:' + C.muted + '; pointer-events:none; }',
    '.axis.hi { top:6px; } .axis.lo { bottom:6px; }',
    '.book { display:grid; grid-template-columns:1fr 1fr; gap:18px; margin-top:12px; }',
    '.book h4 { margin:0 0 5px; font-size:9.5px; color:' + C.muted + ';',
    '  text-transform:uppercase; letter-spacing:.07em; font-weight:600; }',
    '.lvl { position:relative; display:flex; justify-content:space-between;',
    '  padding:1px 5px; border-radius:3px; overflow:hidden;',
    '  font:11px ui-monospace, SFMono-Regular, Menlo, monospace; }',
    '.lvl .bar { position:absolute; top:0; bottom:0; right:0; opacity:.15; }',
    '.lvl.bid .bar { background:' + C.up + '; }',
    '.lvl.ask .bar { background:' + C.down + '; }',
    '.lvl span { position:relative; }',
    '.mid { display:flex; justify-content:space-between; margin-top:10px;',
    '  padding-top:9px; border-top:1px solid ' + C.border + ';',
    '  font:11px ui-monospace, monospace; color:' + C.muted + '; }',
    '.err { margin-top:12px; padding:9px 11px; border-radius:7px;',
    '  background:rgba(248,113,113,.1); border:1px solid rgba(248,113,113,.3);',
    '  color:' + C.down + '; font:12px ui-monospace, monospace; white-space:pre-wrap; }',
    '.foot { margin-top:11px; font-size:10.5px; color:' + C.muted + '; }'
  ].join('\n');

  // --- shell ---------------------------------------------------------------

  function buildShell() {
    var host = document.createElement('div');
    host.id = HOST_ID;
    var root = host.attachShadow({ mode: 'open' });

    var style = document.createElement('style');
    style.textContent = CSS;
    root.appendChild(style);

    var panel = document.createElement('div');
    panel.className = 'panel';
    panel.innerHTML = [
      '<div class="top">',
      '  <h3 class="title">Live Market</h3>',
      '  <span class="chip">READ ONLY</span>',
      '  <span class="spacer"></span>',
      '  <select id="ex"><option value="binance">Binance</option>',
      '    <option value="bybit">Bybit</option></select>',
      '  <input id="sym" value="BTC-USDT" spellcheck="false" aria-label="Symbol">',
      '  <select id="tf" aria-label="Timeframe">',
      '    <option>1m</option><option>5m</option><option>15m</option>',
      '    <option>1h</option><option selected>4h</option>',
      '    <option>1d</option><option>1w</option></select>',
      '  <button id="go">Refresh</button>',
      '  <button id="auto">Auto: off</button>',
      '</div>',
      '<div class="row" id="head">',
      '  <div><div class="v price" id="last">—</div>',
      '    <div class="k" id="pair">waiting</div></div>',
      '  <div class="stat"><div class="k">Bid</div><div class="v" id="bid">—</div></div>',
      '  <div class="stat"><div class="k">Ask</div><div class="v" id="ask">—</div></div>',
      '  <div class="stat"><div class="k">Spread</div><div class="v" id="spr">—</div></div>',
      '  <div class="stat"><div class="k">Spread bps</div><div class="v" id="bps">—</div></div>',
      '  <div class="stat"><div class="k">24h vol</div><div class="v" id="vol">—</div></div>',
      '</div>',
      '<div class="chartbox">',
      '  <svg id="chart" viewBox="0 0 1000 150" preserveAspectRatio="none"></svg>',
      '  <div class="axis hi" id="hi"></div><div class="axis lo" id="lo"></div>',
      '</div>',
      '<div class="book">',
      '  <div><h4>Bids</h4><div id="bids"></div></div>',
      '  <div><h4>Asks</h4><div id="asks"></div></div>',
      '</div>',
      '<div class="mid" id="midrow"><span>—</span><span></span></div>',
      '<div id="errbox"></div>',
      '<div class="foot" id="foot"></div>'
    ].join('');
    root.appendChild(panel);

    var $ = function (id) { return root.getElementById(id); };
    $('ex').onchange = function () { state.exchange = this.value; load(); };
    $('tf').onchange = function () { state.interval = this.value; load(); };
    $('sym').onkeydown = function (e) {
      if (e.key === 'Enter') { state.symbol = this.value.trim(); load(); }
    };
    $('go').onclick = function () { load(); };
    $('auto').onclick = function () {
      var b = this;
      if (state.timer) {
        clearInterval(state.timer); state.timer = null;
        b.textContent = 'Auto: off';
      } else {
        load();
        state.timer = setInterval(load, REFRESH_MS);
        b.textContent = 'Auto: ' + REFRESH_MS / 1000 + 's';
      }
    };

    return { host: host, root: root, $: $ };
  }

  // --- rendering -----------------------------------------------------------

  function drawChart(root, closes) {
    var $ = function (id) { return root.getElementById(id); };
    var svg = $('chart');
    if (!closes || closes.length < 2) {
      svg.innerHTML = ''; $('hi').textContent = ''; $('lo').textContent = '';
      return;
    }
    var W = 1000, H = 150, P = 8;
    var lo = Math.min.apply(null, closes), hi = Math.max.apply(null, closes);
    var span = (hi - lo) || (hi * 0.0001) || 1;
    var rising = closes[closes.length - 1] >= closes[0];
    var stroke = rising ? C.up : C.down;
    var line = '';
    for (var i = 0; i < closes.length; i++) {
      var x = P + (i * (W - 2 * P)) / (closes.length - 1);
      var y = H - P - ((closes[i] - lo) / span) * (H - 2 * P);
      line += (i ? 'L' : 'M') + x.toFixed(1) + ' ' + y.toFixed(1) + ' ';
    }
    var area = line + 'L' + (W - P) + ' ' + (H - P) + ' L' + P + ' ' + (H - P) + ' Z';
    var gid = 'ah' + Math.random().toString(36).slice(2, 8);
    svg.innerHTML =
      '<defs><linearGradient id="' + gid + '" x1="0" y1="0" x2="0" y2="1">' +
      '<stop offset="0%" stop-color="' + stroke + '" stop-opacity="0.3"/>' +
      '<stop offset="100%" stop-color="' + stroke + '" stop-opacity="0"/>' +
      '</linearGradient></defs>' +
      '<path d="' + area + '" fill="url(#' + gid + ')"/>' +
      '<path d="' + line + '" fill="none" stroke="' + stroke +
      '" stroke-width="1.6" vector-effect="non-scaling-stroke"/>';
    // Labels sit outside the SVG: preserveAspectRatio="none" stretches the
    // geometry horizontally and would squash any text drawn inside it.
    $('hi').textContent = px(hi);
    $('lo').textContent = px(lo);
  }

  function drawBook(root, id, levels, kind) {
    var el = root.getElementById(id);
    if (!levels.length) { el.innerHTML = '<div class="lvl"><span>empty</span></div>'; return; }
    var max = Math.max.apply(null, levels.map(function (l) { return l.quantity; })) || 1;
    var cum = 0;
    el.innerHTML = levels.slice(0, 9).map(function (l) {
      cum += l.quantity;
      return '<div class="lvl ' + kind + '">' +
        '<span class="bar" style="width:' + ((l.quantity / max) * 100).toFixed(1) + '%"></span>' +
        '<span>' + px(l.price) + '</span>' +
        '<span style="color:' + C.muted + '">' + fmt(l.quantity, 4) + '</span>' +
        '<span style="color:' + C.muted + '">' + fmt(cum, 3) + '</span></div>';
    }).join('');
  }

  function render(root, d) {
    var $ = function (id) { return root.getElementById(id); };
    var t = d.ticker, b = d.order_book, c = d.candles;
    var dir = t.change_24h_pct >= 0 ? C.up : C.down;

    $('pair').textContent = d.exchange.toUpperCase() + ' · ' + d.symbol + ' · ' + d.interval;
    $('last').textContent = px(t.last);
    $('last').style.color = dir;
    $('bid').textContent = px(t.bid);
    $('ask').textContent = px(t.ask);
    $('spr').textContent = sig(t.spread);
    $('bps').textContent = fmt(t.spread_bps, 4);
    $('vol').textContent = fmt(t.quote_volume_24h, 0);

    drawChart(root, c.closes);
    drawBook(root, 'bids', b.bids, 'bid');
    drawBook(root, 'asks', b.asks, 'ask');

    $('midrow').innerHTML =
      '<span>mid ' + px(b.mid) + ' · spread ' + sig(b.spread) + '</span>' +
      '<span style="color:' + (b.crossed ? C.down : C.muted) + '">' +
      (b.crossed ? 'CROSSED' : b.depth + ' levels each side') + '</span>';

    var L = d.latency_ms;
    $('foot').textContent =
      'Read-only public market data · ' + c.count + ' ' + d.interval +
      ' candles, chronological=' + c.chronological + ' · fetched ' +
      new Date(d.fetched_at_ms).toLocaleTimeString() +
      ' · ticker ' + L.ticker + 'ms, book ' + L.order_book + 'ms, candles ' +
      L.candles + 'ms · no keys, no orders';
  }

  function showError(root, message) {
    var box = root.getElementById('errbox');
    box.innerHTML = '<div class="err">' + esc(message) + '</div>';
  }

  // --- data ----------------------------------------------------------------

  function load() {
    var host = document.getElementById(HOST_ID);
    if (!host || host.__loading) return;
    var root = host.shadowRoot;
    if (!root) return;

    var auth = token();
    if (!auth) {
      showError(root, 'Not signed in. Sign in through the dashboard, then reload.');
      return;
    }

    // The flag lives on the host element, not in module state. The panel lives
    // inside a subtree Nuxt replaces on navigation, so a module-level flag
    // outlives the host it was guarding: once Nuxt swaps the panel out, a stuck
    // flag would starve every replacement that follows.
    host.__loading = true;
    var btn = root.getElementById('go');
    if (btn) btn.disabled = true;

    fetch('/live/market', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', 'Authorization': auth },
      body: JSON.stringify({
        exchange: state.exchange, symbol: state.symbol,
        interval: state.interval, candles: 120
      })
    }).then(function (res) {
      return res.text().then(function (text) {
        var data;
        try { data = JSON.parse(text); }
        catch (e) { throw new Error('HTTP ' + res.status + ': ' + text.slice(0, 200)); }
        if (!res.ok) throw new Error(data.detail || ('HTTP ' + res.status));
        return data;
      });
    }).then(function (d) {
      // Write into whichever panel is mounted *now*. Nuxt may have replaced the
      // one this request started from, and rendering into a detached shadow
      // root is how the data goes missing while the network log says 200.
      write(function (liveRoot) {
        render(liveRoot, d);
        liveRoot.getElementById('errbox').innerHTML = '';
      });
    }).catch(function (e) {
      write(function (liveRoot) {
        showError(liveRoot, e.message || String(e));
      });
    }).then(finish, finish);

    function finish() {
      // Runs on success and on failure, so a throw inside render or inside the
      // error handler cannot wedge the loading flag and kill all future loads.
      host.__loading = false;
      var live = document.getElementById(HOST_ID);
      if (!live || !live.shadowRoot) return;
      var go = live.shadowRoot.getElementById('go');
      if (go) go.disabled = false;
      // The panel on screen may be a replacement that never issued this
      // request; fill it rather than leaving an empty frame.
      var last = live.shadowRoot.getElementById('last');
      if (last && last.textContent === '—') load();
    }
  }

  /** Run `fn` against the shadow root of the currently mounted panel, if any. */
  function write(fn) {
    var live = document.getElementById(HOST_ID);
    if (!live || !live.shadowRoot) return;
    try {
      fn(live.shadowRoot);
    } catch (e) {
      // A rendering fault must not become an unhandled rejection.
    }
  }

  // --- mounting ------------------------------------------------------------

  function onLiveRoute() {
    return (location.hash || '').indexOf(ROUTE_PREFIX) === 0;
  }

  function mount() {
    if (!onLiveRoute()) { unmount(); return; }
    if (document.getElementById(HOST_ID)) return;

    var main = document.querySelector('main');
    if (!main) return;
    var shell = buildShell();
    // Prepend into the page wrapper so the panel sits above the session tabs
    // rather than at the very top of the scroll area.
    var target = main.firstElementChild || main;
    target.insertBefore(shell.host, target.firstChild);
    load();
  }

  function unmount() {
    var host = document.getElementById(HOST_ID);
    if (host) host.remove();
  }

  function safeMount() {
    // A throw in here must not reach the dashboard's own error handling.
    try { mount(); } catch (e) { unmount(); }
  }

  // Nuxt re-renders on navigation without a full page load, so hashchange alone
  // is not enough: the mount target is replaced. The observer catches that.
  window.addEventListener('hashchange', safeMount);
  var observer = new MutationObserver(function () { safeMount(); });
  observer.observe(document.documentElement, { childList: true, subtree: true });
  window.addEventListener('load', safeMount);
  safeMount();
})();
