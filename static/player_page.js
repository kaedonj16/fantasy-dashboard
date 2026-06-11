/* Per-player trade-value page: hydrate the value chart and recent trades.
   Mirrors the in-app player modal's overview chart, but standalone. */
(function () {
  "use strict";

  function formatDateLabel(dateStr) {
    if (!dateStr) return "";
    var m = String(dateStr).match(/^(\d{4})-(\d{2})-(\d{2})/);
    if (!m) return "";
    var months = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"];
    return months[parseInt(m[2], 10) - 1] + " " + parseInt(m[3], 10);
  }

  function renderChart() {
    var hist = window.__ppHistory || [];
    var div = document.getElementById("ppValueChart");
    if (!div || typeof Plotly === "undefined" || hist.length < 2) return;

    var xData = hist.map(function (d) { return formatDateLabel(d.as_of_date); });
    var n = xData.length;
    var y1qb = hist.map(function (d) { return d.value_1qb != null ? d.value_1qb : null; });
    var ysf  = hist.map(function (d) { return d.value_sf != null ? d.value_sf : d.value_1qb; });
    var hasDual = y1qb.some(function (v, i) { return Math.abs((v || 0) - (ysf[i] || 0)) > 1; });

    var muted = getComputedStyle(document.documentElement)
      .getPropertyValue("--text-muted").trim() || "#6b7280";

    var midIdx = Math.floor((n - 1) / 2);
    var tickvals = n <= 2 ? [xData[0], xData[n - 1]] : [xData[0], xData[midIdx], xData[n - 1]];
    var ticktext = n <= 2
      ? [formatDateLabel(hist[0].as_of_date), formatDateLabel(hist[n - 1].as_of_date)]
      : [formatDateLabel(hist[0].as_of_date), formatDateLabel(hist[midIdx].as_of_date), formatDateLabel(hist[n - 1].as_of_date)];

    var trace1qb = {
      x: xData, y: y1qb, type: "scatter", mode: "lines", name: hasDual ? "1QB" : "Value",
      line: { color: "#3b82f6", width: 2, shape: "spline", smoothing: 1.2 },
      fill: hasDual ? "none" : "tozeroy", fillcolor: "rgba(59,130,246,0.1)",
      hovertemplate: "%{x}<br>" + (hasDual ? "1QB" : "Value") + ": %{y:.0f}<extra></extra>"
    };
    var traceSF = {
      x: xData, y: ysf, type: "scatter", mode: "lines", name: "SF",
      line: { color: "#f59e0b", width: 2, shape: "spline", smoothing: 1.2 }, fill: "none",
      hovertemplate: "%{x}<br>SF: %{y:.0f}<extra></extra>"
    };

    var isMobile = window.innerWidth <= 768;
    var layout = {
      margin: { l: 34, r: 18, t: 10, b: 34 },
      height: isMobile ? 210 : 250,
      paper_bgcolor: "transparent", plot_bgcolor: "transparent",
      showlegend: hasDual,
      legend: { orientation: "h", x: 0.5, xanchor: "center", y: 1.12, font: { size: 11, color: muted } },
      xaxis: { showgrid: false, type: "category", tickmode: "array", tickvals: tickvals, ticktext: ticktext,
               tickfont: { size: 11, color: muted }, fixedrange: true },
      yaxis: { showgrid: true, gridcolor: "rgba(128,128,128,0.12)", tickfont: { size: 11, color: muted },
               fixedrange: true, zeroline: false }
    };
    Plotly.newPlot(div, hasDual ? [trace1qb, traceSF] : [trace1qb], layout,
      { displayModeBar: false, responsive: true });
  }

  function escapeHtml(s) {
    return String(s).replace(/[&<>"']/g, function (c) {
      return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c];
    });
  }

  // Mirrors _renderTITrades() in the Trade Intelligence modal so trades look
  // identical across the site (uses the shared .ti-trade-* classes).
  function assetHtml(a) {
    if (a.type === "pick") {
      return "<div class='ti-trade-asset pick'>" + escapeHtml(a.name || "") + "</div>";
    }
    var posTag = a.position && a.position !== "?"
      ? " <span style='font-size:11px;opacity:.6;'>" + escapeHtml(a.position) + "</span>" : "";
    var cls = a.is_focus ? "focus" : "other";
    return "<div class='ti-trade-asset " + cls + "'>" + escapeHtml(a.name || "") + posTag + "</div>";
  }

  function renderTrades() {
    var box = document.getElementById("ppRecentTrades");
    if (!box) return;
    var pid = box.getAttribute("data-player-id");
    var season = box.getAttribute("data-season");
    if (!pid) return;

    fetch("/api/trade-intel/player-trades/" + encodeURIComponent(pid) +
          "?season=" + encodeURIComponent(season) + "&limit=8")
      .then(function (r) { return r.json(); })
      .then(function (data) {
        var trades = (data && data.trades) || [];
        if (!trades.length) {
          box.innerHTML = "<div style='padding:6px 0;'>No logged trades for this player yet this season.</div>";
          return;
        }
        box.innerHTML = trades.map(function (t) {
          var sideA = (t.side_a || []).map(assetHtml).join("");
          var sideB = (t.side_b || []).map(assetHtml).join("");
          var fmt = t.is_superflex ? "SF" : t.is_superflex === false ? "1QB" : "";
          var teams = t.num_teams ? t.num_teams + "-team" : "";
          var ctx = [teams, fmt].filter(Boolean).join(" ");
          var meta = [t.date, ctx].filter(Boolean).join(" · ");
          return "<div class='ti-trade-item'>" +
              "<div class='ti-trade-date'>" + escapeHtml(meta) + "</div>" +
              "<div class='ti-trade-sides'>" +
                "<div><div class='ti-trade-side-label'>Side A</div>" + sideA + "</div>" +
                "<div class='ti-trade-arrow'>⇄</div>" +
                "<div><div class='ti-trade-side-label'>Side B</div>" + sideB + "</div>" +
              "</div>" +
            "</div>";
        }).join("");
      })
      .catch(function () {
        box.innerHTML = "<div style='padding:6px 0;'>Could not load recent trades.</div>";
      });
  }

  // "View X in your league" CTA -> multi-platform sign-in modal that opens the
  // player modal in place (no navigation) with the chosen league's context.
  function wireLeagueCta() {
    var btn = document.querySelector(".pp-league-modal-btn");
    if (!btn) return;
    btn.addEventListener("click", function () {
      openSignInModal(btn.getAttribute("data-player-id"), btn.getAttribute("data-player-name") || "");
    });
  }

  function openSignInModal(pid, name) {
    if (document.getElementById("ppSignin")) return;
    var season = window.__ppSeason || new Date().getFullYear();
    var esc = escapeHtml;

    var saved = null;
    try { saved = JSON.parse(localStorage.getItem("saved_viewer") || "null"); } catch (_) {}
    var continueHtml = "";
    if (saved && saved.league_id && saved.platform) {
      continueHtml =
        "<button class='otc-btn otc-btn-primary' id='ppContinue' style='width:100%;margin-bottom:10px;'>" +
          "Continue as " + esc(saved.username || saved.team_name || "your team") +
        "</button>" +
        "<div class='pp-signin-or'>or sign in to a different league</div>";
    }

    var overlay = document.createElement("div");
    overlay.id = "ppSignin";
    overlay.className = "pp-signin-overlay";
    overlay.innerHTML =
      "<div class='pp-signin-box' role='dialog' aria-label='Sign in'>" +
        "<h3 class='pp-signin-title'>View " + esc(name) + " in your league</h3>" +
        "<p class='pp-signin-sub'>Pick your platform to value " + esc(name) + " with your league's settings.</p>" +
        continueHtml +
        "<div class='pp-signin-platforms'>" +
          "<button type='button' class='pp-plat-btn' data-platform='sleeper'>Sleeper</button>" +
          "<button type='button' class='pp-plat-btn' data-platform='espn'>ESPN</button>" +
          "<button type='button' class='pp-plat-btn' data-platform='yahoo'>Yahoo</button>" +
        "</div>" +
        "<div id='ppStep'></div>" +
        "<div class='pp-signin-err' id='ppSigninErr'></div>" +
        "<div class='pp-signin-foot'><button type='button' class='pp-link-btn' id='ppSigninCancel'>Cancel</button></div>" +
      "</div>";
    document.body.appendChild(overlay);

    overlay.addEventListener("click", function (e) { if (e.target === overlay) overlay.remove(); });
    document.getElementById("ppSigninCancel").addEventListener("click", function () { overlay.remove(); });

    var stepEl = document.getElementById("ppStep");
    var errEl = document.getElementById("ppSigninErr");

    function finishSignIn(platform, leagueId, username, teamName) {
      try {
        localStorage.setItem("saved_viewer", JSON.stringify({
          username: username || "", team_name: teamName || "", platform: platform,
          season: season, league_id: leagueId, ts: Date.now()
        }));
      } catch (_) {}
      try {
        fetch("/api/quick-set-viewer", { method: "POST", headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ username: username || teamName || "", team_name: teamName || "" }) });
      } catch (_) {}
      overlay.remove();
      if (typeof openPlayerModal === "function") {
        openPlayerModal(pid, name, { force: true, platform: platform, season: season, leagueId: leagueId });
      }
    }

    if (document.getElementById("ppContinue")) {
      document.getElementById("ppContinue").addEventListener("click", function () {
        finishSignIn(saved.platform, saved.league_id, saved.username, saved.team_name);
      });
    }

    Array.prototype.forEach.call(overlay.querySelectorAll(".pp-plat-btn"), function (b) {
      b.addEventListener("click", function () {
        Array.prototype.forEach.call(overlay.querySelectorAll(".pp-plat-btn"), function (x) { x.classList.remove("active"); });
        b.classList.add("active");
        errEl.textContent = "";
        renderStep(b.getAttribute("data-platform"));
      });
    });

    function renderStep(platform) {
      if (platform === "sleeper") {
        stepEl.innerHTML =
          "<input class='pp-signin-input' id='ppUser' type='text' placeholder='Sleeper username' autocomplete='username'>" +
          "<div id='ppLeagueWrap' style='display:none;'><select class='pp-signin-select' id='ppLeague'></select></div>" +
          "<button type='button' class='otc-btn otc-btn-primary' id='ppGo' style='width:100%;'>Find my leagues</button>";
        var userEl = document.getElementById("ppUser");
        var go = document.getElementById("ppGo");
        var wrap = document.getElementById("ppLeagueWrap");
        var sel = document.getElementById("ppLeague");
        var st = "lookup";
        userEl.focus();
        async function lookup() {
          var u = (userEl.value || "").trim();
          if (!u) { errEl.textContent = "Enter your username."; return; }
          errEl.textContent = ""; go.disabled = true; go.textContent = "Loading…";
          try {
            var res = await fetch("/api/sleeper-user-leagues?username=" + encodeURIComponent(u));
            var data = await res.json();
            if (!res.ok || !data.ok || !(data.leagues || []).length) throw new Error(data.error || "No leagues found for that username.");
            sel.innerHTML = data.leagues.map(function (lg) {
              return "<option value='" + esc(lg.league_id) + "'>" + esc(lg.label || lg.name || lg.league_id) + "</option>";
            }).join("");
            wrap.style.display = "block"; st = "go"; go.textContent = "View " + name;
          } catch (e2) { errEl.textContent = e2.message || "Could not load leagues."; go.textContent = "Find my leagues"; }
          finally { go.disabled = false; }
        }
        go.addEventListener("click", function () {
          if (st === "lookup") lookup();
          else finishSignIn("sleeper", sel.value, (userEl.value || "").trim(), "");
        });
        userEl.addEventListener("keydown", function (e) { if (e.key === "Enter") { e.preventDefault(); if (st === "lookup") lookup(); } });
      } else if (platform === "espn") {
        stepEl.innerHTML =
          "<input class='pp-signin-input' id='ppLid' type='text' placeholder='ESPN League ID' inputmode='numeric'>" +
          "<input class='pp-signin-input' id='ppTeam' type='text' placeholder='Your team name (optional)'>" +
          "<button type='button' class='otc-btn otc-btn-primary' id='ppGo' style='width:100%;'>View " + esc(name) + "</button>";
        var lidEl = document.getElementById("ppLid");
        var teamEl = document.getElementById("ppTeam");
        var go = document.getElementById("ppGo");
        lidEl.focus();
        go.addEventListener("click", async function () {
          var lid = (lidEl.value || "").trim();
          if (!/^\d+$/.test(lid)) { errEl.textContent = "Enter a valid ESPN League ID (numbers only)."; return; }
          errEl.textContent = ""; go.disabled = true; go.textContent = "Loading…";
          try {
            var res = await fetch("/api/espn-validate-league?league_id=" + encodeURIComponent(lid));
            var data = await res.json();
            if (!res.ok || !data.ok) throw new Error(data.error || "Could not load that ESPN league.");
            finishSignIn("espn", lid, (teamEl.value || "").trim(), (teamEl.value || "").trim());
          } catch (e2) { errEl.textContent = e2.message || "Could not load league."; go.disabled = false; go.textContent = "View " + name; }
        });
      } else if (platform === "yahoo") {
        stepEl.innerHTML =
          "<input class='pp-signin-input' id='ppLid' type='text' placeholder='Yahoo League ID' inputmode='numeric'>" +
          "<input class='pp-signin-input' id='ppTeam' type='text' placeholder='Your team name (optional)'>" +
          "<button type='button' class='otc-btn otc-btn-primary' id='ppGo' style='width:100%;'>Connect Yahoo</button>" +
          "<div class='pp-signin-note'>You'll be redirected to Yahoo to authorize, then returned here.</div>";
        var lidEl = document.getElementById("ppLid");
        var teamEl = document.getElementById("ppTeam");
        lidEl.focus();
        document.getElementById("ppGo").addEventListener("click", function () {
          var lid = (lidEl.value || "").trim();
          if (!/^\d+$/.test(lid)) { errEl.textContent = "Enter a valid Yahoo League ID."; return; }
          var next = "/yahoo/" + season + "/" + lid + "/players?player=" +
            encodeURIComponent(pid) + "&player_name=" + encodeURIComponent(name);
          window.location.href = "/auth/yahoo?league_id=" + encodeURIComponent(lid) +
            "&team_name=" + encodeURIComponent((teamEl.value || "").trim()) +
            "&next=" + encodeURIComponent(next);
        });
      }
    }

    var sleeperBtn = overlay.querySelector(".pp-plat-btn[data-platform='sleeper']");
    if (sleeperBtn) sleeperBtn.click();
  }

  function init() { renderChart(); renderTrades(); wireLeagueCta(); }
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
