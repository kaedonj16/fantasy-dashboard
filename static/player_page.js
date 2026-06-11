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

  // "View X in your league" CTA: sign the returning user in (from saved_viewer),
  // then land on their league players page with the modal auto-opening.
  function wireLeagueCta() {
    var btn = document.querySelector(".pp-league-modal-btn");
    if (!btn) return;
    btn.addEventListener("click", async function () {
      var pid = btn.getAttribute("data-player-id");
      var name = btn.getAttribute("data-player-name") || "";
      var q = "player=" + encodeURIComponent(pid) + "&player_name=" + encodeURIComponent(name);
      var saved = null;
      try { saved = JSON.parse(localStorage.getItem("saved_viewer") || "null"); } catch (_) {}

      if (saved && saved.league_id && saved.platform && saved.season) {
        btn.disabled = true;
        try {
          await fetch("/api/quick-set-viewer", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              username:  saved.username,
              roster_id: saved.roster_id || "",
              user_id:   saved.user_id || "",
              team_name: saved.team_name || "",
            }),
          });
        } catch (_) { /* best effort; navigate anyway */ }
        window.location.href = "/" + saved.platform + "/" + saved.season + "/" +
          saved.league_id + "/players?" + q;
      } else {
        // No saved league: open the modal on the guest players page.
        window.location.href = "/players?" + q;
      }
    });
  }

  function init() { renderChart(); renderTrades(); wireLeagueCta(); }
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
