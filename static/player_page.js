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

  function assetText(side, focusFirst) {
    return side.map(function (a) {
      var nm = a.name || "?";
      return a.is_focus ? "<strong style='color:var(--text);'>" + escapeHtml(nm) + "</strong>" : escapeHtml(nm);
    }).join(", ");
  }

  function escapeHtml(s) {
    return String(s).replace(/[&<>"']/g, function (c) {
      return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c];
    });
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
        var rows = trades.map(function (t) {
          var fmt = (t.is_superflex ? "SF" : "1QB") + (t.num_teams ? " &middot; " + t.num_teams + "-team" : "");
          return "" +
            "<div class='pp-trade-row' style='padding:10px 0;border-bottom:1px solid var(--border);'>" +
              "<div style='display:flex;justify-content:space-between;gap:10px;align-items:baseline;'>" +
                "<div style='flex:1;min-width:0;'>" + assetText(t.side_a) + "</div>" +
                "<div style='color:var(--text-muted);flex-shrink:0;'>for</div>" +
                "<div style='flex:1;min-width:0;text-align:right;'>" + assetText(t.side_b) + "</div>" +
              "</div>" +
              "<div style='font-size:11px;color:var(--text-muted);margin-top:3px;'>" +
                escapeHtml(t.date || "") + " &middot; " + fmt +
              "</div>" +
            "</div>";
        }).join("");
        box.innerHTML = rows;
      })
      .catch(function () {
        box.innerHTML = "<div style='padding:6px 0;'>Could not load recent trades.</div>";
      });
  }

  function init() { renderChart(); renderTrades(); }
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
