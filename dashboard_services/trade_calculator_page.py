# app.py (or trade_calculator_page.py)
from flask import Flask, request, jsonify, render_template_string

app = Flask(__name__)

CURRENT_SEASON = 2025


def build_trade_calculator_body(league_id: str, season: int) -> str:
    return f"""
    <div class="page-layout">
      <main class="page-main">

        <!-- Hidden fields used by JS to load players -->
        <input type="hidden" id="leagueIdInput" value="{league_id}">
        <input type="hidden" id="seasonInput"   value="{season}">

        <div class="card">
          <div class="card-header-row">
            <h2>Trade Calculator</h2>
          </div>
          <div class="card-body main-two-col">
            <!-- Side 1 -->
            <div class="side-card card">
              <div class="side-header"><h2>Team 1 gets…</h2></div>
              <div class="side-body">
                <h3 for="sideASearch">Add player</h3>
                <div class="search-wrapper">
                  <input id="sideASearch"
                         class="search-input"
                         type="text"
                         autocomplete="off"
                         placeholder="Start typing a name..." />
                  <div id="sideADropdown" class="dropdown" style="display:none;"></div>
                </div>
                <div class="chips" id="sideAChips"></div>
              </div>
            </div>

            <!-- Side 2 -->
            <div class="side-card card">
              <div class="side-header"><h2>Team 2 gets…</h2></div>
              <div class="side-body">
                <h3 for="sideBSearch">Add player</h3>
                <div class="search-wrapper">
                  <input id="sideBSearch"
                         class="search-input"
                         type="text"
                         autocomplete="off"
                         placeholder="Start typing a name..." />
                  <div id="sideBDropdown" class="dropdown" style="display:none;"></div>
                </div>
                <div class="chips" id="sideBChips"></div>
              </div>
            </div>
          </div>
          <div class="card trade-summary-card">
          <div class="card-header-row">
            <h2>Trade Summary</h2>
          </div>
          <div class="card-body">
            <div class="trade-totals-row">
              <div class="trade-total">
                <span class="label">Team 1 total</span>
                <span class="value" id="sideATotal">0.0</span>
              </div>
              <div class="trade-total">
                <span class="label">Difference</span>
                <span class="value" id="tradeDiff">0.0</span>
              </div>
              <div class="trade-total">
                <span class="label">Team 2 total</span>
                <span class="value" id="sideBTotal">0.0</span>
              </div>
            </div>

            <div class="trade-bar">
              <div class="trade-bar-track">
                <div class="trade-bar-fair-zone"></div>
                <div class="trade-bar-indicator" id="tradeBarIndicator"></div>
              </div>
              <div class="trade-bar-scale">
                <span>Team 1 favored</span>
                <span>Fair range</span>
                <span>Team 2 favored</span>
              </div>
            </div>

            <div id="tradeVerdict" class="trade-verdict">Add players to both sides to see the trade balance.</div>
            <div id="errorBox" class="error" style="display:none;"></div>
          </div>
        </div>
        </div>


      </main>

      <aside class="page-sidebar">
        <div class="card small">
          <div class="card-header">
            <h3>How this works</h3>
          </div>
          <div class="card-body">
            <p class="hint">
              Players are pulled from your Sleeper league and given a value score based on usage and projections.
              Each side’s total updates automatically as you add or remove players.
              The bar below shows which side is favored and how close the deal is to “fair”.
            </p>
          </div>
        </div>
      </aside>
    </div>

    <script>
    (function() {{
      // --- State ---
      let allPlayers = [];
      let sideASelected = [];
      let sideBSelected = [];

      async function ensurePlayersLoaded() {{
        if (allPlayers.length > 0) return;

        const leagueInput = document.getElementById("leagueIdInput");
        const seasonInput = document.getElementById("seasonInput");
        const errorBox    = document.getElementById("errorBox");

        if (!leagueInput) {{
          throw new Error("leagueIdInput element not found in DOM.");
        }}

        const leagueId = (leagueInput.value || "").trim();
        const season   = (seasonInput && seasonInput.value ? seasonInput.value.trim() : "");

        if (!leagueId) {{
          throw new Error("Enter a league ID before searching players.");
        }}

        const params = new URLSearchParams({{ league_id: leagueId }});
        if (season) params.set("season", season);

        const res = await fetch("/api/league-players?" + params.toString());
        if (!res.ok) {{
          throw new Error("Failed to load players (" + res.status + ").");
        }}
        allPlayers = await res.json();
        console.log("[trade] loaded players:", allPlayers.length);

        if (errorBox) {{
          errorBox.style.display = "none";
          errorBox.textContent = "";
        }}
      }}

      function formatValue(v) {{
        const num = Number(v) || 0;
        return num.toFixed(1);
      }}

      function recomputeTrade() {{
        const sideATotalEl = document.getElementById("sideATotal");
        const sideBTotalEl = document.getElementById("sideBTotal");
        const tradeDiffEl  = document.getElementById("tradeDiff");
        const verdictEl    = document.getElementById("tradeVerdict");
        const barIndicator = document.getElementById("tradeBarIndicator");

        const sumA = sideASelected.reduce((acc, p) => acc + (Number(p.value) || 0), 0);
        const sumB = sideBSelected.reduce((acc, p) => acc + (Number(p.value) || 0), 0);
        const diff = sumA - sumB;

        if (sideATotalEl) sideATotalEl.textContent = formatValue(sumA);
        if (sideBTotalEl) sideBTotalEl.textContent = formatValue(sumB);
        if (tradeDiffEl)  tradeDiffEl.textContent  = formatValue(diff);

        // Bar position: -range (team 2) to +range (team 1)
        const magnitude = Math.max(Math.abs(diff), 10);
        const range = magnitude * 1.5;  // widen a bit so small diffs look subtle
        let pct = (diff / range + 1) / 2; // map [-range, +range] -> [0,1]
        pct = Math.min(1, Math.max(0, pct));
        const leftPct = pct * 100;

        if (barIndicator) {{
          barIndicator.style.left = leftPct + "%";
        }}

        // Verdict text
        if (!verdictEl) return;

        const absDiff = Math.abs(diff);
        if (sumA === 0 && sumB === 0) {{
          verdictEl.textContent = "Add players to both sides to see the trade balance.";
          verdictEl.className = "trade-verdict";
          return;
        }}

        if (absDiff < 5) {{
          verdictEl.textContent = "This trade looks very fair.";
          verdictEl.className = "trade-verdict verdict-fair";
        }} else if (diff > 0) {{
          verdictEl.textContent = "Team 1 is favored by about " + formatValue(absDiff) + " value.";
          verdictEl.className = "trade-verdict verdict-side-a";
        }} else {{
          verdictEl.textContent = "Team 2 is favored by about " + formatValue(absDiff) + " value.";
          verdictEl.className = "trade-verdict verdict-side-b";
        }}
      }}

      function renderChips(side) {{
        const container = document.getElementById(side === "A" ? "sideAChips" : "sideBChips");
        const selected  = side === "A" ? sideASelected : sideBSelected;
        if (!container) return;

        container.innerHTML = "";

        selected.forEach((p, idx) => {{
          const chip = document.createElement("div");
          chip.className = "chip";

          const label = document.createElement("span");
          const metaBits = [];
          if (p.team) metaBits.push(p.team);
          if (p.position) metaBits.push(p.position);
          if (p.age != null) metaBits.push(p.age + " yrs");
          label.textContent = p.name + (metaBits.length ? " — " + metaBits.join(" · ") : "");

          const valueSpan = document.createElement("span");
          valueSpan.className = "chip-value";
          valueSpan.textContent = " · " + formatValue(p.value);

          const btn = document.createElement("button");
          btn.type = "button";
          btn.className = "chip-remove";
          btn.textContent = "×";
          btn.onclick = () => {{
            selected.splice(idx, 1);
            renderChips(side);
          }};

          chip.appendChild(label);
          chip.appendChild(valueSpan);
          chip.appendChild(btn);
          container.appendChild(chip);
        }});

        // Recompute totals & bar every time chips change
        recomputeTrade();
      }}

      function setupSearch(side) {{
        const input    = document.getElementById(side === "A" ? "sideASearch" : "sideBSearch");
        const dropdown = document.getElementById(side === "A" ? "sideADropdown" : "sideBDropdown");
        const errorBox = document.getElementById("errorBox");
        if (!input || !dropdown) return;

        input.addEventListener("input", async function () {{
          const query = input.value.trim().toLowerCase();
          dropdown.innerHTML = "";
          dropdown.style.display = "none";
          dropdown.parentElement.classList.remove("dropdown-open");
          if (!query) return;

          try {{
            await ensurePlayersLoaded();
          }} catch (err) {{
            console.error(err);
            if (errorBox) {{
              errorBox.style.display = "block";
              errorBox.textContent = err.message || "Failed to load players.";
            }}
            return;
          }}

          const matches = allPlayers
            .filter(p => p.name && p.name.toLowerCase().includes(query))
            .slice(0, 20);

          if (!matches.length) return;

          matches.forEach(p => {{
            const item = document.createElement("div");
            item.className = "dropdown-item";
            const meta = [];
            if (p.position) meta.push(p.position);
            if (p.team) meta.push(p.team);
            if (p.age != null) meta.push(p.age + " yrs");
            item.textContent = p.name + (meta.length ? " — " + meta.join(" · ") : "");
            item.onclick = () => {{
              const selected = side === "A" ? sideASelected : sideBSelected;
              if (!selected.find(x => x.id === p.id)) {{
                selected.push(p);
                renderChips(side);
              }}
              input.value = "";
              dropdown.style.display = "none";
              dropdown.parentElement.classList.remove("dropdown-open");
            }};
            dropdown.appendChild(item);
          }});

          dropdown.style.display = "block";
          dropdown.parentElement.classList.add("dropdown-open");
        }});

        input.addEventListener("blur", function () {{
          setTimeout(() => {{
            dropdown.style.display = "none";
            dropdown.parentElement.classList.remove("dropdown-open");
          }}, 150);
        }});
      }}

      function initTradeCalculator() {{
        setupSearch("A");
        setupSearch("B");
        recomputeTrade();
      }}

      if (document.readyState === "loading") {{
        document.addEventListener("DOMContentLoaded", initTradeCalculator);
      }} else {{
        initTradeCalculator();
      }}
    }})();
    </script>
    """
