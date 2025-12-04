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

              <div id="tradeVerdict" class="trade-verdict">
                Add players to both sides to see the trade balance.
              </div>
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

        <!-- All players + values side card -->
        <div class="card small">
          <div class="card-header">
            <h3>All Player Values</h3>
          </div>
          <div class="card-body">
            <label class="mini-label">Filter by position</label>
            <div class="pill-row" id="posFilterRow">
              <button class="pill-toggle pos-filter active" data-pos="ALL">All</button>
              <button class="pill-toggle pos-filter" data-pos="QB">QB</button>
              <button class="pill-toggle pos-filter" data-pos="RB">RB</button>
              <button class="pill-toggle pos-filter" data-pos="WR">WR</button>
              <button class="pill-toggle pos-filter" data-pos="TE">TE</button>
            </div>

            <div id="allPlayersList" class="all-players-list">
              <!-- Filled by JS -->
            </div>
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
      let activePosFilter = "ALL";
      let sortDir = "desc";  // value high -> low

      function formatValue(v) {{
        const num = Number(v) || 0;
        return num.toFixed(1);
      }}

      function renderAllPlayersList() {{
        const container = document.getElementById("allPlayersList");
        if (!container) return;
        container.innerHTML = "";

        if (!allPlayers || allPlayers.length === 0) {{
          const empty = document.createElement("p");
          empty.className = "hint";
          empty.textContent = "Players will appear here once loaded.";
          container.appendChild(empty);
          return;
        }}

        // Filter by position for sidebar list
        let items = allPlayers.filter(p => {{
          if (!p || typeof p !== "object") return false;
          const pos = String(p.position || "").toUpperCase();

          // Ignore picks for this sidebar
          if (pos === "PICK") return false;

          if (activePosFilter === "ALL") return true;
          return pos === activePosFilter;
        }});

        // Sort by value
        items.sort((a, b) => {{
          const va = typeof a.value === "number" ? a.value : 0;
          const vb = typeof b.value === "number" ? b.value : 0;
          return sortDir === "desc" ? vb - va : va - vb;
        }});

        items.forEach(p => {{
          const row = document.createElement("div");
          row.className = "all-players-row";

          const leftWrap = document.createElement("div");
          leftWrap.className = "all-players-left";

          const nameSpan = document.createElement("span");
          nameSpan.className = "all-players-name";
          nameSpan.textContent = p.name || "Unknown";
          leftWrap.appendChild(nameSpan);

          const metaSpan = document.createElement("span");
          metaSpan.className = "all-players-meta";
          const metaBits = [];
          if (p.position) metaBits.push(String(p.pos_rank_label).toUpperCase());
          if (p.team) metaBits.push(p.team);
          if (p.age != null) metaBits.push(p.age + " yrs");
          metaSpan.textContent = metaBits.join(" · ");

          const valueSpan = document.createElement("span");
          valueSpan.className = "all-players-value";
          valueSpan.textContent = formatValue(p.value);

          row.appendChild(leftWrap);
          row.appendChild(metaSpan);
          row.appendChild(valueSpan);

          container.appendChild(row);
        }});
      }}

      function setPosFilter(pos) {{
        activePosFilter = pos;

        document.querySelectorAll(".pos-filter").forEach(btn => {{
          const p = btn.getAttribute("data-pos") || "ALL";
          btn.classList.toggle("active", p === activePosFilter);
        }});

        renderAllPlayersList();
      }}

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

        const data = await res.json();      // list of dicts
        console.log("[trade] loaded players payload sample:", data.slice(0, 5));
        allPlayers = Array.isArray(data) ? data : [];

        if (errorBox) {{
          errorBox.style.display = "none";
          errorBox.textContent = "";
        }}

        renderAllPlayersList();
      }}

      // Use backend /api/trade-eval for totals + verdict
      async function recomputeTrade() {{
        const sideATotalEl = document.getElementById("sideATotal");
        const sideBTotalEl = document.getElementById("sideBTotal");
        const tradeDiffEl  = document.getElementById("tradeDiff");
        const verdictEl    = document.getElementById("tradeVerdict");
        const barIndicator = document.getElementById("tradeBarIndicator");
        const errorBox     = document.getElementById("errorBox");
        const leagueInput  = document.getElementById("leagueIdInput");
        const seasonInput  = document.getElementById("seasonInput");

        const leagueId = leagueInput ? (leagueInput.value || "").trim() : "";
        const season   = seasonInput ? (seasonInput.value || "").trim() : "";

        const sideAIds = sideASelected.map(p => p.id);
        const sideBIds = sideBSelected.map(p => p.id);

        // If both sides empty, just reset UI and bail
        if (sideAIds.length === 0 && sideBIds.length === 0) {{
          if (sideATotalEl) sideATotalEl.textContent = "0.0";
          if (sideBTotalEl) sideBTotalEl.textContent = "0.0";
          if (tradeDiffEl)  tradeDiffEl.textContent  = "0.0";
          if (barIndicator) barIndicator.style.left = "50%";
          if (verdictEl) {{
            verdictEl.textContent = "Add players to both sides to see the trade balance.";
            verdictEl.className = "trade-verdict";
          }}
          if (errorBox) {{
            errorBox.style.display = "none";
            errorBox.textContent = "";
          }}
          return;
        }}

        const payload = {{
          league_id: leagueId || "global",
          season: season ? Number(season) : undefined,
          side_a_players: sideAIds,
          side_b_players: sideBIds,
          side_a_picks: [],
          side_b_picks: []
        }};

        try {{
          const res = await fetch("/api/trade-eval", {{
            method: "POST",
            headers: {{ "Content-Type": "application/json" }},
            body: JSON.stringify(payload)
          }});

          if (!res.ok) {{
            throw new Error("Trade eval failed (" + res.status + ").");
          }}

          const data = await res.json();
          const diff    = Number(data.diff) || 0;
          const aEff    = data.side_a ? Number(data.side_a.effective_total) || 0 : 0;
          const bEff    = data.side_b ? Number(data.side_b.effective_total) || 0 : 0;

          if (sideATotalEl) sideATotalEl.textContent = formatValue(aEff);
          if (sideBTotalEl) sideBTotalEl.textContent = formatValue(bEff);
          if (tradeDiffEl)  tradeDiffEl.textContent  = formatValue(diff);

          // Bar movement: normalize diff to -1..+1 range
          const maxSideTotal = Math.max(Math.abs(aEff), Math.abs(bEff), 1);
          let normalizedDiff = diff / maxSideTotal; // -1..+1-ish
          normalizedDiff = Math.max(-1, Math.min(1, normalizedDiff));
          let pct = (normalizedDiff + 1) / 2; // 0..1
          const leftPct = pct * 100;

          if (barIndicator) {{
            barIndicator.style.left = leftPct + "%";
          }}

          if (verdictEl) {{
            verdictEl.textContent = data.verdict || "";
            verdictEl.className = "trade-verdict";
          }}

          if (errorBox) {{
            errorBox.style.display = "none";
            errorBox.textContent = "";
          }}
        }} catch (err) {{
          console.error("[trade] error in recomputeTrade:", err);
          if (errorBox) {{
            errorBox.style.display = "block";
            errorBox.textContent = err.message || "Failed to evaluate trade.";
          }}
        }}
      }}

      function renderChips(side) {{
        const container = document.getElementById(side === "A" ? "sideAChips" : "sideBChips");
        const selected  = side === "A" ? sideASelected : sideBSelected;
        if (!container) return;

        container.innerHTML = "";

        selected.forEach((p, idx) => {{
          const chip = document.createElement("div");
          chip.className = "trade-player-chip";

          const labelName = document.createElement("span");
          const metaBits = [];
          if (p.pos_rank_label) metaBits.push(p.pos_rank_label);
          if (p.team) metaBits.push(p.team);
          if (p.age != null) metaBits.push(p.age + " yrs");
          labelName.textContent = p.name;

          const detailSpan = document.createElement("span");
          detailSpan.textContent = metaBits.join(" · ");

          const valueDiv = document.createElement("div");
          const valueSpan = document.createElement("span");
          valueSpan.className = "chip-value";
          valueSpan.textContent = formatValue(p.value);

          const btn = document.createElement("button");
          btn.type = "button";
          btn.className = "chip-remove";
          btn.textContent = "×";
          btn.onclick = () => {{
            selected.splice(idx, 1);
            renderChips(side);
          }};

          valueDiv.appendChild(valueSpan);
          valueDiv.appendChild(btn);
          valueDiv.style.display = "inline-flex";
          valueDiv.style.gap = "0.25rem";

          chip.appendChild(labelName);
          chip.appendChild(detailSpan);
          chip.appendChild(valueDiv);
          container.appendChild(chip);
        }});

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
            if (p.pos_rank_label) meta.push(p.pos_rank_label);
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

      async function initTradeCalculator() {{
        await ensurePlayersLoaded();
        setupSearch("A");
        setupSearch("B");

        // wire up position filter buttons
        document.querySelectorAll(".pos-filter").forEach(btn => {{
          btn.addEventListener("click", () => {{
            const pos = btn.getAttribute("data-pos") || "ALL";
            setPosFilter(pos);
          }});
        }});

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
