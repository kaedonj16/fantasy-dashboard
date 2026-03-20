from typing import Optional


def build_trade_calculator_body(league_id: Optional[str], season: Optional[int]) -> str:
    league_val = league_id or ""
    season_val = season if season is not None else ""

    return f"""
    <div class="otc-layout">
      <main class="otc-main">

        <!-- Hidden fields used by JS to load players -->
        <input type="hidden" id="leagueIdInput" value="{league_val}">
        <input type="hidden" id="seasonInput" value="{season_val}">

        <div class="otc-shell">
          <div class="otc-page-head">
            <div class="otc-page-title-wrap">
              <div class="otc-page-kicker">Offseason tool</div>
              <h1 class="otc-page-title">Trade Calculator</h1>
              <p class="otc-page-copy">
                Compare both sides of a deal using BR values, balance, and roster-building context.
              </p>
            </div>
            <div class="otc-page-badge">Dynasty Trade Tool</div>
          </div>

          <div class="otc-builder-grid">
            <!-- Side 1 -->
            <section class="otc-team-card">
              <div class="otc-team-head">
                <h2 class="otc-team-title">Team 1 gets...</h2>
                <div class="otc-team-pill">Side A</div>
              </div>

              <div class="otc-slot">
                <div class="otc-slot-empty">
                  <div class="otc-slot-empty-title">Add player</div>
                  <div class="otc-slot-empty-sub">
                    Search players to build the first side of the trade.
                  </div>
                </div>

                <div class="otc-search-wrap">
                  <div class="search-wrapper">
                    <input id="sideASearch"
                           class="otc-search-input"
                           type="text"
                           autocomplete="off"
                           placeholder="Start typing a name..." />
                    <div id="sideADropdown" class="dropdown otc-search-dropdown" style="display:none;"></div>
                  </div>
                </div>

                <div class="chips" id="sideAChips"></div>
              </div>
            </section>

            <!-- Side 2 -->
            <section class="otc-team-card">
              <div class="otc-team-head">
                <h2 class="otc-team-title">Team 2 gets...</h2>
                <div class="otc-team-pill">Side B</div>
              </div>

              <div class="otc-slot">
                <div class="otc-slot-empty">
                  <div class="otc-slot-empty-title">Add player</div>
                  <div class="otc-slot-empty-sub">
                    Search players to build the second side of the trade.
                  </div>
                </div>

                <div class="otc-search-wrap">
                  <div class="search-wrapper">
                    <input id="sideBSearch"
                           class="otc-search-input"
                           type="text"
                           autocomplete="off"
                           placeholder="Start typing a name..." />
                    <div id="sideBDropdown" class="dropdown otc-search-dropdown" style="display:none;"></div>
                  </div>
                </div>

                <div class="chips" id="sideBChips"></div>
              </div>
            </section>
          </div>

          <section class="otc-summary-card">
            <div class="otc-summary-head">
              <h2 class="otc-summary-title">Trade Summary</h2>
              <div class="otc-summary-sub">Live balance as you add assets</div>
            </div>

            <div class="otc-summary-stats">
              <div class="otc-stat-box">
                <div class="otc-stat-label">Team 1 Total</div>
                <div class="otc-stat-value" id="sideATotal">0.0</div>
              </div>
              <div class="otc-stat-box">
                <div class="otc-stat-label">Difference</div>
                <div class="otc-stat-value" id="tradeDiff">0.0</div>
              </div>
              <div class="otc-stat-box">
                <div class="otc-stat-label">Team 2 Total</div>
                <div class="otc-stat-value" id="sideBTotal">0.0</div>
              </div>
            </div>

            <div class="otc-balance-wrap">
              <div class="otc-balance-bar">
                <div class="otc-balance-fair">FAIR RANGE</div>
                <div class="trade-bar-indicator" id="tradeBarIndicator"></div>
              </div>
              <div class="otc-balance-labels">
                <span>Team 1 favored</span>
                <span>Team 2 favored</span>
              </div>
            </div>

            <div id="tradeVerdict" class="otc-verdict">
              Add players to both sides to see the trade balance.
            </div>
            <div id="errorBox" class="error" style="display:none;"></div>
          </section>
        </div>
      </main>

      <aside class="otc-side">
        <div class="otc-side-panel">
          <div class="otc-side-head">
            <h2 class="otc-side-title">Player Values</h2>
            <div class="otc-side-sub">Filter by position</div>
          </div>

          <div class="otc-filter-row" id="posFilterRow">
            <button class="otc-filter-chip pos-filter is-active" data-pos="ALL">All</button>
            <button class="otc-filter-chip pos-filter" data-pos="QB">QB</button>
            <button class="otc-filter-chip pos-filter" data-pos="RB">RB</button>
            <button class="otc-filter-chip pos-filter" data-pos="WR">WR</button>
            <button class="otc-filter-chip pos-filter" data-pos="TE">TE</button>
          </div>

          <div id="allPlayersList" class="otc-values-list">
            <!-- Filled by JS -->
          </div>
        </div>
      </aside>
    </div>

    <script>
    (function() {{
      let allPlayers = [];
      let sideASelected = [];
      let sideBSelected = [];
      let activePosFilter = "ALL";
      let sortDir = "desc";

      function formatValue(v) {{
        const num = Number(v) || 0;
        return num.toFixed(1);
      }}

      function buildOverallRankMap(players) {{
        const sorted = [...players].sort((a, b) => {{
          const va = typeof a.value === "number" ? a.value : 0;
          const vb = typeof b.value === "number" ? b.value : 0;
          return vb - va;
        }});

        const rankMap = new Map();
        sorted.forEach((p, idx) => {{
          if (p && p.id != null) {{
            rankMap.set(String(p.id), idx + 1);
          }}
        }});
        return rankMap;
      }}

      function buildMetaBits(p) {{
        const metaBits = [];
        if (p.position && String(p.position).toUpperCase() !== "PICK") {{
          if (p.pos_rank_label) metaBits.push(String(p.pos_rank_label).toUpperCase());
        }}
        if (p.team) metaBits.push(p.team);
        if (p.age != null) metaBits.push(p.age + " yrs");
        return metaBits;
      }}

      function buildPlayerValueRow(p, overallRank) {{
        const row = document.createElement("div");
        row.className = "otc-value-row";

        const rankWrap = document.createElement("div");
        rankWrap.className = "otc-value-rank";
        rankWrap.textContent = overallRank ? "#" + overallRank : "—";

        const mainWrap = document.createElement("div");
        mainWrap.className = "otc-value-main";

        const topLine = document.createElement("div");
        topLine.className = "otc-value-topline";

        const nameSpan = document.createElement("div");
        nameSpan.className = "otc-value-name";
        nameSpan.textContent = p.name || "Unknown";

        const valueSpan = document.createElement("div");
        valueSpan.className = "otc-value-score";
        valueSpan.textContent = formatValue(p.value);

        topLine.appendChild(nameSpan);
        topLine.appendChild(valueSpan);

        const metaSpan = document.createElement("div");
        metaSpan.className = "otc-value-sub";
        metaSpan.textContent = buildMetaBits(p).join(" • ");

        mainWrap.appendChild(topLine);
        mainWrap.appendChild(metaSpan);

        row.appendChild(rankWrap);
        row.appendChild(mainWrap);

        return row;
      }}

      function buildDropdownItem(p, overallRank) {{
  const item =
    document.createElement("div");
    item.className = "dropdown-item otc-dropdown-item";

    const
    left = document.createElement("div");
    left.className = "otc-dropdown-left";

    const
    top = document.createElement("div");
    top.className = "otc-dropdown-top";

    const
    rank = document.createElement("span");
    rank.className = "otc-dropdown-rank-inline";
    rank.textContent = overallRank ? "#" + overallRank : "";

  const name = document.createElement("span");
  name.className = "otc-dropdown-name";
  name.textContent = p.name || "Unknown";

  top.appendChild(rank);
  top.appendChild(name);

  const sub = document.createElement("div");
  sub.className = "otc-dropdown-sub";
  sub.textContent = buildMetaBits(p).join(" • ");

  left.appendChild(top);
  left.appendChild(sub);

  const value = document.createElement("div");
  value.className = "otc-dropdown-value";
  value.textContent = formatValue(p.value);

  item.appendChild(left);
  item.appendChild(value);

  return item;
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

        const overallRankMap = buildOverallRankMap(allPlayers);

        let items = allPlayers.filter(p => {{
          if (!p || typeof p !== "object") return false;
          const pos = String(p.position || "").toUpperCase();
          if (activePosFilter === "ALL") return true;
          return pos === activePosFilter;
        }});

        items.sort((a, b) => {{
          const va = typeof a.value === "number" ? a.value : 0;
          const vb = typeof b.value === "number" ? b.value : 0;
          return sortDir === "desc" ? vb - va : va - vb;
        }});

        items.forEach(p => {{
          const overallRank = overallRankMap.get(String(p.id));
          container.appendChild(buildPlayerValueRow(p, overallRank));
        }});
      }}

      function setPosFilter(pos) {{
        activePosFilter = pos;

        document.querySelectorAll(".pos-filter").forEach(btn => {{
          const p = btn.getAttribute("data-pos") || "ALL";
          btn.classList.toggle("is-active", p === activePosFilter);
        }});

        renderAllPlayersList();
      }}

      async function ensurePlayersLoaded() {{
        if (allPlayers.length > 0) return;

        const leagueInput = document.getElementById("leagueIdInput");
        const seasonInput = document.getElementById("seasonInput");
        const errorBox = document.getElementById("errorBox");

        const leagueId = leagueInput ? (leagueInput.value || "").trim() : "";
        const season = (seasonInput && seasonInput.value) ? seasonInput.value.trim() : "";

        const effectiveLeagueId = leagueId || "global";

        const params = new URLSearchParams({{ league_id: effectiveLeagueId }});
        if (season) params.set("season", season);

        const res = await fetch("/api/league-players?" + params.toString());
        if (!res.ok) {{
          throw new Error("Failed to load players (" + res.status + ").");
        }}

        const data = await res.json();
        allPlayers = Array.isArray(data) ? data : [];

        if (errorBox) {{
          errorBox.style.display = "none";
          errorBox.textContent = "";
        }}

        renderAllPlayersList();
      }}

      async function recomputeTrade() {{
        const sideATotalEl = document.getElementById("sideATotal");
        const sideBTotalEl = document.getElementById("sideBTotal");
        const tradeDiffEl = document.getElementById("tradeDiff");
        const verdictEl = document.getElementById("tradeVerdict");
        const barIndicator = document.getElementById("tradeBarIndicator");
        const errorBox = document.getElementById("errorBox");
        const leagueInput = document.getElementById("leagueIdInput");
        const seasonInput = document.getElementById("seasonInput");

        const leagueId = leagueInput ? (leagueInput.value || "").trim() : "";
        const season = seasonInput ? (seasonInput.value || "").trim() : "";

        const sideAIds = sideASelected.map(p => p.id);
        const sideBIds = sideBSelected.map(p => p.id);

        if (sideAIds.length === 0 && sideBIds.length === 0) {{
          if (sideATotalEl) sideATotalEl.textContent = "0.0";
          if (sideBTotalEl) sideBTotalEl.textContent = "0.0";
          if (tradeDiffEl) tradeDiffEl.textContent = "0.0";
          if (barIndicator) barIndicator.style.left = "50%";
          if (verdictEl) {{
            verdictEl.textContent = "Add players to both sides to see the trade balance.";
            verdictEl.className = "otc-verdict";
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
          const diff = Number(data.diff) || 0;
          const aEff = data.side_a ? Number(data.side_a.effective_total) || 0 : 0;
          const bEff = data.side_b ? Number(data.side_b.effective_total) || 0 : 0;

          if (sideATotalEl) sideATotalEl.textContent = formatValue(aEff);
          if (sideBTotalEl) sideBTotalEl.textContent = formatValue(bEff);
          if (tradeDiffEl) tradeDiffEl.textContent = formatValue(diff);

          const maxSideTotal = Math.max(Math.abs(aEff), Math.abs(bEff), 1);
          let normalizedDiff = diff / maxSideTotal;
          normalizedDiff = Math.max(-1, Math.min(1, normalizedDiff));
          let pct = (normalizedDiff + 1) / 2;
          const leftPct = pct * 100;

          if (barIndicator) {{
            barIndicator.style.left = leftPct + "%";
          }}

          if (verdictEl) {{
            verdictEl.textContent = data.verdict || "";
            verdictEl.className = "otc-verdict";
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
        container.className = "otc-selected-list";

        selected.forEach((p, idx) => {{
          const chip = document.createElement("div");
          chip.className = "otc-chip";

          const nameEl = document.createElement("div");
          nameEl.className = "otc-chip-name";
          nameEl.textContent = p.name || "Unknown";

          const metaEl = document.createElement("div");
          metaEl.className = "otc-chip-meta";
          const metaBits = [];
          if (p.pos_rank_label) metaBits.push(p.pos_rank_label);
          if (p.team) metaBits.push(p.team);
          if (p.age != null) metaBits.push(p.age + " yrs");
          metaEl.textContent = metaBits.join(" · ");

          const rightWrap = document.createElement("div");
          rightWrap.className = "otc-chip-value-wrap";

          const valueEl = document.createElement("span");
          valueEl.className = "otc-chip-value";
          valueEl.textContent = formatValue(p.value);

          const removeBtn = document.createElement("button");
          removeBtn.type = "button";
          removeBtn.className = "otc-chip-remove";
          removeBtn.textContent = "×";
          removeBtn.onclick = () => {{
            selected.splice(idx, 1);
            renderChips(side);
          }};

          rightWrap.appendChild(valueEl);
          rightWrap.appendChild(removeBtn);

          chip.appendChild(nameEl);
          chip.appendChild(metaEl);
          chip.appendChild(rightWrap);

          container.appendChild(chip);
        }});

        recomputeTrade();
      }}

      function setupSearch(side) {{
        const input = document.getElementById(side === "A" ? "sideASearch" : "sideBSearch");
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

          const overallRankMap = buildOverallRankMap(allPlayers);

          if (!matches.length) return;

          matches.forEach(p => {{
            const overallRank = overallRankMap.get(String(p.id));
            const item = buildDropdownItem(p, overallRank);

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
