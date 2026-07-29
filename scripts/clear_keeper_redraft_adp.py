"""One-time cleanup: remove keeper-sourced 'redraft' draft-ADP data.

Before the classification fix, keeper leagues (Sleeper league type 1) were
labeled ``draft_type = 'redraft'``. In a keeper league most veterans are kept,
so the draft is mostly rookies + replacements and a top rookie (e.g. Jeremiah
Love) goes ~1.01 — which badly skewed "redraft" ADP.

This deletes every redraft-typed row from all three draft-ADP tables so the
aggregate can't be rebuilt from the bad picks. After running it, re-run league
discovery (which now also finds true-redraft, type-0, leagues) and the draft
crawler; real redraft ADP will repopulate from type-0 leagues only.

Usage:
    python -m scripts.clear_keeper_redraft_adp          # delete
    python -m scripts.clear_keeper_redraft_adp --dry-run  # count only
"""
import sys

from dashboard_services.db import get_conn


def main(dry_run: bool = False) -> int:
    with get_conn() as conn:
        n_agg = conn.execute(
            "SELECT COUNT(*) AS n FROM draft_adp WHERE draft_type = 'redraft'"
        ).fetchone()["n"]
        n_drafts = conn.execute(
            "SELECT COUNT(*) AS n FROM draft_adp_drafts WHERE draft_type = 'redraft'"
        ).fetchone()["n"]

        if dry_run:
            print(f"[dry-run] would delete {n_agg} draft_adp + {n_drafts} "
                  f"draft_adp_drafts (and their picks) redraft rows")
            return 0

        # Picks first (they hang off the drafts), then the drafts, then the
        # aggregate — so compute_adp() can't rebuild redraft from stale picks.
        conn.execute(
            "DELETE FROM draft_adp_picks WHERE draft_id IN "
            "(SELECT draft_id FROM draft_adp_drafts WHERE draft_type = 'redraft')"
        )
        conn.execute("DELETE FROM draft_adp_drafts WHERE draft_type = 'redraft'")
        conn.execute("DELETE FROM draft_adp WHERE draft_type = 'redraft'")

    print(f"Cleared {n_agg} draft_adp + {n_drafts} draft_adp_drafts redraft rows "
          f"(plus their picks). Re-run discovery + the draft crawl to repopulate "
          f"from true redraft (type-0) leagues.")
    return 0


if __name__ == "__main__":
    sys.exit(main(dry_run="--dry-run" in sys.argv))
