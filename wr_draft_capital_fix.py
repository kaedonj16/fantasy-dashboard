def pick_to_draft_capital_score(pick: int, position: str = "WR") -> float:
    """
    Position-adjusted draft capital score (0-100).

    Updated WR calibration: top 5 = 100, top 15 = 90
    """
    if pick <= 0:
        return 0.0

    pos = (position or "WR").upper()
    
    # Position anchors: (elite, good, avg, late)
    _POS_PICK_ANCHORS = {
        "QB": (1, 8, 22, 64),   # QB top-10 expected every year
        "WR": (5, 15, 40, 96),   # WR top-5 is rare; #15 = 90 (user calibration)
        "RB": (10, 25, 55, 120), # RB top-10 is extraordinary (Bijan-tier)
        "TE": (10, 25, 55, 120), # TE top-10 is rare (Pitts/Hockenson-tier)
    }
    
    elite_p, good_p, avg_p, late_p = _POS_PICK_ANCHORS.get(pos, _POS_PICK_ANCHORS["WR"])

    if pick <= elite_p:
        return 100.0
    elif pick <= good_p:
        if pos == "WR":
            # WR special: 100 at #5, 90 at #15
            t = (pick - elite_p) / (good_p - elite_p)
            return round(100.0 - t * 10.0, 2)   # 100 -> 90 for WR
        else:
            t = (pick - elite_p) / (good_p - elite_p)
            return round(100.0 - t * 15.0, 2)   # 100 -> 85 for other positions
    elif pick <= avg_p:
        # For WR: start from 90 instead of 85
        if pos == "WR":
            t = (pick - good_p) / (avg_p - good_p)
            return round(90.0 - t * 30.0, 2)    # 90 -> 60 for WR
        else:
            t = (pick - good_p) / (avg_p - good_p)
            return round(85.0 - t * 25.0, 2)    # 85 -> 60 for others
    elif pick <= late_p:
        t = (pick - avg_p) / (late_p - avg_p)
        return round(60.0 - t * 38.0, 2)    # 60 -> 22
    elif pick <= 220:
        t = (pick - late_p) / (220 - late_p)
        return round(max(2.0, 22.0 - t * 20.0), 2)  # 22 -> 2
    else:
        return 2.0

# Test the new WR scoring
if __name__ == "__main__":
    print("WR Draft Capital Scores (New Calibration):")
    for pick in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 20, 25, 30, 40]:
        score = pick_to_draft_capital_score(pick, "WR")
        print(f"Pick #{pick:2d}: {score:5.1f}")
    
    print("\nComparison with 10% first-round bonus:")
    for pick in [1, 5, 10, 15]:
        base = pick_to_draft_capital_score(pick, "WR")
        with_bonus = min(100, base * 1.10)
        print(f"Pick #{pick:2d}: {base:5.1f} -> {with_bonus:5.1f} (with 10% bonus)")
