"""
Role classification for projected player usage.

Generates hybrid role tags like "WR2 + Red Zone Target" or "RB1 (Bellcow)".
"""

from typing import Dict
from .config import *


class RoleClassifier:
    """
    Classifies player roles based on projected usage patterns.

    Generates hybrid tags combining depth chart position with specializations.
    """

    def classify_role(
        self,
        position: str,
        projected_usage: Dict,
        component_details: Dict
    ) -> str:
        """
        Generate projected role tag.

        Args:
            position: Position ('QB', 'RB', 'WR', 'TE')
            projected_usage: Dictionary with projected targets, carries, snap_share
            component_details: Component detail dictionaries

        Returns:
            Role tag string (e.g., "WR2 + Red Zone Target")
        """
        if position in ['WR', 'TE']:
            return self._classify_receiver_role(position, projected_usage, component_details)
        elif position == 'RB':
            return self._classify_rb_role(projected_usage, component_details)
        elif position == 'QB':
            return self._classify_qb_role(projected_usage)
        else:
            return "Unknown Position"

    def _classify_receiver_role(
        self,
        position: str,
        projected_usage: Dict,
        component_details: Dict
    ) -> str:
        """
        Classify WR/TE role.

        Returns tags like:
        - "WR1"
        - "WR2 + Red Zone Target"
        - "TE1 + Goal Line"
        """
        proj_targets = projected_usage.get('projected_targets', 0)
        proj_snap_share = projected_usage.get('projected_snap_share', 0)

        # Extract red zone usage if available
        role_traj_details = component_details.get('role_trajectory', {})
        # For now, we don't have projected RZ usage easily accessible
        # This would need to be calculated separately

        # Base tier
        if proj_targets >= WR1_TARGETS and proj_snap_share >= WR1_SNAP_SHARE:
            base = f"{position}1"
        elif proj_targets >= WR2_TARGETS:
            base = f"{position}2"
        elif proj_targets >= WR3_TARGETS:
            base = f"{position}3"
        else:
            base = "Rotational"

        # Add modifiers (placeholder - would need additional data)
        modifiers = []

        # Check if high snap share = 3-down player
        if proj_snap_share >= THREE_DOWN_SNAP_THRESHOLD:
            modifiers.append("3-Down")

        # TODO: Add red zone, slot, etc. when data available

        if modifiers:
            return f"{base} + {', '.join(modifiers)}"
        else:
            return base

    def _classify_rb_role(
        self,
        projected_usage: Dict,
        component_details: Dict
    ) -> str:
        """
        Classify RB role.

        Returns tags like:
        - "RB1 (Bellcow)"
        - "RB2 + Passing Down"
        - "Committee Back"
        """
        proj_carries = projected_usage.get('projected_carries', 0)
        proj_targets = projected_usage.get('projected_targets', 0)
        proj_snap_share = projected_usage.get('projected_snap_share', 0)

        # Base tier
        if proj_carries >= RB_BELLCOW_CARRIES and proj_snap_share >= RB_BELLCOW_SNAP_SHARE:
            base = "RB1 (Bellcow)"
        elif proj_carries >= RB1_CARRIES:
            base = "RB1"
        elif proj_carries >= RB2_CARRIES:
            base = "RB2"
        else:
            base = "Committee Back"

        # Add modifiers
        modifiers = []

        if proj_targets >= PASSING_DOWN_TARGETS:
            modifiers.append("3-Down Back")
        elif proj_targets >= PASSING_DOWN_TARGETS * 0.67:
            modifiers.append("Passing Down Role")

        if proj_snap_share >= WORKHORSE_SNAP_THRESHOLD and "Bellcow" not in base:
            modifiers.append("Workhorse")

        if modifiers:
            return f"{base} + {', '.join(modifiers)}"
        else:
            return base

    def _classify_qb_role(
        self,
        projected_usage: Dict
    ) -> str:
        """
        Classify QB role.

        Returns:
        - "QB1 (Locked Starter)"
        - "QB1"
        - "Backup QB"
        """
        proj_snap_share = projected_usage.get('projected_snap_share', 0)

        if proj_snap_share >= QB_LOCKED_STARTER_SNAP:
            return "QB1 (Locked Starter)"
        elif proj_snap_share >= QB_STARTER_SNAP:
            return "QB1"
        else:
            return "Backup QB"
