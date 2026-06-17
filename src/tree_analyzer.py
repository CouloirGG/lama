"""
tree_analyzer.py — Passive tree analysis with swap recommendations.

Loads the POE2 passive tree data, maps player allocations,
compares against top players, and recommends specific perk swaps
with exact stat trade-offs and pathing costs.

Usage:
    from tree_analyzer import TreeAnalyzer
    analyzer = TreeAnalyzer()
    swaps = analyzer.recommend_swaps(player_nodes, top_player_nodes)
"""

import json
import logging
import math
import os
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

logger = logging.getLogger("tree_analyzer")

# Use GGG's official 0.5.0 web export — the SAME tree the canvas renders, so
# swap node ids/positions match the live game (the old PoB 0.4 tree was stale
# and the analyzer's swap pairings no longer held). Managed by tree2.py.
TREE_DATA_PATH = Path(__file__).parent.parent / "resources" / "data" / "tree2" / "data.json"


@dataclass
class TreeNode:
    """A node in the passive tree."""
    id: str
    name: str
    stats: List[str]
    is_notable: bool = False
    is_keystone: bool = False
    is_mastery: bool = False
    ascendancy: str = ""
    group: int = 0
    orbit: int = 0
    orbit_index: int = 0
    x: float = 0.0
    y: float = 0.0
    connections: List[str] = field(default_factory=list)


@dataclass
class SwapRecommendation:
    """A specific perk swap recommendation."""
    refund_notable: str  # notable to refund
    refund_stats: List[str]  # stats lost
    refund_points: int  # points freed (notable + path)

    take_notable: str  # notable to take
    take_stats: List[str]  # stats gained
    take_points: int  # points needed (notable + path to reach it)

    net_points: int  # positive = saves points, negative = costs more
    impact_summary: str  # plain language summary
    priority: float = 0.0  # higher = more impactful
    refund_id: str = ""  # tree node id (0.5.0) of the refund notable
    take_id: str = ""    # tree node id (0.5.0) of the take notable


@dataclass
class TreeAnalysis:
    """Full analysis of a player's passive tree."""
    allocated_notables: List[TreeNode]
    allocated_keystones: List[TreeNode]
    total_allocated: int
    total_notables: int
    total_keystones: int

    # Comparison results
    missing_notables: List[Dict]  # notables top player has, player doesn't
    extra_notables: List[Dict]  # notables player has, top player doesn't
    swap_recommendations: List[SwapRecommendation]

    def to_dict(self) -> dict:
        return {
            "totalAllocated": self.total_allocated,
            "totalNotables": self.total_notables,
            "totalKeystones": self.total_keystones,
            "allocatedNotables": [{"id": n.id, "name": n.name, "stats": n.stats}
                                  for n in self.allocated_notables],
            "allocatedKeystones": [{"id": n.id, "name": n.name, "stats": n.stats}
                                   for n in self.allocated_keystones],
            "missingNotables": self.missing_notables,
            "extraNotables": self.extra_notables,
            "swapRecommendations": [
                {
                    "refund": s.refund_notable,
                    "refundStats": s.refund_stats,
                    "refundPoints": s.refund_points,
                    "take": s.take_notable,
                    "takeStats": s.take_stats,
                    "takePoints": s.take_points,
                    "netPoints": s.net_points,
                    "summary": s.impact_summary,
                    "priority": s.priority,
                    "refundId": s.refund_id,
                    "takeId": s.take_id,
                }
                for s in self.swap_recommendations
            ],
        }


class TreeAnalyzer:
    """Analyzes passive trees and recommends swaps."""

    def __init__(self):
        self._nodes: Dict[str, TreeNode] = {}
        self._graph: Dict[str, Set[str]] = {}
        self._groups: Dict[int, Tuple[float, float]] = {}  # group_id -> (x, y)
        self._orbit_radii: List[float] = []
        self._orbit_angles: List[List[float]] = []  # [orbit][orbitIndex] -> radians
        self._pos_cache: Dict[str, Tuple[float, float]] = {}
        self._loaded = False

    def _node_position(self, node: "TreeNode") -> Optional[Tuple[float, float]]:
        """Node (x, y) — the GGG export provides absolute coordinates directly."""
        return (node.x, node.y)

    def _ensure_loaded(self):
        if self._loaded:
            return
        tree_data = self._load_tree()
        if not tree_data:
            return
        self._parse_tree(tree_data)
        self._loaded = True

    def _load_tree(self) -> Optional[dict]:
        """Load GGG's 0.5.0 tree export (downloaded + cached by tree2)."""
        if not TREE_DATA_PATH.exists():
            try:
                import tree2
                tree2.ensure_assets()
            except Exception as e:
                logger.warning(f"tree2 asset fetch failed: {e}")
        if TREE_DATA_PATH.exists():
            try:
                with open(TREE_DATA_PATH, encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load tree data: {e}")
        return None

    def _parse_tree(self, tree_data: dict):
        """Parse the GGG export into nodes + an undirected adjacency graph."""
        raw_nodes = tree_data.get("nodes", {}) or {}

        for nid, raw in raw_nodes.items():
            nid = str(nid)
            conns = [str(c) for c in (raw.get("out") or [])]
            conns += [str(c) for c in (raw.get("in") or [])]
            node = TreeNode(
                id=nid,
                name=raw.get("name", ""),
                stats=raw.get("stats", []) or [],
                is_notable=bool(raw.get("isNotable")),
                is_keystone=bool(raw.get("isKeystone")),
                is_mastery=bool(raw.get("isMastery")) or "Mastery" in (raw.get("name") or ""),
                ascendancy=raw.get("ascendancyId") or "",
                group=raw.get("group", 0),
                orbit=raw.get("orbit", 0),
                orbit_index=raw.get("orbitIndex", 0),
                x=raw.get("x", 0.0) or 0.0,
                y=raw.get("y", 0.0) or 0.0,
                connections=conns,
            )
            self._nodes[nid] = node

        # Group centres (the GGG export keys groups by string id).
        for gid, grp in (tree_data.get("groups", {}) or {}).items():
            if isinstance(grp, dict):
                self._groups[str(gid)] = (grp.get("x", 0), grp.get("y", 0))

        # Build the undirected graph from out/in (dedup, skip self-loops).
        for nid, node in self._nodes.items():
            self._graph.setdefault(nid, set())
            for cid in node.connections:
                if cid == nid:
                    continue
                self._graph[nid].add(cid)
                self._graph.setdefault(cid, set()).add(nid)

        logger.info(f"Passive tree (0.5.0) loaded: {len(self._nodes)} nodes, "
                     f"{sum(1 for n in self._nodes.values() if n.is_notable)} notables")

    def shortest_path(self, start: str, end: str) -> Optional[List[str]]:
        """BFS shortest path between two nodes."""
        if start == end:
            return [start]
        visited = {start}
        queue = deque([(start, [start])])
        while queue:
            current, path = queue.popleft()
            for neighbor in self._graph.get(current, set()):
                if neighbor == end:
                    return path + [neighbor]
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))
        return None

    def path_cost(self, target: str, allocated: Set[str]) -> int:
        """How many NEW nodes needed to reach target from current allocation."""
        best = 999
        for alloc_id in allocated:
            if alloc_id not in self._graph:
                continue
            path = self.shortest_path(alloc_id, target)
            if path:
                new_nodes = sum(1 for n in path if n not in allocated)
                best = min(best, new_nodes)
        return best if best < 999 else -1

    def analyze(self, player_node_ids: List[int],
                top_player_node_ids: Optional[List[int]] = None) -> TreeAnalysis:
        """Analyze a player's passive tree, optionally comparing to a top player."""
        self._ensure_loaded()

        allocated = set(str(n) for n in player_node_ids)
        top_allocated = set(str(n) for n in top_player_node_ids) if top_player_node_ids else set()

        # Categorize allocated nodes
        alloc_notables = []
        alloc_keystones = []
        for nid in allocated:
            node = self._nodes.get(nid)
            if not node:
                continue
            if node.is_keystone:
                alloc_keystones.append(node)
            elif node.is_notable:
                alloc_notables.append(node)

        analysis = TreeAnalysis(
            allocated_notables=sorted(alloc_notables, key=lambda n: n.name),
            allocated_keystones=sorted(alloc_keystones, key=lambda n: n.name),
            total_allocated=len(allocated),
            total_notables=len(alloc_notables),
            total_keystones=len(alloc_keystones),
            missing_notables=[],
            extra_notables=[],
            swap_recommendations=[],
        )

        if not top_allocated:
            return analysis

        # Find notables in top but not player
        for nid in (top_allocated - allocated):
            node = self._nodes.get(nid)
            if not node or not node.is_notable:
                continue
            if node.ascendancy:
                continue  # skip ascendancy nodes

            cost = self.path_cost(nid, allocated)
            analysis.missing_notables.append({
                "id": nid, "name": node.name, "stats": node.stats,
                "cost": cost,
            })

        # Find notables in player but not top
        for nid in (allocated - top_allocated):
            node = self._nodes.get(nid)
            if not node or not node.is_notable:
                continue
            if node.ascendancy:
                continue

            analysis.extra_notables.append({
                "id": nid, "name": node.name, "stats": node.stats,
            })

        # Generate swap recommendations
        analysis.swap_recommendations = self._generate_swaps(
            allocated, top_allocated, analysis.missing_notables, analysis.extra_notables
        )

        return analysis

    def _generate_swaps(self, allocated: Set[str], top_allocated: Set[str],
                        missing: List[Dict], extra: List[Dict]) -> List[SwapRecommendation]:
        """Generate specific swap recommendations."""
        swaps = []

        # For each missing notable, find the best extra notable to refund
        for miss in missing:
            if miss["cost"] < 0:
                continue  # unreachable

            miss_node = self._nodes.get(miss["id"])
            if not miss_node:
                continue

            for ext in extra:
                ext_node = self._nodes.get(ext["id"])
                if not ext_node:
                    continue

                # Estimate the refund points (how many nodes connect only to this notable)
                # Simplified: count the path from the nearest shared node to this notable
                refund_cost = self.path_cost(ext["id"], top_allocated)
                if refund_cost < 0:
                    refund_cost = 3  # estimate

                take_cost = miss["cost"]
                net = refund_cost - take_cost  # positive = saves points

                # Score the swap — prioritize DPS-relevant stats
                priority = self._score_swap(miss_node.stats, ext_node.stats)

                summary = self._summarize_swap(
                    ext_node.name, ext_node.stats,
                    miss_node.name, miss_node.stats,
                    refund_cost, take_cost, net
                )

                swaps.append(SwapRecommendation(
                    refund_notable=ext_node.name,
                    refund_stats=ext_node.stats,
                    refund_points=refund_cost,
                    take_notable=miss_node.name,
                    take_stats=miss_node.stats,
                    take_points=take_cost,
                    net_points=net,
                    impact_summary=summary,
                    priority=priority,
                    refund_id=str(ext["id"]),
                    take_id=str(miss["id"]),
                ))

        # Filter out impractical swaps (>6 points to reach target)
        swaps = [s for s in swaps if s.take_points <= 6]

        # Penalize swaps that cost extra points
        for s in swaps:
            if s.net_points < 0:
                s.priority += s.net_points * 0.5  # penalty for costing more
            elif s.net_points > 0:
                s.priority += s.net_points * 0.3  # bonus for saving points

        # Sort by priority (highest first)
        swaps.sort(key=lambda s: s.priority, reverse=True)

        # Deduplicate: best swap per target notable AND per refund notable
        seen_targets = set()
        seen_refunds = set()
        deduped = []
        for s in swaps:
            if s.take_notable not in seen_targets and s.refund_notable not in seen_refunds:
                seen_targets.add(s.take_notable)
                seen_refunds.add(s.refund_notable)
                deduped.append(s)

        return deduped[:8]  # top 8 recommendations

    def _score_swap(self, gain_stats: List[str], lose_stats: List[str]) -> float:
        """Score a swap based on stat trade-off. Higher = better swap."""
        score = 0.0

        # DPS-relevant keywords score higher
        dps_keywords = ["damage", "critical", "crit", "cast speed", "attack speed",
                        "elemental", "spell", "multiplier", "penetrat"]
        def_keywords = ["life", "energy shield", "resist", "armour", "evasion",
                        "block", "stun"]

        for stat in gain_stats:
            sl = stat.lower()
            if any(kw in sl for kw in dps_keywords):
                score += 3.0
            elif any(kw in sl for kw in def_keywords):
                score += 1.5
            else:
                score += 0.5

        for stat in lose_stats:
            sl = stat.lower()
            if any(kw in sl for kw in dps_keywords):
                score -= 2.0
            elif any(kw in sl for kw in def_keywords):
                score -= 1.0
            else:
                score -= 0.3

        return score

    def _summarize_swap(self, refund_name: str, refund_stats: List[str],
                        take_name: str, take_stats: List[str],
                        refund_points: int, take_points: int, net: int) -> str:
        """Generate plain-language swap summary."""
        parts = [f"Refund {refund_name} ({refund_points} points)"]
        parts.append(f"→ Take {take_name} ({take_points} points)")

        if net > 0:
            parts.append(f"→ saves {net} point{'s' if net > 1 else ''}")
        elif net < 0:
            parts.append(f"→ costs {-net} extra point{'s' if -net > 1 else ''}")
        else:
            parts.append("→ same point cost")

        # Stat trade-off
        gain_summary = "; ".join(take_stats[:2])
        lose_summary = "; ".join(refund_stats[:2])

        return f"{' '.join(parts)}. Gain: {gain_summary}. Lose: {lose_summary}."

    def get_tree_visual(self, player_node_ids: List[int],
                        top_node_ids: Optional[List[int]] = None,
                        swaps: Optional[List[SwapRecommendation]] = None) -> dict:
        """Return position + edge data for an SVG tree mini-map.

        - ``nodes``: notables/keystones (not small passives), positioned along
          their true orbits so they don't stack on the group centre.
        - ``edges``: the full tree skeleton as ``[x1, y1, x2, y2, allocated]``
          line segments, so the map reads as a tree instead of a dot cloud.
          ``allocated`` is 1 when both endpoints are in the player's tree.

        Nodes are categorized as: allocated, shared, missing (top has, you don't),
        extra (you have, top doesn't), swap_take, swap_refund.
        """
        self._ensure_loaded()

        allocated = set(str(n) for n in player_node_ids) if player_node_ids else set()
        top_allocated = set(str(n) for n in top_node_ids) if top_node_ids else set()

        # Build swap node name sets for highlighting
        swap_take_names = set()
        swap_refund_names = set()
        if swaps:
            for s in swaps:
                swap_take_names.add(s.take_notable)
                swap_refund_names.add(s.refund_notable)

        nodes_out = []
        for nid, node in self._nodes.items():
            # Only show notables and keystones (skip small passives)
            if not node.is_notable and not node.is_keystone:
                continue
            # Skip ascendancy nodes
            if node.ascendancy:
                continue

            pos = self._node_position(node)
            if not pos:
                continue

            # Categorize
            in_player = nid in allocated
            in_top = nid in top_allocated

            if node.name in swap_take_names:
                cat = "swap_take"
            elif node.name in swap_refund_names:
                cat = "swap_refund"
            elif in_player and in_top:
                cat = "shared"
            elif in_player and not in_top:
                cat = "extra"
            elif not in_player and in_top:
                cat = "missing"
            elif in_player:
                cat = "allocated"
            else:
                cat = "unallocated"

            nodes_out.append({
                "id": nid,
                "name": node.name,
                "x": round(pos[0]),
                "y": round(pos[1]),
                "category": cat,
                "isKeystone": node.is_keystone,
                "stats": node.stats[:2],
            })

        # Tree skeleton edges (deduped, ascendancy excluded). Coordinates are
        # carried inline so the frontend needs no small-passive node list.
        edges_out = []
        seen = set()
        for nid, node in self._nodes.items():
            if node.ascendancy:
                continue
            p1 = self._node_position(node)
            if not p1:
                continue
            for cid in node.connections:
                key = (nid, cid) if nid < cid else (cid, nid)
                if key in seen:
                    continue
                other = self._nodes.get(cid)
                if not other or other.ascendancy:
                    continue
                p2 = self._node_position(other)
                if not p2:
                    continue
                seen.add(key)
                alloc = 1 if (nid in allocated and cid in allocated) else 0
                edges_out.append([round(p1[0]), round(p1[1]),
                                  round(p2[0]), round(p2[1]), alloc])

        return {"nodes": nodes_out, "edges": edges_out}
