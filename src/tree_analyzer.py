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
import os
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

logger = logging.getLogger("tree_analyzer")

TREE_DATA_PATH = Path(__file__).parent.parent / "resources" / "data" / "passive_tree.json"
TREE_DOWNLOAD_URL = "https://raw.githubusercontent.com/PathOfBuildingCommunity/PathOfBuilding-PoE2/master/src/TreeData/0_4/tree.json"


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
                }
                for s in self.swap_recommendations
            ],
        }


class TreeAnalyzer:
    """Analyzes passive trees and recommends swaps."""

    def __init__(self):
        self._nodes: Dict[str, TreeNode] = {}
        self._graph: Dict[str, Set[str]] = {}
        self._loaded = False

    def _ensure_loaded(self):
        if self._loaded:
            return
        tree_data = self._load_tree()
        if not tree_data:
            return
        self._parse_tree(tree_data)
        self._loaded = True

    def _load_tree(self) -> Optional[dict]:
        """Load tree.json from disk or download."""
        if TREE_DATA_PATH.exists():
            try:
                with open(TREE_DATA_PATH, encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load tree data: {e}")

        # Download
        try:
            import requests
            logger.info("Downloading passive tree data...")
            resp = requests.get(TREE_DOWNLOAD_URL, timeout=30,
                                headers={"User-Agent": "LAMA/1.0"})
            if resp.status_code == 200:
                TREE_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
                with open(TREE_DATA_PATH, "w", encoding="utf-8") as f:
                    f.write(resp.text)
                return resp.json()
        except Exception as e:
            logger.warning(f"Failed to download tree data: {e}")
        return None

    def _parse_tree(self, tree_data: dict):
        """Parse tree.json into nodes and graph."""
        raw_nodes = tree_data.get("nodes", {})

        for nid, raw in raw_nodes.items():
            node = TreeNode(
                id=nid,
                name=raw.get("name", ""),
                stats=raw.get("stats", []),
                is_notable=raw.get("isNotable", False),
                is_keystone=raw.get("isKeystone", False),
                is_mastery="Mastery" in raw.get("name", ""),
                ascendancy=raw.get("ascendancyName", ""),
                group=raw.get("group", 0),
                connections=[str(c["id"]) for c in raw.get("connections", [])],
            )
            self._nodes[nid] = node

        # Build bidirectional graph
        for nid, node in self._nodes.items():
            if nid not in self._graph:
                self._graph[nid] = set()
            for cid in node.connections:
                self._graph[nid].add(cid)
                if cid not in self._graph:
                    self._graph[cid] = set()
                self._graph[cid].add(nid)

        logger.info(f"Passive tree loaded: {len(self._nodes)} nodes, "
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
