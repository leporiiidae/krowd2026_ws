"""Cone tracking in the global frame and longitudinal pair grouping."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np


@dataclass
class Track:
    track_id: int
    x: float
    y: float
    hits: int = 1              # consecutive observed frames until confirmed, then total
    misses: int = 0            # consecutive unobserved frames
    confirmed: bool = False
    observations: int = 1

    @property
    def xy(self):
        return (self.x, self.y)


class ConeTracker:
    """Nearest-neighbour tracker with first-order position filtering.

    Cones are static and localisation is RTK-grade, so a CONFIRMED track is never
    dropped: a cone that disappears ahead is a momentary miss (keep trusting the
    estimate), and one that disappears behind is simply out of view. Only
    unconfirmed candidates expire after ``miss_tolerance`` frames.
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.tracks: Dict[int, Track] = {}
        self._next_id = 0

    def reset(self) -> None:
        self.tracks.clear()
        self._next_id = 0

    def update(self, observations_xy: Sequence[Sequence[float]]) -> List[Track]:
        p = self.cfg.perception
        obs = np.asarray(observations_xy, dtype=float).reshape(-1, 2)
        ids = list(self.tracks.keys())
        unmatched_obs = set(range(len(obs)))
        matched_tracks = set()

        if ids and len(obs):
            pos = np.array([self.tracks[i].xy for i in ids])
            dist = np.hypot(pos[:, None, 0] - obs[None, :, 0], pos[:, None, 1] - obs[None, :, 1])
            # Greedy global-nearest assignment: closest pairs first.
            for flat in np.argsort(dist, axis=None):
                ti, oi = np.unravel_index(flat, dist.shape)
                if dist[ti, oi] > p.track_match_distance:
                    break
                if ids[ti] in matched_tracks or oi not in unmatched_obs:
                    continue
                track = self.tracks[ids[ti]]
                a = p.track_filter_alpha
                track.x = (1.0 - a) * track.x + a * obs[oi, 0]
                track.y = (1.0 - a) * track.y + a * obs[oi, 1]
                track.hits += 1
                track.observations += 1
                track.misses = 0
                if track.hits >= p.confirm_frames:
                    track.confirmed = True
                matched_tracks.add(ids[ti])
                unmatched_obs.discard(oi)

        for track_id in ids:
            if track_id in matched_tracks:
                continue
            track = self.tracks[track_id]
            track.misses += 1
            if not track.confirmed:
                track.hits = 0  # promotion requires CONSECUTIVE observations
                if track.misses > p.miss_tolerance:
                    del self.tracks[track_id]

        for oi in sorted(unmatched_obs):
            self.tracks[self._next_id] = Track(self._next_id, float(obs[oi, 0]), float(obs[oi, 1]))
            self._next_id += 1

        return self.confirmed()

    def confirmed(self) -> List[Track]:
        return [t for t in self.tracks.values() if t.confirmed]


@dataclass
class ConePair:
    """Cones sharing one longitudinal station (normally two, side by side)."""

    track_ids: List[int]
    s: np.ndarray
    d: np.ndarray

    @property
    def s_min(self) -> float:
        return float(self.s.min())

    @property
    def s_max(self) -> float:
        return float(self.s.max())

    @property
    def s_center(self) -> float:
        return float(self.s.mean())

    @property
    def sd(self) -> np.ndarray:
        return np.column_stack((self.s, self.d))


def group_pairs(track_ids: Sequence[int], cones_sd: np.ndarray, window: float) -> List[ConePair]:
    """Group cones whose s lies within ``window`` of the group's first cone."""
    cones_sd = np.asarray(cones_sd, dtype=float).reshape(-1, 2)
    order = np.argsort(cones_sd[:, 0]) if len(cones_sd) else []
    groups: List[List[int]] = []
    for idx in order:
        if groups and cones_sd[idx, 0] - cones_sd[groups[-1][0], 0] <= window:
            groups[-1].append(int(idx))
        else:
            groups.append([int(idx)])
    return [
        ConePair([int(track_ids[i]) for i in g], cones_sd[g, 0].copy(), cones_sd[g, 1].copy())
        for g in groups
    ]


def find_rows(xy: np.ndarray, heading: float, cfg, origin=None) -> List[List[int]]:
    """Indices of cones that form obstacle rows: N cones side by side ACROSS the road.

    ``heading``/``origin``: travel direction (rad) and vehicle position, same frame as
    ``xy``. A row is a chain of cones whose neighbours are at most ``row_gap_max``
    apart, lying across the travel direction, ``row_min_cones``..``row_max_cones`` long.
    When more are chained (clutter fused to a row), the ``cones_per_group`` cones
    nearest the vehicle's line are the row: groups stand against a lane line, so
    anything fused on is on the OUTER side, beyond that line.
    """
    xy = np.asarray(xy, dtype=float).reshape(-1, 2)
    m = cfg.mission
    if len(xy) < m.row_min_cones:
        return []
    origin = np.zeros(2) if origin is None else np.asarray(origin, dtype=float)
    along_u = np.array([np.cos(heading), np.sin(heading)])
    across_u = np.array([-along_u[1], along_u[0]])
    s = (xy - origin) @ along_u
    d = (xy - origin) @ across_u
    max_len = m.cones_per_group * cfg.perception.cone_diameter + m.row_len_slack

    n = len(xy)
    parent = list(range(n))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(n):
        for j in range(i + 1, n):
            if abs(s[i] - s[j]) <= m.row_along_tol and abs(d[i] - d[j]) <= m.row_gap_max:
                parent[root(i)] = root(j)
    comps = {}
    for i in range(n):
        comps.setdefault(root(i), []).append(i)

    rows = []
    for members in comps.values():
        if len(members) < m.row_min_cones:
            continue
        # Order from the vehicle's line outward, so trimming drops the OUTER cones.
        side = 1.0 if np.mean(d[members]) >= 0 else -1.0
        members = sorted(members, key=lambda i: side * d[i])
        # Touching cones sit one pitch apart; fused clutter breaks that rhythm. Pick the
        # most regular window, so clutter on EITHER end is left out of the row itself.
        pitch = cfg.perception.cone_diameter * cfg.perception.row_pitch_ratio
        best, best_key = None, None
        for size in _sizes(len(members), m):
            for k in range(len(members) - size + 1):
                win = members[k:k + size]
                span = abs(d[win[-1]] - d[win[0]])
                spread = s[win].max() - s[win].min()
                if not (span <= max_len and spread <= m.row_along_tol and _crosses(xy[win], heading, m)):
                    continue
                gaps = np.abs(np.diff(d[win]))
                key = float(np.abs(gaps - pitch).sum())
                if best_key is None or key < best_key:
                    best, best_key = win, key
            if best is not None:
                break
        if best is None:
            continue
        # Fused members INWARD of the row stand in the driving area: they are obstacles
        # too, so they stay. Only those beyond the outer end (shoulder / other lane) go.
        inner_edge = min(side * d[i] for i in best)
        if not _within_reach(inner_edge, float(np.mean(s[best])), cfg):
            continue
        extras = [i for i in members if i not in best and side * d[i] < inner_edge]
        if _isolated(xy, best + extras, s, d, side, m):
            rows.append(best + extras)
    return rows


def _within_reach(inner_edge: float, ahead: float, cfg) -> bool:
    """A group stands inside the lane the vehicle is in; roadside rows are metres off.

    The vehicle's own heading is the only lane reference perception has, and a few
    degrees of yaw shift a far row sideways, so the allowance grows with distance.
    """
    m = cfg.mission
    allowed = m.row_lateral_max + abs(ahead) * np.tan(np.radians(m.row_heading_tol))
    return inner_edge - cfg.cone_radius <= allowed


def _sizes(available: int, m):
    """Window sizes to try: the rule's group size first, then the others."""
    lo, hi = m.row_min_cones, min(available, m.row_max_cones)
    preferred = [m.cones_per_group] if lo <= m.cones_per_group <= hi else []
    return preferred + [k for k in range(hi, lo - 1, -1) if k not in preferred]


def _crosses(xy: np.ndarray, heading: float, m) -> bool:
    """The row lies across the road: its direction is near-perpendicular to travel."""
    centred = xy - xy.mean(axis=0)
    _, _, vt = np.linalg.svd(centred, full_matrices=False)
    angle = abs(np.arctan2(vt[0, 1], vt[0, 0]) - heading) % np.pi
    return np.degrees(abs(angle - np.pi / 2.0)) <= m.row_angle_tol


def _isolated(xy, members, s, d, side, m) -> bool:
    """Obstacle rows stand on open road; vegetation is a DENSE patch of clutter.

    Counts neighbours on the road side only (in front, behind, inward). Beyond the row's
    OUTER end lies the lane line and then the shoulder or the other lane, where growth
    next to a real group is normal. One or two objects near a row never reject it: an
    object on the road is an obstacle itself, and deleting the row for it would hide
    both. Measured on 0919_avoidance_f: real rows 0 neighbours, clutter rows 4-5.
    A track within ``row_dup_radius`` of a member is the same cone (duplicate track).
    """
    others = np.delete(np.arange(len(xy)), members)
    if len(others) == 0:
        return True
    dist = np.hypot(xy[others, None, 0] - xy[members][None, :, 0],
                    xy[others, None, 1] - xy[members][None, :, 1]).min(axis=1)
    outer_edge = max(side * d[i] for i in members)
    beyond = side * d[others] > outer_edge + 0.5 * m.row_gap_max
    near = (dist > m.row_dup_radius) & (dist < m.row_clear_radius) & ~beyond
    # A partial row (fewer cones than a group) is easier to form by chance in a patch of
    # clutter, so it tolerates fewer neighbours.
    limit = m.row_clutter_min if len(members) >= m.cones_per_group else m.row_clutter_min_partial
    return int(np.sum(near)) < limit


class RowFilter:
    """Keeps only tracks that belong to obstacle rows, and remembers accepted rows.

    Cones do not move, so a row accepted once stays accepted: clutter that shows up next
    to it later (seen from a new angle) cannot make a real obstacle disappear, and the
    row survives while the car turns across it mid-maneuver.
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.accepted: set = set()

    def reset(self) -> None:
        self.accepted.clear()

    def update(self, tracks: Sequence[Track], heading: float, origin=None) -> List[Track]:
        if not tracks:
            return []
        xy = np.array([t.xy for t in tracks])
        for row in find_rows(xy, heading, self.cfg, origin):
            self.accepted.update(tracks[i].track_id for i in row)
        return [t for t in tracks if t.track_id in self.accepted]
