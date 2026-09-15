"""
Centroid tracker for video fruit counting.

The tracker keeps persistent IDs for detections across frames so that the
same physical fruit isn't counted multiple times. Two protections against
"phantom tracks" (a single fruit that briefly disappears and reappears as a
new ID):

1. max_distance is generous enough to allow for handheld jitter between
   sampled frames.
2. A track only counts toward the final "unique fruit tracked" number after
   being observed in at least `min_hits` frames. Flickers that appear only
   once don't become new fruit.

The raw `max_track_id_seen` includes unconfirmed flicker tracks and should
NOT be reported as "unique fruit". Use `confirmed_count` for that.
"""

from __future__ import annotations
from detection import Detection


class CentroidTracker:
    def __init__(self, max_distance: float = 150.0, min_hits: int = 3):
        self.max_distance = max_distance
        self.min_hits = min_hits
        self.next_track_id = 1
        # Each entry: [track_id, cx, cy, color, hit_count]
        self._prev: list[list] = []
        self._confirmed: set[int] = set()
        self.max_track_id_seen = 0

    @staticmethod
    def _centroid(d: Detection) -> tuple[float, float]:
        return ((d.x1 + d.x2) / 2.0, (d.y1 + d.y2) / 2.0)

    def update(self, detections: list[Detection]) -> dict[int, int]:
        assignments: dict[int, int] = {}
        used_prev: set[int] = set()

        current_points = [(d.id, *self._centroid(d), d.color_profile) for d in detections]

        for det_id, cx, cy, color in current_points:
            best_idx = -1
            best_track = None
            best_dist = self.max_distance

            for idx, row in enumerate(self._prev):
                if idx in used_prev:
                    continue
                _tid, pcx, pcy, pcolor, _hits = row
                if pcolor != color:
                    continue
                dist = ((cx - pcx) ** 2 + (cy - pcy) ** 2) ** 0.5
                if dist < best_dist:
                    best_dist = dist
                    best_track = _tid
                    best_idx = idx

            if best_track is not None:
                used_prev.add(best_idx)
                assignments[det_id] = best_track
                # Update the matched track in place
                self._prev[best_idx][1] = cx
                self._prev[best_idx][2] = cy
                self._prev[best_idx][3] = color
                self._prev[best_idx][4] += 1
                if self._prev[best_idx][4] >= self.min_hits:
                    self._confirmed.add(best_track)
            else:
                new_id = self.next_track_id
                self.next_track_id += 1
                assignments[det_id] = new_id
                self._prev.append([new_id, cx, cy, color, 1])

        # Drop tracks that weren't matched this frame AND haven't yet been
        # confirmed — these are almost certainly flicker and we don't want
        # them growing the list. Confirmed tracks are always kept so a
        # re-observed fruit reuses its original ID.
        matched_ids = {assignments[d.id] for d in detections}
        self._prev = [
            row for row in self._prev
            if row[0] in matched_ids or row[0] in self._confirmed
        ]

        self.max_track_id_seen = max(self.max_track_id_seen, self.next_track_id - 1)
        return assignments

    @property
    def confirmed_count(self) -> int:
        """Number of tracks observed in >= min_hits frames. This is the
        number to report as 'unique fruit tracked'. The raw
        max_track_id_seen includes flicker phantoms and should not be
        shown to users."""
        return len(self._confirmed)