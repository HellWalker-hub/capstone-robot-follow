import numpy as np
from collections import deque


class CMOH:
    """
    Context Memory for Occlusion Handling.
    Maintains a rolling history of K embeddings per tracklet.
    Matches reappearing candidates via mean cosine similarity against history.
    """

    def __init__(self, k=10, sim_threshold=0.55):
        self.k = k
        self.sim_threshold = sim_threshold
        # target_id -> deque of embeddings
        self._memory: dict[int, deque] = {}

    def register(self, track_id: int):
        if track_id not in self._memory:
            self._memory[track_id] = deque(maxlen=self.k)

    def update(self, track_id: int, embedding: np.ndarray):
        self.register(track_id)
        self._memory[track_id].append(embedding.copy())

    def match(self, query_embedding: np.ndarray, candidate_ids: list[int]) -> tuple[int | None, float]:
        """
        Match query against stored embeddings for given candidate track IDs.
        Returns (best_track_id, similarity) or (None, 0.0) if below threshold.
        """
        best_id, best_sim = None, 0.0
        for tid in candidate_ids:
            if tid not in self._memory or len(self._memory[tid]) == 0:
                continue
            history = np.stack(self._memory[tid])
            # mean embedding match
            mean_emb = history.mean(axis=0)
            mean_emb /= (np.linalg.norm(mean_emb) + 1e-6)
            sim = float(np.dot(query_embedding, mean_emb))
            if sim > best_sim:
                best_sim = sim
                best_id = tid
        if best_sim >= self.sim_threshold:
            return best_id, best_sim
        return None, best_sim

    def get_mean_embedding(self, track_id: int) -> np.ndarray | None:
        if track_id not in self._memory or len(self._memory[track_id]) == 0:
            return None
        history = np.stack(self._memory[track_id])
        mean_emb = history.mean(axis=0)
        mean_emb /= (np.linalg.norm(mean_emb) + 1e-6)
        return mean_emb

    def remove(self, track_id: int):
        self._memory.pop(track_id, None)

    def clear(self):
        self._memory.clear()
