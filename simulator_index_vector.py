from __future__ import annotations

import heapq
import random
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set
from random import Random
import numpy as np


class BasicVectorDB:
    """
    A toy in-memory vector database implemented from scratch.

    The goal is to keep things extremely simple:
    - Maintain an internal list of vectors and their labels
    - Support insertion of individual vectors
    - Provide a brute-force nearest neighbour search to validate inserts
    """

    def __init__(self, dim: int):
        self.dim = dim
        self._vectors: list[np.ndarray] = []
        self._labels: list[str] = []
        self._label_to_index: dict[str, int] = {}

    def insert(self, label: str, vector: np.ndarray) -> None:
        """
        Insert a new vector. We keep things opinionated to stay minimal:
        - Vectors must be numpy arrays with the configured dimensionality
        - Labels must be unique strings
        """
        if label in self._label_to_index:
            raise ValueError(f"Label '{label}' already exists")

        vector = np.asarray(vector, dtype=np.float32).reshape(-1)
        if vector.shape[0] != self.dim:
            raise ValueError(
                f"Expected vector of dim {self.dim}, received {vector.shape[0]}"
            )

        index = len(self._vectors)
        self._vectors.append(vector)
        self._labels.append(label)
        self._label_to_index[label] = index
        print(f"[insert] stored label='{label}' at position={index}")

    def search(self, query: np.ndarray, k: int = 3):
        """
        Very small brute-force nearest neighbour lookup.

        This is not meant to be fast—it simply validates that our data
        is stored correctly by computing Euclidean distances to each
        vector and returning the closest k matches.
        """
        if not self._vectors:
            raise ValueError("Vector DB is empty, insert vectors first")

        query = np.asarray(query, dtype=np.float32).reshape(-1)
        if query.shape[0] != self.dim:
            raise ValueError(
                f"Expected query of dim {self.dim}, received {query.shape[0]}"
            )

        all_vectors = np.stack(self._vectors, axis=0)
        distances = np.linalg.norm(all_vectors - query, axis=1)
        nearest_indices = np.argsort(distances)[:k]
        results = [
            {"label": self._labels[idx], "distance": float(distances[idx])}
            for idx in nearest_indices
        ]
        return results


@dataclass
class HNSWNode:
    label: str
    vector: np.ndarray
    level: int
    neighbors: Dict[int, Set[int]] = field(default_factory=dict)
    deleted: bool = False


class SimpleHNSW:
    """
    Minimal Hierarchical Navigable Small World implementation.

    It mirrors the high-level algorithm while keeping the code compact:
    - Each node knows the maximum layer it participates in plus adjacency lists
    - Insert samples a random level and links to closest neighbours per layer
    - Search walks top layers greedily, then performs a best-first search in
      the ground layer to approximate the nearest neighbours
    """

    def __init__(
        self,
        dim: int,
        m: int = 4,
        level_probability: float = 0.5,
        ef_construction: int = 16,
        seed: Optional[int] = None,
    ):
        self._rng = Random(seed)
        self.dim = dim
        self.m = m
        self.level_probability = level_probability
        self.ef_construction = ef_construction

        self._nodes: List[HNSWNode] = []
        self._label_to_id: Dict[str, int] = {}
        self._entry_point: Optional[int] = None
        self._max_level = -1

    def insert(self, label: str, vector: np.ndarray) -> None:
        if label in self._label_to_id:
            raise ValueError(f"Label '{label}' already exists")

        vector = np.asarray(vector, dtype=np.float32).reshape(-1)
        if vector.shape[0] != self.dim:
            raise ValueError(
                f"Expected vector of dim {self.dim}, received {vector.shape[0]}"
            )

        level = self._sample_level()
        node = HNSWNode(label=label, vector=vector, level=level)
        node_id = len(self._nodes)

        self._nodes.append(node)
        self._label_to_id[label] = node_id

        if self._entry_point is None:
            self._entry_point = node_id
            self._max_level = level
            # print(f"[hnsw insert] '{label}' becomes the entry point at level {level}")
            return

        entry = self._entry_point
        if level > self._max_level:
            self._max_level = level
            self._entry_point = node_id

        current = entry
        for lvl in range(self._max_level, level, -1):
            current = self._greedy_search(vector, current, lvl)

        for lvl in range(min(level, self._max_level), -1, -1):
            candidates = self._layer_candidates(vector, lvl)
            neighbors = self._select_neighbors(vector, candidates, self.m)
            for other_id in neighbors:
                self._link(node_id, other_id, lvl)

    def delete(self, label: str) -> None:
        """
        Delete a node using tombstone approach.
        Marks the node as deleted without removing it from the graph structure.
        If the entry point is deleted, finds a new valid entry point.
        """
        if label not in self._label_to_id:
            raise ValueError(f"Label '{label}' not found")

        node_id = self._label_to_id[label]
        if self._nodes[node_id].deleted:
            raise ValueError(f"Label '{label}' is already deleted")

        self._nodes[node_id].deleted = True
        # print(f"[hnsw delete] marked '{label}' (node_id={node_id}) as deleted")

        # If entry point was deleted, find a new valid entry point
        if self._entry_point == node_id:
            self._reassign_entry_point()

    def delete_with_local_rewiring(self, label: str, representative_strategy: str = "highest_degree") -> dict:
        """
        Delete a node using local rewiring approach.
        
        PSEUDOCODE:
        ===========
        DELETE_NODE_WITH_LOCAL_REWIRING(G, node_id):
            stats = {rewired_edges: 0, representative: None, neighbors_before: {}, neighbors_after: {}}
            
            // Step 1: Collect neighbors at each layer
            FOR layer = max_level DOWN TO 0:
                neighbors[layer] = G.get_neighbors(node_id, layer)
                stats.neighbors_before[layer] = |neighbors[layer]|
            
            // Step 2: Choose representative node
            representative = CHOOSE_REPRESENTATIVE(neighbors[0], strategy)
            stats.representative = representative
            
            // Step 3: Process each layer from top to bottom
            FOR layer = max_level DOWN TO 0:
                layer_neighbors = neighbors[layer]
                
                // Remove edges between node_id and its neighbors
                REMOVE_NODE_FROM_LAYER(G, node_id, layer)
                
                // Local rewiring: connect neighbors to each other
                rewired = LOCAL_REWIRE(layer_neighbors, G, layer, M)
                stats.rewired_edges += rewired
            
            // Step 4: Remove node completely
            G.remove_node(node_id)
            
            // Step 5: Update entry point if needed
            IF entry_point == node_id:
                entry_point = representative OR find_new_entry_point()
            
            RETURN stats
        
        CHOOSE_REPRESENTATIVE(neighbors, strategy):
            IF strategy == "random":
                RETURN random.choice(neighbors)
            ELIF strategy == "highest_degree":
                RETURN argmax(n in neighbors: degree(n))
            ELIF strategy == "centroid":
                centroid = mean(vectors(neighbors))
                RETURN argmin(n in neighbors: distance(n.vector, centroid))
        
        LOCAL_REWIRE(neighbors, G, layer, M):
            rewired_count = 0
            FOR each pair (u, v) in neighbors:
                IF edge(u, v) does NOT exist:
                    candidates = [u, v] + neighbors
                    selected = GREEDY_SELECTION(u, candidates, M)
                    IF v in selected:
                        ADD_EDGE(u, v, layer)
                        rewired_count++
            RETURN rewired_count
        
        REMOVE_NODE_FROM_LAYER(G, node, layer):
            FOR each neighbor in G.get_neighbors(node, layer):
                G.remove_edge(node, neighbor, layer)
                G.remove_edge(neighbor, node, layer)
        
        Args:
            label: Label of the node to delete
            representative_strategy: Strategy for choosing representative
                - "random": Choose randomly
                - "highest_degree": Choose node with most connections
                - "centroid": Choose node closest to centroid of neighbors
        
        Returns:
            Dictionary with stats:
                - rewired_edges_count: Total edges rewired
                - representative_node: ID of chosen representative
                - neighbors_before: Dict mapping layer -> count of neighbors before deletion
                - neighbors_after: Dict mapping layer -> count of neighbors after rewiring
        """
        if label not in self._label_to_id:
            raise ValueError(f"Label '{label}' not found")

        node_id = self._label_to_id[label]
        if self._nodes[node_id].deleted:
            raise ValueError(f"Label '{label}' is already deleted")

        stats = {
            "rewired_edges_count": 0,
            "representative_node": None,
            "neighbors_before": {},
            "neighbors_after": {},
        }

        # Step 1: Collect neighbors at each layer
        node = self._nodes[node_id]
        neighbors_by_layer = {}
        
        for layer in range(node.level, -1, -1):
            layer_neighbors = node.neighbors.get(layer, set()).copy()
            neighbors_by_layer[layer] = [nid for nid in layer_neighbors if not self._nodes[nid].deleted]
            stats["neighbors_before"][layer] = len(neighbors_by_layer[layer])

        # Step 2: Choose representative node (use base layer neighbors)
        base_neighbors = neighbors_by_layer.get(0, [])
        if not base_neighbors:
            # No neighbors, just mark as deleted
            self._nodes[node_id].deleted = True
            if self._entry_point == node_id:
                self._reassign_entry_point()
            return stats

        representative = self._choose_representative(base_neighbors, representative_strategy)
        stats["representative_node"] = representative

        # Step 3: Process each layer from top to bottom
        for layer in range(node.level, -1, -1):
            layer_neighbors = neighbors_by_layer.get(layer, [])
            if not layer_neighbors:
                continue

            # Remove edges between node_id and its neighbors
            self._remove_node_from_layer(node_id, layer)

            # Local rewiring: connect neighbors to each other
            rewired = self._local_rewire_layer(layer_neighbors, layer)
            stats["rewired_edges_count"] += rewired

            # Track neighbors after rewiring
            if layer_neighbors:
                # Count neighbors after rewiring (approximate)
                stats["neighbors_after"][layer] = len(layer_neighbors)

        # Step 4: Mark node as deleted
        self._nodes[node_id].deleted = True

        # Step 5: Update entry point if needed
        if self._entry_point == node_id:
            # Try to use representative if it's at a high enough level
            if self._nodes[representative].level >= self._max_level:
                self._entry_point = representative
            else:
                self._reassign_entry_point()

        # print(
        #     f"[hnsw delete_with_rewiring] removed '{label}' (node_id={node_id}), "
        #     f"representative={self._nodes[representative].label}({representative}), "
        #     f"rewired {stats['rewired_edges_count']} edges"
        # )

        return stats

    def _choose_representative(self, neighbors: List[int], strategy: str) -> int:
        """
        Choose a representative node from neighbors.
        
        Args:
            neighbors: List of neighbor node IDs
            strategy: Strategy to use ("random", "highest_degree", "centroid")
        
        Returns:
            Node ID of chosen representative
        """
        if not neighbors:
            raise ValueError("Cannot choose representative from empty neighbor list")

        if strategy == "random":
            return self._rng.choice(neighbors)

        elif strategy == "highest_degree":
            # Choose node with highest total degree across all layers
            best_node = neighbors[0]
            best_degree = sum(
                len(self._nodes[best_node].neighbors.get(level, set()))
                for level in range(self._nodes[best_node].level + 1)
            )

            for node_id in neighbors[1:]:
                degree = sum(
                    len(self._nodes[node_id].neighbors.get(level, set()))
                    for level in range(self._nodes[node_id].level + 1)
                )
                if degree > best_degree:
                    best_degree = degree
                    best_node = node_id

            return best_node

        elif strategy == "centroid":
            # Compute centroid of neighbor vectors
            neighbor_vectors = [self._nodes[nid].vector for nid in neighbors]
            centroid = np.mean(neighbor_vectors, axis=0)

            # Choose node closest to centroid
            best_node = neighbors[0]
            best_dist = self._distance(self._nodes[best_node].vector, centroid)

            for node_id in neighbors[1:]:
                dist = self._distance(self._nodes[node_id].vector, centroid)
                if dist < best_dist:
                    best_dist = dist
                    best_node = node_id

            return best_node

        else:
            raise ValueError(f"Unknown strategy: {strategy}")

    def _local_rewire_layer(self, neighbors: List[int], layer: int) -> int:
        """
        Rewire neighbors locally at a specific layer.
        
        For each pair of neighbors, if they're not connected, evaluate
        whether to add an edge based on distance and degree constraints.
        
        Args:
            neighbors: List of neighbor node IDs
            layer: Layer to rewire at
        
        Returns:
            Number of edges rewired
        """
        if len(neighbors) <= 1:
            return 0

        rewired_count = 0

        # For each pair of neighbors
        for i, u in enumerate(neighbors):
            for v in neighbors[i + 1:]:
                # Check if edge already exists
                u_neighbors = self._nodes[u].neighbors.get(layer, set())
                if v in u_neighbors:
                    continue  # Edge already exists

                # Check degree constraints
                u_degree = len(u_neighbors)
                v_neighbors = self._nodes[v].neighbors.get(layer, set())
                v_degree = len(v_neighbors)

                # Only add edge if both nodes have room (degree < M)
                if u_degree < self.m and v_degree < self.m:
                    self._link(u, v, layer)
                    rewired_count += 1
                elif u_degree < self.m:
                    # u has room, check if v should connect to u
                    # Use greedy selection: add if v's current neighbors are worse
                    if self._should_add_edge_greedy(v, u, layer):
                        self._link(u, v, layer)
                        rewired_count += 1
                elif v_degree < self.m:
                    # v has room, check if u should connect to v
                    if self._should_add_edge_greedy(u, v, layer):
                        self._link(u, v, layer)
                        rewired_count += 1
                else:
                    # Both at max degree, use greedy selection to potentially replace
                    if self._should_replace_edge_greedy(u, v, layer):
                        rewired_count += 1

        return rewired_count

    def _should_add_edge_greedy(self, node_id: int, candidate: int, layer: int) -> bool:
        """
        Check if candidate should be added to node's neighbors using greedy selection.
        Only adds if candidate is closer than worst current neighbor.
        """
        neighbors = list(self._nodes[node_id].neighbors.get(layer, set()))
        if len(neighbors) < self.m:
            return True  # Has room, add it

        node_vec = self._nodes[node_id].vector
        candidate_dist = self._distance(node_vec, self._nodes[candidate].vector)

        # Find worst (farthest) neighbor
        worst_dist = -1
        for neighbor_id in neighbors:
            dist = self._distance(node_vec, self._nodes[neighbor_id].vector)
            if dist > worst_dist:
                worst_dist = dist

        return candidate_dist < worst_dist

    def _should_replace_edge_greedy(self, u: int, v: int, layer: int) -> bool:
        """
        Check if edge (u, v) should replace worst edge in either u or v's neighborhood.
        Returns True if replacement was made.
        """
        u_vec = self._nodes[u].vector
        v_vec = self._nodes[v].vector
        uv_dist = self._distance(u_vec, v_vec)

        u_neighbors = list(self._nodes[u].neighbors.get(layer, set()))
        v_neighbors = list(self._nodes[v].neighbors.get(layer, set()))

        # Check if we should replace in u's neighborhood
        if len(u_neighbors) >= self.m:
            worst_u = None
            worst_u_dist = -1
            for nid in u_neighbors:
                dist = self._distance(u_vec, self._nodes[nid].vector)
                if dist > worst_u_dist:
                    worst_u_dist = dist
                    worst_u = nid

            if worst_u is not None and uv_dist < worst_u_dist:
                # Remove worst edge and add new one
                self._unlink(u, worst_u, layer)
                self._link(u, v, layer)
                return True

        # Check if we should replace in v's neighborhood
        if len(v_neighbors) >= self.m:
            worst_v = None
            worst_v_dist = -1
            for nid in v_neighbors:
                dist = self._distance(v_vec, self._nodes[nid].vector)
                if dist > worst_v_dist:
                    worst_v_dist = dist
                    worst_v = nid

            if worst_v is not None and uv_dist < worst_v_dist:
                # Remove worst edge and add new one
                self._unlink(v, worst_v, layer)
                self._link(u, v, layer)
                return True

        return False

    def _remove_node_from_layer(self, node_id: int, layer: int) -> None:
        """
        Remove node from a specific layer by removing all edges to its neighbors.
        
        Args:
            node_id: ID of node to remove
            layer: Layer to remove from
        """
        node = self._nodes[node_id]
        neighbors = node.neighbors.get(layer, set()).copy()

        for neighbor_id in neighbors:
            # Remove edge from both directions
            self._unlink(node_id, neighbor_id, layer)

    def _unlink(self, src: int, dst: int, level: int) -> None:
        """
        Remove bidirectional edge between two nodes at a specific level.
        
        Args:
            src: Source node ID
            dst: Destination node ID
            level: Level to remove edge from
        """
        if level in self._nodes[src].neighbors:
            self._nodes[src].neighbors[level].discard(dst)
            if not self._nodes[src].neighbors[level]:
                del self._nodes[src].neighbors[level]

        if level in self._nodes[dst].neighbors:
            self._nodes[dst].neighbors[level].discard(src)
            if not self._nodes[dst].neighbors[level]:
                del self._nodes[dst].neighbors[level]

    def _reassign_entry_point(self) -> None:
        """Find a new entry point if the current one was deleted."""
        # Find the highest-level non-deleted node
        best_id = None
        best_level = -1

        for node_id, node in enumerate(self._nodes):
            if not node.deleted and node.level > best_level:
                best_level = node.level
                best_id = node_id

        if best_id is None:
            # All nodes deleted or no nodes exist
            self._entry_point = None
            self._max_level = -1
            # print("[hnsw delete] no valid entry point remaining")
            pass
        else:
            self._entry_point = best_id
            self._max_level = best_level
            # print(
            #     f"[hnsw delete] reassigned entry point to '{self._nodes[best_id].label}' "
            #     f"(node_id={best_id}, level={best_level})"
            # )

    def search(self, query: np.ndarray, k: int = 3, ef: int = 32, print_path: bool = True):
        if self._entry_point is None:
            raise ValueError("Index is empty, insert vectors first")

        query = np.asarray(query, dtype=np.float32).reshape(-1)
        if query.shape[0] != self.dim:
            raise ValueError(
                f"Expected query of dim {self.dim}, received {query.shape[0]}"
            )

        # Track path through upper layers
        upper_path = []
        entry = self._entry_point
        for lvl in range(self._max_level, 0, -1):
            entry, path_at_level = self._greedy_search(query, entry, lvl, track_path=True)
            upper_path.append((lvl, path_at_level))

        ef = max(ef, k)
        candidates, base_path = self._search_base_layer(query, entry, ef, track_path=True)
        top_k = sorted(candidates, key=lambda x: x[0])[:k]
        # Filter out deleted nodes (should already be filtered, but double-check)
        results = [
            {"label": self._nodes[node_id].label, "distance": float(dist)}
            for dist, node_id in top_k
            if not self._nodes[node_id].deleted
        ]

        if print_path:
            self._print_search_path(upper_path, base_path, entry)

        return results

    def _print_search_path(
        self, upper_path: List[tuple[int, List[int]]], base_path: List[int], entry: int
    ) -> None:
        """Print the search path through the HNSW graph."""
        print("\n" + "-" * 60)
        print("Search Path")
        print("-" * 60)
        
        if upper_path:
            print("Upper Layers (greedy search):")
            for level, path in upper_path:
                if path:
                    path_labels = [f"{self._nodes[node_id].label}({node_id})" for node_id in path]
                    print(f"  Level {level}: {' -> '.join(path_labels)}")
            print()
        
        print("Base Layer (best-first search):")
        if base_path:
            path_labels = [f"{self._nodes[node_id].label}({node_id})" for node_id in base_path]
            print(f"  Path: {' -> '.join(path_labels)}")
            print(f"  Total nodes visited: {len(base_path)}")
        else:
            print("  No path (empty index)")
        
        print("-" * 60 + "\n")

    def print_neighbors(self) -> None:
        """
        Print the neighbor structure for all nodes in the HNSW graph.
        Shows neighbors at each level for each node, along with their labels.
        """
        if not self._nodes:
            print("HNSW index is empty")
            return

        print("\n" + "=" * 60)
        print("HNSW Neighbor Structure")
        print("=" * 60)
        print(f"Entry Point: {self._nodes[self._entry_point].label if self._entry_point is not None else 'None'} "
              f"(node_id={self._entry_point}, level={self._max_level})")
        print()

        for node_id, node in enumerate(self._nodes):
            status = "[DELETED]" if node.deleted else ""
            print(f"Node {node_id}: '{node.label}' (level={node.level}) {status}")
            
            if not node.neighbors:
                print("  No neighbors")
            else:
                # Sort levels in descending order for better readability
                for level in sorted(node.neighbors.keys(), reverse=True):
                    neighbor_ids = sorted(node.neighbors[level])
                    if neighbor_ids:
                        neighbor_labels = [
                            f"{self._nodes[nid].label}({nid})"
                            for nid in neighbor_ids
                            if nid < len(self._nodes)
                        ]
                        print(f"  Level {level}: {neighbor_labels}")
            
            print()

        print("=" * 60)

    def _distance(self, vec_a: np.ndarray, vec_b: np.ndarray) -> float:
        return float(np.linalg.norm(vec_a - vec_b))

    def _sample_level(self) -> int:
        level = 0
        while self._rng.random() < self.level_probability:
            level += 1
        return level

    def _greedy_search(self, query: np.ndarray, entry_id: int, level: int, track_path: bool = False):
        best = entry_id
        path = [entry_id] if track_path else None
        
        # Skip if entry point is deleted (shouldn't happen, but safety check)
        if self._nodes[best].deleted:
            # Find any valid node at this level
            for node_id, node in enumerate(self._nodes):
                if not node.deleted and node.level >= level:
                    best = node_id
                    if track_path:
                        path = [node_id]
                    break
            else:
                return (entry_id, [entry_id]) if track_path else entry_id  # Fallback

        best_distance = self._distance(query, self._nodes[best].vector)
        improved = True

        while improved:
            improved = False
            neighbors = self._nodes[best].neighbors.get(level, set())
            for neighbor in neighbors:
                # Skip deleted neighbors
                if self._nodes[neighbor].deleted:
                    continue
                dist = self._distance(query, self._nodes[neighbor].vector)
                if dist < best_distance:
                    best_distance = dist
                    best = neighbor
                    improved = True
                    if track_path:
                        path.append(neighbor)
        
        if track_path:
            return best, path
        return best

    def _layer_candidates(self, vector: np.ndarray, level: int) -> List[int]:
        return [
            idx
            for idx, node in enumerate(self._nodes[:-1])
            if node.level >= level and not node.deleted
        ]

    def _select_neighbors(
        self, vector: np.ndarray, candidates: List[int], m: int
    ) -> List[int]:
        if not candidates:
            return []
        sorted_candidates = sorted(
            candidates,
            key=lambda idx: self._distance(vector, self._nodes[idx].vector),
        )
        return sorted_candidates[:m]

    def _link(self, src: int, dst: int, level: int) -> None:
        self._nodes[src].neighbors.setdefault(level, set()).add(dst)
        self._nodes[dst].neighbors.setdefault(level, set()).add(src)

        self._prune_neighbors(src, level)
        self._prune_neighbors(dst, level)

    def _prune_neighbors(self, node_id: int, level: int) -> None:
        neighbors = list(self._nodes[node_id].neighbors.get(level, set()))
        if len(neighbors) <= self.m:
            return
        node_vector = self._nodes[node_id].vector
        neighbors.sort(key=lambda idx: self._distance(node_vector, self._nodes[idx].vector))
        pruned = set(neighbors[: self.m])
        self._nodes[node_id].neighbors[level] = pruned

    def _search_base_layer(
        self, query: np.ndarray, entry_id: int, ef: int, track_path: bool = False
    ):
        visited = {entry_id}
        candidate_heap: List[tuple[float, int]] = []
        result_set: List[tuple[float, int]] = []
        path = [] if track_path else None

        # Skip if entry point is deleted (shouldn't happen, but safety check)
        if not self._nodes[entry_id].deleted:
            entry_dist = self._distance(query, self._nodes[entry_id].vector)
            heapq.heappush(candidate_heap, (entry_dist, entry_id))
            heapq.heappush(result_set, (-entry_dist, entry_id))
            if track_path:
                path.append(entry_id)

        while candidate_heap:
            current_dist, current = heapq.heappop(candidate_heap)
            
            # Skip deleted nodes in results
            if self._nodes[current].deleted:
                continue
                
            worst_result_dist = -result_set[0][0]
            if len(result_set) >= ef and current_dist > worst_result_dist:
                break

            for neighbor in self._nodes[current].neighbors.get(0, set()):
                if neighbor in visited:
                    continue
                # Skip deleted neighbors
                if self._nodes[neighbor].deleted:
                    continue
                visited.add(neighbor)
                dist = self._distance(query, self._nodes[neighbor].vector)
                heapq.heappush(candidate_heap, (dist, neighbor))
                heapq.heappush(result_set, (-dist, neighbor))
                if track_path:
                    path.append(neighbor)
                if len(result_set) > ef:
                    heapq.heappop(result_set)

        # Filter out deleted nodes from final results
        results = [
            (-dist, node_id)
            for dist, node_id in result_set
            if not self._nodes[node_id].deleted
        ]
        
        if track_path:
            return results, path
        return results


def main():
    # Configuration parameters
    N = 10000  # Number of vectors
    D = 128  # Dimensions
    K = 5   # Number of nearest neighbors to retrieve
    EF = 20 # Search parameter
    
    print("=" * 70)
    print("HNSW Deletion Comparison: Tombstone vs Local Rewiring")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  N (number of vectors): {N}")
    print(f"  D (dimensions): {D}")
    print(f"  K (nearest neighbors): {K}")
    print(f"  EF (search parameter): {EF}")
    
    # Generate random vectors
    rng = Random()
    data_points = {}
    for i in range(N):
        label = f"vec_{i}"
        vector = np.random.rand(D).astype(np.float32)
        data_points[label] = vector
    
    # Generate a random query vector
    query = np.random.rand(D).astype(np.float32)
    
    print(f"\nGenerated {N} random {D}-dimensional vectors")
    print(f"Query vector: {query[:3]}... (showing first 3 dims)")
    
    # Create two identical HNSW instances for comparison
    # Use same seed to ensure identical structure
    seed = 42
    hnsw_tombstone = SimpleHNSW(dim=D, m=16, level_probability=0.5, ef_construction=16, seed=seed)
    hnsw_rewiring = SimpleHNSW(dim=D, m=16, level_probability=0.5, ef_construction=16, seed=seed)
    
    print(f"\n[Inserting {N} vectors into both HNSW instances...]")
    for label, vector in data_points.items():
        hnsw_tombstone.insert(label, vector)
        hnsw_rewiring.insert(label, vector)
    
    # Get baseline search results before deletion
    print(f"\n[Baseline search before deletion]")
    baseline_results = hnsw_tombstone.search(query, k=K, ef=EF, print_path=False)
    
    # Choose a random node to delete (not the first or last to avoid edge cases)
    node_to_delete = rng.choice(list(data_points.keys())[1:-1])
    print(f"\n[Deleting node: '{node_to_delete}']")
    
    # Delete using tombstone approach with timing
    print(f"\n[Method 1: Tombstone Deletion]")
    start_time = time.perf_counter()
    hnsw_tombstone.delete(node_to_delete)
    tombstone_deletion_time = time.perf_counter() - start_time
    
    start_time = time.perf_counter()
    tombstone_results = hnsw_tombstone.search(query, k=K, ef=EF, print_path=False)
    tombstone_search_time = time.perf_counter() - start_time
    
    # Delete using local rewiring approach with timing
    print(f"\n[Method 2: Local Rewiring Deletion]")
    start_time = time.perf_counter()
    stats = hnsw_rewiring.delete_with_local_rewiring(node_to_delete, representative_strategy="random")
    rewiring_deletion_time = time.perf_counter() - start_time
    
    start_time = time.perf_counter()
    rewiring_results = hnsw_rewiring.search(query, k=K, ef=EF, print_path=False)
    rewiring_search_time = time.perf_counter() - start_time
    
    # Compare results
    print("\n" + "=" * 70)
    print("COMPARISON ANALYSIS")
    print("=" * 70)
    
    print(f"\nBaseline (before deletion) - Top {K} results:")
    for rank, match in enumerate(baseline_results, start=1):
        print(f"  {rank}. {match['label']:10s} distance={match['distance']:.6f}")
    
    print(f"\nTombstone Deletion - Top {K} results:")
    for rank, match in enumerate(tombstone_results, start=1):
        print(f"  {rank}. {match['label']:10s} distance={match['distance']:.6f}")
    
    print(f"\nLocal Rewiring Deletion - Top {K} results:")
    for rank, match in enumerate(rewiring_results, start=1):
        print(f"  {rank}. {match['label']:10s} distance={match['distance']:.6f}")
    
    # Check if results match
    print("\n" + "-" * 70)
    print("RESULT COMPARISON:")
    print("-" * 70)
    
    tombstone_labels = [r['label'] for r in tombstone_results]
    rewiring_labels = [r['label'] for r in rewiring_results]
    
    labels_match = tombstone_labels == rewiring_labels
    distances_match = all(
        abs(t['distance'] - r['distance']) < 1e-6
        for t, r in zip(tombstone_results, rewiring_results)
    )
    
    if labels_match and distances_match:
        print("✓ SUCCESS: Both deletion methods return IDENTICAL results!")
        print(f"  - Labels match: {labels_match}")
        print(f"  - Distances match: {distances_match}")
    else:
        print("✗ DIFFERENCE: Results differ between deletion methods")
        if not labels_match:
            print(f"  - Labels differ:")
            print(f"    Tombstone: {tombstone_labels}")
            print(f"    Rewiring:  {rewiring_labels}")
        if not distances_match:
            print(f"  - Distances differ:")
            for i, (t, r) in enumerate(zip(tombstone_results, rewiring_results)):
                if abs(t['distance'] - r['distance']) >= 1e-6:
                    print(f"    Rank {i+1}: Tombstone={t['distance']:.6f}, Rewiring={r['distance']:.6f}")
    
    # Verify deleted node is not in results
    deleted_in_tombstone = any(r['label'] == node_to_delete for r in tombstone_results)
    deleted_in_rewiring = any(r['label'] == node_to_delete for r in rewiring_results)
    
    print(f"\n[Verification]")
    print(f"  Deleted node '{node_to_delete}' in tombstone results: {deleted_in_tombstone}")
    print(f"  Deleted node '{node_to_delete}' in rewiring results: {deleted_in_rewiring}")
    
    if not deleted_in_tombstone and not deleted_in_rewiring:
        print("  ✓ Deleted node correctly excluded from both methods")
    else:
        print("  ✗ WARNING: Deleted node still appears in results!")
    
    print(f"\n[Rewiring Statistics]")
    print(f"  Representative node: {hnsw_rewiring._nodes[stats['representative_node']].label}")
    print(f"  Total edges rewired: {stats['rewired_edges_count']}")
    
    # Performance comparison
    print("\n" + "-" * 70)
    print("PERFORMANCE COMPARISON:")
    print("-" * 70)
    
    print(f"\nDeletion Time:")
    print(f"  Tombstone:  {tombstone_deletion_time * 1000:.4f} ms")
    print(f"  Rewiring:   {rewiring_deletion_time * 1000:.4f} ms")
    
    if tombstone_deletion_time < rewiring_deletion_time:
        speedup = rewiring_deletion_time / tombstone_deletion_time
        print(f"  → Tombstone is {speedup:.2f}x faster")
    elif rewiring_deletion_time < tombstone_deletion_time:
        speedup = tombstone_deletion_time / rewiring_deletion_time
        print(f"  → Rewiring is {speedup:.2f}x faster")
    else:
        print(f"  → Both methods take the same time")
    
    print(f"\nSearch Time (after deletion):")
    print(f"  Tombstone:  {tombstone_search_time * 1000:.4f} ms")
    print(f"  Rewiring:   {rewiring_search_time * 1000:.4f} ms")
    
    if tombstone_search_time < rewiring_search_time:
        search_speedup = rewiring_search_time / tombstone_search_time
        print(f"  → Tombstone search is {search_speedup:.2f}x faster")
    elif rewiring_search_time < tombstone_search_time:
        search_speedup = tombstone_search_time / rewiring_search_time
        print(f"  → Rewiring search is {search_speedup:.2f}x faster")
    else:
        print(f"  → Both search methods take the same time")
    
    print(f"\nTotal Time (deletion + search):")
    tombstone_total = tombstone_deletion_time + tombstone_search_time
    rewiring_total = rewiring_deletion_time + rewiring_search_time
    print(f"  Tombstone:  {tombstone_total * 1000:.4f} ms")
    print(f"  Rewiring:   {rewiring_total * 1000:.4f} ms")
    
    if tombstone_total < rewiring_total:
        total_speedup = rewiring_total / tombstone_total
        print(f"  → Tombstone total is {total_speedup:.2f}x faster")
    elif rewiring_total < tombstone_total:
        total_speedup = tombstone_total / rewiring_total
        print(f"  → Rewiring total is {total_speedup:.2f}x faster")
    else:
        print(f"  → Both methods take the same total time")
    
    print("\n" + "=" * 70)
    print("Analysis Complete")
    print("=" * 70)


if __name__ == "__main__":
    main()