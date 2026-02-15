#include <algorithm>
#include <queue>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <optional>
#include <random>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

struct HNSWNode {
    std::string label;
    std::vector<float> vector;
    int level;
    std::unordered_map<int, std::unordered_set<int>> neighbors;
    bool deleted{false};
};

enum class RUVariant {
    LR_RN,  // Local Rewiring - Representative Neighbor (your algorithm)
    MNRU    // Multi-Neighbor Replace Update (paper-style)
};

class SimpleHNSW {
  public:
    SimpleHNSW(int dim,
               int m = 4,
               float level_probability = 0.5f,
               int ef_construction = 16,
               std::optional<uint64_t> seed = std::nullopt)
        : dim_(dim),
          m_(m),
          level_probability_(level_probability),
          ef_construction_(ef_construction),
          rng_(seed.has_value() ? seed.value() : std::random_device{}()) {}

    void insert(const std::string &label, const std::vector<float> &vector) {
        
        if (label_to_id_.count(label) and nodes_[label_to_id_[label]].deleted == false) {
            throw std::runtime_error("Label already exists: " + label);
        }
        if (static_cast<int>(vector.size()) != dim_) {
            throw std::runtime_error("Vector dimensionality mismatch");
        }

        int level = sample_level();
        HNSWNode node{label, vector, level};
        int node_id = static_cast<int>(nodes_.size());
        nodes_.push_back(node);
        label_to_id_[label] = node_id;

        if (entry_point_.has_value() == false) {
            entry_point_ = node_id;
            max_level_ = level;
            return;
        }

        int current = entry_point_.value();
        if (level > max_level_) {
            max_level_ = level;
            entry_point_ = node_id;
        }

        for (int lvl = max_level_; lvl > level; --lvl) {
            current = greedy_search(vector, current, lvl, false).first;
        }

        for (int lvl = std::min(level, max_level_); lvl >= 0; --lvl) {
            // std::cout << "Searching base layer at level " << label << "\n";
            // auto [cand_set, _] = search_base_layer(vector, current, ef_construction_, false);
            auto [cand_set, _] = search_layer(vector, current, lvl, ef_construction_, false);
            std::vector<int> candidates;
            for (auto &p: cand_set) candidates.push_back(p.second);
            auto neighbors = select_neighbors_heuristic(vector, candidates, m_);
            for (int other_id : neighbors) {
                link(node_id, other_id, lvl, 1);
            }
        }
    }

    // void replace_update_lr(const std::string &label,
    //     const std::vector<float> &new_vec) {
    //     if (!allow_replace_deleted_) {
    //         // fall back to normal insert
    //         insert(label, new_vec);
    //         return;
    //     }

    //     if (static_cast<int>(new_vec.size()) != dim_) {
    //         throw std::runtime_error("Vector dimensionality mismatch");
    //     }

    //     // If label already exists and is not deleted => reject
    //     if (label_to_id_.count(label) && !nodes_[label_to_id_[label]].deleted) {
    //         throw std::runtime_error("Label already exists: " + label);
    //     }

    //     // If we have a deleted slot, reuse it (hnswlib replace_deleted path)
    //     if (!deleted_pool_.empty()) {
    //         int reused_id = deleted_pool_.back();
    //         deleted_pool_.pop_back();

    //         // remove old label mapping
    //         std::string old_label = nodes_[reused_id].label;
    //         if (!old_label.empty()) {
    //             auto it = label_to_id_.find(old_label);
    //             if (it != label_to_id_.end() && it->second == reused_id) {
    //                 label_to_id_.erase(it);
    //             }
    //         }

    //         // overwrite data
    //         nodes_[reused_id].label   = label;
    //         nodes_[reused_id].vector  = new_vec;
    //         nodes_[reused_id].deleted = false;

    //         // update mapping
    //         label_to_id_[label] = reused_id;

            
    //         rebuild_neighbors_lr_rn_all_levels(reused_id);

    //         // entry point might have been deleted earlier; ensure it's valid
    //         if (!entry_point_.has_value() || nodes_[entry_point_.value()].deleted) {
    //             reassign_entry_point();
    //         }
    //         return;
    //     }

    //     // No reusable slot -> normal insert
    //     insert(label, new_vec);
    // }

    void replace_update_mnru(const std::string &label,
        const std::vector<float> &new_vec) {
        if (!allow_replace_deleted_) {
            // fall back to normal insert
            insert(label, new_vec);
            return;
        }

        if (static_cast<int>(new_vec.size()) != dim_) {
            throw std::runtime_error("Vector dimensionality mismatch");
        }

        // If label already exists and is not deleted => reject
        if (label_to_id_.count(label) && !nodes_[label_to_id_[label]].deleted) {
            throw std::runtime_error("Label already exists: " + label);
        }

        // If we have a deleted slot, reuse it (hnswlib replace_deleted path)
        if (!deleted_pool_.empty()) {
            int reused_id = deleted_pool_.back();
            deleted_pool_.pop_back();

            // remove old label mapping
            std::string old_label = nodes_[reused_id].label;
            if (!old_label.empty()) {
                auto it = label_to_id_.find(old_label);
                if (it != label_to_id_.end() && it->second == reused_id) {
                    label_to_id_.erase(it);
                }
            }

            // overwrite data
            nodes_[reused_id].label   = label;
            nodes_[reused_id].vector  = new_vec;
            nodes_[reused_id].deleted = false;

            // update mapping
            label_to_id_[label] = reused_id;

            
            rebuild_neighbors_mnru_all_levels(reused_id);

            // entry point might have been deleted earlier; ensure it's valid
            if (!entry_point_.has_value() || nodes_[entry_point_.value()].deleted) {
                reassign_entry_point();
            }
            return;
        }

        // No reusable slot -> normal insert
        insert(label, new_vec);
    }

    void insert_replace_deleted(const std::string &label,
                            const std::vector<float> &vector) {
        // if (!allow_replace_deleted_) {
        //     // fall back to normal insert
        //     insert(label, vector);
        //     return;
        // }

        // if (static_cast<int>(vector.size()) != dim_) {
        //     throw std::runtime_error("Vector dimensionality mismatch");
        // }

        // // If label already exists and is not deleted => reject
        // if (label_to_id_.count(label) && !nodes_[label_to_id_[label]].deleted) {
        //     throw std::runtime_error("Label already exists: " + label);
        // }

        // If we have a deleted slot, reuse it (hnswlib replace_deleted path)
        // if (!deleted_pool_.empty()) {
        int reused_id = label_to_id_[label];
        // deleted_pool_.pop_back();

        // remove old label mapping
        std::string old_label = nodes_[reused_id].label;
        if (!old_label.empty()) {
            auto it = label_to_id_.find(old_label);
            if (it != label_to_id_.end() && it->second == reused_id) {
                label_to_id_.erase(it);
            }
        }

        // overwrite data
        nodes_[reused_id].label   = label;
        nodes_[reused_id].vector  = vector;
        nodes_[reused_id].deleted = false;

        // update mapping
        label_to_id_[label] = reused_id;

        // IMPORTANT: do local repair similar to hnswlib updatePoint(..., prob=1.0)
        update_point_local(reused_id, /*updateNeighborProbability=*/1.0f, /*alpha=*/1.1f);

        // entry point might have been deleted earlier; ensure it's valid
        // if (!entry_point_.has_value() || nodes_[entry_point_.value()].deleted) {
        reassign_entry_point();
        // }
        return;

        // No reusable slot -> normal insert
        // insert(label, vector);
    }

    // --- MNRU Algorithm 3: update(deletedPoint, data, label) ---
    // Reuse a deleted node-id, inherit its level, and reinsert like standard HNSW.
    struct MNRUUpdateStats {
        int reused_id{-1};
        int inherited_level{-1};
        int links_added{0};
        int layers_processed{0};
    };

    MNRUUpdateStats mnru_update_deleted_id(
        int deleted_id,
        const std::string& new_label,
        const std::vector<float>& new_vec,
        float alpha = 1.0f)
        
    {
        // auto it = label_to_id_.find(reused_label);
        // if (it == label_to_id_.end()) {
        //     throw std::runtime_error("mnru_update_deleted_id: label not found: " + reused_label);
        // }
        // int deleted_id = it->second;
        if (deleted_id < 0 || deleted_id >= (int)nodes_.size())
            throw std::runtime_error("mnru_update_deleted_id: invalid deleted_id");

        if (!nodes_[deleted_id].deleted)
            throw std::runtime_error("mnru_update_deleted_id: target id is not deleted");

        if ((int)new_vec.size() != dim_)
            throw std::runtime_error("mnru_update_deleted_id: dimensionality mismatch");

        // If label exists and is alive -> reject
        if (label_to_id_.count(new_label) && !nodes_[label_to_id_[new_label]].deleted)
            throw std::runtime_error("mnru_update_deleted_id: label already exists: " + new_label);

        MNRUUpdateStats st;
        st.reused_id = deleted_id;

        // 0) Remove old label mapping (if any)
        std::string old_label = nodes_[deleted_id].label;
        if (!old_label.empty()) {
            auto it = label_to_id_.find(old_label);
            if (it != label_to_id_.end() && it->second == deleted_id) {
                label_to_id_.erase(it);
            }
        }

        // 1) Clear old connectivity completely (important: treat as "fresh insert")
        //    Unlink at all levels the deleted node had.
        //    (We must do this before overwriting node.level if we rely on it.)
        // int inherited_level = nodes_[deleted_id].level;
        // for (int lvl = inherited_level; lvl >= 0; --lvl) {
        //     auto it_old = nodes_[deleted_id].neighbors.find(lvl);
        //     if (it_old != nodes_[deleted_id].neighbors.end()) {
        //         std::vector<int> old(it_old->second.begin(), it_old->second.end());
        //         // std::cout << "unlinking " << deleted_id << " " << old << " " << lvl << "\n";
        //         for (int v : old) {
        //             // If the deleted node had no neighbors at this level, it could potentially create empty or missing links
        //             unlink(deleted_id, v, lvl);
        //             // if (nodes_[v].neighbors.count(lvl) > 1 || lvl == 0) {
        //             //     unlink(deleted_id, v, lvl);
        //             // }
        //         }
        //     }
        //     // nodes_[deleted_id].neighbors.erase(lvl);
        // }
        
        // std::cout << "\n Inherited Level " << inherited_level << "\n max layer" << max_level_ << "\n neighbors" << nodes_[deleted_id].neighbors;
        int inherited_level = nodes_[deleted_id].level;

        // for (int lvl = inherited_level; lvl >= 0; --lvl) {
        //     auto it = nodes_[deleted_id].neighbors.find(lvl);
        //     if (it == nodes_[deleted_id].neighbors.end()) continue;

        //     std::vector<int> old(it->second.begin(), it->second.end());

        //     // Filter alive + valid-at-level
        //     std::vector<int> alive;
        //     alive.reserve(old.size());
        //     for (int v : old) {
        //         if (v == deleted_id) continue;
        //         if (nodes_[v].deleted) continue;
        //         if (nodes_[v].level < lvl) continue;
        //         alive.push_back(v);
        //     }

        //     // If 0/1 neighbors at this level, nothing much to rewire
        //     // (but we still must remove the edge(s) to deleted_id cleanly)
        //     int r = -1;
        //     if (alive.size() >= 2) {
        //         // pick representative: closest to deleted_id (or any stable choice)
        //         r = alive[0];
        //         float best = distance(nodes_[deleted_id].vector, nodes_[r].vector);
        //         for (int v : alive) {
        //             float d = distance(nodes_[deleted_id].vector, nodes_[v].vector);
        //             if (d < best) { best = d; r = v; }
        //         }
        //     }

        //     // REWIRE: ensure nodes won't be isolated after unlink
        //     // Only matters for lvl>0; at lvl==0 isolation is catastrophic.
        //     if (r != -1) {
        //         for (int v : alive) {
        //             if (v == r) continue;

        //             // If v would become empty after removing deleted_id, give it a replacement
        //             // (count includes deleted_id currently)
        //             size_t deg_v = nodes_[v].neighbors.count(lvl) ? nodes_[v].neighbors[lvl].size() : 0;

        //             bool would_isolate = (deg_v <= 1); // i.e., only linked to deleted_id
        //             if (would_isolate || lvl == 0) {
        //                 link(r, v, lvl, alpha);
        //             }
        //         }

        //         // Optional: prune locally (keeps degree caps sane)
        //         prune_neighbors(r, lvl, alpha);
        //         for (int v : alive) prune_neighbors(v, lvl, alpha);
        //     }

        //     // NOW unlink everything (no conditional skipping)
        //     for (int v : alive) {
        //         unlink(deleted_id, v, lvl);
        //     }

        //     // Remove deleted node’s level adjacency
        //     nodes_[deleted_id].neighbors.erase(lvl);
        // }
        // auto start_update_deleted_neighbors = std::chrono::steady_clock::now();
        // for (int lvl = inherited_level; lvl >= 0; --lvl) {
        //     auto it_old = nodes_[deleted_id].neighbors.find(lvl);
        //     if (it_old != nodes_[deleted_id].neighbors.end()) {
        //         std::vector<int> old(it_old->second.begin(), it_old->second.end());
        //         // std::cout << "unlinking " << deleted_id << " " << old << " " << lvl << "\n";
        //         for (int v : old) {
        //             // If the deleted node had no neighbors at this level, it could potentially create empty or missing links
        //             unlink(deleted_id, v, lvl);
        //             // if (nodes_[v].neighbors.count(lvl) > 1 || lvl == 0) {
        //             //     unlink(deleted_id, v, lvl);
        //             // }
        //         }
        //     }
        //     nodes_[deleted_id].neighbors.erase(lvl);
        // }
        // auto end_update_deleted_neighbors = std::chrono::steady_clock::now();
        // auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end_update_deleted_neighbors - start_update_deleted_neighbors);
        // std::cout << "unlinking took " << elapsed.count() << " ms\n";
        // Ensure no leftover levels
        nodes_[deleted_id].neighbors.clear();
        // int level = sample_level();
        // 2) Overwrite payload, inherit level
        nodes_[deleted_id].label   = new_label;
        nodes_[deleted_id].vector  = new_vec;
        nodes_[deleted_id].level   = inherited_level; // inherit from deleted point
        nodes_[deleted_id].deleted = false;
        label_to_id_[new_label]    = deleted_id;

        // int level = sample_level();
        // HNSWNode node{new_label, new_vec, level};
        // int node_id = static_cast<int>(nodes_.size());
        // nodes_.push_back(node);
        // label_to_id_[new_label] = node_id;
        // nodes_[deleted_id] = node;


        st.inherited_level = inherited_level;

        // // 3) Ensure we have a valid entry point
        // if (!entry_point_.has_value() || nodes_[entry_point_.value()].deleted) {
        //     reassign_entry_point();
        // }
        // if (!entry_point_.has_value()) {
        //     // Index had only deleted nodes before; this node becomes the entry point.
        //     entry_point_ = deleted_id;
        //     max_level_   = inherited_level;
        //     return st;
        // }

        // // L = current max level of index
        // int L = max_level_;
        // int Lmax = inherited_level;
        // // int Lmax = level;

        // int ep = entry_point_.value();

        // // 4) From top layer down to (Lmax+1): SEARCH-LAYER(data, ep, 1, lc)
        // //    With ef=1 this is essentially greedy descent in that layer.
        // for (int lc = L; lc > Lmax; --lc) {
        //     // Greedy step at lc
        //     // auto [w, _path] = search_layer(new_vec, ep, lc, 1, /*track_path=*/false);
        //     // if (!w.empty()) {
        //     //     ep = w.front().second;
        //     // }
        //     ep = greedy_search(new_vec, ep, lc, /*track_path=*/false).first;
        // }

        // // 5) For layer = Lmax .. 0:
        // //    w <- SEARCH-LAYER(data, ep, efConstruction, layer)
        // //    neighbors <- SELECT-NEIGHBORS(data, w, M, layer)
        // //    addBidirectionalConnections(neighbors, data, layer)
        // //    ep <- getNearestElement(w, data)  (for next layer)
        // for (int layer = Lmax; layer >= 0; --layer) {
        //     st.layers_processed++;

        //     // Use your generic search at this layer (bounded by ef_construction_)
        //     auto [w, _path] = search_layer(new_vec, ep, ef_construction_, /*track_path=*/false);

        //     // Convert w (pair<dist,id>) into candidate ids
        //     std::vector<int> cand_ids;
        //     cand_ids.reserve(w.size());
        //     for (auto &p : w) cand_ids.push_back(p.second);

        //     // Use M0 = 2*M at level 0 (this is important for stability)
        //     // int M_layer = (layer == 0) ? (2 * m_) : m_;

        //     // Neighbor selection
        //     auto neighs = select_neighbors_heuristic(new_vec, cand_ids, m_);
        //     if (neighs.empty()) {
        //         // fallback
        //         std::cout << "I am here, didn't find any neighbors";
        //         neighs = select_neighbors_simple(new_vec, cand_ids, m_);
        //     }

        //     // Connect bidirectionally (link() prunes both sides)
        //     for (int nb : neighs) {
        //         if (nb == deleted_id) continue;
        //         if (nodes_[nb].deleted) continue;
        //         if (nodes_[nb].level < layer) continue;
        //         link(deleted_id, nb, layer);
        //         // link(nb, deleted_id, layer);
        //         st.links_added++;
        //     }

        //     // Update ep for next layer: nearest element in w
        //     // (w is sorted ascending at end of search_layer; if not guaranteed, pick min explicitly)
        //     if (!w.empty()) {
        //         ep = w.front().second;
        //     }
        // }

        // // 6) If this new node has a higher level than current max, update entry point
        // if (nodes_[deleted_id].level > max_level_) {
        //     max_level_ = nodes_[deleted_id].level;
        //     entry_point_ = deleted_id;
        // }
        auto start = std::chrono::steady_clock::now();
        repair_connections_for_update(deleted_id, alpha);
        auto end = std::chrono::steady_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        // std::cout << "Time taken: " << duration.count() << " ms" << std::endl;
        // if (!entry_point_.has_value() || nodes_[entry_point_.value()].deleted) {
        reassign_entry_point();
            // }
        return st;
    }

    // Convenience overload: reuse ANY deleted slot (like hnswlib replace_deleted),
    // but performs full MNRU Algo-3 insertion, not the local update.
    MNRUUpdateStats mnru_update_replace_any_deleted(
        const std::string& new_label,
        const std::vector<float>& new_vec,
        float alpha)
    {
        if (deleted_pool_.empty()) {
            // If no deleted slots, fall back to normal insert
            insert(new_label, new_vec);
            return {};
        }
        int deleted_id = deleted_pool_.back();
        deleted_pool_.pop_back();
        return mnru_update_deleted_id(deleted_id, new_label, new_vec, alpha);
    }



    void replace_update(const std::string &label,
                    const std::vector<float> &new_vec) {
        auto it = label_to_id_.find(label);
        if (it == label_to_id_.end()) {
            throw std::runtime_error("replace_update: label not found: " + label);
        }
        int node_id = it->second;
        if (static_cast<int>(new_vec.size()) != dim_) {
            throw std::runtime_error("replace_update: dimensionality mismatch");
        }

        HNSWNode &node = nodes_[node_id];
        node.vector = new_vec;
        node.deleted = false;

        // hnswlib-like: local repair (node + neighbors), no global rebuild
        update_point_local(node_id, /*updateNeighborProbability=*/1.0f, /*alpha=*/1.1f);

        if (!entry_point_.has_value() || nodes_[entry_point_.value()].deleted) {
            reassign_entry_point();
        }
    }

    void delete_node(const std::string &label) {
        if (!label_to_id_.count(label)) {
            throw std::runtime_error("Label not found: " + label);
        }
        int node_id = label_to_id_[label];
        if (nodes_[node_id].deleted) {
            throw std::runtime_error("Label already deleted: " + label);
        }
        nodes_[node_id].deleted = true;

        if (allow_replace_deleted_) {
            deleted_pool_.push_back(node_id);
        }
        if (entry_point_.has_value() && entry_point_.value() == node_id) {
            reassign_entry_point();
        }
    }

    struct SearchResult {
        std::string label;
        float distance;
    };

    std::vector<SearchResult> search(const std::vector<float> &query,
                                     int k,
                                     int ef,
                                     bool track_path = false) {
        if (!entry_point_.has_value()) {
            throw std::runtime_error("Index is empty");
        }
        if (static_cast<int>(query.size()) != dim_) {
            throw std::runtime_error("Query dimensionality mismatch");
        }

        std::vector<std::pair<int, std::vector<int>>> upper_path;
        int entry = entry_point_.value();
        auto start_greedy_search = std::chrono::steady_clock::now();
        for (int lvl = max_level_; lvl > 0; --lvl) {
            auto [best, path] = greedy_search(query, entry, lvl, track_path);
            entry = best;
            if (track_path) {
                upper_path.emplace_back(lvl, path);
            }
        }
        auto end_greedy_search = std::chrono::steady_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_greedy_search - start_greedy_search);
        // std::cout << "Time taken for greedy search: " << duration.count() << " ms" << std::endl;

        auto start_base_search = std::chrono::steady_clock::now();
        auto [candidates, base_path] =
            search_layer(query, entry, 0, std::max(k, ef), track_path);
        auto end_base_search = std::chrono::steady_clock::now();
        duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_base_search - start_base_search);
        // std::cout << "Time taken for base search: " << duration.count() << " ms" << std::endl;

        std::sort(candidates.begin(), candidates.end(),
                  [](const auto &a, const auto &b) { return a.first < b.first; });

        std::vector<SearchResult> results;
        auto start_final_search = std::chrono::steady_clock::now();
        for (size_t i = 0; i < candidates.size() && static_cast<int>(i) < k; ++i) {
            int node_id = candidates[i].second;
            if (!nodes_[node_id].deleted) {
                results.push_back({nodes_[node_id].label, candidates[i].first});
            }
        }
        auto end_final_search = std::chrono::steady_clock::now();
        duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_final_search - start_final_search);
        // std::cout << "Time taken for final search: " << duration.count() << " ms" << std::endl;
        return results;
    }

    // local rewiring deletion (simplified)
    struct RewireStats {
        int representative_node{-1};
        int rewired_edges_count{0};
    };

    // MNRU deletion stats
    struct MNRUStats {
        int nodes_touched{0};
        int total_candidates{0};
        int distance_evals{0};
    };

    RewireStats delete_with_local_rewiring(const std::string &label, const std::vector<float>& new_vec) {
        if (!label_to_id_.count(label)) {
            throw std::runtime_error("Label not found: " + label);
        }
        int node_id = label_to_id_[label];
        float alpha = 1.1f;
        if (nodes_[node_id].deleted) {
            throw std::runtime_error("Label already deleted: " + label);
        }

        RewireStats stats;
        auto &node = nodes_[node_id];
        if (!node.neighbors.count(0)) {
            nodes_[node_id].deleted = true;
            if (entry_point_.value() == node_id) {
                reassign_entry_point();
            }
            return stats;
        }

        std::vector<int> base_neighbors;
        for (int n : node.neighbors[0]) {
            if (!nodes_[n].deleted) {
                base_neighbors.push_back(n);
            }
        }

        if (base_neighbors.empty()) {
            nodes_[node_id].deleted = true;
            if (entry_point_.value() == node_id) {
                reassign_entry_point();
            }
            return stats;
        }

        // Choose representative randomly (matching Python "random" strategy)
        std::uniform_int_distribution<int> dist(0, static_cast<int>(base_neighbors.size()) - 1);
        stats.representative_node = base_neighbors[dist(rng_)];
        // std::cout << "Top level: " << max_level_ << std::endl;
        for (int lvl = node.level; lvl >= 0; --lvl) {
            std::vector<int> layer_neighbors;
            // std::cout << "Level: " << lvl << std::endl;
            // std::cout << "Node: " << node_id << std::endl;
            // std::cout << "Representative: " << stats.representative_node << std::endl;
            // std::cout << "------------------" << std::endl;
            if (node.neighbors.count(lvl)) {
                for (int n : node.neighbors[lvl]) {
                    if (!nodes_[n].deleted) {
                        layer_neighbors.push_back(n);
                        // std::cout << "Neighbors: " << n << std::endl;
                    }
                }
            }
            // std::cout << "------------------" << std::endl;
            // for (int v : layer_neighbors) {
            //     auto it_v = nodes_[v].neighbors.find(lvl);
            //     if (it_v != nodes_[v].neighbors.end()) {
            //         it_v->second.erase(node_id);
            //         if (it_v->second.empty()) nodes_[v].neighbors.erase(lvl);
            //     }
            // }
            // std::unordered_map<int, std::vector<int>> new_layer_neighbors;
            // new_layer_neighbors.reserve(layer_neighbors.size());
            // for (int v : layer_neighbors) {
            //     nodes_[v].neighbors[lvl].insert(stats.representative_node);
            //     nodes_[stats.representative_node].neighbors[lvl].insert(v);
            // }

            

            // std::vector <int> pruned_neighbors = prune_alpha_rng(stats.representative_node, layer_neighbors, alpha);
            // nodes_[stats.representative_node].neighbors[lvl].insert(pruned_neighbors.begin(), pruned_neighbors.end());

            // for (int v: pruned_neighbors) {
            //     if (nodes_[v].deleted) continue;
            //     link(stats.representative_node, v, lvl, alpha);
            // }

            // prune_neighbors(stats.representative_node, lvl);
            // new_layer_neighbors = local_rewire_layer_representative(stats.representative_node,
            //     layer_neighbors,
            //     lvl);

            // for (auto &p : new_layer_neighbors) {
            //     int u = p.first;

            //     std::unordered_set<int> newSet;
            //     newSet.reserve(p.second.size());
            //     for (int v : p.second) newSet.insert(v);

            //     nodes_[u].neighbors[lvl] = std::move(newSet);
            // }
            // Ensure mutual link consistency at this level (bounded to m_)
            // for (int u : layer_neighbors) {
            //     // prune_neighbors(u, lvl);
            //     const auto &u_set = nodes_[u].neighbors[lvl];   
            //     std::vector<int> lvl_neighbors;
            //     for (int v : u_set) {
            //         if (!nodes_[v].deleted) {
            //             lvl_neighbors.push_back(v);
            //         }
            //     }
            //     std::vector <int> pruned_neighbors_u = prune_alpha_rng(u, lvl_neighbors, alpha);
            //     nodes_[u].neighbors[lvl].insert(pruned_neighbors_u.begin(), pruned_neighbors_u.end());
            // }

            // node.neighbors.erase(lvl);
            rebuild_node_neighbors_from_candidates(stats.representative_node, lvl, layer_neighbors, alpha);
        }

        node.deleted = true;
        if (allow_replace_deleted_) {
            deleted_pool_.push_back(node_id);
        }
        mnru_update_deleted_id(node_id, label, new_vec, alpha);
        // if (entry_point_.value() == node_id) {
        //     reassign_entry_point();
        // }
        
        return stats;
    }

    void update_with_MNRU(const std::string &label, const std::vector<float> &new_vec, float alpha) {
        if (!label_to_id_.count(label)) {
            throw std::runtime_error("Label not found: " + label);
        }
        int node_id = label_to_id_[label];
        auto &del_node = nodes_[node_id];

        int level = del_node.level;
        for (int lvl = 0; lvl >= level; lvl ++) {
            auto it_del_N1 = del_node.neighbors.find(lvl);
            std::vector<int> N1;
            for (int nbr_N1: it_del_N1 -> second) {
                if (nbr_N1 == node_id) continue;
                if (nodes_[nbr_N1].deleted) continue;
                N1.push_back(nbr_N1);
            }
            if (N1.empty()) {
                del_node.neighbors.erase(lvl);
                continue;
            }

            std::vector<int> N2;
            for (int nbr_N1: N1) {
                auto it_N1 = nodes_[nbr_N1].neighbors.find(lvl);
                if (it_N1 == nodes_[nbr_N1].neighbors.end()) continue;
                if (it_N1->second.count(node_id)) {
                    N2.push_back(nbr_N1);
                }
            }
        }
    }

    // MNRU-style deletion: rebuild neighborhoods of 1-hop neighbors using 1- and 2-hop candidates
    MNRUStats delete_with_MNRU(const std::string &label, const std::vector<float>& new_vec, float alpha) {
        if (!label_to_id_.count(label)) {
            throw std::runtime_error("Label not found: " + label);
        }
        int node_id = label_to_id_[label];
        auto &del_node = nodes_[node_id];
        if (del_node.deleted) {
            throw std::runtime_error("Label already deleted: " + label);
        }

        MNRUStats stats;

        // For each level the node participates in
        for (int level = del_node.level; level >= 0; --level) {
            auto it_del = del_node.neighbors.find(level);
            if (it_del == del_node.neighbors.end()) continue;

            // N1 = neighbors(deletedPoint, level) (filter deleted + valid level)
            std::vector<int> N1;
            N1.reserve(it_del->second.size());
            for (int v : it_del->second) {
                if (v == node_id) continue;
                if (nodes_[v].deleted) continue;
                if (nodes_[v].level <= level) continue;
                N1.push_back(v);
            }
            if (N1.empty()) {
                del_node.neighbors.erase(level);
                continue;
            }

            // N2 = subset of N1 that actually have back-edge v -> deletedPoint at this level
            // (if you maintain symmetric edges always, N2 == N1, but keeping this matches pseudo-code)
            std::vector<int> N2;
            N2.reserve(N1.size());
            for (int v : N1) {
                auto it_v = nodes_[v].neighbors.find(level);
                if (it_v == nodes_[v].neighbors.end()) continue;
                if (it_v->second.count(node_id)) {
                    N2.push_back(v);
                    it_v->second.erase(node_id);
                    if (it_v->second.empty()) nodes_[v].neighbors.erase(level);
                }
            }
            if (N2.empty()) {
                del_node.neighbors.erase(level);
                continue;
            }

            // Step 0: remove deleted node from N2 adjacency first (avoid it leaking into candidates)
            // for (int v : N2) {
            //     auto it_v = nodes_[v].neighbors.find(level);
            //     if (it_v != nodes_[v].neighbors.end()) {
            //         it_v->second.erase(node_id);
            //         if (it_v->second.empty()) nodes_[v].neighbors.erase(level);
            //     }
            // }

            // // ---- NEW: Build 2-hop candidate pool around the deletion region (per level) ----
            // // Option A (broader): via N1
            std::unordered_set<int> N2hop;
            N2hop.reserve(N1.size() * (size_t)m_);

            // Build 2-hop from neighbors of nodes in N1 (or use N2 for stricter locality)
            // for (int x : N1) {  // change to: for (int x : N2) if you want stricter
            //     auto it_x = nodes_[x].neighbors.find(level);
            //     if (it_x == nodes_[x].neighbors.end()) continue;

            //     for (int y : it_x->second) {
            //         if (y == node_id || y == x) continue;
            //         if (nodes_[y].deleted) continue;
            //         if (nodes_[y].level < level) continue;
            //         N2hop.insert(y);
            //     }
            // }

            // Build new neighbor sets for nodes in N2
            std::unordered_map<int, std::vector<int>> new_layer_neighbors;
            new_layer_neighbors.reserve(N2.size());

            for (int u : N2) {
                if (nodes_[u].deleted) continue;
                if (nodes_[u].level <= level) continue;

                // C = neighbors(u, level) ∪ N1  (minus {deleted,node_id,u})
                std::unordered_set<int> Cset;
                Cset.reserve(m_ * 4 + N1.size());
                

                // add current neighbors(u,level)
                auto it_u = nodes_[u].neighbors.find(level);
                if (it_u != nodes_[u].neighbors.end()) {
                    for (int x : it_u->second) {
                        if (x == node_id || x == u) continue;
                        if (nodes_[x].deleted) continue;
                        if (nodes_[x].level <= level) continue;
                        Cset.insert(x);
                    }
                }

                // add N1
                for (int x : N1) {
                    if (x == node_id || x == u) continue;
                    if (nodes_[x].deleted) continue;
                    if (nodes_[x].level <= level) continue;
                    Cset.insert(x);
                }

                // for (int x : N2hop) {
                //     if (x == node_id || x == u) continue;
                //     if (nodes_[x].deleted) continue;
                //     if (nodes_[x].level < level) continue;
                //     Cset.insert(x);
                // }

                // turn into vector
                std::vector<int> C;
                C.reserve(Cset.size());
                for (int x : Cset) C.push_back(x);

                stats.nodes_touched++;
                stats.total_candidates += (int)C.size();
                
                // for (int v : C) {
                //     link(u, v, level, alpha);
                // }
                // α-RNG prune
                std::vector<int> pruned = prune_alpha_rng(u, C, alpha);
                // std::vector<int> pruned = prune_neighbors(u, level);
                new_layer_neighbors.emplace(u, std::move(pruned));
                // link(u, level, C);
            }

            // Apply new neighbor sets for this level
            for (auto &p : new_layer_neighbors) {
                int u = p.first;
                std::unordered_set<int> newSet;
                newSet.reserve(p.second.size());
                for (int v : p.second) newSet.insert(v);
                nodes_[u].neighbors[level] = std::move(newSet);
            }

            // Ensure mutual link consistency at this level (bounded to m_)
            for (const auto &p : new_layer_neighbors) {
                // int u = p.first;
                // const auto &u_set = nodes_[u].neighbors[level];

                std::unordered_set<int> newSet;
                newSet.reserve(p.second.size());
                for (int v : p.second) newSet.insert(v);

                int u = p.first;
                for (int v : newSet) {
                    if (nodes_[v].deleted) continue;
                    if (nodes_[v].level <= level) continue;
                    auto &v_set = nodes_[v].neighbors[level];
                    v_set.insert(u);
                    std::vector<int> candidates_v;
                    for (int x : v_set) candidates_v.push_back(x);
                    std::vector<int> pruned_v = prune_alpha_rng(v, candidates_v, alpha);
                    // nodes_[u].neighbors[level].clear();
                    nodes_[v].neighbors[level].insert(pruned_v.begin(), pruned_v.end());
                }
                
                // for (int v : u_set) {
                //     if (nodes_[v].deleted) continue;
                //     if (nodes_[v].level != level) continue;
                //     std::vector<int> candidates_v;
                //     // auto &v_set = nodes_[v].neighbors[level];

                //     // if (v_set.count(u)) continue;
                    
                //     // v_set.insert(u);
                    
                //     // for (int x : v_set) candidates_v.push_back(x);
                //     candidates_v.push_back(v);
                // }
                // std::vector<int> pruned_v = prune_alpha_rng(u, candidates_v, alpha);
                //  nodes_[u].neighbors[level].clear();
                // nodes_[u].neighbors[level].insert(pruned_v.begin(), pruned_v.end());

                    // if ((int)v_set.size() < m_) {
                    //     v_set.insert(u); 
                    // } else {
                    //     // Replace worst neighbor of v if u is better
                    //     int worst = -1;
                    //     float worst_dist = -1.0f;

                    //     for (int x : v_set) {
                    //         float d = distance(nodes_[v].vector, nodes_[x].vector);
                    //         stats.distance_evals++;
                    //         if (d > worst_dist) { worst_dist = d; worst = x; }
                    //     }

                    //     float d_new = distance(nodes_[v].vector, nodes_[u].vector);
                    //     stats.distance_evals++;

                    //     if (worst != -1 && d_new < worst_dist) {
                    //         v_set.erase(worst);
                    //         v_set.insert(u);
                    //     }
                    // }
            }

            // Remove this level's edges from deleted node
            del_node.neighbors.erase(level);
        }

        // Logically delete node
        del_node.deleted = true;
        if (allow_replace_deleted_) deleted_pool_.push_back(node_id);
        // Update MN-RU
        mnru_update_deleted_id(node_id, label, new_vec, alpha);
        

        // if (entry_point_.has_value() && entry_point_.value() == node_id) {
        //     reassign_entry_point();
        // }

        return stats;
    }


    // Hybrid MN-RU deletion:
    // - Level 0: rebuild neighbors of the deleted node's neighbors using
    //   an MN-RU style local reconstruction based on ANN search.
    // - Upper levels: lazy repair (simply unlink the node, no rewiring).
    RewireStats delete_with_mnru_minimal(const std::string &label) {
        RewireStats stats;
    
        // 1) Find node
        auto it = label_to_id_.find(label);
        if (it == label_to_id_.end()) {
            throw std::runtime_error("Label not found: " + label);
        }
        int node_id = it->second;
        HNSWNode &node = nodes_[node_id];
    
        if (node.deleted) {
            throw std::runtime_error("Label already deleted: " + label);
        }
    
        // 2) Collect base-layer neighbors (level 0) for representative choice
        std::vector<int> base_neighbors;
        if (node.neighbors.count(0)) {
            for (int n : node.neighbors[0]) {
                if (!nodes_[n].deleted) {
                    base_neighbors.push_back(n);
                }
            }
        }
    
        if (base_neighbors.empty()) {
            // No neighbors; just mark deleted and fix entry point
            node.deleted = true;
            if (entry_point_.has_value() && entry_point_.value() == node_id) {
                reassign_entry_point();
            }
            return stats;
        }
    
        // 3) Choose representative: min-degree at level 0
        int rep = -1;
        size_t best_deg = std::numeric_limits<size_t>::max();
        for (int n : base_neighbors) {
            size_t deg = 0;
            auto it_lvl0 = nodes_[n].neighbors.find(0);
            if (it_lvl0 != nodes_[n].neighbors.end()) {
                deg = it_lvl0->second.size();
            }
            if (deg < best_deg) {
                best_deg = deg;
                rep = n;
            }
        }
    
        stats.representative_node = rep;
    
        // 4) For each level, reconnect neighbors of v to rep (minimally)
        for (int lvl = node.level; lvl >= 0; --lvl) {
            std::vector<int> neighs;
    
            // copy current neighbors at this level before unlinking
            auto it_lvl = node.neighbors.find(lvl);
            if (it_lvl != node.neighbors.end()) {
                for (int n : it_lvl->second) {
                    if (!nodes_[n].deleted && n != rep) {
                        neighs.push_back(n);
                    }
                }
            }
    
            // Remove v from this layer
            remove_node_from_layer(node_id, lvl);
    
            // Reconnect each neighbor to rep (if useful and allowed)
            for (int n : neighs) {
                if (nodes_[n].deleted) continue;
                // If edge already exists at this level, skip
                if (nodes_[n].neighbors.count(lvl) &&
                    nodes_[n].neighbors[lvl].count(rep)) {
                    continue;
                }
    
                // Very lightweight: only add if degrees allow, no distance-based rebuild
                size_t deg_rep = nodes_[rep].neighbors.count(lvl)
                                     ? nodes_[rep].neighbors[lvl].size()
                                     : 0;
                size_t deg_n   = nodes_[n].neighbors.count(lvl)
                                     ? nodes_[n].neighbors[lvl].size()
                                     : 0;
    
                if (deg_rep < static_cast<size_t>(m_) &&
                    deg_n   < static_cast<size_t>(m_)) {
                    link(rep, n, lvl, 1);            // symmetric edge, prunes if needed
                    stats.rewired_edges_count++;
                }
                // else: do nothing; minimal update means we don't try to force extra edges
            }
        }
    
        // 5) Mark v deleted and possibly fix entry point
        node.deleted = true;
        if (entry_point_.has_value() && entry_point_.value() == node_id) {
            reassign_entry_point();
        }
    
        return stats;
    }

    void debug_connectivity() const {
        int N = static_cast<int>(nodes_.size());
    
        // Count alive nodes
        int alive = 0;
        for (int i = 0; i < N; ++i) {
            if (!nodes_[i].deleted) alive++;
        }
    
        // BFS from entry point on level 0 over non-deleted nodes
        if (!entry_point_.has_value()) {
            std::cout << "No entry point\n";
            return;
        }
    
        std::vector<char> visited(N, 0);
        std::vector<int> queue;
        queue.reserve(N);
    
        int start = entry_point_.value();
        if (!nodes_[start].deleted) {
            visited[start] = 1;
            queue.push_back(start);
        }
    
        for (size_t qi = 0; qi < queue.size(); ++qi) {
            int u = queue[qi];
            auto it = nodes_[u].neighbors.find(0);
            if (it == nodes_[u].neighbors.end()) continue;
            for (int v : it->second) {
                if (nodes_[v].deleted) continue;
                if (!visited[v]) {
                    visited[v] = 1;
                    queue.push_back(v);
                }
            }
        }
    
        int reachable = 0;
        for (int i = 0; i < N; ++i) {
            if (!nodes_[i].deleted && visited[i]) reachable++;
        }
    
        std::cout << "Alive nodes:     " << alive << " / " << N << "\n";
        std::cout << "Reachable nodes: " << reachable << " from entry point\n";
        std::cout << "Unreachable alive nodes: " << (alive - reachable) << "\n";
    }

    void debug_degree_hist_level0() const {
        std::vector<int> hist(21, 0); // 0..20 (20 = "20 or more")
        for (int i = 0; i < (int)nodes_.size(); ++i) {
            if (nodes_[i].deleted) continue;
            int d = 0;
            auto it = nodes_[i].neighbors.find(0);
            if (it != nodes_[i].neighbors.end()) {
                d = static_cast<int>(it->second.size());
            }
            if (d >= 20) d = 20;
            hist[d]++;
        }
        for (int d = 0; d <= 20; ++d) {
            std::cout << "deg=" << d << ": " << hist[d] << " nodes\n";
        }
    }
    

  private:
    int dim_;
    int m_;
    float level_probability_;
    int ef_construction_;
    std::mt19937 rng_;
    std::vector<HNSWNode> nodes_;
    std::unordered_map<std::string, int> label_to_id_;
    std::optional<int> entry_point_;
    int max_level_{-1};
    bool allow_replace_deleted_{true};
    std::vector<int> deleted_pool_;

    float distance(const std::vector<float> &a,
                   const std::vector<float> &b) const {
        float sum = 0.0f;
        for (int i = 0; i < dim_; ++i) {
            float diff = a[i] - b[i];
            sum += diff * diff;
        }
        return std::sqrt(sum);
    }

    int max_degree(int level) const {
        return (level == 0) ? (2 * m_) : m_;
    }

    // α-RNG prune for a single node u at a given level.
    // candidates: set of candidate neighbor ids (u and deleted should already be excluded).
    // Keeps up to m_ nodes.
    std::vector<int> prune_alpha_rng(int u,
                                    const std::vector<int>& candidates,
                                    float alpha) {
        struct Item { float duv; int v; };
        std::vector<Item> sorted;
        sorted.reserve(candidates.size());

        for (int v : candidates) {
            float d = distance(nodes_[u].vector, nodes_[v].vector);
            // stats.distance_evals++;
            sorted.push_back({d, v});
        }

        std::sort(sorted.begin(), sorted.end(),
                [](const Item& a, const Item& b){ return a.duv < b.duv; });

        std::vector<int> selected;
        selected.reserve(std::min((int)sorted.size(), m_));

        std::vector<int> rejected;
        rejected.reserve(sorted.size());

        // α-RNG rule:
        // accept v if for all already selected w: dist(u,v) <= alpha * dist(v,w)
        // (otherwise v is redundant via w)
        for (const auto& cand : sorted) {
            bool keep = true;
            for (int w : selected) {
                float dvw = distance(nodes_[cand.v].vector, nodes_[w].vector);
                // stats.distance_evals++;
                if (cand.duv > alpha * dvw) {
                    keep = false;
                    break;
                }
            }
            if (keep) {
                selected.push_back(cand.v);
                if ((int)selected.size() >= m_) break;
            } else {
                rejected.push_back(cand.v);
            }
        }

        // Pass 2: backfill to ensure degree ~ m_
        // (critical for recall stability under many deletions)
        for (int v : rejected) {
            selected.push_back(v);
            if ((int)selected.size() >= m_) break;
        }

        return selected;
    }


    int sample_level() {
        std::uniform_real_distribution<float> dist(0.0f, 1.0f);
        int level = 0;
        while (dist(rng_) < level_probability_) {
            ++level;
        }
        return level;
    }

    void repair_connections_for_update(int node_id, float alpha) {
        if (nodes_[node_id].deleted) return;
        if (!entry_point_.has_value()) return;

        const auto &q = nodes_[node_id].vector;

        // If entry point is deleted, pick a new one
        if (nodes_[entry_point_.value()].deleted) {
            reassign_entry_point();
            if (!entry_point_.has_value()) return;
        }

        int ep = entry_point_.value();
        int level = nodes_[node_id].level;
        if (level > max_level_) {
            max_level_ = level;
            entry_point_ = node_id;
        }

        // 1) Navigate from top to just above node's top layer
        for (int lvl = max_level_; lvl > nodes_[node_id].level; --lvl) {
            ep = greedy_search(q, ep, lvl, /*track_path=*/false).first;
        }

        // 2) For each level node participates in, do a local search and connect
        for (int lvl = nodes_[node_id].level; lvl >= 0; --lvl) {
            // Search within this layer for candidates around q
            auto [top, _path] = search_layer(q, ep, lvl, ef_construction_, /*track_path=*/false);

            std::vector<int> candidates;
            candidates.reserve(top.size());
            for (auto &p : top) {
                int cand = p.second;
                if (cand == node_id) continue;
                if (nodes_[cand].deleted) continue;
                if (nodes_[cand].level < lvl) continue;
                candidates.push_back(cand);
            }

            // Pick neighbors and connect
            int cap = m_;
            // auto chosen = prune_alpha_rng(node_id, candidates, alpha);
            // if (chosen.empty()) {
            //     chosen = select_neighbors_simple(q, candidates, cap);
            // }

            // Ensure node has at least something if possible (avoid isolating it)
            // If chosen is empty but we have candidates, fall back to the nearest.
            // if (chosen.empty() && !candidates.empty()) {
            //     chosen.push_back(candidates[0]);
            // }

            for (int v : candidates) {
                // std::cout << "connecting " << node_id << " to " << v << " at level " << lvl << std::endl;
                // nodes_[node_id].neighbors[lvl].insert(v);
                // nodes_[v].neighbors[lvl].insert(node_id);
                // auto it_dst = nodes_[v].neighbors.find(lvl);
                // auto nbrs_dst = it_dst->second;
                // std::vector<int> candiates_dst;
                // candiates_dst.reserve(nbrs_dst.size());
                // for (int nbr : nbrs_dst) candiates_dst.push_back(nbr);

                // std::vector<int> pruned_dst = prune_alpha_rng(v, candiates_dst, alpha);
                // nodes_[v].neighbors[lvl] = std::unordered_set<int>(pruned_dst.begin(), pruned_dst.end());
                link(node_id, v, lvl, alpha);
                // nodes_[node_id].neighbors[lvl].insert(v);
                // nodes_[v].neighbors[lvl].insert(node_id);
            }

            // Improve ep for next lower level (standard HNSW behavior)
            // Use the closest found (top is already sorted asc in your search_layer)
            if (!top.empty()) {
                ep = top[0].second;
            }
        }
    }


    void update_point_local(int node_id, float updateNeighborProbability, float alpha) {
        if (nodes_[node_id].deleted) return;

        std::uniform_real_distribution<float> uni(0.0f, 1.0f);

        for (int level = nodes_[node_id].level; level >= 0; --level) {
            // 1) rebuild the updated node at this level
            auto cands = collect_two_hop_candidates(node_id, level);
            rebuild_node_neighbors_from_candidates(node_id, level, cands, alpha);

            // 2) optionally rebuild some neighbors as well (hnswlib does this)
            auto it = nodes_[node_id].neighbors.find(level);
            if (it == nodes_[node_id].neighbors.end()) continue;

            std::vector<int> one_hop(it->second.begin(), it->second.end());
            for (int n : one_hop) {
                if (nodes_[n].deleted) continue;
                if (nodes_[n].level < level) continue;

                if (uni(rng_) <= updateNeighborProbability) {
                    auto cands_n = collect_two_hop_candidates(n, level);
                    rebuild_node_neighbors_from_candidates(n, level, cands_n, alpha);
                }
            }
        }
        repair_connections_for_update(node_id, alpha);
    }

    void rebuild_node_neighbors_from_candidates(int u, int level,
                                           const std::vector<int>& candidates, float alpha) {
        if (nodes_[u].deleted) return;

        // Remove existing edges at this level (symmetrically)
        auto it_old = nodes_[u].neighbors.find(level);
        if (it_old != nodes_[u].neighbors.end()) {
            std::vector<int> old(it_old->second.begin(), it_old->second.end());
            for (int v : old) unlink(u, v, level);
        }

        // Score candidates (exclude self / deleted / nodes without this level)
        std::vector<std::pair<float,int>> scored;
        scored.reserve(candidates.size());

        for (int c : candidates) {
            if (c == u) continue;
            if (nodes_[c].deleted) continue;
            if (nodes_[c].level < level) continue;
            float d = distance(nodes_[u].vector, nodes_[c].vector);
            scored.emplace_back(d, c);
        }

        if (scored.empty()) return;

        std::sort(scored.begin(), scored.end(),
                [](auto& a, auto& b){ return a.first < b.first; });

        // Limit to ef_construction_ best candidates (like hnswlib’s bounded candidate set)
        if ((int)scored.size() > ef_construction_) scored.resize(ef_construction_);

        std::vector<int> cand_ids;
        cand_ids.reserve(scored.size());
        for (auto &p : scored) cand_ids.push_back(p.second);

        // Choose neighbors (heuristic or simple)
        int cap = m_;
        // auto chosen = select_neighbors_heuristic(nodes_[u].vector, cand_ids, cap);
        // std::vector<int> pruned = prune_alpha_rng(u, cand_ids, alpha);
        // if (chosen.empty()) {
        //     // fallback: simple top-m_
        //     chosen = select_neighbors_simple(nodes_[u].vector, cand_ids, cap);
        // }

        

        // for (int node : chosen) {
        //     const auto &u_set = nodes_[node].neighbors[level];

        //     for (int v : u_set) {
        //         if (nodes_[v].deleted) continue;
        //         if (nodes_[v].level < level) continue;

        //         auto &v_set = nodes_[v].neighbors[level];

        //         if (v_set.count(node)) continue;

        //         if ((int)v_set.size() < m_) {
        //             v_set.insert(node);
        //         } else {
        //             // Replace worst neighbor of v if u is better
        //             int worst = -1;
        //             float worst_dist = -1.0f;

        //             for (int x : v_set) {
        //                 float d = distance(nodes_[v].vector, nodes_[x].vector);
        //                 if (d > worst_dist) { worst_dist = d; worst = x; }
        //             }

        //             float d_new = distance(nodes_[v].vector, nodes_[u].vector);

        //             if (worst != -1 && d_new < worst_dist) {
        //                 v_set.erase(worst);
        //                 v_set.insert(node);
        //             }
        //         }
        //     }
        // }

        // // Add edges (use link_new_node-like semantics so u stays connected)
        for (int v : cand_ids) {
            // if (nodes_[v].deleted) continue;
            // nodes_[u].neighbors[level].insert(v);
            // nodes_[v].neighbors[level].insert(u);
            // auto it_dst = nodes_[v].neighbors.find(level);
            // auto nbrs_dst = it_dst->second;
            // std::vector<int> candiates_dst;
            // candiates_dst.reserve(nbrs_dst.size());
            // for (int nbr : nbrs_dst) candiates_dst.push_back(nbr);

            // std::vector<int> pruned_dst = prune_alpha_rng(v, candiates_dst, alpha);
            // nodes_[v].neighbors[level] = std::unordered_set<int>(pruned_dst.begin(), pruned_dst.end());
            // std::cout << "Linking " << u << " and " << v << " at level " << level << std::endl;
            link(u, v, level, alpha);
        }

        
        
    }

    std::vector<int> collect_two_hop_candidates(int node_id, int level) {
        std::unordered_set<int> cand;
        cand.reserve(m_ * m_ + m_ + 8);

        if (nodes_[node_id].deleted) return {};

        cand.insert(node_id);

        auto it1 = nodes_[node_id].neighbors.find(level);
        if (it1 != nodes_[node_id].neighbors.end()) {
            for (int n1 : it1->second) {
                if (nodes_[n1].deleted) continue;
                cand.insert(n1);

                auto it2 = nodes_[n1].neighbors.find(level);
                if (it2 != nodes_[n1].neighbors.end()) {
                    for (int n2 : it2->second) {
                        if (nodes_[n2].deleted) continue;
                        cand.insert(n2);
                    }
                }
            }
        }

        // remove self not needed, we’ll filter later
        std::vector<int> out;
        out.reserve(cand.size());
        for (int x : cand) out.push_back(x);
        return out;
    }

    std::pair<int, std::vector<int>>
    greedy_search(const std::vector<float> &query,
                  int entry_id,
                  int level,
                  bool track_path) {
        int best = entry_id;
        std::vector<int> path;
        if (track_path) {
            path.push_back(best);
        }

        if (nodes_[best].deleted) {
            for (size_t i = 0; i < nodes_.size(); ++i) {
                if (!nodes_[i].deleted && nodes_[i].level >= level) {
                    best = static_cast<int>(i);
                    if (track_path) {
                        path = {best};
                    }
                    break;
                }
            }
        }

        float best_distance = distance(query, nodes_[best].vector);
        bool improved = true;

        while (improved) {
            improved = false;
            auto it = nodes_[best].neighbors.find(level);
            if (it == nodes_[best].neighbors.end()) {
                break;
            }
            for (int neighbor : it->second) {
                if (nodes_[neighbor].deleted) {
                    continue;
                }
                float dist_val = distance(query, nodes_[neighbor].vector);
                if (dist_val < best_distance) {
                    best_distance = dist_val;
                    best = neighbor;
                    improved = true;
                    if (track_path) {
                        path.push_back(best);
                    }
                }
            }
        }
        return {best, path};
    }

    std::vector<int> layer_candidates(int exclude_id,
                                      const std::vector<float> &vector,
                                      int level) const {
        std::vector<int> candidates;
        for (size_t i = 0; i < nodes_.size() - 1; ++i) {
            if (static_cast<int>(i) == exclude_id) {
                continue;
            }
            const auto &node = nodes_[i];
            if (!node.deleted && node.level >= level) {
                candidates.push_back(static_cast<int>(i));
            }
        }
        return candidates;
    }

    std::vector<int> select_neighbors(const std::vector<float> &vector,
                                      const std::vector<int> &candidates,
                                      int m) const {
        std::vector<std::pair<float, int>> scored;
        scored.reserve(candidates.size());
        for (int idx : candidates) {
            scored.emplace_back(distance(vector, nodes_[idx].vector), idx);
        }
        std::sort(scored.begin(), scored.end(),
                  [](const auto &a, const auto &b) { return a.first < b.first; });

        std::vector<int> neighbors;
        for (size_t i = 0; i < scored.size() && static_cast<int>(neighbors.size()) < m;
             ++i) {
            neighbors.push_back(scored[i].second);
        }
        return neighbors;
    }

    std::vector<int> select_neighbors_heuristic(
        const std::vector<float>& query,
        const std::vector<int>& candidates,
        int M) const
    {
        if (candidates.empty()) return {};
    
        // Score candidates by distance
        std::vector<std::pair<float,int>> scored;
        scored.reserve(candidates.size());
        for (int c: candidates) {
            float d = distance(query, nodes_[c].vector);
            scored.emplace_back(d, c);
        }
        std::sort(scored.begin(), scored.end(),
                  [](auto& a, auto& b){ return a.first < b.first; });
    
        // Heuristic: select neighbors whose distance is strictly
        // better than any already chosen.
        std::vector<int> result;

        std::vector<int> rejected;
        rejected.reserve(scored.size());

        for (auto& p: scored) {
            float d = p.first;
            bool good = true;
            for (int r: result) {
                float dr = distance(nodes_[r].vector, nodes_[p.second].vector);
                if (dr < d) {      // heuristic condition
                    good = false;
                    break;
                }
            }
            if (good) {
                result.push_back(p.second);
                if ((int)result.size() >= M) break;
            }
            else {
                rejected.push_back(p.second);
            }
        }

        for (int v : rejected) {
            result.push_back(v);
            if ((int)result.size() >= m_) break;
        }

        return result;
    }

    std::vector<int> select_neighbors_simple(
        const std::vector<float> &query,
        const std::vector<int> &candidates,
        int M) const
    {
        std::vector<std::pair<float,int>> scored;
        scored.reserve(candidates.size());
        for (int c : candidates) {
            scored.emplace_back(distance(query, nodes_[c].vector), c);
        }
        std::sort(scored.begin(), scored.end(),
                  [](auto &a, auto &b) { return a.first < b.first; });
    
        std::vector<int> out;
        for (size_t i = 0; i < scored.size() && (int)out.size() < M; ++i)
            out.push_back(scored[i].second);
        return out;
    }

    void link(int src, int dst, int level, float alpha) {
        nodes_[src].neighbors[level].insert(dst);
        nodes_[dst].neighbors[level].insert(src);

        // candidates source
        auto it_src = nodes_[src].neighbors.find(level);
        auto nbrs_src = it_src->second;
        std::vector<int> candiates_src;
        candiates_src.reserve(nbrs_src.size());
        for (int nbr : nbrs_src) candiates_src.push_back(nbr);

        // candidates destination
        auto it_dst = nodes_[dst].neighbors.find(level);
        auto nbrs_dst = it_dst->second;
        std::vector<int> candiates_dst;
        candiates_dst.reserve(nbrs_dst.size());
        for (int nbr : nbrs_dst) candiates_dst.push_back(nbr);

        // prune neighbors
        // prune_neighbors(src, level);
        // prune_neighbors(dst, level);
        std::vector<int> pruned_src = prune_alpha_rng(src, candiates_src, alpha);
        std::vector<int> pruned_dst = prune_alpha_rng(dst, candiates_dst, alpha);
        nodes_[src].neighbors[level] = std::unordered_set<int>(pruned_src.begin(), pruned_src.end());
        nodes_[dst].neighbors[level] = std::unordered_set<int>(pruned_dst.begin(), pruned_dst.end());
    }

    void prune_neighbors(int node_id, int level) {
        auto it = nodes_[node_id].neighbors.find(level);
        if (it == nodes_[node_id].neighbors.end()) return;

        int cap = m_;
        auto &nbrs = it->second;
        if ((int)nbrs.size() <= cap) return;

        std::vector<std::pair<float,int>> scored;
        scored.reserve(nbrs.size());
        for (int nb : nbrs) scored.emplace_back(distance(nodes_[node_id].vector, nodes_[nb].vector), nb);

        std::sort(scored.begin(), scored.end(), [](auto& a, auto& b){ return a.first < b.first; });

        nbrs.clear();
        for (int i = 0; i < (int)scored.size() && i < cap; ++i) nbrs.insert(scored[i].second);
    }

    std::pair<std::vector<std::pair<float,int>>, std::vector<int>>
    search_base_layer(const std::vector<float> &query,
                    int entry_id,
                    int ef,
                    bool track_path)
    {
        std::vector<int> path;

        // Min-heap for candidates (closest distance first)
        using Pair = std::pair<float,int>;
        auto cmp_min = [](const Pair &a, const Pair &b) {
            return a.first > b.first;  // smaller distance has higher priority
        };
        std::priority_queue<Pair, std::vector<Pair>, decltype(cmp_min)> candidates(cmp_min);

        // Max-heap for result set (we keep the ef best; top() is the *worst* among them)
        auto cmp_max = [](const Pair &a, const Pair &b) {
            return a.first < b.first;  // larger distance has higher priority
        };
        std::priority_queue<Pair, std::vector<Pair>, decltype(cmp_max)> results(cmp_max);

        std::unordered_set<int> visited;

        if (!nodes_[entry_id].deleted) {
            float dist0 = distance(query, nodes_[entry_id].vector);
            candidates.push({dist0, entry_id});
            results.push({dist0, entry_id});
            visited.insert(entry_id);
            if (track_path) path.push_back(entry_id);
        } else {
            // Optional: handle deleted entry (e.g., pick another non-deleted node)
            // for (size_t i = 0; i < nodes_.size(); ++i) {
            //     if (!nodes_[i].deleted && nodes_[i].level == 0) {
            //         float dist0 = distance(query, nodes_[entry_id].vector);
            //         candidates.push({dist0, entry_id});
            //         results.push({dist0, entry_id});
            //         visited.insert(entry_id);
            //         if (track_path) path.push_back(entry_id);
            //         break;
            //     }
            // }
            visited.insert(entry_id);
        }

        while (!candidates.empty()) {
            auto curr = candidates.top();
            float currDist = curr.first;
            int   currId   = curr.second;
            candidates.pop();

            // Worst distance among current results
            float worstDist = results.top().first;

            // Standard HNSW stopping rule
            if (results.size() >= static_cast<size_t>(ef) && currDist > worstDist)
                break;

            // Explore neighbors at level 0
            auto it = nodes_[currId].neighbors.find(0);
            if (it == nodes_[currId].neighbors.end()) continue;

            for (int nb : it->second) {
                if (nodes_[nb].deleted) continue;
                if (visited.count(nb)) continue;
                visited.insert(nb);
                if (track_path) path.push_back(nb);

                float d = distance(query, nodes_[nb].vector);

                // Candidate for expansion only if it can improve result set
                if (results.size() < static_cast<size_t>(ef) || d < results.top().first) {
                    // std::cout << "I am here expanding the candidate set" << std::endl;
                    candidates.push({d, nb});
                    results.push({d, nb});
                    if (results.size() > static_cast<size_t>(ef)) {
                        results.pop();  // drop worst
                    }
                }
            }
        }

        // Extract results into a vector and sort by distance ascending
        std::vector<std::pair<float,int>> out;
        out.reserve(results.size());
        while (!results.empty()) {
            out.push_back(results.top());
            results.pop();
        }
        std::sort(out.begin(), out.end(),
                [](const auto &a, const auto &b) { return a.first < b.first; });

        return {out, path};
    }
    // std::pair<std::vector<std::pair<float, int>>, std::vector<int>>
    // search_base_layer(const std::vector<float> &query,
    //                   int entry_id,
    //                   int ef,
    //                   bool track_path) {
    //     std::vector<int> path;
    //     std::vector<std::pair<float, int>> result_set;
    //     if (!nodes_[entry_id].deleted) {
    //         float entry_dist = distance(query, nodes_[entry_id].vector);
    //         result_set.emplace_back(entry_dist, entry_id);
    //         if (track_path) {
    //             path.push_back(entry_id);
    //         }
    //     }
    //     std::vector<std::pair<float, int>> candidate_heap = result_set;
    //     std::unordered_set<int> visited{entry_id};

    //     while (!candidate_heap.empty()) {
    //         std::pop_heap(candidate_heap.begin(), candidate_heap.end(),
    //                       [](const auto &a, const auto &b) {
    //                           return a.first > b.first;
    //                       });
    //         auto current = candidate_heap.back();
    //         candidate_heap.pop_back();

    //         float worst_dist = result_set.back().first;
    //         if (static_cast<int>(result_set.size()) >= ef &&
    //             current.first > worst_dist) {
    //             break;
    //         }

    //         for (int neighbor : nodes_[current.second].neighbors[0]) {
    //             if (visited.count(neighbor) || nodes_[neighbor].deleted) {
    //                 continue;
    //             }
    //             visited.insert(neighbor);
    //             float dist_val = distance(query, nodes_[neighbor].vector);
    //             candidate_heap.emplace_back(dist_val, neighbor);
    //             std::push_heap(candidate_heap.begin(), candidate_heap.end(),
    //                            [](const auto &a, const auto &b) {
    //                                return a.first > b.first;
    //                            });
    //             result_set.emplace_back(dist_val, neighbor);
    //             std::push_heap(result_set.begin(), result_set.end(),
    //                            [](const auto &a, const auto &b) {
    //                                return a.first < b.first;
    //                            });
    //             if (track_path) {
    //                 path.push_back(neighbor);
    //             }
    //             if (static_cast<int>(result_set.size()) > ef) {
    //                 std::pop_heap(result_set.begin(), result_set.end(),
    //                               [](const auto &a, const auto &b) {
    //                                   return a.first < b.first;
    //                               });
    //                 result_set.pop_back();
    //             }
    //         }
    //     }
    //     return {result_set, path};
    // }
    // std::pair<std::vector<std::pair<float,int>>, std::vector<int>>
    // search_base_layer(const std::vector<float>& query,
    //                 int entry_id, int ef, bool track_path)
    // {
    //     std::vector<int> path;
        
    //     // Min-heap by distance
    //     auto cmp = [](auto& a, auto& b){ return a.first > b.first; };
    //     std::priority_queue<
    //         std::pair<float,int>,
    //         std::vector<std::pair<float,int>>,
    //         decltype(cmp)> candidates(cmp);

    //     std::vector<std::pair<float,int>> top(1);

    //     float dist = distance(query, nodes_[entry_id].vector);
    //     top[0] = {dist, entry_id};

    //     candidates.push({dist, entry_id});

    //     std::unordered_set<int> visited;
    //     visited.insert(entry_id);

    //     if (track_path) path.push_back(entry_id);
        
    //     while (!candidates.empty()) {
    //         auto curr = candidates.top();
    //         candidates.pop();

    //         float currDist = curr.first;
    //         int   currId   = curr.second;

    //         // Stop condition
    //         if (currDist > top.back().first)
    //             break;
            
            
    //         for (int nb : nodes_[currId].neighbors.at(0)) {
    //             if (visited.count(nb) || nodes_[nb].deleted)
    //                 continue;
    //             visited.insert(nb);

    //             if (track_path) path.push_back(nb);

    //             float d = distance(query, nodes_[nb].vector);

    //             // candidate for exploring
    //             candidates.push({d, nb});

    //             // candidate for final top
    //             top.push_back({d, nb});
    //             std::push_heap(top.begin(), top.end(),
    //                         [](auto& a, auto& b){ return a.first < b.first; });

    //             // prune to ef
    //             if ((int)top.size() > ef) {
    //                 std::pop_heap(top.begin(), top.end(),
    //                             [](auto& a, auto& b){ return a.first < b.first; });
    //                 top.pop_back();
    //             }
    //         }
    //     }

    //     std::sort(top.begin(), top.end(),
    //             [](auto& a, auto& b){ return a.first < b.first; });
    //     std::cout << "Searching base layer at entry_id " << entry_id << "\n";
    //     return {top, path};
    // }

    std::pair<std::vector<std::pair<float,int>>, std::vector<int>>
    search_layer(const std::vector<float>& query,
                int entry_id,
                int level,
                int ef,
                bool track_path)
    {
        std::vector<int> path;

        using Pair = std::pair<float,int>;

        // min-heap: candidates to explore (best first)
        auto cmp_min = [](const Pair& a, const Pair& b){ return a.first > b.first; };
        std::priority_queue<Pair, std::vector<Pair>, decltype(cmp_min)> candidates(cmp_min);

        // max-heap: current best results (worst on top)
        auto cmp_max = [](const Pair& a, const Pair& b){ return a.first < b.first; };
        std::priority_queue<Pair, std::vector<Pair>, decltype(cmp_max)> results(cmp_max);

        std::unordered_set<int> visited;
        visited.reserve(ef * 4);

        // seed
        if (!nodes_[entry_id].deleted && nodes_[entry_id].level >= level) {
            float d0 = distance(query, nodes_[entry_id].vector);
            candidates.push({d0, entry_id});
            results.push({d0, entry_id});
            visited.insert(entry_id);
            if (track_path) path.push_back(entry_id);
        } else {
            visited.insert(entry_id);
        }

        while (!candidates.empty()) {
            auto curr = candidates.top(); candidates.pop();
            float currDist = curr.first;
            int   currId   = curr.second;

            float worstDist = results.top().first;

            if (results.size() >= (size_t)ef && currDist > worstDist) break;

            auto it = nodes_[currId].neighbors.find(level);
            if (it == nodes_[currId].neighbors.end()) continue;

            for (int nb : it->second) {
                if (nodes_[nb].deleted) continue;
                if (nodes_[nb].level < level) continue;
                if (visited.count(nb)) continue;
                visited.insert(nb);
                if (track_path) path.push_back(nb);

                float d = distance(query, nodes_[nb].vector);

                if (results.size() < (size_t)ef || d < results.top().first) {
                    candidates.push({d, nb});
                    results.push({d, nb});
                    if (results.size() > (size_t)ef) results.pop();
                }
            }
        }

        std::vector<Pair> out;
        out.reserve(results.size());
        while (!results.empty()) { out.push_back(results.top()); results.pop(); }
        std::sort(out.begin(), out.end(), [](auto& a, auto& b){ return a.first < b.first; });

        return {out, path};
    }

    // std::pair<std::vector<std::pair<float,int>>, std::vector<int>>
    // search_layer(const std::vector<float>& query,
    //             int entry_id, int level, int ef, bool track_path)
    // {
    //     // std::vector<int> path;

    //     // auto cmp = [](auto& a, auto& b){ return a.first > b.first; };
    //     // std::priority_queue<
    //     //     std::pair<float,int>,
    //     //     std::vector<std::pair<float,int>>,
    //     //     decltype(cmp)> candidates(cmp);

    //     // std::vector<std::pair<float,int>> top;

    //     // float dist0 = distance(query, nodes_[entry_id].vector);
    //     // top.push_back({dist0, entry_id});
    //     // candidates.push({dist0, entry_id});

    //     // std::unordered_set<int> visited;
    //     // visited.insert(entry_id);
    //     // if (track_path) path.push_back(entry_id);
    //     std::vector<int> path;

    //     using Pair = std::pair<float,int>;

    //     // min-heap: candidates to explore (best first)
    //     auto cmp_min = [](const Pair& a, const Pair& b){ return a.first > b.first; };
    //     std::priority_queue<Pair, std::vector<Pair>, decltype(cmp_min)> candidates(cmp_min);

    //     // max-heap: current best results (worst on top)
    //     auto cmp_max = [](const Pair& a, const Pair& b){ return a.first < b.first; };
    //     std::priority_queue<Pair, std::vector<Pair>, decltype(cmp_max)> results(cmp_max);

    //     std::unordered_set<int> visited;
    //     visited.reserve(ef * 4);

    //     // seed
    //     if (!nodes_[entry_id].deleted && nodes_[entry_id].level >= level) {
    //         float d0 = distance(query, nodes_[entry_id].vector);
    //         candidates.push({d0, entry_id});
    //         results.push({d0, entry_id});
    //         visited.insert(entry_id);
    //         if (track_path) path.push_back(entry_id);
    //     } else {
    //         visited.insert(entry_id);
    //     }


    //     while (!candidates.empty()) {
    //         auto curr = candidates.top(); candidates.pop();
    //         float currDist = curr.first;
    //         int   currId   = curr.second;

    //         if (currDist > top.top().first) break;

    //         auto it = nodes_[currId].neighbors.find(level);
    //         if (it == nodes_[currId].neighbors.end()) continue;

    //         for (int nb : it->second) {
    //             if (visited.count(nb) || nodes_[nb].deleted) continue;
    //             visited.insert(nb);
    //             if (track_path) path.push_back(nb);

    //             float d = distance(query, nodes_[nb].vector);

    //             candidates.push({d, nb});

    //             top.push_back({d, nb});
    //             std::push_heap(top.begin(), top.end(),
    //                         [](auto& a, auto& b){ return a.first < b.first; });

    //             if ((int)top.size() > ef) {
    //                 std::pop_heap(top.begin(), top.end(),
    //                             [](auto& a, auto& b){ return a.first < b.first; });
    //                 top.pop_back();
    //             }
    //         }
    //     }

    //     // std::sort(top.begin(), top.end(),
    //     //         [](auto& a, auto& b){ return a.first < b.first; });

    //     // return {top, path};
    //     std::vector<Pair> out;
    //     out.reserve(results.size());
    //     while (!results.empty()) { out.push_back(results.top()); results.pop(); }
    //     std::sort(out.begin(), out.end(), [](auto& a, auto& b){ return a.first < b.first; });

    //     return {out, path};
    // }

    void rebuild_neighbors_mnru_all_levels(int node_id) {
        HNSWNode &node = nodes_[node_id];
        for (int level = node.level; level >= 0; --level) {
            rebuild_neighbors_mnru(node_id, level);
        }
    }

    void reassign_entry_point() {
        int best_id = -1;
        int best_level = -1;
        for (size_t i = 0; i < nodes_.size(); ++i) {
            if (!nodes_[i].deleted && nodes_[i].level > best_level) {
                best_level = nodes_[i].level;
                best_id = static_cast<int>(i);
            }
        }
        if (best_id == -1) {
            entry_point_.reset();
            max_level_ = -1;
        } else {
            entry_point_ = best_id;
            max_level_ = best_level;
        }
    }

    void remove_node_from_layer(int node_id, int level) {
        auto it = nodes_[node_id].neighbors.find(level);
        if (it == nodes_[node_id].neighbors.end()) {
            return;
        }
        auto neighbors = it->second;
        for (int neighbor : neighbors) {
            unlink(node_id, neighbor, level);
        }
        nodes_[node_id].neighbors.erase(level);
    }

    void unlink(int src, int dst, int level) {
        nodes_[src].neighbors[level].erase(dst);
        if (nodes_[src].neighbors[level].empty()) {
            nodes_[src].neighbors.erase(level);
        }
        nodes_[dst].neighbors[level].erase(src);
        if (nodes_[dst].neighbors[level].empty()) {
            nodes_[dst].neighbors.erase(level);
        }
    }

    int local_rewire_layer(const std::vector<int> &neighbors, int level) {
        if (neighbors.size() <= 1) {
            return 0;
        }
        int rewired = 0;
        for (size_t i = 0; i < neighbors.size(); ++i) {
            for (size_t j = i + 1; j < neighbors.size(); ++j) {
                int u = neighbors[i];
                int v = neighbors[j];
                if (nodes_[u].neighbors[level].count(v)) {
                    continue;
                }
                int u_degree = nodes_[u].neighbors[level].size();
                int v_degree = nodes_[v].neighbors[level].size();
                if (u_degree < m_ && v_degree < m_) {
                    link(u, v, level, 1);
                    ++rewired;
                } else if (u_degree < m_) {
                    if (should_add_edge_greedy(v, u, level)) {
                        link(u, v, level, 1);
                        ++rewired;
                    }
                } else if (v_degree < m_) {
                    if (should_add_edge_greedy(u, v, level)) {
                        link(u, v, level, 1);
                        ++rewired;
                    }
                } else {
                    if (should_replace_edge_greedy(u, v, level)) {
                        ++rewired;
                    }
                }
            }
        }
        return rewired;
    }

    std::unordered_map<int, std::vector<int>> local_rewire_layer_representative(int representative, const std::vector<int> &neighbors, int level) { 
        int rewired = 0; 
        std::unordered_map<int, std::vector<int>> layer_neighbors;
        layer_neighbors.reserve(neighbors.size());
        auto it_rep = nodes_[representative].neighbors.find(level);
        if (it_rep == nodes_[representative].neighbors.end()) {
            return layer_neighbors;
        }
        for (int u : neighbors) 
        { 
            if (u == representative) continue;
            if (nodes_[u].deleted) continue;
            if (nodes_[u].level < level) continue;

            // u must have this level too, otherwise skip (THIS was your crash).
            auto it_u = nodes_[u].neighbors.find(level);
            if (it_u == nodes_[u].neighbors.end()) continue;

            // already connected? (use it_rep instead of operator[])
            if (it_rep->second.count(u)) continue;

            const auto &nbrs = it_u->second;
            std::vector<int> candidates_src;
            for (int nbr : nbrs) candidates_src.push_back(nbr);
            candidates_src.push_back(representative);

            // std::vector<int> pruned_src = prune_alpha_rng(u, candidates_src, 1);
            layer_neighbors.emplace(u, std::move(candidates_src));
            
        } 
        return layer_neighbors; 
    }

    // void rebuild_neighbors_lr_rn_all_levels(int node_id) {
    //     HNSWNode &node = nodes_[node_id];

    //     for (int level = node.level; level >= 0; --level) {
    //         std::vector<int> layer_neighbors;

    //         auto it = node.neighbors.find(level);
    //         if (it != node.neighbors.end()) {
    //             for (int n : it->second) {
    //                 if (!nodes_[n].deleted) {
    //                     layer_neighbors.push_back(n);
    //                 }
    //             }
    //         }

    //         // Ensure node_id is connected to its neighborhood at this level
    //         local_rewire_layer_representative(node_id, layer_neighbors, level);
    //     }
    // }
    

    bool should_add_edge_greedy(int node_id, int candidate, int level) {
        auto &neighbors = nodes_[node_id].neighbors[level];
        if (static_cast<int>(neighbors.size()) < m_) {
            return true;
        }
        float candidate_dist =
            distance(nodes_[node_id].vector, nodes_[candidate].vector);
        float worst_dist = -1.0f;
        int worst_neighbor = -1;
        for (int neighbor : neighbors) {
            float dist_val =
                distance(nodes_[node_id].vector, nodes_[neighbor].vector);
            if (dist_val > worst_dist) {
                worst_dist = dist_val;
                worst_neighbor = neighbor;
            }
        }
        if (candidate_dist < worst_dist && worst_neighbor != -1) {
            unlink(node_id, worst_neighbor, level);
            return true;
        }
        return false;
    }

    bool should_replace_edge_greedy(int u, int v, int level) {
        bool replaced = false;
        auto attempt_replace = [&](int node, int other) {
            auto &neighbors = nodes_[node].neighbors[level];
            float uv_dist = distance(nodes_[node].vector, nodes_[other].vector);
            float worst_dist = -1.0f;
            int worst_neighbor = -1;
            for (int neighbor : neighbors) {
                float dist_val =
                    distance(nodes_[node].vector, nodes_[neighbor].vector);
                if (dist_val > worst_dist) {
                    worst_dist = dist_val;
                    worst_neighbor = neighbor;
                }
            }
            if (worst_neighbor != -1 && uv_dist < worst_dist) {
                unlink(node, worst_neighbor, level);
                link(node, other, level, 1);
                return true;
            }
            return false;
        };

        replaced = attempt_replace(u, v) || attempt_replace(v, u);
        return replaced;
    }

    // MN-RU style neighbor reconstruction for a single node at a given level.
    // For simplicity, we:
    //   - clear existing neighbors at this level
    //   - run an ANN search around this node's own vector
    //   - connect to up to m_ closest non-deleted nodes
    int rebuild_neighbors_mnru(int node_id, int level) {
        if (nodes_[node_id].deleted) return 0;

        // Remove all current neighbors at this level.
        auto it = nodes_[node_id].neighbors.find(level);
        if (it != nodes_[node_id].neighbors.end()) {
            // copy to avoid iterator invalidation
            std::vector<int> old_neighbors(it->second.begin(), it->second.end());
            for (int neigh : old_neighbors) {
                unlink(node_id, neigh, level);
            }
        }

        // Use ANN search to find candidate neighbors for reconstruction.
        // We ask for more than m_ candidates, then keep the closest m_.
        int K = std::max(m_ * 2, m_);
        auto results = search(nodes_[node_id].vector, K, ef_construction_, false);

        int added = 0;
        for (const auto &res : results) {
            auto it_label = label_to_id_.find(res.label);
            if (it_label == label_to_id_.end()) continue;
            int other_id = it_label->second;
            if (other_id == node_id) continue;
            if (nodes_[other_id].deleted) continue;
            link(node_id, other_id, level, 1);
            ++added;
            if (added >= m_) break;
        }

        return added;
    }

    
};

// Forward declarations
void test_top_layer_deletion();
void test_sequential_deletion_degradation();

std::vector<std::vector<float>> generate_gaussian_clusters(
    int N, int D, int clusters = 50, float cluster_spread = 0.1f, float separation = 1.5f)
{
    std::mt19937 rng(42);
    std::normal_distribution<float> norm(0.0, 1.0);

    // Generate cluster centers
    std::vector<std::vector<float>> centers(clusters, std::vector<float>(D));
    for (int c = 0; c < clusters; ++c) {
        for (int d = 0; d < D; ++d) {
            centers[c][d] = norm(rng) * separation;
        }
    }

    // Generate points
    std::vector<std::vector<float>> vectors(N, std::vector<float>(D));
    for (int i = 0; i < N; ++i) {
        int c = i % clusters;  // simple assignment (balanced clusters)
        for (int d = 0; d < D; ++d) {
            vectors[i][d] = centers[c][d] + norm(rng) * cluster_spread;
        }
    }

    return vectors;
}

std::vector<std::vector<float>> generate_gaussian_shells(
    int N, int D, int shells = 10, float separation = 3.0f, float noise = 0.05f)
{
    std::mt19937 rng(42);
    std::normal_distribution<float> norm(0.0, 1.0);

    std::vector<std::vector<float>> vectors(N, std::vector<float>(D));
    
    for (int i = 0; i < N; ++i) {
        int s = i % shells;                 // choose shell
        float radius = 1.0f + s * separation;

        // sample normal vector
        std::vector<float> v(D);
        float len = 0.0f;
        for (int d = 0; d < D; ++d) {
            float x = norm(rng);
            v[d] = x;
            len += x * x;
        }
        len = std::sqrt(len);

        // scale to radius + noise
        for (int d = 0; d < D; ++d) {
            v[d] = v[d] / len * radius + norm(rng) * noise;
        }

        vectors[i] = std::move(v);
    }
    return vectors;
}


std::vector<std::vector<float>> generate_cluster_with_noise(
    int N, int D, int clusters = 50, float noise_ratio = 0.1)
{
    auto cluster_vecs = generate_gaussian_clusters(N, D, clusters);

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> uni(0.0f, 1.0f);

    int noise_points = N * noise_ratio;

    for (int i = 0; i < noise_points; ++i) {
        int idx = rng() % N;
        for (int d = 0; d < D; ++d) {
            cluster_vecs[idx][d] = uni(rng);
        }
    }

    return cluster_vecs;
}

std::vector<std::vector<float>> generate_sift_like_vectors(int N, int D)
{
    std::mt19937 rng(42);
    std::normal_distribution<float> norm(0.0, 0.1);

    std::vector<std::vector<float>> vectors(N, std::vector<float>(D));

    for (int i = 0; i < N; ++i) {
        for (int d = 0; d < D; ++d) {
            vectors[i][d] = norm(rng);
        }
    }

    return vectors;
}

std::vector<std::vector<float>> load_fvecs(const std::string &filename) {
    FILE *f = fopen(filename.c_str(), "rb");
    if (!f) throw std::runtime_error("Could not open file: " + filename);

    std::vector<std::vector<float>> data;

    while (true) {
        int dim;
        if (fread(&dim, sizeof(int), 1, f) != 1) break;

        std::vector<float> vec(dim);
        size_t read = fread(vec.data(), sizeof(float), dim, f);
        if (read != (size_t)dim) break;

        data.push_back(std::move(vec));
    }

    fclose(f);
    return data;
}

std::vector<std::vector<int>> load_ivecs(const std::string &filename) {
    FILE *f = fopen(filename.c_str(), "rb");
    if (!f) throw std::runtime_error("Could not open file: " + filename);

    std::vector<std::vector<int>> data;

    while (true) {
        int dim;
        if (fread(&dim, sizeof(int), 1, f) != 1) break;

        std::vector<int> vec(dim);
        size_t read = fread(vec.data(), sizeof(int), dim, f);
        if (read != (size_t)dim) break;

        data.push_back(std::move(vec));
    }

    fclose(f);
    return data;
}





int main() {
    // Start total execution timer
    auto program_start = std::chrono::steady_clock::now();
    
    // const int N = 50000;
    // const int D = 128;
    // const int K = 5;
    // const int EF = 20;

    // std::cout << "======================================================\n";
    // std::cout << "HNSW Deletion Comparison: Tombstone vs Local Rewiring\n";
    // std::cout << "======================================================\n";
    // std::cout << "\nConfiguration:\n";
    // std::cout << "  N: " << N << "\n";
    // std::cout << "  D: " << D << "\n";
    // std::cout << "  K: " << K << "\n";
    // std::cout << "  EF: " << EF << "\n";

    // std::mt19937 rng(42);
    // std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    // std::vector<std::string> labels;
    // std::vector<std::vector<float>> vectors;
    // labels.reserve(N);
    // vectors.reserve(N);

    // for (int i = 0; i < N; ++i) {
    //     std::string label = "vec_" + std::to_string(i);
    //     std::vector<float> vec(D);
    //     for (int d = 0; d < D; ++d) {
    //         vec[d] = dist(rng);
    //     }
    //     labels.push_back(label);
    //     vectors.push_back(std::move(vec));
    // }

    // std::vector<float> query(D);
    // for (int d = 0; d < D; ++d) {
    //     query[d] = dist(rng);
    // }

    // SimpleHNSW hnsw_tombstone(D, 16, 0.5f, 16, 42);
    // SimpleHNSW hnsw_rewiring(D, 16, 0.5f, 16, 42);

    // std::cout << "\n[Inserting " << N << " vectors into both indexes...]\n";
    // for (int i = 0; i < N; ++i) {
    //     hnsw_tombstone.insert(labels[i], vectors[i]);
    //     hnsw_rewiring.insert(labels[i], vectors[i]);
    // }

    // std::cout << "\n[Baseline search before deletion]\n";
    // auto baseline_results =
    //     hnsw_tombstone.search(query, K, EF, /*track_path=*/false);

    // std::uniform_int_distribution<int> dist_idx(1, N - 2);
    // std::string node_to_delete = labels[dist_idx(rng)];
    // std::cout << "\n[Deleting node: '" << node_to_delete << "']\n";

    // auto start = std::chrono::steady_clock::now();
    // hnsw_tombstone.delete_node(node_to_delete);
    // auto end = std::chrono::steady_clock::now();
    // double tombstone_delete_ms =
    //     std::chrono::duration<double, std::milli>(end - start).count();

    // start = std::chrono::steady_clock::now();
    // auto tombstone_results =
    //     hnsw_tombstone.search(query, K, EF, /*track_path=*/false);
    // end = std::chrono::steady_clock::now();
    // double tombstone_search_ms =
    //     std::chrono::duration<double, std::milli>(end - start).count();

    // start = std::chrono::steady_clock::now();
    // auto stats = hnsw_rewiring.delete_with_local_rewiring(node_to_delete);
    // end = std::chrono::steady_clock::now();
    // double rewiring_delete_ms =
    //     std::chrono::duration<double, std::milli>(end - start).count();

    // start = std::chrono::steady_clock::now();
    // auto rewiring_results =
    //     hnsw_rewiring.search(query, K, EF, /*track_path=*/false);
    // end = std::chrono::steady_clock::now();
    // double rewiring_search_ms =
    //     std::chrono::duration<double, std::milli>(end - start).count();

    // auto print_results = [&](const std::string &title,
    //                          const std::vector<SimpleHNSW::SearchResult> &results) {
    //     std::cout << "\n" << title << "\n";
    //     int rank = 1;
    //     for (const auto &res : results) {
    //         std::cout << "  " << rank++ << ". " << res.label
    //                   << " distance=" << res.distance << "\n";
    //     }
    // };

    // print_results("Baseline (before deletion)", baseline_results);
    // print_results("Tombstone Deletion", tombstone_results);
    // print_results("Local Rewiring Deletion", rewiring_results);

    // auto extract_labels = [](const std::vector<SimpleHNSW::SearchResult> &results) {
    //     std::vector<std::string> labels;
    //     labels.reserve(results.size());
    //     for (const auto &res : results) {
    //         labels.push_back(res.label);
    //     }
    //     return labels;
    // };

    // auto labels_equal = [](const std::vector<std::string> &a,
    //                        const std::vector<std::string> &b) {
    //     if (a.size() != b.size()) {
    //         return false;
    //     }
    //     for (size_t i = 0; i < a.size(); ++i) {
    //         if (a[i] != b[i]) {
    //             return false;
    //         }
    //     }
    //     return true;
    // };

    // auto distances_match_fn = [](const std::vector<SimpleHNSW::SearchResult> &a,
    //                              const std::vector<SimpleHNSW::SearchResult> &b,
    //                              float eps = 1e-6f) {
    //     if (a.size() != b.size()) {
    //         return false;
    //     }
    //     for (size_t i = 0; i < a.size(); ++i) {
    //         if (std::fabs(a[i].distance - b[i].distance) >= eps) {
    //             return false;
    //         }
    //     }
    //     return true;
    // };

    // auto tombstone_labels = extract_labels(tombstone_results);
    // auto rewiring_labels = extract_labels(rewiring_results);
    // bool labels_match = labels_equal(tombstone_labels, rewiring_labels);
    // bool distances_match = distances_match_fn(tombstone_results, rewiring_results);

    // std::cout << "\n" << std::string(70, '-')
    //           << "\nRESULT COMPARISON:\n"
    //           << std::string(70, '-') << "\n";
    // if (labels_match && distances_match) {
    //     std::cout << "✓ SUCCESS: Both deletion methods return IDENTICAL results!\n";
    //     std::cout << "  - Labels match: true\n";
    //     std::cout << "  - Distances match: true\n";
    // } else {
    //     std::cout << "✗ DIFFERENCE: Results differ between deletion methods\n";
    //     if (!labels_match) {
    //         std::cout << "  - Labels differ:\n";
    //         std::cout << "    Tombstone: ";
    //         for (const auto &lab : tombstone_labels) {
    //             std::cout << lab << " ";
    //         }
    //         std::cout << "\n    Rewiring:  ";
    //         for (const auto &lab : rewiring_labels) {
    //             std::cout << lab << " ";
    //         }
    //         std::cout << "\n";
    //     }
    //     if (!distances_match) {
    //         std::cout << "  - Distances differ:\n";
    //         size_t limit = std::min(tombstone_results.size(), rewiring_results.size());
    //         for (size_t i = 0; i < limit; ++i) {
    //             if (std::fabs(tombstone_results[i].distance -
    //                           rewiring_results[i].distance) >= 1e-6f) {
    //                 std::cout << "    Rank " << (i + 1)
    //                           << ": Tombstone=" << tombstone_results[i].distance
    //                           << ", Rewiring=" << rewiring_results[i].distance
    //                           << "\n";
    //             }
    //         }
    //     }
    // }

    // auto contains_label = [](const std::vector<SimpleHNSW::SearchResult> &results,
    //                          const std::string &label) {
    //     for (const auto &res : results) {
    //         if (res.label == label) {
    //             return true;
    //         }
    //     }
    //     return false;
    // };

    // bool deleted_in_tombstone = contains_label(tombstone_results, node_to_delete);
    // bool deleted_in_rewiring = contains_label(rewiring_results, node_to_delete);

    // std::cout << "\n[Verification]\n";
    // std::cout << "  Deleted node '" << node_to_delete
    //           << "' in tombstone results: " << std::boolalpha
    //           << deleted_in_tombstone << "\n";
    // std::cout << "  Deleted node '" << node_to_delete
    //           << "' in rewiring results: " << std::boolalpha
    //           << deleted_in_rewiring << "\n";
    // if (!deleted_in_tombstone && !deleted_in_rewiring) {
    //     std::cout << "  ✓ Deleted node correctly excluded from both methods\n";
    // } else {
    //     std::cout << "  ✗ WARNING: Deleted node still appears in results!\n";
    // }

    // std::cout << "\n[Performance Comparison]\n";
    // auto compare_times = [](const std::string &label, double a, double b) {
    //     std::cout << label << "\n";
    //     std::cout << "  Tombstone: " << a << " ms\n";
    //     std::cout << "  Rewiring:  " << b << " ms\n";
    //     if (a < b) {
    //         std::cout << "  → Tombstone is " << (b / a) << "x faster\n";
    //     } else if (b < a) {
    //         std::cout << "  → Rewiring is " << (a / b) << "x faster\n";
    //     } else {
    //         std::cout << "  → Both methods take the same time\n";
    //     }
    // };

    // compare_times("\nDeletion Time:", tombstone_delete_ms, rewiring_delete_ms);
    // compare_times("\nSearch Time:", tombstone_search_ms, rewiring_search_ms);

    // double tombstone_total = tombstone_delete_ms + tombstone_search_ms;
    // double rewiring_total = rewiring_delete_ms + rewiring_search_ms;
    // compare_times("\nTotal Time:", tombstone_total, rewiring_total);

    // std::cout << "\nRewired edges: " << stats.rewired_edges_count << "\n";
    // std::cout << "Representative node: " << stats.representative_node << "\n";

    // std::cout << "\nAnalysis complete.\n";
    
    // // Run top-layer deletion test
    // test_top_layer_deletion();
    
    // Run sequential deletion degradation test
    test_sequential_deletion_degradation();
    
    // Calculate and display total execution time
    auto program_end = std::chrono::steady_clock::now();
    double total_execution_time = std::chrono::duration<double, std::milli>(program_end - program_start).count();
    
    std::cout << "\n" << std::string(70, '=') << "\n";
    std::cout << "TOTAL EXECUTION TIME: " << total_execution_time << " ms";
    if (total_execution_time >= 1000.0) {
        std::cout << " (" << (total_execution_time / 1000.0) << " seconds)";
    }
    std::cout << "\n" << std::string(70, '=') << "\n";
    
    return 0;
}

// Test case for deleting top-layer nodes
// void test_top_layer_deletion() {
//     std::cout << "\n" << std::string(70, '=')
//               << "\nTEST: Deleting Top-Layer Nodes"
//               << "\n" << std::string(70, '=') << "\n";

//     const int N = 1000;
//     const int D = 64;
//     const int K = 5;
//     const int EF = 20;

//     // Generate random vectors
//     std::mt19937 rng(42);
//     std::uniform_real_distribution<float> dist(0.0f, 1.0f);
//     std::vector<std::pair<std::string, std::vector<float>>> data_points;
//     for (int i = 0; i < N; ++i) {
//         std::vector<float> vec(D);
//         for (int j = 0; j < D; ++j) {
//             vec[j] = dist(rng);
//         }
//         data_points.emplace_back("vec_" + std::to_string(i), vec);
//     }

//     // Generate query
//     std::vector<float> query(D);
//     for (int j = 0; j < D; ++j) {
//         query[j] = dist(rng);
//     }

//     // Create two identical indexes
//     SimpleHNSW hnsw_tombstone(D, 8, 0.5f, 16, 42);
//     SimpleHNSW hnsw_rewiring(D, 8, 0.5f, 16, 42);

//     std::cout << "\n[Inserting " << N << " vectors...]\n";
//     for (const auto &[label, vec] : data_points) {
//         hnsw_tombstone.insert(label, vec);
//         hnsw_rewiring.insert(label, vec);
//     }

//     // Get baseline search
//     auto baseline = hnsw_tombstone.search(query, K, EF, false);
//     std::cout << "\n[Baseline search - Top " << K << " results:]\n";
//     for (size_t i = 0; i < baseline.size() && i < K; ++i) {
//         std::cout << "  " << (i + 1) << ". " << baseline[i].label
//                   << " distance=" << baseline[i].distance << "\n";
//     }

//     // Simple heuristic: pick nodes that are likely at high levels
//     // In practice, you'd check the actual node.level, but for this test we'll use
//     // a simpler approach: delete nodes that are far from the query (likely at higher levels)
//     std::vector<std::pair<float, std::string>> distances;
//     for (const auto &[label, vec] : data_points) {
//         float dist = 0.0f;
//         for (int i = 0; i < D; ++i) {
//             float diff = query[i] - vec[i];
//             dist += diff * diff;
//         }
//         distances.emplace_back(std::sqrt(dist), label);
//     }
    
//     // Sort by distance (farthest first - these are more likely to be at higher levels)
//     std::sort(distances.begin(), distances.end(),
//               [](const auto &a, const auto &b) { return a.first > b.first; });
    
//     // Pick top 3 farthest nodes as candidates for top-layer deletion
//     std::vector<std::string> nodes_to_delete;
//     for (size_t i = 0; i < std::min(3UL, distances.size()); ++i) {
//         nodes_to_delete.push_back(distances[i].second);
//     }

//     std::cout << "\n[Testing deletion of " << nodes_to_delete.size()
//               << " candidate top-layer nodes]\n";
    
//     // Track cumulative times
//     double total_tombstone_delete = 0.0;
//     double total_rewiring_delete = 0.0;
//     double total_tombstone_search = 0.0;
//     double total_rewiring_search = 0.0;
    
//     for (const auto &node_label : nodes_to_delete) {
//         std::cout << "\n--- Deleting: " << node_label << " ---\n";

//         // Delete using tombstone
//         auto start = std::chrono::steady_clock::now();
//         hnsw_tombstone.delete_node(node_label);
//         auto end = std::chrono::steady_clock::now();
//         double tombstone_delete_time = std::chrono::duration<double, std::milli>(end - start).count();
//         total_tombstone_delete += tombstone_delete_time;

//         // Delete using rewiring
//         start = std::chrono::steady_clock::now();
//         // auto stats = hnsw_rewiring.delete_with_local_rewiring(node_label, );
//         end = std::chrono::steady_clock::now();
//         double rewiring_delete_time = std::chrono::duration<double, std::milli>(end - start).count();
//         total_rewiring_delete += rewiring_delete_time;

//         // Search after deletion
//         start = std::chrono::steady_clock::now();
//         auto tombstone_results = hnsw_tombstone.search(query, K, EF, false);
//         end = std::chrono::steady_clock::now();
//         double tombstone_search_time = std::chrono::duration<double, std::milli>(end - start).count();
//         total_tombstone_search += tombstone_search_time;
        
//         start = std::chrono::steady_clock::now();
//         auto rewiring_results = hnsw_rewiring.search(query, K, EF, false);
//         end = std::chrono::steady_clock::now();
//         double rewiring_search_time = std::chrono::duration<double, std::milli>(end - start).count();
//         total_rewiring_search += rewiring_search_time;

//         // Verify deleted node is not in results
//         auto contains = [](const std::vector<SimpleHNSW::SearchResult> &results,
//                           const std::string &label) {
//             for (const auto &r : results) {
//                 if (r.label == label) return true;
//             }
//             return false;
//         };

//         bool deleted_in_tombstone = contains(tombstone_results, node_label);
//         bool deleted_in_rewiring = contains(rewiring_results, node_label);

//         std::cout << "  Deletion time - Tombstone: " << tombstone_delete_time << " ms, "
//                   << "Rewiring: " << rewiring_delete_time << " ms\n";
//         std::cout << "  Search time - Tombstone: " << tombstone_search_time << " ms, "
//                   << "Rewiring: " << rewiring_search_time << " ms\n";
//         std::cout << "  Deleted node in results - Tombstone: " << std::boolalpha
//                   << deleted_in_tombstone << ", Rewiring: " << deleted_in_rewiring << "\n";
//         std::cout << "  Rewired edges: " << stats.rewired_edges_count << "\n";

//         if (deleted_in_tombstone || deleted_in_rewiring) {
//             std::cout << "  ✗ WARNING: Deleted node still in results!\n";
//         } else {
//             std::cout << "  ✓ Deleted node correctly excluded\n";
//         }

//         // Check if results match
//         auto extract_labels = [](const std::vector<SimpleHNSW::SearchResult> &results) {
//             std::vector<std::string> labels;
//             for (const auto &r : results) {
//                 labels.push_back(r.label);
//             }
//             return labels;
//         };

//         auto tombstone_labels = extract_labels(tombstone_results);
//         auto rewiring_labels = extract_labels(rewiring_results);
//         bool labels_match = (tombstone_labels == rewiring_labels);

//         if (labels_match) {
//             std::cout << "  ✓ Results match between methods\n";
//         } else {
//             std::cout << "  ✗ Results differ between methods\n";
//         }
//     }

//     // Final search comparison
//     std::cout << "\n[Final Search Comparison]\n";
//     auto final_tombstone = hnsw_tombstone.search(query, K, EF, false);
//     auto final_rewiring = hnsw_rewiring.search(query, K, EF, false);

//     std::cout << "Tombstone results:\n";
//     for (size_t i = 0; i < final_tombstone.size(); ++i) {
//         std::cout << "  " << (i + 1) << ". " << final_tombstone[i].label
//                   << " distance=" << final_tombstone[i].distance << "\n";
//     }
//     std::cout << "Rewiring results:\n";
//     for (size_t i = 0; i < final_rewiring.size(); ++i) {
//         std::cout << "  " << (i + 1) << ". " << final_rewiring[i].label
//                   << " distance=" << final_rewiring[i].distance << "\n";
//     }

//     // Performance comparison summary
//     std::cout << "\n" << std::string(70, '-')
//               << "\nPERFORMANCE COMPARISON (Top-Layer Deletion):\n"
//               << std::string(70, '-') << "\n";
    
//     auto compare_times = [](const std::string &label, double a, double b) {
//         std::cout << label << "\n";
//         std::cout << "  Tombstone: " << a << " ms\n";
//         std::cout << "  Rewiring:  " << b << " ms\n";
//         if (a < b) {
//             std::cout << "  → Tombstone is " << (b / a) << "x faster\n";
//         } else if (b < a) {
//             std::cout << "  → Rewiring is " << (a / b) << "x faster\n";
//         } else {
//             std::cout << "  → Both methods take the same time\n";
//         }
//     };

//     compare_times("\nTotal Deletion Time (all " + std::to_string(nodes_to_delete.size()) + " nodes):",
//                   total_tombstone_delete, total_rewiring_delete);
//     compare_times("\nTotal Search Time (after all deletions):",
//                   total_tombstone_search, total_rewiring_search);

//     double tombstone_total = total_tombstone_delete + total_tombstone_search;
//     double rewiring_total = total_rewiring_delete + total_rewiring_search;
//     compare_times("\nTotal Time (deletion + search):", tombstone_total, rewiring_total);

//     std::cout << "\nAverage deletion time per node:\n";
//     double avg_tombstone = total_tombstone_delete / nodes_to_delete.size();
//     double avg_rewiring = total_rewiring_delete / nodes_to_delete.size();
//     std::cout << "  Tombstone: " << avg_tombstone << " ms\n";
//     std::cout << "  Rewiring:  " << avg_rewiring << " ms\n";

//     std::cout << "\nAverage search time per query:\n";
//     double avg_tombstone_search = total_tombstone_search / nodes_to_delete.size();
//     double avg_rewiring_search = total_rewiring_search / nodes_to_delete.size();
//     std::cout << "  Tombstone: " << avg_tombstone_search << " ms\n";
//     std::cout << "  Rewiring:  " << avg_rewiring_search << " ms\n";

//     std::cout << "\n" << std::string(70, '=') << "\n";
//     std::cout << "Top-layer deletion test complete.\n";
// }

// // Test accuracy and performance degradation over sequential deletions
// void test_sequential_deletion_degradation() {
//     std::cout << "\n" << std::string(70, '=')
//               << "\nTEST: Sequential Deletion Degradation Analysis"
//               << "\n" << std::string(70, '=') << "\n";

//     const int N = 50000;  // Large dataset to support up to 10k deletions
//     const int D = 128;
//     const int K = 10;    // Retrieve more neighbors for better accuracy measurement
//     const int EF = 50;   // Higher EF for better accuracy

//     // Generate random vectors
//     std::mt19937 rng(42);
//     std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    
//     std::vector<std::string> labels;
//     std::vector<std::vector<float>> vectors;
//     labels.reserve(N);
//     vectors.reserve(N);
    
//     for (int i = 0; i < N; ++i) {
//         std::string label = "vec_" + std::to_string(i);
//         std::vector<float> vec(D);
//         for (int d = 0; d < D; ++d) {
//             vec[d] = dist(rng);
//         }
//         labels.push_back(label);
//         vectors.push_back(std::move(vec));
//     }

//     // Generate query
//     std::vector<float> query(D);
//     for (int d = 0; d < D; ++d) {
//         query[d] = dist(rng);
//     }

//     // Test with cumulative deletion milestones (10, then 100 total, then 1000 total, then 10000 total)
//     std::vector<int> deletion_milestones = {10, 100, 1000, 10000};
    
//     std::cout << "\nConfiguration:\n";
//     std::cout << "  Dataset size: " << N << " vectors\n";
//     std::cout << "  Dimensions: " << D << "\n";
//     std::cout << "  Deletion milestones (cumulative): ";
//     for (size_t i = 0; i < deletion_milestones.size(); ++i) {
//         std::cout << deletion_milestones[i];
//         if (i < deletion_milestones.size() - 1) std::cout << ", ";
//     }
//     std::cout << "\n";
//     std::cout << "  Note: Using same graph, accumulating deletions\n";

//     // Get baseline results (ground truth from brute force) BEFORE any deletions
//     std::cout << "\n[Computing baseline (ground truth) results from full dataset...]\n";
//     std::vector<std::pair<float, std::string>> all_distances;
//     for (int i = 0; i < N; ++i) {
//         float dist = 0.0f;
//         for (int d = 0; d < D; ++d) {
//             float diff = query[d] - vectors[i][d];
//             dist += diff * diff;
//         }
//         all_distances.emplace_back(std::sqrt(dist), labels[i]);
//     }
//     std::sort(all_distances.begin(), all_distances.end(),
//               [](const auto &a, const auto &b) { return a.first < b.first; });
    
//     std::vector<std::string> baseline_labels;
//     for (int i = 0; i < std::min(K, static_cast<int>(all_distances.size())); ++i) {
//         baseline_labels.push_back(all_distances[i].second);
//     }

//     std::cout << "Baseline top-" << K << " labels: ";
//     for (size_t i = 0; i < baseline_labels.size(); ++i) {
//         std::cout << baseline_labels[i];
//         if (i < baseline_labels.size() - 1) std::cout << ", ";
//     }
//     std::cout << "\n";

//     // Create indexes ONCE and reuse them
//     SimpleHNSW hnsw_tombstone(D, 16, 0.5f, 16, 42);
//     SimpleHNSW hnsw_rewiring(D, 16, 0.5f, 16, 42);
//     SimpleHNSW hnsw_mnru(D, 16, 0.5f, 16, 42);

//     std::cout << "\n[Inserting " << N << " vectors into both indexes...]\n";
//     for (int i = 0; i < N; ++i) {
//         hnsw_tombstone.insert(labels[i], vectors[i]);
//         hnsw_rewiring.insert(labels[i], vectors[i]);
//         hnsw_mnru.insert(labels[i], vectors[i]);
//     }

//     // Select random nodes to delete (shuffle once, use sequentially)
//     std::vector<int> indices(N);
//     std::iota(indices.begin(), indices.end(), 0);
//     std::shuffle(indices.begin(), indices.end(), rng);
//     std::vector<bool> deleted_flags(N, false);
    
//     // Results storage
//     struct DegradationResult {
//         int cumulative_deletions;
//         double tombstone_delete_time;
//         double rewiring_delete_time;
//         double mnru_delete_time;

//         double tombstone_search_time;
//         double rewiring_search_time;
//         double mnru_search_time;

//         double tombstone_accuracy;  // Recall@K
//         double rewiring_accuracy;
//         double mnru_accuracy;

//         int tombstone_correct;
//         int rewiring_correct;
//         int mnru_correct;
//     };
//     std::vector<DegradationResult> results;

//     int current_deletions = 0;
//     for (int target_deletions : deletion_milestones) {
//         if (target_deletions >= N) {
//             std::cout << "\n[Skipping " << target_deletions 
//                       << " deletions - exceeds dataset size]\n";
//             continue;
//         }

//         int deletions_to_perform = target_deletions - current_deletions;
//         std::cout << "\n" << std::string(70, '-') << "\n";
//         std::cout << "Deleting " << deletions_to_perform << " more nodes "
//                   << "(cumulative: " << target_deletions << ")\n";
//         std::cout << std::string(70, '-') << "\n";

//         // Perform deletions and measure time
//         double total_tombstone_delete = 0.0;
//         double total_rewiring_delete = 0.0;

//         for (int i = current_deletions; i < target_deletions; ++i) {
//             const std::string &label = labels[indices[i]];

//             auto start = std::chrono::steady_clock::now();
//             hnsw_tombstone.delete_node(label);
//             auto end = std::chrono::steady_clock::now();
//             total_tombstone_delete += 
//                 std::chrono::duration<double, std::milli>(end - start).count();

//             start = std::chrono::steady_clock::now();
//             hnsw_rewiring.delete_with_local_rewiring(label);
//             end = std::chrono::steady_clock::now();
//             total_rewiring_delete += 
//                 std::chrono::duration<double, std::milli>(end - start).count();
//         }

//         // Search and measure time
//         auto start = std::chrono::steady_clock::now();
//         auto tombstone_results = hnsw_tombstone.search(query, K, EF, false);
//         auto end = std::chrono::steady_clock::now();
//         double tombstone_search_time = 
//             std::chrono::duration<double, std::milli>(end - start).count();

//         start = std::chrono::steady_clock::now();
//         auto rewiring_results = hnsw_rewiring.search(query, K, EF, false);
//         end = std::chrono::steady_clock::now();
//         double rewiring_search_time = 
//             std::chrono::duration<double, std::milli>(end - start).count();

//         // Calculate accuracy (Recall@K) - compare against ground truth baseline
//         auto extract_labels = [](const std::vector<SimpleHNSW::SearchResult> &results) {
//             std::vector<std::string> labels;
//             for (const auto &r : results) {
//                 labels.push_back(r.label);
//             }
//             return labels;
//         };

//         auto tombstone_result_labels = extract_labels(tombstone_results);
//         auto rewiring_result_labels = extract_labels(rewiring_results);

//         // Count how many baseline (ground truth) labels are in results
//         std::unordered_set<std::string> baseline_set(baseline_labels.begin(), 
//                                                       baseline_labels.end());
        
//         int tombstone_correct = 0;
//         for (const auto &label : tombstone_result_labels) {
//             if (baseline_set.count(label)) {
//                 tombstone_correct++;
//             }
//         }

//         int rewiring_correct = 0;
//         for (const auto &label : rewiring_result_labels) {
//             if (baseline_set.count(label)) {
//                 rewiring_correct++;
//             }
//         }

//         double tombstone_accuracy = static_cast<double>(tombstone_correct) / K;
//         double rewiring_accuracy = static_cast<double>(rewiring_correct) / K;

//         DegradationResult result;
//         result.cumulative_deletions = target_deletions;
//         result.tombstone_delete_time = total_tombstone_delete;
//         result.rewiring_delete_time = total_rewiring_delete;
//         result.tombstone_search_time = tombstone_search_time;
//         result.rewiring_search_time = rewiring_search_time;
//         result.tombstone_accuracy = tombstone_accuracy;
//         result.rewiring_accuracy = rewiring_accuracy;
//         result.tombstone_correct = tombstone_correct;
//         result.rewiring_correct = rewiring_correct;
//         results.push_back(result);

//         std::cout << "\nResults after " << target_deletions << " cumulative deletions:\n";
//         std::cout << "  Deletion time (last " << deletions_to_perform << " nodes) - "
//                   << "Tombstone: " << total_tombstone_delete 
//                   << " ms, Rewiring: " << total_rewiring_delete << " ms\n";
//         std::cout << "  Search time - Tombstone: " << tombstone_search_time 
//                   << " ms, Rewiring: " << rewiring_search_time << " ms\n";
//         std::cout << "  Accuracy (Recall@" << K << " vs ground truth) - "
//                   << "Tombstone: " << (tombstone_accuracy * 100.0) << "% (" 
//                   << tombstone_correct << "/" << K << "), "
//                   << "Rewiring: " << (rewiring_accuracy * 100.0) << "% (" 
//                   << rewiring_correct << "/" << K << ")\n";

//         current_deletions = target_deletions;
//     }

//     // Summary table
//     std::cout << "\n" << std::string(70, '=') << "\n";
//     std::cout << "DEGRADATION SUMMARY TABLE\n";
//     std::cout << std::string(70, '=') << "\n";
//     std::cout << std::fixed << std::setprecision(2);
//     std::cout << "\n" << std::setw(8) << "Deletions" 
//               << std::setw(15) << "Tomb Del(ms)" 
//               << std::setw(15) << "Rew Del(ms)"
//               << std::setw(15) << "Tomb Srch(ms)"
//               << std::setw(15) << "Rew Srch(ms)"
//               << std::setw(12) << "Tomb Acc%"
//               << std::setw(12) << "Rew Acc%"
//               << "\n";
//     std::cout << std::string(100, '-') << "\n";

//     for (const auto &result : results) {
//         std::cout << std::setw(8) << result.cumulative_deletions
//                   << std::setw(15) << result.tombstone_delete_time
//                   << std::setw(15) << result.rewiring_delete_time
//                   << std::setw(15) << result.tombstone_search_time
//                   << std::setw(15) << result.rewiring_search_time
//                   << std::setw(12) << (result.tombstone_accuracy * 100.0)
//                   << std::setw(12) << (result.rewiring_accuracy * 100.0)
//                   << "\n";
//     }

//     // Calculate degradation rates
//     if (results.size() >= 2) {
//         std::cout << "\n" << std::string(70, '-') << "\n";
//         std::cout << "PERFORMANCE DEGRADATION ANALYSIS\n";
//         std::cout << std::string(70, '-') << "\n";

//         auto first = results[0];
//         auto last = results.back();

//         std::cout << "\nTombstone Method:\n";
//         double delete_degradation = (last.tombstone_delete_time / first.tombstone_delete_time);
//         double search_degradation = (last.tombstone_search_time / first.tombstone_search_time);
//         double accuracy_degradation = (last.tombstone_accuracy / first.tombstone_accuracy);
        
//         std::cout << "  Deletion time: " << first.tombstone_delete_time << " ms -> " 
//                   << last.tombstone_delete_time << " ms (" 
//                   << delete_degradation << "x slower)\n";
//         std::cout << "  Search time: " << first.tombstone_search_time << " ms -> " 
//                   << last.tombstone_search_time << " ms (" 
//                   << search_degradation << "x slower)\n";
//         std::cout << "  Accuracy: " << (first.tombstone_accuracy * 100.0) << "% -> " 
//                   << (last.tombstone_accuracy * 100.0) << "% (" 
//                   << accuracy_degradation << "x ratio)\n";

//         std::cout << "\nRewiring Method:\n";
//         double rew_delete_degradation = (last.rewiring_delete_time / first.rewiring_delete_time);
//         double rew_search_degradation = (last.rewiring_search_time / first.rewiring_search_time);
//         double rew_accuracy_degradation = (last.rewiring_accuracy / first.rewiring_accuracy);
        
//         std::cout << "  Deletion time: " << first.rewiring_delete_time << " ms -> " 
//                   << last.rewiring_delete_time << " ms (" 
//                   << rew_delete_degradation << "x slower)\n";
//         std::cout << "  Search time: " << first.rewiring_search_time << " ms -> " 
//                   << last.rewiring_search_time << " ms (" 
//                   << rew_search_degradation << "x slower)\n";
//         std::cout << "  Accuracy: " << (first.rewiring_accuracy * 100.0) << "% -> " 
//                   << (last.rewiring_accuracy * 100.0) << "% (" 
//                   << rew_accuracy_degradation << "x ratio)\n";
//     }

//     std::cout << "\n" << std::string(70, '=') << "\n";
//     std::cout << "Sequential deletion degradation test complete.\n";
// }

// ==========================================================
// Sequential Deletion Degradation Benchmark
// ==========================================================
// void test_sequential_deletion_degradation() {

//     std::cout << "\n" << std::string(70, '=')
//               << "\nTEST: Sequential Deletion Degradation Analysis"
//               << "\n" << std::string(70, '=') << "\n";

//     const int N = 50000;  
//     const int D = 128;
//     const int K = 10;  
//     const int EF = 50;   

//     std::mt19937 rng(42);
//     std::uniform_real_distribution<float> dist(0.0f, 1.0f);

//     // ---------------------------------------------
//     // Generate dataset
//     // ---------------------------------------------
//     std::vector<std::string> labels;
//     std::vector<std::vector<float>> vectors;
//     labels.reserve(N);
//     vectors.reserve(N);

//     for (int i = 0; i < N; ++i) {
//         std::string label = "vec_" + std::to_string(i);
//         std::vector<float> vec(D);
//         for (int d = 0; d < D; ++d) {
//             vec[d] = dist(rng);
//         }
//         labels.push_back(label);
//         vectors.push_back(std::move(vec));
//     }

//     // ---------------------------------------------
//     // Choose fixed query vector
//     // ---------------------------------------------
//     std::vector<float> query(D);
//     for (int d = 0; d < D; ++d) {
//         query[d] = dist(rng);
//     }

//     // ---------------------------------------------
//     // Compute Baseline Ground Truth
//     // ---------------------------------------------
//     std::cout << "\n[Computing ground-truth baseline...]\n";

//     std::vector<std::pair<float, std::string>> gt;
//     gt.reserve(N);

//     for (int i = 0; i < N; ++i) {
//         float sum = 0.f;
//         for (int d = 0; d < D; ++d) {
//             float diff = query[d] - vectors[i][d];
//             sum += diff * diff;
//         }
//         gt.emplace_back(std::sqrt(sum), labels[i]);
//     }

//     std::sort(gt.begin(), gt.end(),
//               [](auto &a, auto &b) { return a.first < b.first; });

//     std::unordered_set<std::string> gt_set;
//     for (int i = 0; i < K; ++i) {
//         gt_set.insert(gt[i].second);
//     }

//     std::cout << "Ground truth top-" << K << " computed.\n";

//     // ---------------------------------------------
//     // Build 3 indexes: Tombstone, Rewiring, MN-RU hybrid
//     // ---------------------------------------------
//     std::cout << "\n[Building HNSW Indexes...]\n";

//     SimpleHNSW h_tomb(D, 16, 0.5f, 16, 42);
//     SimpleHNSW h_rw(D, 16, 0.5f, 16, 42);
//     SimpleHNSW h_mnru(D, 16, 0.5f, 16, 42);

//     for (int i = 0; i < N; ++i) {
//         h_tomb.insert(labels[i], vectors[i]);
//         h_rw.insert(labels[i], vectors[i]);
//         h_mnru.insert(labels[i], vectors[i]);
//     }

//     // ---------------------------------------------
//     // Prepare deletion order
//     // ---------------------------------------------
//     std::vector<int> order(N);
//     std::iota(order.begin(), order.end(), 0);
//     std::shuffle(order.begin(), order.end(), rng);

//     // Evaluate at cumulative deletion steps
//     std::vector<int> milestones = {10, 100, 1000, 10000};

//     struct Row {
//         int del;
//         double t_del, rw_del, mn_del;
//         double t_sr, rw_sr, mn_sr;
//         int t_hit, rw_hit, mn_hit;
//     };

//     std::vector<Row> table;

//     int deleted = 0;

//     // ---------------------------------------------
//     // MAIN LOOP
//     // ---------------------------------------------
//     for (int target : milestones) {

//         int batch = target - deleted;
//         std::cout << "\n--- Deletions up to " << target << " ---\n";

//         double t_del_ms = 0, rw_del_ms = 0, mn_del_ms = 0;

//         for (int i = deleted; i < target; ++i) {
//             std::string label = labels[order[i]];

//             // Tombstone
//             auto s = std::chrono::high_resolution_clock::now();
//             h_tomb.delete_node(label);
//             auto e = std::chrono::high_resolution_clock::now();
//             t_del_ms += std::chrono::duration<double, std::milli>(e - s).count();

//             // Local Rewiring
//             s = std::chrono::high_resolution_clock::now();
//             h_rw.delete_with_local_rewiring(label);
//             e = std::chrono::high_resolution_clock::now();
//             rw_del_ms += std::chrono::duration<double, std::milli>(e - s).count();

//             // MN-RU Hybrid
//             s = std::chrono::high_resolution_clock::now();
//             h_mnru.delete_with_mnru_minimal(label);     // (You already added this helper)
//             e = std::chrono::high_resolution_clock::now();
//             mn_del_ms += std::chrono::duration<double, std::milli>(e - s).count();
//         }

//         deleted = target;

//         // -----------------------------------------------------
//         // Perform search on all 3 indexes
//         // -----------------------------------------------------
//         auto run_search = [&](SimpleHNSW &h, double &time_ms, int &hits) {
//             auto s = std::chrono::high_resolution_clock::now();
//             auto res = h.search(query, K, EF, false);
//             auto e = std::chrono::high_resolution_clock::now();
//             time_ms = std::chrono::duration<double, std::milli>(e - s).count();

//             hits = 0;
//             for (auto &r : res) {
//                 if (gt_set.count(r.label)) hits++;
//             }
//         };

//         double t_sr = 0, rw_sr = 0, mn_sr = 0;
//         int t_hit = 0, rw_hit = 0, mn_hit = 0;

//         run_search(h_tomb, t_sr, t_hit);
//         run_search(h_rw, rw_sr, rw_hit);
//         run_search(h_mnru, mn_sr, mn_hit);

//         table.push_back(
//             {target, t_del_ms, rw_del_ms, mn_del_ms,
//                      t_sr,    rw_sr,    mn_sr,
//                      t_hit,   rw_hit,   mn_hit});
//     }

//     // ---------------------------------------------
//     // Print summary table
//     // ---------------------------------------------
//     std::cout << "\n\n" << std::string(80, '=')
//               << "\nDEGRADATION SUMMARY TABLE\n"
//               << std::string(80, '=') << "\n";

//     std::cout << std::fixed << std::setprecision(2);

//     std::cout << std::setw(10) << "Del"
//               << std::setw(12) << "T-Del"
//               << std::setw(12) << "RW-Del"
//               << std::setw(12) << "MN-Del"
//               << std::setw(12) << "T-Srch"
//               << std::setw(12) << "RW-Srch"
//               << std::setw(12) << "MN-Srch"
//               << std::setw(10) << "T-Hit"
//               << std::setw(10) << "RW-Hit"
//               << std::setw(10) << "MN-Hit"
//               << "\n";

//     for (auto &r : table) {
//         std::cout << std::setw(10) << r.del
//                   << std::setw(12) << r.t_del
//                   << std::setw(12) << r.rw_del
//                   << std::setw(12) << r.mn_del
//                   << std::setw(12) << r.t_sr
//                   << std::setw(12) << r.rw_sr
//                   << std::setw(12) << r.mn_sr
//                   << std::setw(10) << r.t_hit
//                   << std::setw(10) << r.rw_hit
//                   << std::setw(10) << r.mn_hit
//                   << "\n";
//     }

//     std::cout << "\n" << std::string(80, '=') << "\n";
//     std::cout << "Sequential deletion test complete.\n";
// }
// void test_sequential_deletion_degradation() {
//     std::cout << "\n" << std::string(70, '=')
//               << "\nTEST: Sequential Deletion Degradation Analysis"
//               << "\n" << std::string(70, '=') << "\n";

//     const int N = 50000;  
//     const int D = 128;
//     const int K = 10;
//     const int EF = 50;

//     // ---------- DATASET GENERATION ----------
//     std::mt19937 rng(42);
//     std::uniform_real_distribution<float> dist(0.0f, 1.0f);

//     std::vector<std::string> labels;
//     // std::vector<std::vector<float>> vectors;
//     labels.reserve(N);
//     // vectors.reserve(N);

//     for (int i = 0; i < N; ++i) {
//         std::string label = "vec_" + std::to_string(i);
//         // std::vector<float> vec(D);
//         // for (int d = 0; d < D; ++d) vec[d] = dist(rng);
//         labels.push_back(label);
//         // vectors.push_back(std::move(vec));
//     }

//     auto vectors = generate_sift_like_vectors(N, D);

//     // Create a fixed query
//     // std::vector<float> query(D);
//     // for (int d = 0; d < D; ++d) query[d] = dist(rng);
//     // query = generate_sift_like_vectors(1, D)[0];
//     // int Q = 50;
//     // auto queries = generate_sift_like_vectors(Q, D);

//     // Deletion milestones
//     std::vector<int> milestones = {10, 100, 1000, 10000};

//     // --------- Generate permutation of delete order ---------
//     std::vector<int> indices(N);
//     std::iota(indices.begin(), indices.end(), 0);
//     std::shuffle(indices.begin(), indices.end(), rng);

//     // --------- Initialize three indexes ---------
//     SimpleHNSW hnsw_tomb(D, 16, 0.5f, 16, 42);
//     SimpleHNSW hnsw_lr(D,   16, 0.5f, 16, 42);
//     SimpleHNSW hnsw_mnru(D, 16, 0.5f, 16, 42);

//     std::cout << "\n[Inserting " << N << " vectors into all indexes...]\n";

//     for (int i = 0; i < N; ++i) {
//         hnsw_tomb.insert(labels[i], vectors[i]);
//         hnsw_lr.insert(labels[i], vectors[i]);
//         hnsw_mnru.insert(labels[i], vectors[i]);
//     }

//     // -------- Results structure --------
//     struct ResultRow {
//         int deletions;
//         double tomb_del, lr_del, mnru_del;
//         double tomb_srch, lr_srch, mnru_srch;
//         double tomb_recall, lr_recall, mnru_recall;
//     };

//     std::vector<ResultRow> table;
//     int current_del = 0;

//     // -----------------------------------------------------
//     //                MAIN LOOP OVER MILESTONES
//     // -----------------------------------------------------
//     // for (int target_del : milestones) {
//     //     int del_count = target_del - current_del;

//     //     std::cout << "\n" << std::string(70, '-')
//     //               << "\nDeleting " << del_count << " nodes (cumulative: "
//     //               << target_del << ")\n"
//     //               << std::string(70, '-') << "\n";

//     //     double tomb_del_ms = 0.0;
//     //     double lr_del_ms = 0.0;
//     //     double mnru_del_ms = 0.0;

//     //     for (int i = current_del; i < target_del; ++i) {
//     //         std::string lbl = labels[indices[i]];

//     //         auto start = std::chrono::steady_clock::now();
//     //         hnsw_tomb.delete_node(lbl);
//     //         auto end = std::chrono::steady_clock::now();
//     //         tomb_del_ms += std::chrono::duration<double, std::milli>(end - start).count();

//     //         start = std::chrono::steady_clock::now();
//     //         hnsw_lr.delete_with_local_rewiring(lbl);
//     //         end = std::chrono::steady_clock::now();
//     //         lr_del_ms += std::chrono::duration<double, std::milli>(end - start).count();

//     //         start = std::chrono::steady_clock::now();
//     //         hnsw_mnru.delete_with_MNRU(lbl);   // <--- YOU WILL ADD THIS METHOD
//     //         end = std::chrono::steady_clock::now();
//     //         mnru_del_ms += std::chrono::duration<double, std::milli>(end - start).count();
//     //     }

//     //     // --------- RUN SEARCH ---------
//     //     auto measure_search = [&](SimpleHNSW &idx) {
//     //         auto start = std::chrono::steady_clock::now();
//     //         auto res = idx.search(query, K, EF, false);
//     //         auto end = std::chrono::steady_clock::now();
//     //         double ms = std::chrono::duration<double, std::milli>(end - start).count();
//     //         return std::make_pair(res, ms);
//     //     };

//     //     auto [tomb_res, tomb_srch_ms] = measure_search(hnsw_tomb);
//     //     auto [lr_res,   lr_srch_ms]   = measure_search(hnsw_lr);
//     //     auto [mnru_res, mnru_srch_ms] = measure_search(hnsw_mnru);

//     //     // -----------------------------------------------------
//     //     //        RECOMPUTE GROUND TRUTH AFTER DELETIONS
//     //     // -----------------------------------------------------
//     //     std::vector<std::pair<float, std::string>> gt_all;
//     //     gt_all.reserve(N - target_del);

//     //     std::unordered_set<std::string> deleted_set;
//     //     for (int i = 0; i < target_del; ++i)
//     //         deleted_set.insert(labels[indices[i]]);

//     //     for (int i = 0; i < N; ++i) {
//     //         if (deleted_set.count(labels[i])) continue;

//     //         float distval = 0.0f;
//     //         for (int d = 0; d < D; ++d) {
//     //             float diff = query[d] - vectors[i][d];
//     //             distval += diff * diff;
//     //         }
//     //         gt_all.emplace_back(std::sqrt(distval), labels[i]);
//     //     }

//     //     std::sort(gt_all.begin(), gt_all.end(),
//     //               [](auto &a, auto &b) { return a.first < b.first; });

//     //     std::unordered_set<std::string> gt_set;
//     //     for (int i = 0; i < K && i < (int)gt_all.size(); ++i)
//     //         gt_set.insert(gt_all[i].second);

//     //     auto compute_recall = [&](auto &results) {
//     //         int hits = 0;
//     //         for (int i = 0; i < (int)results.size() && i < K; i++) {
//     //             if (gt_set.count(results[i].label)) hits++;
//     //         }
//     //         return (double)hits / K;
//     //     };

//     //     double tomb_recall = compute_recall(tomb_res);
//     //     double lr_recall   = compute_recall(lr_res);
//     //     double mnru_recall = compute_recall(mnru_res);

//     //     // Store result row
//     //     table.push_back({
//     //         target_del,
//     //         tomb_del_ms, lr_del_ms, mnru_del_ms,
//     //         tomb_srch_ms, lr_srch_ms, mnru_srch_ms,
//     //         tomb_recall, lr_recall, mnru_recall
//     //     });

//     //     current_del = target_del;
//     // }
//     for (int target_del : milestones) {
//         // --------- RUN SEARCH FOR MULTIPLE QUERIES ---------
//         int Q = 50;  // number of queries to average over
//         auto query_set = generate_sift_like_vectors(Q, D);

//         auto measure_search_query = [&](SimpleHNSW &idx, const std::vector<float> &q) {
//             auto start = std::chrono::steady_clock::now();
//             auto res = idx.search(q, K, EF, false);
//             auto end = std::chrono::steady_clock::now();
//             double ms = std::chrono::duration<double, std::milli>(end - start).count();
//             return std::make_pair(res, ms);
//         };

//         int del_count = target_del - current_del;
//         std::cout << "\n" << std::string(70, '-')
//                   << "\nDeleting " << del_count << " nodes (cumulative: "
//                   << target_del << ")\n"
//                   << std::string(70, '-') << "\n";

//         double tomb_del_ms = 0.0;
//         double lr_del_ms = 0.0;
//         double mnru_del_ms = 0.0;
//         for (int i = current_del; i < target_del; ++i) {
//             std::string lbl = labels[indices[i]];

//             auto start = std::chrono::steady_clock::now();
//             hnsw_tomb.delete_node(lbl);
//             auto end = std::chrono::steady_clock::now();
//             tomb_del_ms += std::chrono::duration<double, std::milli>(end - start).count();

//             start = std::chrono::steady_clock::now();
//             hnsw_lr.delete_with_local_rewiring(lbl);
//             end = std::chrono::steady_clock::now();
//             lr_del_ms += std::chrono::duration<double, std::milli>(end - start).count();

//             start = std::chrono::steady_clock::now();
//             hnsw_mnru.delete_with_MNRU(lbl);   // <--- YOU WILL ADD THIS METHOD
//             end = std::chrono::steady_clock::now();
//             mnru_del_ms += std::chrono::duration<double, std::milli>(end - start).count();
//         }

//         // Running sums (averaged later)
//         double tomb_srch_sum = 0.0, lr_srch_sum = 0.0, mnru_srch_sum = 0.0;
//         double tomb_recall_sum = 0.0, lr_recall_sum = 0.0, mnru_recall_sum = 0.0;

//         // for recomputing ground truth
//         std::unordered_set<std::string> deleted_set;
//         for (int i = 0; i < target_del; ++i)
//             deleted_set.insert(labels[indices[i]]);

//         // -------- Loop over Q queries --------
//         for (int qi = 0; qi < Q; qi++) {

//             const auto &query = query_set[qi];

//             // ---- Compute Ground Truth for this query ----
//             std::vector<std::pair<float, std::string>> gt_all;
//             gt_all.reserve(N - target_del);

//             for (int i = 0; i < N; ++i) {
//                 if (deleted_set.count(labels[i])) continue;

//                 float distval = 0.0f;
//                 for (int d = 0; d < D; ++d) {
//                     float diff = query[d] - vectors[i][d];
//                     distval += diff * diff;
//                 }
//                 gt_all.emplace_back(std::sqrt(distval), labels[i]);
//             }

//             std::sort(gt_all.begin(), gt_all.end(),
//                     [](auto &a, auto &b) { return a.first < b.first; });

//             std::unordered_set<std::string> gt_set;
//             for (int i = 0; i < K && i < (int)gt_all.size(); ++i)
//                 gt_set.insert(gt_all[i].second);

//             auto compute_recall = [&](auto &results) {
//                 int hits = 0;
//                 for (int i = 0; i < (int)results.size() && i < K; i++) {
//                     if (gt_set.count(results[i].label)) hits++;
//                 }
//                 return (double)hits / K;
//             };

//             // ---- Perform search for this query ----
//             auto [tomb_res, tomb_ms] = measure_search_query(hnsw_tomb, query);
//             auto [lr_res,   lr_ms]   = measure_search_query(hnsw_lr,   query);
//             auto [mnru_res, mnru_ms] = measure_search_query(hnsw_mnru, query);

//             // accumulate times
//             tomb_srch_sum += tomb_ms;
//             lr_srch_sum   += lr_ms;
//             mnru_srch_sum += mnru_ms;

//             // accumulate recall
//             tomb_recall_sum += compute_recall(tomb_res);
//             lr_recall_sum   += compute_recall(lr_res);
//             mnru_recall_sum += compute_recall(mnru_res);
//         }

//         // Averages
//         double tomb_srch_avg = tomb_srch_sum / Q;
//         double lr_srch_avg   = lr_srch_sum   / Q;
//         double mnru_srch_avg = mnru_srch_sum / Q;

//         double tomb_recall_avg = tomb_recall_sum / Q;
//         double lr_recall_avg   = lr_recall_sum   / Q;
//         double mnru_recall_avg = mnru_recall_sum / Q;

//         // Store result row
//         table.push_back({
//             target_del,
//             tomb_del_ms, lr_del_ms, mnru_del_ms,
//             tomb_srch_avg, lr_srch_avg, mnru_srch_avg,
//             tomb_recall_avg, lr_recall_avg, mnru_recall_avg
//         });

//         // move to next deletion block
//         current_del = target_del;
//     }
//     // -----------------------------------------------------
//     //                PRINT SUMMARY TABLE
//     // -----------------------------------------------------
//     std::cout << "\n" << std::string(70, '=')
//               << "\nSEQUENTIAL DELETION SUMMARY"
//               << "\n" << std::string(70, '=') << "\n";

//     std::cout << std::setw(10) << "Del"
//               << std::setw(12) << "Tomb_D"
//               << std::setw(12) << "LR_D"
//               << std::setw(12) << "MNRU_D"
//               << std::setw(12) << "Tomb_S"
//               << std::setw(12) << "LR_S"
//               << std::setw(12) << "MNRU_S"
//               << std::setw(12) << "Tomb_R"
//               << std::setw(12) << "LR_R"
//               << std::setw(12) << "MNRU_R"
//               << "\n";

//     for (auto &r : table) {
//         std::cout << std::setw(10) << r.deletions
//                   << std::setw(12) << r.tomb_del
//                   << std::setw(12) << r.lr_del
//                   << std::setw(12) << r.mnru_del
//                   << std::setw(12) << r.tomb_srch
//                   << std::setw(12) << r.lr_srch
//                   << std::setw(12) << r.mnru_srch
//                   << std::setw(12) << r.tomb_recall
//                   << std::setw(12) << r.lr_recall
//                   << std::setw(12) << r.mnru_recall
//                   << "\n";
//     }

//     std::cout << "\n[Done]\n";
// }

void test_sequential_deletion_degradation() {
    std::cout << "\n" << std::string(70, '=')
              << "\nTEST: Sequential Deletion Degradation Analysis (SIFT)"
              << "\n" << std::string(70, '=') << "\n";

    // ------------------ LOAD SIFT DATASET ------------------
    auto base_vectors   = load_fvecs("./sift/sift_base.fvecs");
    auto query_vectors  = load_fvecs("./sift/sift_query.fvecs");
    auto gt_neighbors   = load_ivecs("./sift/sift_groundtruth.ivecs");

    if (base_vectors.empty() || query_vectors.empty() || gt_neighbors.empty()) {
        throw std::runtime_error("SIFT dataset files could not be loaded correctly.");
    }

    std::unordered_map<std::string, std:: string> updated_labels;

    int total_N = static_cast<int>(base_vectors.size());
    int D       = static_cast<int>(base_vectors[0].size());
    static const std::unordered_set<int> checkpoints = {10, 100, 1000, 5000, 10000};

    // Cap N for speed (you can change this)
    const int MAX_N = 100000;
    int N = std::min(total_N, MAX_N);

    // ANN parameters
    const int K  = 10;   // top-K neighbors to retrieve + eval recall@K
    const int EF = 400;   // efSearch

    // Number of queries to average over
    const int Q = query_vectors.size();    // you can increase to 100, etc.
    if (static_cast<int>(query_vectors.size()) < Q ||
        static_cast<int>(gt_neighbors.size())   < Q) {
        throw std::runtime_error("Not enough queries / ground-truth entries in SIFT files.");
    }

    // std::vector<std::vector<float>> query_vectors;

    // for (int i = 0; i < sift_queries.size() && query_vectors.size() < Q; i++) {
    //     bool ok = true;
    //     for (int k = 0; k < K; k++) {
    //         int nn_index = gt_neighbors[i][k];
    //         if (nn_index >= N) {
    //             ok = false;
    //             break;
    //         }
    //     }
    //     if (ok) query_vectors.push_back(sift_queries[i]);
    // }

    std::cout << "Loaded SIFT base: "   << total_N  << " vectors (using N=" << N << ")\n";
    std::cout << "Loaded SIFT queries: " << query_vectors.size()
              << ", dimension: " << D << "\n";
    std::cout << "Using Q=" << Q << " queries, K=" << K << ", EF=" << EF << "\n";

    // Copy first N base vectors
    std::vector<std::vector<float>> vectors(base_vectors.begin(),
                                            base_vectors.begin() + N);

    // Create labels "vec_0", "vec_1", ...
    std::vector<std::string> labels;
    labels.reserve(N);
    for (int i = 0; i < N; ++i) {
        labels.push_back("vec_" + std::to_string(i));
    }

    // Deletion milestones (cumulative)
    // std::vector<int> milestones = {10, 100, 1000, 10000, 50000, 100000, 200000, 500000};
    std::vector<int> milestones = {10, 100, 1000, 10000};
    // Sanity: ensure they don't exceed N
    milestones.erase(
        std::remove_if(milestones.begin(), milestones.end(),
                       [N](int m) { return m >= N; }),
        milestones.end()
    );

    if (milestones.empty()) {
        std::cout << "No valid deletion milestones (all >= N). Exiting test.\n";
        return;
    }

    // --------- Generate permutation of delete order ---------
    std::mt19937 rng(42);
    std::vector<int> indices(N);
    std::iota(indices.begin(), indices.end(), 0);
    std::shuffle(indices.begin(), indices.end(), rng);

    // --------- Initialize three indexes ---------
    // SimpleHNSW hnsw_tomb(D, 16, 0.5f, 400, 42);
    SimpleHNSW hnsw_lr(D, 16, 0.5f, 400, 42);
    // SimpleHNSW hnsw_mnru(D, 16, 0.5f, 400, 42);

    std::cout << "\n[Inserting " << N << " vectors into all indexes...]\n";
    for (int i = 0; i < N; ++i) {
        // hnsw_tomb.insert(labels[i], vectors[i]);
        hnsw_lr.insert(labels[i],   vectors[i]);
        // hnsw_mnru.insert(labels[i], vectors[i]);
        if (checkpoints.count(i)) {
            std::cout << "Inserted vector " << i << " into all indexes\n";
        }
    }
    // hnsw_lr.force_connectivity_level0()
    std::cout << "Debugging connectivity of Tombstone index...\n";
    hnsw_lr.debug_connectivity();
    std::cout << "Debugging degree of Tombstone index...\n";
    hnsw_lr.debug_degree_hist_level0();

    // -------- Results structure --------
    // struct ResultRow {
    //     int deletions;
    //     double tomb_del, lr_del, mnru_del;
    //     double tomb_srch, lr_srch, mnru_srch;
    //     double tomb_recall, lr_recall, mnru_recall;
    // };

    // struct ResultRow {
    //     int deletions;
    //     // double tomb_del;
    //     // double tomb_srch_del;
    //     // double tomb_recall_del;
    //     double tomb_update;
    //     double tomb_srch_update;
    //     double tomb_recall_update;
    // };

    struct ResultRow {
        int deletions;
        // double lr_del;
        // double lr_srch_del;
        // double lr_recall_del;
        double lr_update; 
        double lr_srch_update;
        double lr_recall_update;
    };

    // struct ResultRow {
    //     int deletions;
    //     // double mnru_del;
    //     // double mnru_srch;
    //     // double mnru_recall;
    //     double mnru_update;
    //     double mnru_srch_update;
    //     double mnru_recall_update;
    // };

    std::vector<ResultRow> table;
    int current_del = 0;

    // Helper: run search for a single query and measure time
    auto measure_search_query = [&](SimpleHNSW &idx,
                                    const std::vector<float> &q) {
        auto start = std::chrono::steady_clock::now();
        auto res   = idx.search(q, K, EF, /*track_path=*/false);
        auto end   = std::chrono::steady_clock::now();
        double ms  = std::chrono::duration<double, std::milli>(end - start).count();
        return std::make_pair(res, ms);
    };

    // -----------------------------------------------------
    //                MAIN LOOP OVER MILESTONES
    // -----------------------------------------------------
    for (int target_del : milestones) {
        int del_count = target_del - current_del;

        std::cout << "\n" << std::string(70, '-')
                  << "\nDeleting " << del_count << " nodes (cumulative: "
                  << target_del << ")\n"
                  << std::string(70, '-') << "\n";

        // double tomb_del_ms = 0.0;
        // double tomb_update_ms = 0.0;
        // double tomb_srch_ms_delete = 0.0;
        // double tomb_recall_delete = 0.0;

        // double lr_del_ms = 0.0;
        double lr_update_ms = 0.0;
        // double lr_srch_delete = 0.0;
        // double lr_recall_delete = 0.0;


        // double mnru_del_ms = 0.0;
        // double mnru_update_ms = 0.0;
        // double mnru_srch_ms_delete = 0.0;
        // double mnru_recall_delete = 0.0;

        // --------- Perform deletions on all 3 indexes ---------
        // for (int i = current_del; i < target_del; ++i) {
        //     const std::string &lbl = labels[indices[i]];

        //     // Tombstone
        //     // auto start = std::chrono::steady_clock::now();
        //     // hnsw_tomb.delete_node(lbl);
        //     // auto end   = std::chrono::steady_clock::now();
        //     // tomb_del_ms += std::chrono::duration<double, std::milli>(end - start).count();

        //     // Local Rewiring
        //     // auto start = std::chrono::steady_clock::now();
        //     // hnsw_lr.delete_with_local_rewiring(lbl);
        //     // auto end   = std::chrono::steady_clock::now();
        //     // lr_del_ms += std::chrono::duration<double, std::milli>(end - start).count();

        //     // MNRU
        //     // auto start = std::chrono::steady_clock::now();
        //     // hnsw_mnru.delete_with_MNRU(lbl, 1);
        //     // auto end   = std::chrono::steady_clock::now(); 
        //     // mnru_del_ms += std::chrono::duration<double, std::milli>(end - start).count();
        // }
        // std::cout << "Debugging connectivity of Tombstone index after deletion...\n, target_del: " << target_del << "\n";
        // hnsw_mnru.debug_connectivity();
        // std::cout << "Debugging degree of Tombstone index after deletion...\n, target_del: " << target_del << "\n";
        // hnsw_mnru.debug_degree_hist_level0();

        // for (int qi = 0; qi < Q; ++qi) {
        //     const auto &qvec        = query_vectors[qi];
        //     const auto &gt_row       = gt_neighbors[qi];  // sorted by true distance

        //     // ---- Build ground truth set for this query (top-K, clipped to N) ----
        //     std::unordered_set<int> gt_set;
        //     int taken = 0;
        //     for (int id : gt_row) {
        //         if (id < N) {
        //             gt_set.insert(id);
        //             if (++taken >= K) break;
        //         }
        //     }

        //     auto compute_recall = [&](const std::vector<SimpleHNSW::SearchResult> &results) {
        //         int hits = 0;
        //         for (int i = 0; i < K && i < (int)results.size(); ++i) {
        //             // label is "vec_<id>"
        //             std::string label = results[i].label;
        //             auto it = updated_labels.find(label);
        //             const std::string &canonical_lbl =
        //                     (it != updated_labels.end()) ? it->second : label;
        //             std::string label_id = canonical_lbl.substr(4);
        //             int id = stoi(label_id);
        //             if (gt_set.count(id)) ++hits;
        //         }
        //         return static_cast<double>(hits) / K;
        //     };

        //     // ---- Search all 3 indexes ----
        //     // auto [tomb_res, tomb_ms]  = measure_search_query(hnsw_tomb, qvec);
        //     // auto [lr_res,   lr_ms]    = measure_search_query(hnsw_lr,   qvec);
        //     auto [mnru_res, mnru_ms]  = measure_search_query(hnsw_mnru, qvec);

        //     // lr_srch_delete += lr_ms;
        //     // tomb_srch_ms_delete   += tomb_ms;
        //     mnru_srch_ms_delete += mnru_ms;

        //     // lr_recall_delete += compute_recall(lr_res);
        //     // tomb_recall_delete   += compute_recall(tomb_res);
        //     mnru_recall_delete += compute_recall(mnru_res);
        // }


        for (int i = current_del; i < target_del; ++i) {
            const std::string lbl = labels[indices[i]];
            // const std::string lbl_updated = lbl + "_updated";
            // updated_labels[lbl_updated] = lbl; 
            auto start = std::chrono::steady_clock::now();
            hnsw_lr.delete_with_local_rewiring(lbl, vectors[indices[i]]);
            // hnsw_tomb.insert(lbl_updated, vectors[indices[i]]);
            // hnsw_tomb.insert_replace_deleted(lbl, vectors[indices[i]]);
            // hnsw_mnru.delete_with_MNRU(lbl, vectors[indices[i]], 1.1);
            auto end   = std::chrono::steady_clock::now();
            // tomb_update_ms += std::chrono::duration<double, std::milli>(end - start).count();
            // mnru_update_ms += std::chrono::duration<double, std::milli>(end - start).count();
            lr_update_ms += std::chrono::duration<double, std::milli>(end - start).count();
        }
        std::cout << "Debugging connectivity of Tombstone index after update...\n, target_del: " << target_del << "\n";
        hnsw_lr.debug_connectivity();
        std::cout << "Debugging degree of Tombstone index after update...\n, target_del: " << target_del << "\n";
        hnsw_lr.debug_degree_hist_level0();

        // --------- EVALUATION OVER Q QUERIES ---------
        // double tomb_srch_sum_update   = 0.0;
        double lr_srch_update     = 0.0;
        // double mnru_srch_update   = 0.0;
        // double tomb_recall_sum_update = 0.0;
        double lr_recall_update   = 0.0;
        // double mnru_recall_update = 0.0;

        for (int qi = 0; qi < Q; ++qi) {
            const auto &qvec        = query_vectors[qi];
            const auto &gt_row       = gt_neighbors[qi];  // sorted by true distance

            // ---- Build ground truth set for this query (top-K, clipped to N) ----
            std::unordered_set<int> gt_set;
            int taken = 0;
            for (int id : gt_row) {
                if (id < N) {
                    gt_set.insert(id);
                    if (++taken >= K) break;
                }
            }

            auto compute_recall = [&](const std::vector<SimpleHNSW::SearchResult> &results) {
                int hits = 0;
                for (int i = 0; i < K && i < (int)results.size(); ++i) {
                    // label is "vec_<id>"
                    std::string label = results[i].label;
                    auto it = updated_labels.find(label);
                    const std::string &canonical_lbl =
                            (it != updated_labels.end()) ? it->second : label;
                    std::string label_id = canonical_lbl.substr(4);
                    int id = stoi(label_id);
                    if (gt_set.count(id)) ++hits;
                }
                return static_cast<double>(hits) / K;
            };

            // ---- Search all 3 indexes ----
            // auto [tomb_res, tomb_ms]  = measure_search_query(hnsw_tomb, qvec);
            auto [lr_res,   lr_ms]    = measure_search_query(hnsw_lr,   qvec);
            // auto [mnru_res, mnru_ms]  = measure_search_query(hnsw_mnru, qvec);

            lr_srch_update += lr_ms;
            // tomb_srch_sum_update   += tomb_ms;
            // mnru_srch_update += mnru_ms; 

            lr_recall_update += compute_recall(lr_res);
            // tomb_recall_sum_update   += compute_recall(tomb_res);
            // mnru_recall_update += compute_recall(mnru_res);
        }

        // Averages over Q queries
        // double tomb_srch_avg_delete  = tomb_srch_ms_delete   / Q;
        // double tomb_recall_avg_delete = tomb_recall_delete / Q;
        // double tomb_srch_avg_update = tomb_srch_sum_update / Q;
        // double tomb_recall_avg_update = tomb_recall_sum_update / Q;


        // double lr_srch_avg_delete  = lr_srch_delete   / Q;
        // double lr_recall_avg_delete = lr_recall_delete / Q;
        double lr_srch_avg_update = lr_srch_update / Q;
        double lr_recall_avg_update = lr_recall_update / Q;


        // double mnru_srch_avg_delete  = mnru_srch_ms_delete   / Q;
        // double mnru_recall_avg_delete = mnru_recall_delete / Q;
        // double mnru_srch_avg_update = mnru_srch_update / Q;
        // double mnru_recall_avg_update = mnru_recall_update / Q;


        // double lr_srch_avg     = lr_srch_sum     / Q; 
        // double mnru_srch_avg   = mnru_srch_sum   / Q; 
        // double tomb_recall_avg = tomb_recall_sum / Q;
        // double mnru_recall_avg   = mnru_recall_sum   / Q;
        // double lr_recall_avg = lr_recall_sum     / Q; 

        
        // table.push_back({
        //     target_del,
        //     mnru_del_ms,
        //     mnru_srch_avg,
        //     mnru_recall_avg,
        //     mnru_update_ms
        // });
 
        // table.push_back({
        //     target_del,
        //     // tomb_del_ms,
        //     // tomb_srch_avg_delete,
        //     // tomb_recall_avg_delete,
        //     tomb_update_ms,
        //     tomb_srch_avg_update,
        //     tomb_recall_avg_update
        // });

        table.push_back({
            target_del,
            // lr_del_ms,
            // lr_srch_avg_delete,
            // lr_recall_avg_delete,
            lr_update_ms,
            lr_srch_avg_update,
            lr_recall_avg_update
        });

        // table.push_back({
        //     target_del,
        //     // mnru_del_ms,
        //     // mnru_srch_avg_delete,
        //     // mnru_recall_avg_delete,
        //     mnru_update_ms,
        //     mnru_srch_avg_update,
        //     mnru_recall_avg_update
        // });
        
        current_del = target_del;
    }

    // -----------------------------------------------------
    //                PRINT SUMMARY TABLE
    // -----------------------------------------------------
    std::cout << "\n" << std::string(70, '=')
              << "\nSEQUENTIAL DELETION SUMMARY"
              << "\n" << std::string(70, '=') << "\n";

    std::cout << std::setw(20) << "Number of deletes"
            //   << std::setw(42) << "Total tombstone delete time in ms"
            //   << std::setw(20) << "Tombstone D"
            //   << std::setw(20) << "LR_RN D"
            //   << std::setw(20) << "MNRU D"
            //   << std::setw(42) << "Tombstone search average time in ms (post deletion)"
            //   << std::setw(20) << "Tombstone S_D"
            //   << std::setw(20) << "LR_RN S_D"
            //   << std::setw(20) << "MNRU S_D"
            //   << std::setw(42) << "Tombstone recall average (post deletion)"
            //   << std::setw(20) << "LR_RN R_D"
            //   << std::setw(20) << "Tombstone R_D"
            //   << std::setw(20) << "MNRU R_D"
            //   << std::setw(20) << "Tombstone U"
            //   << std::setw(20) << "Tombstone S_U"
            //   << std::setw(20) << "Tombstone R_U"
              << std::setw(20) << "LR_RN U"
              << std::setw(20) << "LR_RN S_U"
              << std::setw(20) << "LR_RN R_U"
            //   << std::setw(20) << "MNRU U"
            //   << std::setw(20) << "MNRU S_U"
            //   << std::setw(20) << "MNRU R_U"
              << "\n";

    for (const auto &r : table) {
        std::cout << std::setw(20) << r.deletions
                //   << std::setw(20) << r.tomb_del
                //   << std::setw(20) << r.mnru_del  
                //   << std::setw(20) << r.lr_del
                //   << std::setw(20) << r.tomb_srch_del
                //   << std::setw(20) << r.mnru_srch
                //   << std::setw(20) << r.lr_srch_del
                //   << std::setw(20) << r.tomb_recall_del
                //   << std::setw(20) << r.mnru_recall
                //   << std::setw(20) << r.lr_recall_del
                //   << std::setw(20) << r.tomb_update
                //   << std::setw(20) << r.mnru_update
                  << std::setw(20) << r.lr_update
                //   << std::setw(20) << r.tomb_srch_update
                //   << std::setw(20) << r.tomb_recall_update
                  << std::setw(20) << r.lr_srch_update
                  << std::setw(20) << r.lr_recall_update
                //   << std::setw(20) << r.mnru_srch_update
                //   << std::setw(20) << r.mnru_recall_update
                  << "\n";
    }

    std::cout << "\n[Done]\n";
}


