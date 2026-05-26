//! Path-compressed (radix/PATRICIA) prefix tree mapping prefixes -> backends.
//!
//! Arena-based (nodes in a `Vec`, referenced by index) to avoid shared-ownership
//! borrow-checker pain while keeping the Python semantics: longest CONTIGUOUS
//! prefix match, holders recorded along a backend's path, edge-splitting on
//! divergent inserts, and a per-backend LRU bounded by `backend_cache_blocks`.

use std::collections::{HashMap, HashSet};

struct Node {
    edge: Vec<u64>,                 // run of block-hashes into this node
    children: HashMap<u64, usize>,  // first hash of child edge -> node index
    holders: HashMap<String, u64>,  // backend -> last_seen clock
}

impl Node {
    fn new(edge: Vec<u64>) -> Self {
        Node { edge, children: HashMap::new(), holders: HashMap::new() }
    }
}

pub struct RadixTree {
    nodes: Vec<Node>,                          // index 0 = root
    cap: usize,                                // per-backend block budget
    clock: u64,
    lru: HashMap<String, HashMap<usize, u64>>, // backend -> {node_idx -> last_seen}
}

impl RadixTree {
    pub fn new(backend_cache_blocks: usize) -> Self {
        RadixTree {
            nodes: vec![Node::new(Vec::new())],
            cap: backend_cache_blocks,
            clock: 0,
            lru: HashMap::new(),
        }
    }

    /// backend -> longest contiguous-from-front matched prefix length (blocks).
    pub fn match_prefix(&self, hashes: &[u64]) -> HashMap<String, usize> {
        let mut best: HashMap<String, usize> = HashMap::new();
        let mut node = 0usize;
        let mut alive: Option<HashSet<String>> = None;
        let mut depth = 0usize;
        let mut i = 0usize;
        let n = hashes.len();
        while i < n {
            let child = match self.nodes[node].children.get(&hashes[i]) {
                Some(&c) => c,
                None => break,
            };
            let e = &self.nodes[child].edge;
            let le = e.len();
            let mut j = 0usize;
            while j < le && i + j < n && e[j] == hashes[i + j] {
                j += 1;
            }
            let holders: HashSet<String> = self.nodes[child].holders.keys().cloned().collect();
            alive = Some(match alive {
                None => holders,
                Some(a) => a.intersection(&holders).cloned().collect(),
            });
            let cur = alive.as_ref().unwrap();
            if cur.is_empty() {
                break;
            }
            depth += j;
            for b in cur {
                best.insert(b.clone(), depth);
            }
            if j < le {
                break; // diverged inside the edge
            }
            i += j;
            node = child;
        }
        best
    }

    pub fn insert(&mut self, hashes: &[u64], backend: &str) {
        if hashes.is_empty() {
            return;
        }
        self.clock += 1;
        let mut node = 0usize;
        let mut i = 0usize;
        let n = hashes.len();
        while i < n {
            let first = hashes[i];
            match self.nodes[node].children.get(&first).copied() {
                None => {
                    let leaf = self.new_node(hashes[i..].to_vec());
                    self.nodes[node].children.insert(first, leaf);
                    self.hold(backend, leaf);
                    break;
                }
                Some(child) => {
                    let le = self.nodes[child].edge.len();
                    let mut j = 0usize;
                    while j < le && i + j < n && self.nodes[child].edge[j] == hashes[i + j] {
                        j += 1;
                    }
                    if j == le {
                        self.hold(backend, child);
                        i += j;
                        node = child;
                        continue;
                    }
                    // split `child` at offset j
                    let head_edge: Vec<u64> = self.nodes[child].edge[..j].to_vec();
                    let tail_edge: Vec<u64> = self.nodes[child].edge[j..].to_vec();
                    let holders_copy = self.nodes[child].holders.clone();
                    let split = self.new_node(head_edge);
                    self.nodes[child].edge = tail_edge;
                    let tail_first = self.nodes[child].edge[0];
                    self.nodes[split].children.insert(tail_first, child);
                    self.nodes[split].holders = holders_copy.clone();
                    self.nodes[node].children.insert(first, split);
                    for b in holders_copy.keys().cloned().collect::<Vec<_>>() {
                        self.track(&b, split);
                    }
                    self.hold(backend, split);
                    if i + j < n {
                        let leaf = self.new_node(hashes[i + j..].to_vec());
                        self.nodes[split].children.insert(hashes[i + j], leaf);
                        self.hold(backend, leaf);
                    }
                    break;
                }
            }
        }
        self.evict(backend);
    }

    pub fn remove_backend(&mut self, backend: &str) {
        if let Some(map) = self.lru.remove(backend) {
            for nid in map.keys() {
                self.nodes[*nid].holders.remove(backend);
            }
        }
    }

    pub fn held_blocks(&self, backend: &str) -> usize {
        self.lru
            .get(backend)
            .map(|m| m.keys().map(|&nid| self.nodes[nid].edge.len()).sum())
            .unwrap_or(0)
    }

    // --- internals ---
    fn new_node(&mut self, edge: Vec<u64>) -> usize {
        self.nodes.push(Node::new(edge));
        self.nodes.len() - 1
    }

    fn hold(&mut self, backend: &str, node: usize) {
        self.nodes[node].holders.insert(backend.to_string(), self.clock);
        self.track(backend, node);
    }

    fn track(&mut self, backend: &str, node: usize) {
        self.lru.entry(backend.to_string()).or_default().insert(node, self.clock);
    }

    fn evict(&mut self, backend: &str) {
        loop {
            let total: usize = match self.lru.get(backend) {
                Some(m) => m.keys().map(|&nid| self.nodes[nid].edge.len()).sum(),
                None => return,
            };
            let map = self.lru.get(backend).unwrap();
            if total <= self.cap || map.is_empty() {
                break;
            }
            // evict the least-recently-seen held node
            let victim = map.iter().min_by_key(|(_, &ls)| ls).map(|(&nid, _)| nid).unwrap();
            self.lru.get_mut(backend).unwrap().remove(&victim);
            self.nodes[victim].holders.remove(backend);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn m(t: &RadixTree, h: &[u64]) -> HashMap<String, usize> {
        t.match_prefix(h)
    }

    #[test]
    fn insert_then_full_match() {
        let mut t = RadixTree::new(100);
        t.insert(&[1, 2, 3], "b1");
        assert_eq!(m(&t, &[1, 2, 3]).get("b1"), Some(&3));
    }

    #[test]
    fn longest_prefix_match() {
        let mut t = RadixTree::new(100);
        t.insert(&[1, 2, 3], "b1");
        assert_eq!(m(&t, &[1, 2, 9]).get("b1"), Some(&2));
    }

    #[test]
    fn no_match_different_first_block() {
        let mut t = RadixTree::new(100);
        t.insert(&[1, 2, 3], "b1");
        assert!(m(&t, &[9, 9, 9]).is_empty());
    }

    #[test]
    fn branching_distinguishes_backends() {
        let mut t = RadixTree::new(100);
        t.insert(&[1, 2, 3], "b1");
        t.insert(&[1, 2, 4], "b2");
        let r = m(&t, &[1, 2, 3]);
        assert_eq!(r.get("b1"), Some(&3));
        assert_eq!(r.get("b2"), Some(&2));
    }

    #[test]
    fn edge_split_on_divergent_insert() {
        let mut t = RadixTree::new(100);
        t.insert(&[1, 2, 3, 4], "b1");
        t.insert(&[1, 2, 9], "b2");
        let r1 = m(&t, &[1, 2, 3, 4]);
        assert_eq!(r1.get("b1"), Some(&4));
        assert_eq!(r1.get("b2"), Some(&2));
        let r2 = m(&t, &[1, 2, 9]);
        assert_eq!(r2.get("b2"), Some(&3));
        assert_eq!(r2.get("b1"), Some(&2));
    }

    #[test]
    fn oversized_prefix_is_evicted() {
        let mut t = RadixTree::new(2); // single 3-block node exceeds cap
        t.insert(&[1, 2, 3], "b1");
        assert_eq!(t.held_blocks("b1"), 0);
        assert!(m(&t, &[1, 2, 3]).is_empty());
    }

    #[test]
    fn lru_evicts_least_recently_used() {
        let mut t = RadixTree::new(4);
        t.insert(&[1, 2], "b1");
        t.insert(&[3, 4], "b1");
        t.insert(&[1, 2], "b1"); // touch
        t.insert(&[5, 6], "b1"); // evict [3,4]
        assert_eq!(m(&t, &[1, 2]).get("b1"), Some(&2));
        assert_eq!(m(&t, &[5, 6]).get("b1"), Some(&2));
        assert!(m(&t, &[3, 4]).is_empty());
    }

    #[test]
    fn remove_backend_membership_eviction() {
        let mut t = RadixTree::new(100);
        t.insert(&[1, 2, 3], "b1");
        t.insert(&[1, 2, 3], "b2");
        t.remove_backend("b1");
        let r = m(&t, &[1, 2, 3]);
        assert_eq!(r.get("b2"), Some(&3));
        assert_eq!(r.get("b1"), None);
    }
}
