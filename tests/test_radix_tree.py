from gateway.radix_tree import RadixTree


def test_insert_then_full_match():
    t = RadixTree(backend_cache_blocks=100)
    t.insert([1, 2, 3], "b1")
    assert t.match([1, 2, 3]) == {"b1": 3}


def test_longest_prefix_match():
    t = RadixTree(100)
    t.insert([1, 2, 3], "b1")
    # query shares 2 blocks then diverges
    assert t.match([1, 2, 9]) == {"b1": 2}


def test_no_match_on_different_first_block():
    t = RadixTree(100)
    t.insert([1, 2, 3], "b1")
    assert t.match([9, 9, 9]) == {}


def test_branching_distinguishes_backends():
    t = RadixTree(100)
    t.insert([1, 2, 3], "b1")
    t.insert([1, 2, 4], "b2")          # shares first 2 blocks with b1
    m = t.match([1, 2, 3])
    assert m["b1"] == 3                  # b1 holds the full path
    assert m["b2"] == 2                  # b2 only shares the first 2 blocks


def test_multiple_backends_same_prefix():
    t = RadixTree(100)
    t.insert([1, 2], "b1")
    t.insert([1, 2], "b2")
    assert t.match([1, 2]) == {"b1": 2, "b2": 2}


def test_oversized_prefix_is_evicted():
    t = RadixTree(backend_cache_blocks=2)    # a single 3-block node exceeds the cap
    t.insert([1, 2, 3], "b1")
    assert t.held_blocks("b1") == 0          # node-granular eviction drops the whole segment
    assert t.match([1, 2, 3]) == {}          # prefix is now cold


def test_lru_evicts_least_recently_used_prefix():
    t = RadixTree(backend_cache_blocks=4)    # holds two 2-block prefixes
    t.insert([1, 2], "b1")
    t.insert([3, 4], "b1")                    # at cap (4 blocks)
    t.insert([1, 2], "b1")                    # touch [1,2] -> most recently used
    t.insert([5, 6], "b1")                    # over cap -> evict LRU ([3,4])
    assert t.match([1, 2]) == {"b1": 2}
    assert t.match([5, 6]) == {"b1": 2}
    assert t.match([3, 4]) == {}             # evicted


def test_edge_split_on_divergent_insert():
    t = RadixTree(100)
    t.insert([1, 2, 3, 4], "b1")
    t.insert([1, 2, 9], "b2")                # diverges after [1,2] -> splits the edge
    assert t.match([1, 2, 3, 4]) == {"b1": 4, "b2": 2}
    assert t.match([1, 2, 9]) == {"b2": 3, "b1": 2}
    assert t.match([1, 2]) == {"b1": 2, "b2": 2}


def test_remove_backend_membership_eviction():
    t = RadixTree(100)
    t.insert([1, 2, 3], "b1")
    t.insert([1, 2, 3], "b2")
    t.remove_backend("b1")
    assert t.match([1, 2, 3]) == {"b2": 3}
