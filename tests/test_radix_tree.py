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


def test_front_eviction_breaks_contiguity():
    t = RadixTree(backend_cache_blocks=2)   # cap forces eviction of the front node
    t.insert([1, 2, 3], "b1")
    assert t.held_blocks("b1") == 2          # bounded to cap
    # front block evicted -> KV prefix is cold -> no contiguous match
    assert t.match([1, 2, 3]) == {}


def test_remove_backend_membership_eviction():
    t = RadixTree(100)
    t.insert([1, 2, 3], "b1")
    t.insert([1, 2, 3], "b2")
    t.remove_backend("b1")
    assert t.match([1, 2, 3]) == {"b2": 3}
