from gateway.hashing import block_hashes

BC = 64  # block_chars


def test_deterministic():
    p = "X" * 200
    assert block_hashes(p, BC, 100) == block_hashes(p, BC, 100)


def test_full_blocks_only():
    # 3 full blocks + 30 leftover chars -> 3 hashes (partial block dropped)
    p = "a" * (BC * 3 + 30)
    assert len(block_hashes(p, BC, 100)) == 3


def test_cutoff_truncates():
    p = "a" * (BC * 10)
    assert len(block_hashes(p, BC, 4)) == 4


def test_chaining_is_prefix_exact():
    # shared first block, divergent second block
    a = ("X" * BC) + ("A" * BC)
    b = ("X" * BC) + ("B" * BC)
    ha, hb = block_hashes(a, BC, 100), block_hashes(b, BC, 100)
    assert ha[0] == hb[0]        # identical prefix block -> identical hash
    assert ha[1] != hb[1]        # divergent block -> different hash


def test_chaining_depends_on_history():
    # same second-block content, different first block -> different chained hash
    a = ("X" * BC) + ("Z" * BC)
    b = ("Y" * BC) + ("Z" * BC)
    ha, hb = block_hashes(a, BC, 100), block_hashes(b, BC, 100)
    assert ha[0] != hb[0]
    assert ha[1] != hb[1]        # chaining propagates the earlier difference


def test_empty_prompt():
    assert block_hashes("", BC, 100) == []
