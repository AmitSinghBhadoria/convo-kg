from src.contracts import Utterance
from src.chunking import chunk_transcript, count_tokens

def U(text, spk="S0"):
    return Utterance(speaker=spk, text=text, start=0.0, end=1.0)

def words(text):                 # pure fake counter: 1 token per word (no tiktoken needed)
    return len(text.split())

def test_splits_at_next_turn_boundary_never_midturn():
    utts = [U("a a a a a"), U("b b b b b"), U("c c c c c")]      # 5 "tokens" each
    chunks = chunk_transcript(utts, target_tokens=8, count=words)
    assert [c.index for c in chunks] == [0, 1]
    assert [u.text for u in chunks[0].utterances] == ["a a a a a"]       # +5 -> 10>8 -> cut
    assert [u.text for u in chunks[1].utterances] == ["b b b b b", "c c c c c"]
    assert chunks[0].indices == [0] and chunks[1].indices == [1, 2]

def test_oversized_single_turn_kept_intact():
    utts = [U("x " * 20), U("y y")]                             # first turn = 20 > target
    chunks = chunk_transcript(utts, target_tokens=8, count=words)
    assert [u.text for u in chunks[0].utterances] == ["x " * 20]        # intact, alone
    assert chunks[1].utterances[0].text == "y y"

def test_context_is_previous_chunks_last_turn():
    utts = [U("a a a a a"), U("b b b b b"), U("c c c c c")]
    chunks = chunk_transcript(utts, target_tokens=8, count=words)
    assert chunks[0].context is None and chunks[0].context_index is None
    assert chunks[1].context.text == "a a a a a"
    assert chunks[1].context_index == 0

def test_empty_input():
    assert chunk_transcript([], target_tokens=8, count=words) == []

def test_count_tokens_real_tiktoken_is_positive():
    assert count_tokens("hello world") > 0
