from src.evaltools import normalize, similarity

def test_similarity_bounds_and_sense():
    assert similarity("PMS minimum is 50 lakh", "PMS minimum is 50 lakh") > 0.99
    assert similarity("totally different words here", "PMS minimum is 50 lakh") < 0.4
    assert normalize("  PMS,  Minimum!  ") == "pms minimum"
