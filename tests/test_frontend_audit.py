from pathlib import Path

HTML = Path("frontend/index.html").read_text()


def test_no_known_mock_literals_survive():
    # Specific mock values from the original prototype must be gone.
    for lit in ["12 / 12", "11 / 12", "₹50 L", "₹1 Cr", "₹500 SIP", "6%", "41%",
                "sim:[0.98", "recall:[0.95", "qa:[0.93",
                "0.98,0.96,0.93", "SEBI mandated", "sixty percent PMS"]:
        assert lit not in HTML, f"mock literal survived: {lit!r}"


def test_no_hardcoded_snr_values():
    # The real SNR similarities must come from /api/experiment, never re-typed.
    for v in ["0.5479", "0.5932", "0.6253", "0.4525", "0.2873"]:
        assert v not in HTML, f"hardcoded SNR similarity in frontend: {v}"


def test_data_fields_are_wired_not_literal_populated():
    # Structural: the controller's data fields must be empty/null literals (filled by
    # fetch), not mock-populated arrays. Proves "no mock survives" beyond known strings.
    import re
    for field, empty in [("graphNodes", "[]"), ("graphEdges", "[]"), ("chart", "null")]:
        m = re.search(rf"\b{field}\s*=\s*([^;\n]+)", HTML)
        assert m, f"{field} declaration not found"
        assert m.group(1).strip() == empty, f"{field} is literal-populated, not wired: {m.group(1)[:40]}"
    # ask() and the data loads must go through fetch().
    assert "fetch('/api/ask'" in HTML
    assert "fetch('/api/graph'" in HTML
    assert "fetch('/api/experiment'" in HTML


def test_no_cdn_refs_offline():
    # Vendored offline — no live CDN that could fail on stage. (Google Fonts <link>
    # is replaced by vendored fonts in this task if still present.)
    assert "https://unpkg.com" not in HTML
    assert "https://fonts.googleapis.com" not in HTML and "https://fonts.gstatic.com" not in HTML
