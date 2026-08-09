"""Tests for textile-nlp-dict — standalone (no pytest dependency)."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

passed = 0
failed = 0


def check(name, fn):
    global passed, failed
    try:
        fn()
        passed += 1
        print(f"  OK  {name}")
    except Exception as e:
        failed += 1
        print(f"  FAIL  {name}: {e}")


def main():
    global passed, failed
    print("=" * 50)
    print("Testing textile-nlp-dict")
    print("=" * 50)

    # ─── Lexicon ───
    from textile_dict.core import Lexicon

    def t_load():
        lex = Lexicon()
        assert lex.version == "v2.8"
        assert lex.total_terms == 2545
        assert len(lex.layers) == 7
    check("load Lexicon", t_load)

    lex = Lexicon()

    def t_search():
        results = lex.search("纺纱")
        assert len(results) > 0
        assert all("term" in r for r in results)
    check("search", t_search)

    def t_get_layer():
        layer = lex.get_layer("layer_3_textile_chain")
        assert isinstance(layer, dict)
    check("get_layer", t_get_layer)

    def t_invalid_layer():
        try:
            lex.get_layer("nonexistent")
            raise AssertionError("expected KeyError")
        except KeyError:
            pass
    check("invalid layer raises KeyError", t_invalid_layer)

    def t_all_terms():
        terms = list(lex.all_terms())
        assert len(terms) > 1000
        assert all(isinstance(t, str) for t in terms)
    check("all_terms", t_all_terms)

    def t_get_term():
        entries = lex.get_term("涡流纺")
        assert len(entries) > 0
        assert entries[0]["layer"] == "layer_3_textile_chain"
    check("get_term", t_get_term)

    def t_category():
        terms = lex.terms_by_category("layer_3_textile_chain", "3_织造")
        assert isinstance(terms, list)
    check("terms_by_category", t_category)

    # ─── jieba loading ───
    from textile_dict import load_jieba_dict, get_version

    def t_version():
        assert get_version() == "v2.8"
    check("get_version", t_version)

    def t_jieba():
        assert load_jieba_dict() is True
    check("load_jieba_dict", t_jieba)

    # ─── domains ───
    from textile_dict.domains import industry_chain, policy_all, green_compliance

    def t_industry_chain():
        terms = industry_chain()
        assert len(terms) > 0
    check("domains.industry_chain", t_industry_chain)

    def t_policy():
        terms = policy_all()
        assert len(terms) > 0
    check("domains.policy_all", t_policy)

    def t_green():
        terms = green_compliance()
        assert len(terms) > 0
    check("domains.green_compliance", t_green)

    # ─── Summary ───
    print("-" * 50)
    total = passed + failed
    print(f"Results: {passed}/{total} passed, {failed} failed")
    if failed:
        print("SOME TESTS FAILED")
        sys.exit(1)
    else:
        print("ALL TESTS PASSED")
    print("=" * 50)


if __name__ == "__main__":
    main()
