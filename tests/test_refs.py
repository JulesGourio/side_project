"""Document keys and cited-code detection agree across the three notebooks and the document_recall scorer."""
import pathlib, tempfile
REPO = pathlib.Path(__file__).resolve().parents[1]
import re
cases = {"PRLAT549.FR": "PRLAT549", "PRLAT549_GB": "PRLAT549", "prlat-549 fr": "PRLAT549", "IN_APO_006": "INAPO6",
         "IN_APO_0006": "INAPO6", "P0043NF_BG": "P43NF", "Q0062MI": "Q62MI", "H0049MR": "H49MR", "QP-1457.pdf": "QP1457",
         "REF: PRLAT549_FR": None, "MR-1226 EN": "MR1226", "Q0196QP_FR": "Q196QP"}
bold = "Voir **PRLAT549.FR**, **Q0062MI**, **v1.2**, **Note**, **QP-1457** et https://x/i.aspx?ref=Q0196QP_FR#:~:text=a"
for f in ["Score_Production_QA.py", "Evaluate_Knowledge_Assistant.py", "Build_Golden_Dataset.py"]:
    src = (REPO / f).read_text()
    ns = {"re": re, "LANG_SUFFIXES": eval(re.search(r"LANG_SUFFIXES = (\[.*?\])", src).group(1))}
    block = src[src.index("_EXT = re.compile"):]
    block = block[:block.index("def code_like")] + block[block.index("def code_like"):].split("\n\n\n")[0]
    exec(block, ns)
    got = {k: ns["base_ref"](k) for k in cases}
    bad = {k: (got[k], v) for k, v in cases.items() if v and got[k] != v}
    print(f, "base_ref mismatches:", bad or "none", "| code_like:", sorted(ns["code_like"](bold)))
# self-contained document_recall key
src = (REPO / "Evaluate_Knowledge_Assistant.py").read_text()
k = re.search(r"    def key\(code\):\n(.*?)\n\n", src, re.S).group(0)
ns = {"re": re}; exec("import re\n" + "\n".join(l[4:] for l in k.splitlines()), ns)
bad = {c: (ns["key"](c), v) for c, v in {**cases, "REF: PRLAT549_FR": "PRLAT549"}.items() if ns["key"](c) != v}
print("document_recall key mismatches:", bad or "none")
