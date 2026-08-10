"""Analiza results.jsonl -> tabele do RESULTS.md (stdout, markdown)."""
import json
from collections import Counter, defaultdict

PATH = "/Users/justi/dotfiles/llm-confidence/results.jsonl"

rows = [json.loads(l) for l in open(PATH, encoding="utf-8") if l.strip()]
rows = [r for r in rows if "error" not in r]
by = defaultdict(list)
for r in rows:
    by[r["cond"]].append(r)


def near(x, step=0.05, eps=1e-9):
    return abs(x / step - round(x / step)) < eps


def brier(pairs):  # pairs: (p_tak, label_bool)
    return sum((p - (1.0 if y else 0.0)) ** 2 for p, y in pairs) / len(pairs)


print(f"# rekordów: {len(rows)} (C1={len(by['C1'])}, C2={len(by['C2'])}, C3={len(by['C3'])})\n")

for cond in ("C1", "C2"):
    rs = [r for r in by[cond] if r["conf"] is not None and r["answer"]]
    confs = [r["conf"] for r in rs]
    cnt = Counter(confs)
    top = cnt.most_common()
    n = len(confs)
    top3 = sum(c for _, c in top[:3])
    mult05 = sum(1 for c in confs if near(c))
    print(f"## {cond}: n={n} poprawnie sparsowanych")
    print(f"- unikalnych wartości: {len(cnt)}")
    print(f"- top-3 wartości pokrywają: {100*top3/n:.0f}%")
    print(f"- wielokrotności 0.05: {100*mult05/n:.0f}%")
    print("- pełny histogram (wartość: liczność):")
    for v, c in sorted(cnt.items()):
        print(f"    {v:>5}: {'#'*c} {c}")
    # powtarzalność per zdanie
    per = defaultdict(list)
    for r in rs:
        per[r["id"]].append(r["conf"])
    same = sum(1 for v in per.values() if len(set(v)) == 1 and len(v) >= 2)
    multi = sum(1 for v in per.values() if len(v) >= 2)
    print(f"- zdania z identycznym conf we wszystkich powtórzeniach: {same}/{multi}")
    # per tier
    for tier in ("easy", "medium", "hard"):
        t = [r for r in rs if r["tier"] == tier]
        acc = sum(1 for r in t if (r["answer"] == "TAK") == r["label"]) / len(t)
        mc = sum(r["conf"] for r in t) / len(t)
        print(f"- {tier:6s}: śr. conf {mc:.3f} | accuracy odpowiedzi {100*acc:.0f}%")
    pairs = [((r["conf"] if r["answer"] == "TAK" else 1 - r["conf"]), r["label"]) for r in rs]
    print(f"- Brier: {brier(pairs):.4f}")
    # reliability: deklarowany conf vs trafność
    buckets = defaultdict(lambda: [0, 0])
    for r in rs:
        b = min(int(r["conf"] * 10) / 10, 0.9)
        buckets[b][0] += 1
        buckets[b][1] += 1 if (r["answer"] == "TAK") == r["label"] else 0
    print("- kalibracja (deklarowany conf -> realna trafność):")
    for b in sorted(buckets):
        tot, ok = buckets[b]
        print(f"    conf {b:.1f}-{b+0.1:.1f}: trafność {100*ok/tot:.0f}% (n={tot})")
    print()

# C3
rs = [r for r in by["C3"] if r["answer"]]
per = defaultdict(list)
meta = {}
for r in rs:
    per[r["id"]].append(r["answer"] == "TAK")
    meta[r["id"]] = (r["label"], r["tier"])
fracs = {i: sum(v) / len(v) for i, v in per.items() if len(v) >= 8}
cnt = Counter(round(f, 1) for f in fracs.values())
print(f"## C3 (frakcja głosów TAK z k=10): zdań={len(fracs)}")
print(f"- użytych wartości frakcji (możliwych 11): {len(set(fracs.values()))}")
print("- histogram frakcji:")
for v, c in sorted(cnt.items()):
    print(f"    {v:>4}: {'#'*c} {c}")
pairs = [(fracs[i], meta[i][0]) for i in fracs]
print(f"- Brier (frakcja jako p(TAK)): {brier(pairs):.4f}")
for tier in ("easy", "medium", "hard"):
    t = [i for i in fracs if meta[i][1] == tier]
    dec = [1 if (fracs[i] >= 0.5) == meta[i][0] else 0 for i in t]
    sp = [fracs[i] if meta[i][0] else 1 - fracs[i] for i in t]
    print(f"- {tier:6s}: śr. p(poprawna) {sum(sp)/len(sp):.3f} | trafność większości {100*sum(dec)/len(dec):.0f}%")
niepewne = [i for i in fracs if 0.2 <= fracs[i] <= 0.8]
print(f"- zdania z frakcją w (0.2..0.8) - realnie niepewne: {len(niepewne)}: {sorted(niepewne)}")
