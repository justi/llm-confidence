"""Pomiar rozkładu 'confidence 0..1' zwracanego przez LLM.

Warunki:
  C1  ODPOWIEDŹ TAK/NIE + CONFIDENCE 0..1 (typowy prompt), 3 powtórzenia/zdanie
  C2  jak C1, ale z jawną prośbą o dokładność 0.01, 3 powtórzenia/zdanie
  C3  sama binarna odpowiedź TAK/NIE, 10 sampli/zdanie -> frakcja głosów jako confidence

Wyjście: results.jsonl (append, 1 rekord = 1 wywołanie).
"""
import json, re, sys, time, urllib.request

MODEL = "qwen-fast"
URL = "http://localhost:11434/api/chat"
STMTS_PATH = "/Users/justi/dotfiles/llm-confidence/statements.json"
OUT_PATH = "/Users/justi/dotfiles/llm-confidence/results.jsonl"

P_C1 = ('Oceń, czy poniższe zdanie jest prawdziwe.\n'
        'Zdanie: "{s}"\n'
        'Odpowiedz DOKŁADNIE w dwóch liniach, bez niczego więcej:\n'
        'ODPOWIEDZ: TAK albo NIE\n'
        'CONFIDENCE: liczba z przedziału 0..1')
P_C2 = ('Oceń, czy poniższe zdanie jest prawdziwe.\n'
        'Zdanie: "{s}"\n'
        'Odpowiedz DOKŁADNIE w dwóch liniach, bez niczego więcej:\n'
        'ODPOWIEDZ: TAK albo NIE\n'
        'CONFIDENCE: liczba z przedziału 0..1 z dokładnością do 0.01 (np. 0.73)')
P_C3 = ('Czy poniższe zdanie jest prawdziwe?\n'
        'Zdanie: "{s}"\n'
        'Odpowiedz DOKŁADNIE jednym słowem: TAK albo NIE.')

RE_ANS = re.compile(r'\b(TAK|NIE)\b', re.IGNORECASE)
RE_NUM = re.compile(r'CONFIDENCE[^0-9]*([01](?:[.,]\d+)?)', re.IGNORECASE)
RE_NUM_ANY = re.compile(r'([01](?:[.,]\d+)?)')


def call(prompt, num_predict):
    body = json.dumps({
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "think": False,
        "stream": False,
        "options": {"num_predict": num_predict},
    }).encode()
    req = urllib.request.Request(URL, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=180) as r:
        return json.loads(r.read())["message"]["content"].strip()


def parse_c12(raw):
    m_ans = RE_ANS.search(raw)
    m_num = RE_NUM.search(raw) or RE_NUM_ANY.search(raw.split("\n")[-1])
    ans = m_ans.group(1).upper() if m_ans else None
    conf = float(m_num.group(1).replace(",", ".")) if m_num else None
    if conf is not None and not (0.0 <= conf <= 1.0):
        conf = None
    return ans, conf


def main():
    smoke = "--smoke" in sys.argv
    stmts = []
    data = json.load(open(STMTS_PATH, encoding="utf-8"))
    for tier in ("easy", "medium", "hard"):
        for s in data[tier]:
            stmts.append({**s, "tier": tier})
    if smoke:
        stmts = [stmts[0], stmts[30]]

    plan = []
    for s in stmts:
        for rep in range(1 if smoke else 3):
            plan.append(("C1", s, rep))
        for rep in range(1 if smoke else 3):
            plan.append(("C2", s, rep))
        for rep in range(2 if smoke else 10):
            plan.append(("C3", s, rep))

    total = len(plan)
    out = open(OUT_PATH, "a", encoding="utf-8")
    for i, (cond, s, rep) in enumerate(plan, 1):
        print(f"[conf] {i}/{total} {cond} {s['id']} rep{rep}", flush=True)
        prompt = {"C1": P_C1, "C2": P_C2, "C3": P_C3}[cond].format(s=s["text"])
        npred = 16 if cond == "C3" else 80
        t0 = time.time()
        try:
            raw = call(prompt, npred)
        except Exception as e:
            print(f"[conf] ERROR {cond} {s['id']} rep{rep}: {e}", flush=True)
            time.sleep(2)
            try:
                raw = call(prompt, npred)
            except Exception as e2:
                out.write(json.dumps({"cond": cond, "id": s["id"], "rep": rep,
                                      "error": str(e2)}, ensure_ascii=False) + "\n")
                out.flush()
                continue
        ans, conf = (parse_c12(raw) if cond in ("C1", "C2")
                     else (RE_ANS.search(raw).group(1).upper() if RE_ANS.search(raw) else None, None))
        out.write(json.dumps({
            "cond": cond, "id": s["id"], "tier": s["tier"], "label": s["label"],
            "rep": rep, "answer": ans, "conf": conf, "raw": raw[:200],
            "sec": round(time.time() - t0, 2),
        }, ensure_ascii=False) + "\n")
        out.flush()
    out.close()
    print("[conf] DONE", flush=True)


if __name__ == "__main__":
    main()
