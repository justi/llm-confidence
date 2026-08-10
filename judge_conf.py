"""C4: drugi przebieg LLM jako sedzia odpowiedzi z C1 (ten sam model).

C4a: sedzia DEKLARUJE 0..1, na ile oceniana odpowiedz jest poprawna (3 rep/zdanie)
C4b: sedzia GLOSUJE binarnie TAK/NIE czy odpowiedz poprawna (10 sampli/zdanie)

Oceniana odpowiedz = odpowiedz C1 rep0 danego zdania (z results.jsonl).
Wyjscie: results_judge.jsonl
"""
import json, re, sys, time, urllib.request

MODEL = "qwen-fast"
URL = "http://localhost:11434/api/chat"
DIR = "/Users/justi/dotfiles/llm-confidence"

P_C4A = ('Zdanie: "{s}"\n'
         'Ktoś odpowiedział, że to zdanie jest {ans_pl}.\n'
         'Oceń, na ile ta odpowiedź jest poprawna.\n'
         'Odpowiedz DOKŁADNIE jedną linią, bez niczego więcej:\n'
         'CONFIDENCE: liczba z przedziału 0..1')
P_C4B = ('Zdanie: "{s}"\n'
         'Ktoś odpowiedział, że to zdanie jest {ans_pl}.\n'
         'Czy ta odpowiedź jest poprawna? Odpowiedz DOKŁADNIE jednym słowem: TAK albo NIE.')

RE_ANS = re.compile(r'\b(TAK|NIE)\b', re.IGNORECASE)
RE_NUM = re.compile(r'([01](?:[.,]\d+)?)')


def call(prompt, num_predict):
    body = json.dumps({"model": MODEL, "messages": [{"role": "user", "content": prompt}],
                       "think": False, "stream": False,
                       "options": {"num_predict": num_predict}}).encode()
    req = urllib.request.Request(URL, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=180) as r:
        return json.loads(r.read())["message"]["content"].strip()


def main():
    stmts = {}
    data = json.load(open(f"{DIR}/statements.json", encoding="utf-8"))
    for tier in ("easy", "medium", "hard"):
        for s in data[tier]:
            stmts[s["id"]] = {**s, "tier": tier}
    judged = {}  # id -> odpowiedz C1 rep0
    for line in open(f"{DIR}/results.jsonl", encoding="utf-8"):
        r = json.loads(line)
        if r.get("cond") == "C1" and r.get("rep") == 0 and r.get("answer"):
            judged[r["id"]] = r["answer"]

    plan = []
    for sid, s in stmts.items():
        for rep in range(3):
            plan.append(("C4a", s, judged[sid], rep))
        for rep in range(10):
            plan.append(("C4b", s, judged[sid], rep))

    total = len(plan)
    out = open(f"{DIR}/results_judge.jsonl", "a", encoding="utf-8")
    for i, (cond, s, ans, rep) in enumerate(plan, 1):
        print(f"[judge] {i}/{total} {cond} {s['id']} rep{rep}", flush=True)
        ans_pl = "PRAWDZIWE" if ans == "TAK" else "FAŁSZYWE"
        prompt = (P_C4A if cond == "C4a" else P_C4B).format(s=s["text"], ans_pl=ans_pl)
        npred = 40 if cond == "C4a" else 16
        t0 = time.time()
        try:
            raw = call(prompt, npred)
        except Exception as e:
            print(f"[judge] ERROR {cond} {s['id']} rep{rep}: {e}", flush=True)
            time.sleep(2)
            raw = call(prompt, npred)
        if cond == "C4a":
            m = RE_NUM.search(raw)
            conf = float(m.group(1).replace(",", ".")) if m else None
            if conf is not None and not (0.0 <= conf <= 1.0):
                conf = None
            verdict = None
        else:
            m = RE_ANS.search(raw)
            verdict = m.group(1).upper() if m else None
            conf = None
        out.write(json.dumps({
            "cond": cond, "id": s["id"], "tier": s["tier"], "label": s["label"],
            "judged_answer": ans, "rep": rep, "verdict": verdict, "conf": conf,
            "raw": raw[:150], "sec": round(time.time() - t0, 2),
        }, ensure_ascii=False) + "\n")
        out.flush()
    out.close()
    print("[judge] DONE", flush=True)


if __name__ == "__main__":
    main()
