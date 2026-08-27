"""Tier-4: pomiar kalibracji na zamrozonym statements_tier4.json.

Uzycie: python3 run_tier4.py <openai|ollama> [--smoke]
  openai -> endpoint z .endpoint.env (qwen3.8-27b), temperature 0.7
  ollama -> localhost:11434, model qwen-fast (think off; temp 0.7 z Modelfile)

Warunki: C1 (deklaracja confidence, 3 powtorzenia) + C3 (glos TAK/NIE, 10 sampli).
Wyjscie: results_tier4_<backend>.jsonl
"""
import json, re, sys, time, urllib.request

DIR = "/Users/justi/dotfiles/llm-confidence"
BACKEND = sys.argv[1]

if BACKEND == "openai":
    ENV = {}
    for line in open(f"{DIR}/.endpoint.env", encoding="utf-8"):
        line = line.strip()
        if line and "=" in line:
            k, v = line.split("=", 1)
            ENV[k] = v

P_C1 = ('Oceń, czy poniższe zdanie jest prawdziwe.\n'
        'Zdanie: "{s}"\n'
        'Odpowiedz DOKŁADNIE w dwóch liniach, bez niczego więcej:\n'
        'ODPOWIEDZ: TAK albo NIE\n'
        'CONFIDENCE: liczba z przedziału 0..1')
P_C3 = ('Czy poniższe zdanie jest prawdziwe?\n'
        'Zdanie: "{s}"\n'
        'Odpowiedz DOKŁADNIE jednym słowem: TAK albo NIE.')

RE_ANS = re.compile(r'\b(TAK|NIE)\b', re.IGNORECASE)
RE_NUM = re.compile(r'CONFIDENCE[^0-9]*([01](?:[.,]\d+)?)', re.IGNORECASE)
RE_NUM_ANY = re.compile(r'([01](?:[.,]\d+)?)')


def call(prompt, max_tokens):
    if BACKEND == "openai":
        body = json.dumps({"model": ENV["ENDPOINT_MODEL"],
                           "messages": [{"role": "user", "content": prompt}],
                           "temperature": 0.7, "max_tokens": max_tokens}).encode()
        req = urllib.request.Request(ENV["ENDPOINT_URL"] + "/chat/completions", data=body,
                                     headers={"Content-Type": "application/json",
                                              "Authorization": "Bearer " + ENV["ENDPOINT_KEY"]})
        with urllib.request.urlopen(req, timeout=300) as r:
            d = json.loads(r.read())
        return (d["choices"][0]["message"]["content"] or "").strip()
    body = json.dumps({"model": "qwen-fast",
                       "messages": [{"role": "user", "content": prompt}],
                       "think": False, "stream": False,
                       "options": {"num_predict": max_tokens}}).encode()
    req = urllib.request.Request("http://localhost:11434/api/chat", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.loads(r.read())["message"]["content"].strip()


def main():
    smoke = "--smoke" in sys.argv
    data = json.load(open(f"{DIR}/statements_tier4.json", encoding="utf-8"))
    stmts = data["statements"][:2] if smoke else data["statements"]

    plan = []
    for s in stmts:
        for rep in range(1 if smoke else 3):
            plan.append(("C1", s, rep))
        for rep in range(2 if smoke else 10):
            plan.append(("C3", s, rep))

    total = len(plan)
    out = open(f"{DIR}/results_tier4_{BACKEND}.jsonl", "a", encoding="utf-8")
    for i, (cond, s, rep) in enumerate(plan, 1):
        print(f"[t4-{BACKEND}] {i}/{total} {cond} {s['id']} rep{rep}", flush=True)
        prompt = (P_C1 if cond == "C1" else P_C3).format(s=s["text"])
        mtok = 700 if (BACKEND == "openai" and cond == "C1") else (400 if BACKEND == "openai" else (80 if cond == "C1" else 16))
        t0 = time.time()
        try:
            raw = call(prompt, mtok)
        except Exception as e:
            print(f"[t4-{BACKEND}] ERROR {cond} {s['id']} rep{rep}: {e}", flush=True)
            time.sleep(3)
            try:
                raw = call(prompt, mtok)
            except Exception as e2:
                out.write(json.dumps({"cond": cond, "id": s["id"], "rep": rep,
                                      "error": str(e2)}, ensure_ascii=False) + "\n")
                out.flush()
                continue
        if cond == "C1":
            m_ans = RE_ANS.search(raw)
            m_num = RE_NUM.search(raw) or RE_NUM_ANY.search(raw.split("\n")[-1])
            ans = m_ans.group(1).upper() if m_ans else None
            conf = float(m_num.group(1).replace(",", ".")) if m_num else None
            if conf is not None and not (0.0 <= conf <= 1.0):
                conf = None
        else:
            m = RE_ANS.search(raw)
            ans, conf = (m.group(1).upper() if m else None), None
        out.write(json.dumps({
            "cond": cond, "id": s["id"], "label": s["label"], "rep": rep,
            "answer": ans, "conf": conf, "raw": raw[:150],
            "sec": round(time.time() - t0, 2),
        }, ensure_ascii=False) + "\n")
        out.flush()
    out.close()
    print(f"[t4-{BACKEND}] DONE", flush=True)


if __name__ == "__main__":
    main()
