"""Ten sam pomiar co run_conf.py (C1/C2/C3), ale przez endpoint OpenAI-compatible.

Konfiguracja endpointu w gitignorowanym pliku .endpoint.env (URL, KEY, MODEL) -
sekrety nie wchodza do repo. Sampling: temperature 0.7 (jak w pomiarze bazowym).
Model rozumujacy: reasoning wraca osobnym polem i nie zanieczyszcza content.

Wyjscie: results_qwen38.jsonl
"""
import json, re, sys, time, urllib.request

DIR = "/Users/justi/dotfiles/llm-confidence"

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


def call(prompt, max_tokens):
    body = json.dumps({
        "model": ENV["ENDPOINT_MODEL"],
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7,
        "max_tokens": max_tokens,
    }).encode()
    req = urllib.request.Request(
        ENV["ENDPOINT_URL"] + "/chat/completions", data=body,
        headers={"Content-Type": "application/json",
                 "Authorization": "Bearer " + ENV["ENDPOINT_KEY"]})
    with urllib.request.urlopen(req, timeout=300) as r:
        d = json.loads(r.read())
    ch = d["choices"][0]
    return (ch["message"]["content"] or "").strip(), ch.get("finish_reason"), d.get("usage", {})


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
    data = json.load(open(f"{DIR}/statements.json", encoding="utf-8"))
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
    out = open(f"{DIR}/results_qwen38.jsonl", "a", encoding="utf-8")
    for i, (cond, s, rep) in enumerate(plan, 1):
        print(f"[q38] {i}/{total} {cond} {s['id']} rep{rep}", flush=True)
        prompt = {"C1": P_C1, "C2": P_C2, "C3": P_C3}[cond].format(s=s["text"])
        mtok = 400 if cond == "C3" else 700
        t0 = time.time()
        try:
            raw, fin, usage = call(prompt, mtok)
        except Exception as e:
            print(f"[q38] ERROR {cond} {s['id']} rep{rep}: {e}", flush=True)
            time.sleep(3)
            try:
                raw, fin, usage = call(prompt, mtok)
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
            "finish": fin, "completion_tokens": usage.get("completion_tokens"),
            "sec": round(time.time() - t0, 2),
        }, ensure_ascii=False) + "\n")
        out.flush()
    out.close()
    print("[q38] DONE", flush=True)


if __name__ == "__main__":
    main()
