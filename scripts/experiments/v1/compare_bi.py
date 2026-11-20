import json, struct, sys

def bits(x):
    return struct.pack(">d", float(x)).hex()

def cmp_list(a, b, label):
    a = a or []
    b = b or []
    n = min(len(a), len(b))
    first = None
    for i in range(n):
        if a[i] is None or b[i] is None:
            first = i
            break
        if bits(a[i]) != bits(b[i]):
            first = i
            break
    diffs = [abs(x-y) for x,y in zip(a,b) if x is not None and y is not None]
    print(f"{label}: len_a={len(a)} len_b={len(b)} "
          f"bitwise_equal={len(a)==len(b) and first is None} "
          f"first_mismatch={first} "
          f"max_abs_diff={max(diffs) if diffs else None}")
    return len(a)==len(b) and first is None

def main():
    pa, pb = sys.argv[1], sys.argv[2]
    a = json.load(open(pa, encoding="utf-8"))
    b = json.load(open(pb, encoding="utf-8"))
    print("A env:", json.dumps(a.get("env"), ensure_ascii=False))
    print("B env:", json.dumps(b.get("env"), ensure_ascii=False))
    ok = True
    ok &= (a["single"]["token_ids"] == b["single"]["token_ids"])
    print(f"single token_ids_equal={a['single']['token_ids'] == b['single']['token_ids']}")
    ok &= cmp_list(a["single"]["logprobs"], b["single"]["logprobs"], "single logprobs")
    ok &= (a["long_prompt"]["token_ids"] == b["long_prompt"]["token_ids"])
    print(f"long token_ids_equal={a['long_prompt']['token_ids'] == b['long_prompt']['token_ids']}")
    ok &= cmp_list(a["long_prompt"]["logprobs"], b["long_prompt"]["logprobs"], "long logprobs")
    print("OVERALL_BITWISE_EQUAL=", ok)

if __name__ == "__main__":
    main()
