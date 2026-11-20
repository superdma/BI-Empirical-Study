import os, json, struct, urllib.request, time
BASE=os.environ.get("BASE","http://127.0.0.1:8000"); MODEL=os.environ.get("MODEL_NAME",os.path.expanduser("~/models/Qwen3-30B-A3B"))
OUT=os.environ.get("OUT",os.path.expanduser("~/bi_long_diag.json"))
def post(p,pl,t=900):
    r=urllib.request.Request(BASE+p,data=json.dumps(pl).encode(),headers={"Content-Type":"application/json"})
    with urllib.request.urlopen(r,timeout=t) as x: return json.loads(x.read().decode())
def tok(s): return list(post("/tokenize",{"model":MODEL,"prompt":s})["tokens"])
def gen(ps,salt,mt=8):
    pl={"model":MODEL,"prompt":ps,"max_tokens":mt,"temperature":0.0,"logprobs":5,"seed":0,"return_token_ids":True,"cache_salt":salt}
    t0=time.time(); r=post("/v1/completions",pl); return [c for c in r["choices"] if c["index"]==0][0], time.time()-t0
def bits(x): return struct.pack(">d",float(x)).hex()
BASE_TXT="The following is a long passage. "+"The quick brown fox jumps over the lazy dog. "*1500
END="\nOnce upon a time, in a small village, there was a"
base_toks=tok(BASE_TXT)
res={"cases":{},"cmp":{}}
def mk(L):
    head=base_toks[:max(8,L-30)]
    tail=tok(END)
    return head+tail
def run(n,ps,salt):
    c,dt=gen(ps,salt); res["cases"][n]={"salt":salt,"n":len(ps),"sec":dt,"ids":c.get("token_ids"),"lp":c["logprobs"]["token_logprobs"],"top":c["logprobs"]["top_logprobs"]}
    print(n,"n_tokens",len(ps),"first_lp",c["logprobs"]["token_logprobs"][0])
def cmp(a,b):
    x,y=res["cases"][a],res["cases"][b]
    d={"ids":x["ids"]==y["ids"],"bitwise":all(bits(p)==bits(q) for p,q in zip(x["lp"],y["lp"])),
       "max":max((abs(p-q) for p,q in zip(x["lp"],y["lp"])),default=None),
       "first":next((i for i,(p,q) in enumerate(zip(x["lp"],y["lp"])) if bits(p)!=bits(q)),None),
       "top":x["top"]==y["top"]}
    res["cmp"][f"{a}__{b}"]=d; print(f"{a}__{b}",d)
for L in [1900,7600]:
    P=mk(L); k=f"diag{L}"
    run(f"P{L}_freshA",[P],k)
    run(f"P{L}_hit",[P],k)
    run(f"P{L}_freshB",[P],f"{k}_b")
    cmp(f"P{L}_freshA",f"P{L}_freshB")
    cmp(f"P{L}_freshA",f"P{L}_hit")
    print(f"P{L} first 6 fresh:", res["cases"][f"P{L}_freshA"]["lp"])
    print(f"P{L} first 6 hit  :", res["cases"][f"P{L}_hit"]["lp"])
json.dump(res,open(OUT,"w"),indent=2)
