import os,json,struct,urllib.request,time,re
BASE=os.environ.get("BASE","http://127.0.0.1:8021"); MODEL=os.environ.get("MODEL_NAME",os.path.expanduser("~/models/Qwen3-30B-A3B"))
OUT=os.environ.get("OUT",os.path.expanduser("~/bi_long_diag2.json"))
def post(p,pl,t=900):
    r=urllib.request.Request(BASE+p,data=json.dumps(pl).encode(),headers={"Content-Type":"application/json"})
    with urllib.request.urlopen(r,timeout=t) as x: return json.loads(x.read().decode())
def tok(s): return list(post("/tokenize",{"model":MODEL,"prompt":s})["tokens"])
def metric():
    with urllib.request.urlopen(BASE+"/metrics",timeout=30) as x: txt=x.read().decode()
    for line in txt.splitlines():
        if line.startswith("vllm:prefix_cache_hits_total"): return float(line.split()[-1])
    return None
def gen(ps,salt,mt=8):
    pl={"model":MODEL,"prompt":ps,"max_tokens":mt,"temperature":0.0,"logprobs":5,"seed":0,"return_token_ids":True,"cache_salt":salt}
    t0=time.time(); r=post("/v1/completions",pl); c=[c for c in r["choices"] if c["index"]==0][0]
    return c,r.get("usage"),time.time()-t0
def bits(x): return struct.pack(">d",float(x)).hex()
BASE_TXT="The following is a long passage. "+"The quick brown fox jumps over the lazy dog. "*1500
END="\nOnce upon a time, in a small village, there was a"
base=tok(BASE_TXT); tail=tok(END)
P=(base[:7570]+tail)
res={"prompt_tokens":len(P),"cases":{},"cmp":{}}
def run(n,salt):
    h0=metric(); c,u,dt=gen([P],salt); h1=metric()
    res["cases"][n]={"salt":salt,"hits_before":h0,"hits_after":h1,"usage":u,"sec":dt,
                     "ids":c.get("token_ids"),"lp":c["logprobs"]["token_logprobs"],"top":c["logprobs"]["top_logprobs"]}
    print(n,"cache_hits",h0,"->",h1,"prompt_tokens",u.get("prompt_tokens"),"first_lp",c["logprobs"]["token_logprobs"][0],"ids",c.get("token_ids"))
def cmp(a,b):
    x,y=res["cases"][a],res["cases"][b]
    d={"ids":x["ids"]==y["ids"],"bitwise":all(bits(p)==bits(q) for p,q in zip(x["lp"],y["lp"])),
       "first":next((i for i,(p,q) in enumerate(zip(x["lp"],y["lp"])) if bits(p)!=bits(q)),None),
       "max":max((abs(p-q) for p,q in zip(x["lp"],y["lp"])),default=None)}
    res["cmp"][f"{a}__{b}"]=d; print(f"{a}__{b}",d)
for n,s in [("freshA","saltX"),("hit1","saltX"),("hit2","saltX"),("freshB","saltY"),("hit3","saltY")]:
    run(n,s)
cmp("freshA","freshB"); cmp("freshA","hit1"); cmp("hit1","hit2"); cmp("freshB","hit3"); cmp("freshA","hit3")
json.dump(res,open(OUT,"w"),indent=2)
