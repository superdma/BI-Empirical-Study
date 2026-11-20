"""Read two Nsight Systems exports and compare what surrounds the norm launch.

torch.profiler reports grid, block, shared memory and registers. Nsight
Systems adds the stream, the graph id (if the launch was replayed from a
graph) and the kernels that overlap in time, which is what this script
compares between the 255 and the 256 run.

Usage:
    python analyze_nsys.py --a p255.sqlite --b p256.sqlite \
      --out results/tier4/bi_nsys_diff.json
"""
import argparse
import json
import sqlite3
from collections import Counter


def columns(conn, table):
    return [r[1] for r in conn.execute(f"PRAGMA table_info({table})")]


def tables(conn):
    return [r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'")]


def rms_rows(conn):
    cols = columns(conn, "CUPTI_ACTIVITY_KIND_KERNEL")
    has_short = "shortName" in cols
    name_col = "shortName" if has_short else "demangledName"
    query = f"""SELECT k.start, k.end, k.streamId, k.gridX, k.blockX,
                       {('k.graphId' if 'graphId' in cols else 'NULL')},
                       {('k.correlationId' if 'correlationId' in cols else 'NULL')},
                       s.value
                FROM CUPTI_ACTIVITY_KIND_KERNEL k
                LEFT JOIN StringIds s ON s.id = k.{name_col}
                WHERE lower(s.value) LIKE '%rms%norm%'
                ORDER BY k.start"""
    return [{"start": r[0], "end": r[1], "stream": r[2], "grid": r[3], "block": r[4],
             "graph": r[5], "corr": r[6], "name": r[7]}
            for r in conn.execute(query)]


def overlap_counts(conn, rows):
    """Count kernels running at the same time as each rms launch."""
    cols = columns(conn, "CUPTI_ACTIVITY_KIND_KERNEL")
    name_col = "shortName" if "shortName" in cols else "demangledName"
    out = []
    for r in rows:
        q = f"""SELECT s.value, k.streamId, k.start, k.end
                FROM CUPTI_ACTIVITY_KIND_KERNEL k
                LEFT JOIN StringIds s ON s.id = k.{name_col}
                WHERE k.start < ? AND k.end > ?"""
        hits = [{"name": x[0], "stream": x[1]} for x in
                conn.execute(q, (r["end"], r["start"]))]
        hit_names = Counter(h["name"].split("(")[0] if h["name"] else "?" for h in hits)
        out.append({"grid": r["grid"], "block": r["block"], "stream": r["stream"],
                    "graph": r["graph"], "concurrent": sum(hit_names.values()),
                    "concurrent_names": dict(hit_names.most_common(5))})
    return out


def api_summary(conn):
    names = tables(conn)
    out = {"tables": names}
    if "CUPTI_ACTIVITY_KIND_RUNTIME" not in names:
        return out
    q = """SELECT s.value, COUNT(*) FROM CUPTI_ACTIVITY_KIND_RUNTIME r
           JOIN StringIds s ON s.id = r.nameId GROUP BY s.value
           ORDER BY COUNT(*) DESC LIMIT 25"""
    out["api_calls"] = {r[0]: r[1] for r in conn.execute(q)}
    graph = {k: v for k, v in out["api_calls"].items() if "Graph" in k}
    out["graph_api"] = graph
    return out


def summarize(path):
    conn = sqlite3.connect(path)
    try:
        rows = rms_rows(conn)
        grids = Counter((r["grid"], r["block"]) for r in rows)
        return {
            "rms_launches": len(rows),
            "grid_block_hist": {f"{g}/{b}": n for (g, b), n in sorted(grids.items())},
            "streams": dict(Counter(r["stream"] for r in rows)),
            "graphs": dict(Counter(str(r["graph"]) for r in rows)),
            "api": api_summary(conn),
            "rows": rows[-400:],
            "overlap": overlap_counts(conn, rows[-120:]),
        }
    finally:
        conn.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--out", default="bi_nsys_diff.json")
    args = ap.parse_args()

    a = summarize(args.a)
    b = summarize(args.b)
    result = {"a": args.a, "b": args.b, "a_summary": a, "b_summary": b}

    # Compare the tail of each trace: the decode launches with grid 255/256.
    def tail(rows, grid):
        return [r for r in rows if r["grid"] == grid][-60:]

    result["a_tail"] = tail(a["rows"], 255)
    result["b_tail"] = tail(b["rows"], 256)
    with open(args.out, "w") as handle:
        json.dump(result, handle, indent=1)
    for tag, s in (("a", a), ("b", b)):
        print(tag, "rms launches", s["rms_launches"], "grid/block", s["grid_block_hist"])
        print(tag, "streams", s["streams"], "graphs", s["graphs"])
        print(tag, "graph api", s["api"].get("graph_api"))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
