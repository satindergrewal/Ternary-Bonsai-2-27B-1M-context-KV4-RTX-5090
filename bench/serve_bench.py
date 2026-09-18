#!/usr/bin/env python3
"""Quick serve bench: prose/code decode, prefill, prompt-cache behavior.

Usage: python3 bench/serve_bench.py [model_alias] [PORT]
Reads timings from the llama-server response (no log scraping).
"""
import json, os, sys, urllib.request

alias = sys.argv[1] if len(sys.argv) > 1 else "Bonsai-2-27B-Ablit"
port = sys.argv[2] if len(sys.argv) > 2 else os.environ.get("PORT", "8013")
URL = "http://localhost:%s/v1/chat/completions" % port

PROSE = ("Explain how a modern CPU pipeline handles branch mispredictions, "
         "including the role of the BTB, speculative execution, and how "
         "recovery works on a flush. Be thorough and technical.")
CODE = ("Write a production-grade Python implementation of an LRU cache with "
        "O(1) get and put, thread safety, TTL support, and detailed comments.")

def ask(tag, prompt, maxtok):
    r = json.loads(urllib.request.urlopen(urllib.request.Request(URL,
        data=json.dumps({"model": alias, "messages": [{"role": "user", "content": prompt}],
                         "max_tokens": maxtok, "temperature": 0}).encode()),
        timeout=600).read())
    t = r.get("timings", {})
    print("%-12s decode %7.1f t/s | prefill %8.1f t/s | cached: %5d | gen: %d"
          % (tag, t.get("predicted_per_second", 0), t.get("prompt_per_second", 0),
             t.get("cache_n", 0), t.get("predicted_n", 0)))

ask("prose cold", PROSE, 400)
ask("prose cached", PROSE, 8)   # second hit: prefix replays from cache
ask("code cold", CODE, 400)
ask("prose mid", "Summarize the tradeoffs between optimistic and pessimistic "
    "locking in distributed databases, then recommend one for a read-heavy "
    "workload and justify it.", 300)
