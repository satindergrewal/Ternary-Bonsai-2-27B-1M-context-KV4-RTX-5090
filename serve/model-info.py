import json, os, sys, urllib.request
port = os.environ.get("PORT", "8013")
d = json.load(urllib.request.urlopen("http://localhost:%s/v1/models" % port, timeout=5))
m = d["data"][0]
print("model id:", m["id"])
print("ctx:", m["meta"]["n_ctx"], "| quant:", m["meta"]["ftype"])
