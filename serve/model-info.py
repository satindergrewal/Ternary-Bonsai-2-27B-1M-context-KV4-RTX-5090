import json,sys,urllib.request
d=json.load(urllib.request.urlopen("http://localhost:8013/v1/models",timeout=5))
m=d["data"][0]
print("model id:",m["id"])
print("ctx:",m["meta"]["n_ctx"],"| quant:",m["meta"]["ftype"])
