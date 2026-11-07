"""Seal the current local source release before a new collection session."""
import hashlib
import json
from gpu_pre2.i2.provenance import ROOT, source_hashes

files=source_hashes()
release=hashlib.sha256(json.dumps(files,sort_keys=True).encode()).hexdigest()
path=ROOT/'data/manifests/CODE_RELEASE.json'
path.write_text(json.dumps({'release_id':release,'files':files},indent=2)+'\n')
print(release)
