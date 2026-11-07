"""Explicit E2 hardware descriptors with historical software compatibility evidence."""
import hashlib
import json
from .storage import dataset

def load_profile(device):
    root=dataset('E2_'+device,stage='E2')
    path=root/'hardware_profile.json'
    profile=json.loads(path.read_text())
    if profile.get('protocol')!='e2_calibration_v3' or profile.get('status')!='completed':
        raise ValueError('Unknown or incomplete E2 profile')
    return profile,hashlib.sha256(path.read_bytes()).hexdigest()

def compatibility(device, collection):
    profile,digest=load_profile(device)
    checks={}
    for old,new in [('torch','torch'),('cuda_runtime','cuda'),('gpu_uuid','gpu_uuid')]:
        value=collection.get(new)
        checks[new]='unverified' if value is None else 'match' if value==profile.get(old) else 'mismatch'
    return {'device':device,'profile_id':profile['profile_id'],'profile_sha256':digest,
            'checks':checks,'status':'historical_profile',
            'compatible':True if all(x=='match' for x in checks.values()) else None if 'mismatch' not in checks.values() else False,
            'strict_reference_verified':False,'recalibration_automatically_started':False}
