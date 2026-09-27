"""Bounded immutable downloads into a content-addressed cache (or offline verification)."""
import argparse
from pathlib import Path
import urllib.request
from external_corpus_inventory import validate, digest, MAX_BYTES


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise ValueError('redirect refused; review canonical repository inventory')


def download(url, limit):
    opener=urllib.request.build_opener(NoRedirect())
    with opener.open(url, timeout=20) as response:
        if response.status != 200 or response.geturl() != url:
            raise ValueError('unexpected download response')
        return response.read(limit+1)


def checked(data, sha, size):
    if len(data)!=size or digest(data)!=sha:
        raise ValueError('downloaded/cached bytes differ from pinned inventory')
    return data


def obtain(row, cache, license_file=False, offline=False, fetch=download):
    prefix='license_' if license_file else ''
    sha,size,path=row[prefix+'sha256'],row[prefix+'bytes'],row[prefix+'path']
    target=cache/sha
    if target.is_symlink():raise ValueError('cache symlink refused')
    if target.exists():
        with target.open('rb') as stream:return checked(stream.read(size+1),sha,size)
    if offline:raise ValueError('offline cache entry missing')
    url=f"https://raw.githubusercontent.com/{row['repository']}/{row['revision']}/{path}"
    data=checked(fetch(url,size),sha,size)
    with target.open('xb') as stream:stream.write(data)
    return data


def fetch_inventory(inventory,cache,offline=False,fetch=download):
    if cache.is_symlink():raise ValueError('cache directory symlink refused')
    cache.mkdir(exist_ok=True)
    for row in inventory['sources']:
        obtain(row,cache,offline=offline,fetch=fetch)
        obtain(row,cache,license_file=True,offline=offline,fetch=fetch)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('inventory',type=Path);p.add_argument('sha256');p.add_argument('cache',type=Path);p.add_argument('--offline',action='store_true');a=p.parse_args()
    with a.inventory.open('rb') as f:inventory=validate(f.read(MAX_BYTES+1),a.sha256)
    fetch_inventory(inventory,a.cache,a.offline)
    print('verified',len(inventory['sources']),'sources and their pinned license files')
