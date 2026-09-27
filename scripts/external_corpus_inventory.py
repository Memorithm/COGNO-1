"""Strict immutable GitHub inventory; content identity is not provenance authentication."""
import hashlib
import json
import re
from pathlib import PurePosixPath

MAX_BYTES = 1_048_576
SOURCE_LIMIT = 16_384
LICENSE_LIMIT = 65_536


def unique(pairs):
    out = {}
    for key, value in pairs:
        if key in out:
            raise ValueError('duplicate JSON key')
        out[key] = value
    return out


def digest(data):
    return hashlib.sha256(data).hexdigest()


def path(value):
    if not isinstance(value, str) or len(value) > 256 or not re.fullmatch(r'[A-Za-z0-9_./-]+', value):
        raise ValueError('invalid repository path')
    if value.startswith('/') or any(p in ('', '.', '..') for p in value.split('/')):
        raise ValueError('noncanonical repository path')
    return str(PurePosixPath(value))


def validate(data, expected):
    if len(data) > MAX_BYTES or digest(data) != expected:
        raise ValueError('inventory capacity or hash mismatch')
    obj = json.loads(data, object_pairs_hook=unique)
    if set(obj) != {'schema', 'sources'} or type(obj['schema']) is not int or obj['schema'] != 1:
        raise ValueError('unknown inventory schema')
    if not isinstance(obj['sources'], list) or not 1 <= len(obj['sources']) <= 256:
        raise ValueError('source count')
    ids, identities = set(), set()
    fields = {'id', 'repository', 'revision', 'path', 'sha256', 'bytes', 'license',
              'license_path', 'license_sha256', 'license_bytes', 'expected_label', 'expected_diagnostics'}
    for row in obj['sources']:
        if not isinstance(row, dict) or set(row) != fields:
            raise ValueError('source schema')
        if not re.fullmatch(r'[a-z0-9][a-z0-9_-]{0,63}', row['id']) or row['id'] in ids:
            raise ValueError('duplicate/invalid id')
        ids.add(row['id'])
        if not re.fullmatch(r'[a-z0-9][a-z0-9_.-]*/[a-z0-9][a-z0-9_.-]*', row['repository']):
            raise ValueError('repository must use canonical lowercase owner/name')
        if not re.fullmatch(r'[0-9a-f]{40}', row['revision']):
            raise ValueError('full immutable commit required')
        for key in ('path', 'license_path'):
            path(row[key])
        identity = (row['repository'], row['revision'], row['path'])
        if identity in identities:
            raise ValueError('duplicate upstream source')
        identities.add(identity)
        for key, limit in [('bytes', SOURCE_LIMIT), ('license_bytes', LICENSE_LIMIT)]:
            if type(row[key]) is not int or not 1 <= row[key] <= limit:
                raise ValueError('invalid byte count')
        for key in ('sha256', 'license_sha256'):
            if not re.fullmatch(r'[0-9a-f]{64}', row[key]):
                raise ValueError('invalid SHA-256')
        if row['license'] not in ('MIT', 'Apache-2.0') or type(row['expected_label']) is not int or row['expected_label'] not in (0, 1):
            raise ValueError('license or label policy')
        codes = row['expected_diagnostics']
        if not isinstance(codes, list) or any(not isinstance(x, str) or not re.fullmatch(r'E[0-9]{4}', x) for x in codes) or codes != sorted(set(codes)):
            raise ValueError('sorted unique rustc error codes required')
        if bool(codes) == bool(row['expected_label']):
            raise ValueError('failure requires diagnostic codes; success requires none')
    return obj
