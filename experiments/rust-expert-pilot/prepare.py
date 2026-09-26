"""Original synthetic Rust pairs, compiler-labelled; never executes input programs."""
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parent
# A complete concept family belongs to exactly one split. Labels are expectations
# checked by rustc, never injected into model input. 0=reject, 1=compile.
PAIRS = [
    ('train', 'move', 'fn f(){let a=String::new();let b=a.clone();drop(a);drop(b);}', 'fn f(){let a=String::new();let b=a;drop(a);drop(b);}'),
    ('train', 'borrow', 'fn f(){let mut a=0;let b=&mut a;*b+=1;let c=&mut a;*c+=1;}', 'fn f(){let mut a=0;let b=&mut a;let c=&mut a;*b+=1;*c+=1;}'),
    ('train', 'lifetime', "fn f<'a>(x:&'a str)->&'a str{x}", "fn f<'a>(x:&str)->&'a str{x}"),
    ('train', 'send', 'fn f<T:Send>(){} fn g(){f::<std::sync::Arc<u8>>();}', 'fn f<T:Send>(){} fn g(){f::<std::rc::Rc<u8>>();}'),
    ('train', 'sync', 'fn f<T:Sync>(){} fn g(){f::<std::sync::Mutex<u8>>();}', 'fn f<T:Sync>(){} fn g(){f::<std::cell::Cell<u8>>();}'),
    ('train', 'sized', 'fn f<T:?Sized>(_:&T){} fn g(x:&str){f(x);}', 'fn f<T:Sized>(_:&T){} fn g(x:&str){f(x);}'),
    ('train', 'associated', 'fn f<T:Iterator<Item=u8>>(x:T)->Vec<u8>{x.collect()}', 'fn f<T:Iterator<Item=u16>>(x:T)->Vec<u8>{x.collect()}'),
    ('train', 'const_generic', 'fn f<const N:usize>(x:[u8;N])->[u8;N]{x}', 'fn f<const N:usize>(x:[u8;N])->[u8;3]{x}'),
    ('validation', 'dyn_compatibility', 'trait A{fn f(&self);} fn g(_: &dyn A){}', 'trait A{fn f<T>(&self);} fn g(_: &dyn A){}'),
    ('validation', 'pin_unpin', 'fn f(x:std::pin::Pin<Box<u8>>){let _=std::pin::Pin::into_inner(x);}', 'fn f(x:std::pin::Pin<Box<std::marker::PhantomPinned>>){let _=std::pin::Pin::into_inner(x);}'),
    ('test', 'gat', "trait A{type B<'a> where Self:'a;fn f<'a>(&'a self)->Self::B<'a>;}", "trait A{type B<'a>;fn f<'a>(&'a self)->Self::B<'a>;}"),
    ('test', 'hrtb', "fn f<F:for<'a> Fn(&'a u8)>(f:F){let x=0;f(&x);}", "fn f<F:Fn(&'static u8)>(f:F){let x=0;f(&x);}"),
]


def main():
    version = subprocess.check_output(['rustc', '+1.97.1', '--version'], text=True).strip()
    records, fingerprints = [], set()
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / 'input.rs'
        for split, family, good, bad in PAIRS:
            for label, code in [(1, good), (0, bad)]:
                digest = hashlib.sha256(code.encode()).hexdigest()
                if digest in fingerprints:
                    raise ValueError('duplicate source')
                fingerprints.add(digest)
                path.write_text(code)
                result = subprocess.run(['rustc', '+1.97.1', '--edition=2024', '--crate-type=lib', '--emit=metadata', '--error-format=json', str(path), '-o', str(Path(tmp)/'out.rmeta')], capture_output=True, text=True, timeout=20)
                errors = [json.loads(line) for line in result.stderr.splitlines() if line.startswith('{')]
                codes = [e['code']['code'] if e.get('code') else e['message'] for e in errors if e.get('level') == 'error' and e.get('spans')]
                if result.returncode not in (0, 1) or int(result.returncode == 0) != label or (not label and not codes):
                    raise ValueError((family, label, result.stderr))
                records.append(dict(split=split, family=family, label=label, source=code, sha256=digest, diagnostics=codes, classification='Internal', provenance='original synthetic example; rustc validated', compiler=version))
    (ROOT/'corpus.jsonl').write_text(''.join(json.dumps(r, sort_keys=True)+'\n' for r in records))
    (ROOT/'corpus.tsv').write_text(''.join(f"{r['split']}\t{r['family']}\t{r['label']}\t{r['source']}\n" for r in records))
    print(f'Validated {len(records)} snippets, {len(PAIRS)} disjoint concept families with {version}')

if __name__ == '__main__':
    main()
