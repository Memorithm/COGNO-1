"""Conservative standalone-source admission and comment-only derivation, not a Rust parser."""
import re
from external_corpus_inventory import SOURCE_LIMIT,digest


def derive(raw):
    if not 1<=len(raw)<=SOURCE_LIMIT:raise ValueError('source byte bound')
    try:
        text=raw.decode('utf-8')
    except UnicodeDecodeError as exc:
        raise ValueError('source must be valid UTF-8') from exc
    if '\r' in text or '/*' in text or '*/' in text or re.search(r'\b(?:b|c)?r#*"',text):
        raise ValueError('unsupported CR, block comment, or raw string')
    output=[];removed=[]
    for number,line in enumerate(text.splitlines(keepends=True),1):
        quoted=False;escaped=False;cut=None
        for i,char in enumerate(line):
            if escaped:escaped=False;continue
            if quoted and char=='\\':escaped=True;continue
            if char=='"':quoted=not quoted;continue
            if not quoted and line[i:i+2]=='//':cut=i;break
        if quoted:raise ValueError('multiline/unsupported quoted literal')
        if cut is not None:
            comment=line[cut:]
            if re.search(r'(aux-build|compile-flags|revision|needs-|only-|ignore-|run-flags|edition:|rustc-env|assembly-output)',comment):
                raise ValueError('upstream harness directive requires manual dependency review')
            removed.append(number)
            line=line[:cut].rstrip()+'\n'
        output.append(line)
    source=''.join(output).encode()
    code=source.decode()
    forbidden=[r'\bextern\s+crate\b',r'\b(?:include|include_str|include_bytes|env|option_env)\s*!',
               r'\bmod\s+\w+\s*;',r'#\s*!?\s*\[\s*(?:feature|path|cfg|cfg_attr)',r'\bproc_macro\b']
    if any(re.search(pattern,code) for pattern in forbidden):raise ValueError('external dependency or unsupported compiler configuration')
    roots=re.findall(r'\b([a-z_][a-z_0-9]*)\s*::',code)
    if any(root not in {'std','core','alloc','self','super','crate','ptr','thread','sync','atomic','mpsc','cell','rc'} for root in roots):
        raise ValueError('unreviewed lowercase namespace dependency')
    if not source.strip():raise ValueError('empty derived source')
    return source,dict(transform='standalone-line-comments-v1',raw_sha256=digest(raw),
                       source_sha256=digest(source),removed_comment_lines=removed,
                       limitations='conservative subset; lexical rejection is not semantic parsing')

