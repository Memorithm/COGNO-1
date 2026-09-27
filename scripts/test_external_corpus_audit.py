import unittest
from external_corpus_audit import derive
class AuditTests(unittest.TestCase):
    def test_comment_removal_preserves_quoted_slashes_and_code(self):
        s,b=derive(b'// TODO answer\nfn main(){println!("https://x");} //~ ERROR\n')
        self.assertEqual(s,b'\nfn main(){println!("https://x");}\n');self.assertEqual(b['removed_comment_lines'],[1,2])
    def test_dependencies_and_harness_refused(self):
        for s in [b'include!("x");',b'extern crate other;',b'mod dependency;',b'//@ aux-build: x.rs\nfn main(){}',b'serde::x();',b'#![feature(test)]\nfn main(){}']:
            with self.assertRaises(ValueError):derive(s)
    def test_unsupported_lexical_cases_fail_closed(self):
        for s in [b'/* x */ fn main(){}',b'let x=r#"x"#;',b'let x="line\nbreak";',b'//only comment']:
            with self.assertRaises(ValueError):derive(s)
    def test_all_raw_string_prefixes_refused_before_comment_scanning(self):
        for prefix in (b'r', b'br', b'cr'):
            for hashes in (b'', b'#', b'##'):
                with self.subTest(prefix=prefix, hashes=hashes):
                    source = b'fn main(){let _=' + prefix + hashes + b'"https://x"' + hashes + b';}\n'
                    with self.assertRaisesRegex(ValueError, 'raw string'):
                        derive(source)
            with self.subTest(prefix=prefix, internal_quote=True):
                source = b'fn main(){let _=' + prefix + b'#"quoted "//still literal"#;}\n'
                with self.assertRaisesRegex(ValueError, 'raw string'):
                    derive(source)
