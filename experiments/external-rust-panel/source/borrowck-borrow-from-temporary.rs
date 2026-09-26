fn id<T>(x: T) -> T { x }
struct Foo(isize);
fn foo<'a>() -> &'a isize {
    let &Foo(ref x) = &id(Foo(3));
    x
}
pub fn main() {
}
