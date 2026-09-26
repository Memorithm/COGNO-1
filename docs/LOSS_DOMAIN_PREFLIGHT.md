# Loss-domain preflight

InfoNCE requires a positive finite temperature whose reciprocal is representable
in f32. The smallest positive f32 passes a positivity check but its reciprocal
is infinity. Construction and evaluation now reject that case explicitly.
Evaluation rechecks the public mutable temperature on all three entry points.
The scalar-list path checks count and target bounds before allocating a stack
node, so rejected admission does not consume tape capacity.

Pairwise ranking requires a finite nonnegative margin. Construction and both
evaluation entry points enforce this even after mutation of the public field.
Zero remains valid. Successful valid-domain computations are unchanged.

Tests use a tape with one remaining node and verify it remains available after
rejection. Existing connected-gradient and loss tests cover accepted paths.
No clipping, replacement value, or altered numerical objective is introduced.
