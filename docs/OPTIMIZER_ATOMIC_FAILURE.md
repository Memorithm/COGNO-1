# Atomic numerical optimizer failures

AdamW and AMSGrad validate every public moment buffer, parameter, gradient and
hyperparameter before updating. Mismatched buffers return a shape error instead
of indexing past their end. Negative second moments, nonfinite inputs, overflowed
moments, exhausted step counters and nonfinite candidates reject the entire step.

An update is staged on owned copies and committed only after every element
succeeds. This adds linear temporary storage and copying; it is a correctness
tradeoff, not a speed claim. Finite successful arithmetic and default formulas
are unchanged. Atomicity here applies to one optimizer tensor, not a collection
of independently called optimizer objects.

The regression tests place invalid data at the final element, check unchanged
parameters and state, then retry and compare with a fresh optimizer. They also
exercise each malformed public moment buffer and exhausted counters.
