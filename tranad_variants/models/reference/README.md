# Reference

Subclass of the sibling baseline `TranAD`, with no changes to initialization,
forward computation or state keys. Two independently parameterized decoders,
two passes through the shared encoder, original self-conditioning and shared
linear-sigmoid reconstruction head. This is the control in every comparison.

Training uses the original epoch-weighted two-reconstruction objective. Scoring
evaluates 21 alpha mixtures. The initial reference checkpoint is trained anew
alongside the variants; old v10/v12 results remain external comparison evidence.
