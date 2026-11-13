# Project conventions

Keep the current implementation in **tranad_plus/**. Improve that implementation
in place; do not create sibling copies such as tranad_v15/ or tranad_plus_v16/.
Version labels identify campaign plans, provenance and immutable results under
results/vNN/. Shared architecture implementations belong in tranad_plus/core/;
campaign tools belong in tranad_plus/campaigns/ by purpose.

Preserve all existing datasets, checkpoints and result artifacts. Source cleanup
does not authorize deleting research data. Keep source provenance verifiable
when reorganizing code. Never claim an integer kernel, hardware speedup or
measurement that has not actually been verified.
