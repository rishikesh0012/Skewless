# Screenshot reference

These portfolio screenshots were captured from the running application with the default trip and `distance_unit` scenario. Only the feature architecture changes between them.

## `broken-distance-skew.png`

- Select **Broken — Duplicated paths**.
- Keep **Distance unit — miles interpreted as km** selected.
- Run the parity check.
- Shows the architecture strip, `$29.19` predicted fare, `8 / 9` parity summary, and highlighted `trip_distance_miles` mismatch.

## `correct-perfect-parity.png`

- Keep the same trip and distance-unit scenario.
- Switch to **Correct — Shared transform**.
- Run the parity check again.
- Shows the shared-path architecture strip, `$20.29` predicted fare, `9 / 9` parity summary, and matching feature values.

Both files use PNG format and matching viewport dimensions. If the application UI changes, recapture both images together so their framing remains consistent.
