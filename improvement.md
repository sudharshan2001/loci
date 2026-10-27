# Improvements (later)

Ideas and fixes deferred so they don't block the milestone schedule.

## Performance

- **Move Glimmer GGUFs to the NVMe drive.** `checkpoints/` sits on `/dev/sda1`, which is slow (~13 MB/s
  when reading and writing at once), so every `llama-server` start reads ~13–15 GB from it. Move the GGUFs
  to the NVMe root drive (~139 GB free) and keep `checkpoints/glimmer` as a symlink, so no paths change.
  Measured: first Q3 load took 7 min 43 s (while the Q4 download was writing to the same disk). Every
  `name`, `attributes` and `ask` run pays this unless the server is kept up, so this is high priority.

## Quality

- **Screen on/off survives blurring.** Before redaction, record only the mean brightness and variance inside
  each screen mask (no content) and store it as a hint; blurred screens currently look dark, so Glimmer says
  "off" for lit screens.
- **Duplicate objects at frame edges** (three chairs for one): second merge pass for same-label objects whose
  union boxes intersect.
- **Long anchors** (a desk seen in parts): build the anchor's top surface from its points (plane fit) instead
  of a box, so things at the far end are still "on the desk".
- **Answer wording**: avoid repeating the area ("your desk in the bedroom is in the bedroom").
