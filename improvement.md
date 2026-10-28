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
- ~~Duplicate objects at frame edges~~: second merge pass added; remaining duplicates come from broken 3D.
- **Long anchors** (a desk seen in parts): build the anchor's top surface from its points (plane fit) instead
  of a box, so things at the far end are still "on the desk".
- **Answer wording**: avoid repeating the area ("your desk in the bedroom is in the bedroom").

## Next ideas

- **Time slider** in the viewer across captures of the same room (needs a second home capture; room frame
  and `what_changed` are ready).
- **Chunked MapAnything** with overlapping windows for walks longer than 80 keyframes.
- **Appearance matching** (SAM crop features) to link the same object across captures when it moved far.
- **Open/closed for carts** (the coconut cart has no shutter): ask "is someone selling, goods out?".
- **Parallel `look` calls** to cut the 70–80 s answers that check several candidates.
- **Answer stability**: lower the agent's temperature (e.g. 0.6) for tool-calling turns and take a majority of
  2–3 `look` calls; measure pass rate over 3 eval runs, not 1.
- **Garden / terrace / balcony** as area labels, so notes said outdoors link to the right place.
