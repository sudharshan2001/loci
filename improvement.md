# Improvements (later)

Ideas and fixes deferred so they don't block the milestone schedule.

## Performance

- **Move Glimmer GGUFs to the NVMe drive.** `checkpoints/` sits on `/dev/sda1`, which is slow (~13 MB/s
  when reading and writing at once), so every `llama-server` start reads ~13–15 GB from it. Move the GGUFs
  to the NVMe root drive (~139 GB free) and keep `checkpoints/glimmer` as a symlink, so no paths change.
