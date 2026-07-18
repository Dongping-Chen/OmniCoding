# RL coordinator security boundary

The coordinator accepts requests that eventually launch coding agents with
shell access. Treat it as a privileged control-plane service.

Implemented controls:

- bearer authentication on every HTTP route;
- exact inference-origin and model allowlists;
- bounded samples, turns, token settings, timeouts, queued jobs, and global
  in-flight trajectories;
- dataset-root containment for every staged media path;
- copied media and gold-free worker payloads: dataset roots and answer keys stay
  in the coordinator, and reward grading happens only after worker results return;
- a fixed non-secret environment allowlist for Slurm jobs;
- completion-marker-based scratch cleanup that never age-deletes active or
  scheduler-unknown job directories;
- no local dataset path in the health response.

Remaining deployment boundary:

Kira executes shell commands as the Slurm job's operating-system identity.
Workspace scoping is not a filesystem sandbox: an adversarial or compromised
model could try absolute paths, inspect readable host files, or access the
network allowed to that job. Public or multi-tenant deployment therefore
requires a container/job sandbox with a read-only base image, an explicit
task-media mount, an empty secret store, network egress policy, resource limits,
and a disposable OS identity. Until that executor exists, use the RL path only
with trusted operators on an isolated cluster account.

Never place cloud, Hugging Face write, coordinator, or tunnel credentials in
the Slurm worker environment. The public Relax recipe passes only the path to
a current-user-owned, mode-0600 coordinator token file into Ray; it never
serializes the token value into command arguments or job metadata. Rotate any
credential that was previously kept
in an experiment `.env` before publishing the repository.
