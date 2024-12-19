# Configuration file for the current farm

# This input parameter is cluster specific - maximum number of jobs to submit (in either meta or simple modes).
# If using the -auto feature, leave room for at least one more job (the one which will be doing the resubmit).
# E.g., if the cluster job limit is 999, set this to 998 or less.
NJOBS_MAX=998

# Minimum and maximum allowed runtimes for farm jobs (seconds):
RUNTIME_MIN=30
RUNTIME_MAX=3000000

# Case is considered to fail if the actual runtime is shorter than this number of seconds:
dt_failed=5

# The meta-job fails if first N_failed_max cases all fail:
N_failed_max=5


export NJOBS_MAX RUNTIME_MIN RUNTIME_MAX dt_failed N_failed_max
