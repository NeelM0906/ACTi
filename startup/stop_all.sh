#!/bin/bash
# Stop all ACTi tmux services. Does not touch nginx or saved data.
set +e
for s in acti-inference acti-proxy acti-ui acti-status; do
  if tmux has-session -t "$s" 2>/dev/null; then
    tmux kill-session -t "$s"
    echo "  stopped: $s"
  fi
done
echo "ACTi services stopped. Run startup/07_start_all.sh to bring them back up."
