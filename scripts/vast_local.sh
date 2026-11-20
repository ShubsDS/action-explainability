#!/usr/bin/env bash
# Runs LOCALLY, from the repo root: rent a Vast machine, set it up with scripts/vast_serve.sh, start
# vLLM and the queue (scripts/vast_relay.py), keep an SSH tunnel open, and stop or destroy it.
# The conversation experiment did these steps from throwaway files that a reboot then deleted;
# this keeps them. Nothing here samples: sampling is `bin/td python -m td.experiments.<...>` with
# TD_VLLM_RELAY_URL=http://127.0.0.1:8001 once `tunnel` is up.
#
#   scripts/vast_local.sh offers                      # 4 x 80 GB machines with enough disk, cheapest first
#   scripts/vast_local.sh create <offer id> <label>   # RENTS (billing starts); records id, time and credit
#   scripts/vast_local.sh wait                        # until the machine runs and answers over SSH
#   scripts/vast_local.sh push                        # copy the two scripts and the server's API key
#   scripts/vast_local.sh setup                       # start vast_serve.sh setup on the machine (about 40 min)
#   scripts/vast_local.sh status                      # setup log, processes, queue health
#   scripts/vast_local.sh serve                       # start vLLM, wait until it answers
#   scripts/vast_local.sh relay <stop_at unix time> [idle minutes] [in flight]   # (re)start the queue
#   scripts/vast_local.sh deadline <stop_at unix time>   # move the machine's self-stop time
#   scripts/vast_local.sh tunnel                      # reconnecting tunnel to ports 8000 and 8001 (foreground)
#   scripts/vast_local.sh charged                     # dollars charged since `create`, from the account's credit
#   scripts/vast_local.sh ssh '<command>'             # run a command on the machine
#   scripts/vast_local.sh stop | destroy              # stop keeps the disk ($0.13/h); destroy ends all billing
#
# Keys: VAST_API_KEY and TD_VLLM_API_KEY come from .env and are never printed or put on a command line.
# State (instance id, SSH target, start time, credit at start) lives in data/vast/, which git ignores.
set -uo pipefail
cd "$(dirname "$0")/.."
STATE=data/vast
mkdir -p "$STATE"
set -a; . ./.env; set +a
SSH_KEY=$HOME/.ssh/td_vast
IMAGE=nvidia/cuda:12.4.1-devel-ubuntu22.04
DISK_GB=460

# Remote `pgrep -f` patterns start with a bracketed letter: without it the pattern matches the remote
# shell that carries it in its own command line (the first `serve` reported "already running").
vast() {   # the Vast CLI with the key taken from the environment; the key is scrubbed from its output
  "$HOME/.local/bin/vastai" "$@" 2>&1 | sed "s/$VAST_API_KEY/<vast key>/g" | grep -v -i "warn\|deprecat"
}
iid() { cat "$STATE/instance_id" 2>/dev/null || { echo "no machine recorded in $STATE/instance_id" >&2; exit 1; }; }
credit() { vast show user --raw | python3 -I -c "import json,sys; t=sys.stdin.read(); print(round(json.loads(t[t.index('{'):])['credit'], 4))"; }
vssh() {
  read -r HOST PORT < "$STATE/ssh_target"
  ssh -i "$SSH_KEY" -o UserKnownHostsFile="$STATE/known_hosts" -o StrictHostKeyChecking=accept-new -o ConnectTimeout=15 \
      -o ServerAliveInterval=30 -o LogLevel=ERROR -p "$PORT" "root@$HOST" "$@" 2>&1 | grep -v "Welcome to vast\|Have fun"
  return "${PIPESTATUS[0]}"   # the remote command's status, not grep's (which is 1 whenever there is no output)
}

case "${1:-}" in
  offers)
    vast search offers "num_gpus=4 gpu_ram>=79 disk_space>=$DISK_GB verified=true rentable=true inet_down>=700 reliability>0.97" \
        --storage $DISK_GB -o dph_total --raw | python3 -I -c "
import json, sys
t = sys.stdin.read(); d = json.loads(t[t.index('['):])
print(len(d), 'offers; price includes', $DISK_GB, 'GB of disk')
for o in d[:15]:
    print(f\"{o['id']:>10} {o['gpu_name'][:16]:16} {o['gpu_ram']/1024:4.0f} GB  \${o['dph_total']:.2f}/h  down {o.get('inet_down',0):5.0f} Mb/s  \"
          f\"disk {o.get('disk_space',0):5.0f} GB  reliability {o.get('reliability2',0):.3f}  {(o.get('geolocation') or '')[:16]:16} max {o.get('duration',0)/86400:.0f} days\")"
    ;;
  create)
    [ $# -eq 3 ] || { echo "usage: $0 create <offer id> <label>"; exit 2; }
    [ -f "$STATE/instance_id" ] && { echo "a machine is already recorded ($(cat "$STATE/instance_id")); destroy it or remove $STATE/instance_id"; exit 1; }
    credit > "$STATE/credit_at_start"
    date -u +"%Y-%m-%d %H:%M:%S UTC" > "$STATE/rental_start"
    # Without --raw the reply has no instance-scoped key in it; keep only the contract number.
    out=$(vast create instance "$2" --image "$IMAGE" --disk $DISK_GB --ssh --direct --label "$3")
    id=$(printf '%s' "$out" | grep -o "new_contract'*\"*: *[0-9]*" | grep -o "[0-9]*$" | head -1)
    [ -n "$id" ] || { echo "create failed: $(printf '%s' "$out" | cut -c1-300)"; rm -f "$STATE/rental_start" "$STATE/credit_at_start"; exit 1; }
    echo "$id" > "$STATE/instance_id"
    echo "rented: instance $id at $(cat "$STATE/rental_start"); credit before \$$(cat "$STATE/credit_at_start")"
    ;;
  wait)
    ID=$(iid)
    for _ in $(seq 1 120); do
      st=$(vast show instance "$ID" --raw | python3 -I -c "
import json,sys
t=sys.stdin.read()
try:
    d=json.loads(t[t.index('{'):]); print(d.get('actual_status'), '|', (d.get('status_msg') or '')[:90].replace(chr(10),' '), '| \$%.2f/h' % (d.get('dph_total') or 0))
except Exception: print('?', '|', t[:80].replace(chr(10),' '))")
      echo "$(date -u +%H:%M:%S) $st"
      case "$st" in running*) break ;; esac
      sleep 10
    done
    case "$st" in running*) ;; *) echo "NOT RUNNING after 20 minutes"; exit 1 ;; esac
    URL=$(vast ssh-url "$ID" | grep -o 'ssh://[^ ]*' | tail -1)
    HOSTPORT=${URL#ssh://root@}; echo "${HOSTPORT%:*} ${HOSTPORT##*:}" > "$STATE/ssh_target"
    for _ in $(seq 1 30); do vssh -n 'echo ssh ok' | grep -q "ssh ok" && { echo "ssh ok"; exit 0; }; sleep 10; done
    echo "machine runs but SSH does not answer"; exit 1
    ;;
  push)
    read -r HOST PORT < "$STATE/ssh_target"
    scp -i "$SSH_KEY" -o UserKnownHostsFile="$STATE/known_hosts" -o LogLevel=ERROR -P "$PORT" \
        scripts/vast_serve.sh scripts/vast_relay.py "root@$HOST:/root/" && echo "scripts copied"
    # The server's key goes over SSH stdin: never on a command line, never printed.
    printf '%s' "$TD_VLLM_API_KEY" | vssh 'mkdir -p /workspace/td && umask 077 && cat > /workspace/td/.vllm_key && echo "key bytes on machine: $(wc -c < /workspace/td/.vllm_key)"'
    ;;
  setup)
    vssh -n 'mkdir -p /workspace/td; setsid nohup bash /root/vast_serve.sh setup > /workspace/td/setup.log 2>&1 < /dev/null & disown; sleep 3; pgrep -f "[v]ast_serve.sh setup" > /dev/null && echo "setup started"'
    ;;
  status)
    vssh -n 'echo "--- setup.log"; cut -c1-200 /workspace/td/setup.log 2>/dev/null | tail -12
      echo "--- sizes"; du -sh /workspace/td/base /workspace/td/peft/* 2>/dev/null
      tail -c 300 /workspace/td/download.log 2>/dev/null | tr "\r" "\n" | tail -1 | cut -c1-160
      echo "--- processes"; pgrep -fa "[v]ast_serve|[h]f download|[b]uild_lora|[v]llm serve|[v]ast_relay" | cut -c1-110
      echo "--- vLLM"; curl -s -o /dev/null -w "health %{http_code}\n" --max-time 5 http://127.0.0.1:8000/health
      grep "Avg generation throughput" /workspace/td/serve.log 2>/dev/null | tail -1 | sed "s/.*Avg generation/gen/" | cut -c1-200
      echo "--- queue"; curl -s --max-time 5 http://127.0.0.1:8001/health; echo; grep "relay up\|moved\|STOPPING" /workspace/td/relay.log 2>/dev/null | tail -2'
    ;;
  serve)
    vssh -n 'pgrep -f "[v]llm serve" > /dev/null && { echo "vLLM already running"; exit 0; }
      setsid nohup bash /root/vast_serve.sh serve > /workspace/td/serve.log 2>&1 < /dev/null & disown; sleep 2; echo "server process started"'
    for _ in $(seq 1 90); do
      code=$(vssh -n 'curl -s -o /dev/null -w "%{http_code}" --max-time 5 http://127.0.0.1:8000/health')
      [ "$code" = "200" ] && { echo "$(date -u +%H:%M:%S) vLLM answers"; exit 0; }
      vssh -n 'pgrep -f "[v]llm serve" > /dev/null' || { echo "vLLM died:"; vssh -n 'tail -5 /workspace/td/serve.log | cut -c1-300'; exit 1; }
      sleep 20
    done
    echo "vLLM not ready after 30 minutes"; exit 1
    ;;
  relay)
    [ $# -ge 2 ] || { echo "usage: $0 relay <stop_at unix time> [idle minutes, default 45] [in flight, default 16]"; exit 2; }
    # The steps go to the machine on stdin. As one command-line string they contained the queue's own
    # launch line, so `pgrep -f` matched the shell that carried them and the `kill` ended it.
    vssh bash -s -- "$2" "${3:-45}" "${4:-16}" <<'REMOTE'
export PATH=$HOME/.local/bin:$PATH
[ -x /workspace/td/venv-vllm/bin/vastai ] || uv pip install --quiet --python /workspace/td/venv-vllm/bin/python vastai 2>&1 | tail -1
/workspace/td/venv-vllm/bin/python -c 'import fastapi, uvicorn, httpx' || uv pip install --quiet --python /workspace/td/venv-vllm/bin/python fastapi uvicorn httpx
echo "machine-scoped credentials found: $(tr '\0' '\n' < /proc/1/environ | grep -c '^CONTAINER_API_KEY=\|^CONTAINER_ID=')"
P=$(pgrep -f 'bin/python /root/vast_relay.py' || true)
[ -n "$P" ] && kill $P && sleep 2
rm -f /workspace/td/results/*.tmp /workspace/td/stop_at
cd /root
WORK=/workspace/td STOP_AT=$1 RELAY_CONCURRENCY=$3 IDLE_STOP_MIN=$2 setsid nohup /workspace/td/venv-vllm/bin/python /root/vast_relay.py >> /workspace/td/relay.log 2>&1 < /dev/null &
disown
sleep 6
grep 'relay up\|moved\|STOPPING' /workspace/td/relay.log | tail -1
curl -s http://127.0.0.1:8001/health; echo
REMOTE
    ;;
  deadline)
    [ $# -eq 2 ] || { echo "usage: $0 deadline <stop_at unix time>"; exit 2; }
    vssh -n "echo $2 > /workspace/td/stop_at; sleep 65; grep 'moved\|STOPPING' /workspace/td/relay.log | tail -1"
    ;;
  tunnel)
    rm -f "$STATE/tunnel_stop"
    while [ ! -f "$STATE/tunnel_stop" ]; do
      read -r HOST PORT < "$STATE/ssh_target"
      echo "$(date -u +%H:%M:%S) tunnel up"
      ssh -i "$SSH_KEY" -o UserKnownHostsFile="$STATE/known_hosts" -o StrictHostKeyChecking=accept-new -o ConnectTimeout=15 \
          -o ServerAliveInterval=15 -o ServerAliveCountMax=3 -o ExitOnForwardFailure=yes -o LogLevel=ERROR \
          -N -L 8000:127.0.0.1:8000 -L 8001:127.0.0.1:8001 -p "$PORT" "root@$HOST"
      echo "$(date -u +%H:%M:%S) tunnel dropped (rc $?); reconnecting in 5 s"
      sleep 5
    done
    echo "tunnel stopped"
    ;;
  charged)
    now=$(credit); start=$(cat "$STATE/credit_at_start")
    python3 -I -c "print(f'charged since create: \${$start - $now:.2f} (credit \${$now:.2f}; rented at $(cat "$STATE/rental_start"))')"
    ;;
  ssh) shift; vssh "$@" ;;
  stop) vast stop instance "$(iid)" | tail -2; date -u +"stop requested %Y-%m-%d %H:%M:%S UTC" ;;
  destroy)
    touch "$STATE/tunnel_stop"
    vast destroy instance "$(iid)" | tail -2
    mv "$STATE/instance_id" "$STATE/instance_id.destroyed.$(date -u +%Y%m%dT%H%M%S)"
    date -u +"destroyed %Y-%m-%d %H:%M:%S UTC"
    ;;
  *) sed -n 2,24p "$0"; exit 2 ;;
esac
