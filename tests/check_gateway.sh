#!/usr/bin/env bash
# Проверки после прогона плейбука в CI: ufw и nginx перед заглушкой vLLM.
# Переменные: TLS (http | self-signed | own-cert), VLLM_API_KEY.
set -euo pipefail

domain=llm.example.test
auth="Authorization: Bearer ${VLLM_API_KEY}"

check() {
  if ! "$@"; then
    echo "FAIL: $*" >&2
    exit 1
  fi
}

status() { curl -sk -o /dev/null -w '%{http_code}' "$@"; }

expect() {
  local want=$1 got
  shift
  got=$(status "$@")
  if [ "$got" != "$want" ]; then
    echo "FAIL: ожидался $want, получен $got (${*: -1})" >&2
    exit 1
  fi
  echo "ok: $want ${*: -1}"
}

# ufw включен, входящие закрыты, порт API открыт.
ufw_status=$(sudo ufw status verbose)
echo "$ufw_status"
check grep -q 'Status: active' <<<"$ufw_status"
check grep -q 'deny (incoming)' <<<"$ufw_status"
check grep -qE '^80/tcp +ALLOW IN +Anywhere' <<<"$ufw_status"

if [ "$TLS" = http ]; then
  base=http://127.0.0.1
else
  base=https://127.0.0.1
  check grep -qE '^443/tcp +ALLOW IN +Anywhere' <<<"$ufw_status"
  expect 301 http://127.0.0.1/v1/models
  # Сертификат выписан на домен: curl сверяет имя с subjectAltName.
  check curl -sf --resolve "$domain:443:127.0.0.1" --cacert /etc/ssl/certs/llm-api.crt \
    -o /dev/null "https://$domain/health"
  echo "ok: сертификат действителен для $domain"
  check test "$(sudo stat -c %a /etc/ssl/private/llm-api.key)" = 600
fi

expect 200 "$base/health"
expect 401 "$base/v1/models"
expect 401 -H "Authorization: Bearer wrong-key" "$base/v1/models"
expect 200 -H "$auth" "$base/v1/models"
expect 404 "$base/docs"
expect 404 "$base/metrics"

# Rate limit 10r/s с burst 20: из 60 запросов подряд часть должна получить 429.
limited=0
for _ in $(seq 60); do
  if [ "$(status -H "$auth" "$base/v1/models")" = 429 ]; then
    limited=$((limited + 1))
  fi
done
echo "429: $limited из 60"
check test "$limited" -gt 0
