# GPU-сервер с LLM: Ansible + vLLM

Ansible-плейбук, который превращает чистый Ubuntu-сервер с GPU NVIDIA в рабочий LLM API,
совместимый с OpenAI API: ставит драйвер, Docker с NVIDIA Container Toolkit, поднимает
vLLM в контейнере, закрывает его за nginx с ключом и rate limit, включает мониторинг GPU
и в конце сам проверяет, что модель отвечает.

## Архитектура

```
 клиент ──HTTP(S)──▶ ufw ──▶ nginx :80/:443 ──▶ 127.0.0.1:8000 vLLM (Docker, GPU)
                     │        - Bearer-ключ обязателен           - OpenAI API /v1/*
                     │        - rate limit, лимит тела            - ключ VLLM_API_KEY
                     │        - наружу только /v1/* и /health     - кеш моделей на томе
                     │
                     └─(только metrics_allowed_cidrs, по умолчанию закрыто)─▶ dcgm-exporter :9400, node_exporter :9100
```

| Роль | Что делает | Тег |
|---|---|---|
| `common` | базовые пакеты, часовой пояс, ufw: при первом прогоне сбрасывает правила, с которыми пришел образ, затем открывает SSH и порт nginx, порты метрик только для `metrics_allowed_cidrs` (по умолчанию закрыты) | `common` |
| `nvidia_driver` | проверяет GPU на шине; если рабочий драйвер уже есть (образ провайдера), не трогает его, иначе ставит драйвер заданной ветки, фиксирует его (apt hold) и перезагружает сервер; выключает MIG, если он остался включен (иначе CUDA не видит карту); в конце проверяет `nvidia-smi` | `nvidia` |
| `docker` | Docker CE из официального репозитория, NVIDIA Container Toolkit, `nvidia-ctk runtime configure`, тестовый `nvidia-smi` в контейнере CUDA 13.0 (заодно проверяет, что драйвер подходит образу vLLM) | `docker` |
| `vllm` | docker compose с образом `vllm/vllm-openai` закрепленной версии, ключ API, кеш HF на томе, `restart: unless-stopped`, healthcheck, порт только на 127.0.0.1 | `vllm` |
| `api_gateway` | nginx перед vLLM: rate limit, лимит тела запроса, отсечение запросов без ключа, стриминг, опционально HTTPS по домену (свой или самоподписанный сертификат) | `gateway` |
| `monitoring` | dcgm-exporter (метрики GPU) и node_exporter (метрики хоста), включается флагом | `monitoring` |
| `verify` | `nvidia-smi`, ожидание `/health`, 401 без ключа, `/v1/models` и тестовый `/v1/chat/completions` через nginx | `verify` |

Ключевые решения:

- **vLLM**: высокая пропускная способность (continuous batching, PagedAttention) и готовый
  OpenAI-совместимый API, клиенты подключаются без доработок.
- **Образ `vllm/vllm-openai:v0.31.0`** (CUDA 13.0): версия закреплена, `latest` не используется.
  Нужен драйвер 580+, это ветка по умолчанию. Вариант `v0.31.0-cu129` для старых драйверов
  не подходит: torchcodec в нем собран под CUDA 13, и vLLM падает при старте
  (`libnvrtc.so.13: cannot open shared object file`).
- **Драйвер ветки 580, пакет `nvidia-driver-580-server-open`**: production-ветка NVIDIA,
  покрывает CUDA 12.x и 13.x. Открытые модули ядра обязательны для Blackwell и рекомендованы
  NVIDIA для Turing и новее. Пакеты фиксируются через apt hold, чтобы unattended-upgrades
  не обновил драйвер под работающей моделью (несовпадение версий модуля ядра и библиотек
  ломает `nvidia-smi` до перезагрузки).
- **Рабочий драйвер не трогается**: у GPU-провайдеров образы часто идут с драйвером.
  Если `nvidia-smi` уже работает, роль ничего не ставит и не перезагружает сервер
  (`nvidia_driver_keep_existing: true`). Подходит ли драйвер образу vLLM, проверяет тест
  `nvidia-smi` в контейнере CUDA 13.0 в роли `docker`.
- **Модель по умолчанию `Qwen/Qwen3.8-27B-FP8`**: Qwen3.8-27B в официальной FP8-версии, около
  31 ГБ весов вместо 56 ГБ в BF16, поэтому помещается на одну GPU от 48 ГБ. Без гейта на
  Hugging Face. Модель по умолчанию сначала размышляет (thinking): vLLM запущен с
  `--reasoning-parser qwen3` и отдает размышления отдельным полем `reasoning_content`,
  выключаются они в запросе (пример ниже). Меняется одной переменной.
- **vLLM слушает только 127.0.0.1**: единственная точка входа nginx. Ключ проверяет сам vLLM
  (переменная `VLLM_API_KEY`, аналог `--api-key`: так ключ не виден в `ps` и аргументах
  контейнера), nginx дополнительно отбрасывает запросы без заголовка `Authorization`
  до того, как они дойдут до GPU.
- **Экспортеры в `network_mode: host`**: порты, опубликованные Docker через `ports:`,
  обходят ufw. С host-сетью доступ к метрикам действительно ограничивает ufw.
- **Автозапуск**: Docker включен в systemd, контейнеры с `restart: unless-stopped`,
  после перезагрузки сервера все поднимается само.

## Требования

**Сервер**

- Ubuntu 22.04 или 24.04 (x86_64). Ubuntu 26.04 можно разрешить через
  `supported_ubuntu_versions`, но на ней плейбук не проверялся.
- GPU NVIDIA архитектуры Turing и новее (T4, A10, A100, L4, L40S, H100, RTX 30/40/50, Blackwell),
  от 48 ГБ VRAM для модели по умолчанию (L40S, RTX 6000 Ada, A6000, A100, H100; FP8 нативно
  на Ada и Hopper). Для карт меньше модель меняется одной переменной (пример ниже).
  Для Volta и Pascal (V100, P100) задать `nvidia_driver_flavor: "server"`: открытые модули
  их не поддерживают.
- Доступ в интернет: репозитории Ubuntu, Docker, NVIDIA, Docker Hub, Hugging Face.
- 100+ ГБ свободного места: образ vLLM около 26 ГБ на диске, модель по умолчанию около 31 ГБ.
- Пользователь с sudo и SSH-доступом по ключу.
- Secure Boot выключен, либо готовность подписать модуль NVIDIA (иначе драйвер не загрузится).

**Машина, с которой запускается Ansible**

- Linux, macOS или WSL; `ansible-core` 2.16+ (в CI прогоняется на 2.21 и на 2.16 из пакета Ubuntu 24.04).
- В WSL проект на диске Windows (`/mnt/c`, `/mnt/d`) лежит в каталоге с правами на запись
  для всех, и Ansible игнорирует `ansible.cfg` из такого каталога. Либо скопировать проект
  в домашний каталог WSL, либо указать конфиг явно: `export ANSIBLE_CONFIG=$PWD/ansible.cfg`.
- Коллекция `community.general` (ufw, timezone): `ansible-galaxy collection install -r requirements.yml`.
  Свежие версии коллекции требуют ansible-core 2.18+, для 2.16 и 2.17 подходит 11.x:
  `ansible-galaxy collection install 'community.general:>=11.0.0,<12.0.0'`.

## Быстрый старт

```bash
# 1. Коллекции
ansible-galaxy collection install -r requirements.yml

# 2. Инвентарь
cp inventory/hosts.example.yml inventory/hosts.yml
#    указать ansible_host и ansible_user своего сервера
#    один раз зайти по SSH, чтобы ключ хоста попал в known_hosts (host_key_checking включен)
ssh <user>@<host> true

# 3. Секреты
cp group_vars/all/vault.yml.example group_vars/all/vault.yml
#    вписать vllm_api_key (например, вывод openssl rand -hex 32), при необходимости hf_token
ansible-vault encrypt group_vars/all/vault.yml

# 4. Запуск
ansible-playbook site.yml --ask-vault-pass
```

Первый прогон занимает от 15 минут до часа в зависимости от канала: установка драйвера
с перезагрузкой, скачивание образа vLLM и модели (около 31 ГБ). Повторный прогон ничего
не меняет (идемпотентность) и не перезагружает сервер.

Запуск по частям через теги:

```bash
ansible-playbook site.yml --ask-vault-pass --tags vllm,verify     # обновить модель/параметры и проверить
ansible-playbook site.yml --ask-vault-pass --tags verify          # только проверка
ansible-playbook site.yml --ask-vault-pass --skip-tags monitoring
```

## Переменные

Все настройки в [group_vars/all/main.yml](group_vars/all/main.yml), секреты в
`group_vars/all/vault.yml` (зашифрован ansible-vault). Для отдельного сервера значения
можно переопределить в `host_vars/<имя>.yml`.

| Переменная | По умолчанию | Назначение |
|---|---|---|
| `vllm_api_key` | (vault, обязательно) | ключ API, 16+ символов |
| `hf_token` | `""` (vault) | токен Hugging Face для моделей с гейтом |
| `nvidia_driver_version` | `"580"` | ветка драйвера или `auto` (выбор `ubuntu-drivers`) |
| `nvidia_driver_flavor` | `"server-open"` | `server-open`, `server` (Volta/Pascal), `open` или `""` |
| `nvidia_driver_keep_existing` | `true` | не трогать уже работающий драйвер |
| `nvidia_reboot_allowed` | `true` | разрешить перезагрузку после установки драйвера |
| `nvidia_driver_hold` | `true` | apt hold на пакеты драйвера |
| `nvidia_mig_disable` | `true` | выключить MIG, если он включен (с перезагрузкой) |
| `docker_gpu_test_image` | `"nvidia/cuda:13.0.2-base-ubuntu24.04"` | образ для проверки GPU в Docker |
| `vllm_image_tag` | `"v0.31.0"` | версия образа vLLM |
| `vllm_model` | `"Qwen/Qwen3.8-27B-FP8"` | модель с Hugging Face |
| `vllm_served_model_name` | `"qwen3.8-27b"` | имя модели в поле `model` запросов |
| `vllm_gpu_memory_utilization` | `0.90` | доля памяти GPU под vLLM |
| `vllm_max_model_len` | `32768` | максимальная длина контекста |
| `vllm_tensor_parallel_size` | `1` | число GPU для одной модели |
| `vllm_gpus` | `"all"` | какие GPU отдать контейнеру (`"0,1"`) |
| `vllm_extra_args` | `["--reasoning-parser", "qwen3"]` | дополнительные аргументы vLLM |
| `gateway_port` | `80` | порт nginx |
| `gateway_server_name` | `"_"` | домен API (`server_name` в nginx); `_` = любое имя и IP |
| `gateway_tls_enabled` | `false` | включить HTTPS (443) с редиректом с 80 |
| `gateway_tls_cert` / `gateway_tls_key` | `/etc/ssl/...` | где сертификат и ключ лежат на сервере |
| `gateway_tls_cert_src` / `gateway_tls_key_src` | `""` | свой сертификат и ключ (ansible-vault) на машине с Ansible, плейбук скопирует их на сервер |
| `gateway_tls_self_signed` | `true` | выпустить самоподписанный сертификат на домен, если файла нет и свой не задан |
| `gateway_rate_limit` / `gateway_rate_burst` | `10r/s` / `20` | rate limit на IP клиента |
| `gateway_client_max_body_size` | `1m` | максимальный размер тела запроса |
| `api_allowed_cidrs` | `[]` | откуда разрешен доступ к API (пусто = отовсюду) |
| `ssh_ufw_rule` | `"allow"` | правило ufw для SSH: `allow` или `limit` |
| `ufw_reset_on_first_run` | `true` | при первом прогоне сбросить правила ufw, с которыми пришел образ |
| `monitoring_enabled` | `true` | ставить dcgm-exporter и node_exporter |
| `metrics_allowed_cidrs` | `[]` | откуда разрешен сбор метрик, например `["10.0.5.20/32"]` (пусто = только с самого сервера) |
| `supported_ubuntu_versions` | `["22.04", "24.04"]` | разрешенные версии Ubuntu |

Примеры замены модели для отдельного сервера:

```yaml
# host_vars/gpu-01.yml: GPU 80 ГБ, полная BF16-версия той же модели
vllm_model: "Qwen/Qwen3.8-27B"
---
# host_vars/gpu-02.yml: GPU 24 ГБ, модель поменьше без размышлений
vllm_model: "Qwen/Qwen2.5-7B-Instruct"
vllm_served_model_name: "qwen2.5-7b-instruct"
vllm_extra_args: []
vllm_max_model_len: 16384
---
# host_vars/gpu-03.yml: модель с гейтом (нужен hf_token в vault)
vllm_model: "meta-llama/Llama-3.1-8B-Instruct"
vllm_served_model_name: "llama-3.1-8b-instruct"
vllm_extra_args: []
```

## Домен и HTTPS

По умолчанию `gateway_server_name: "_"` и TLS выключен: nginx отвечает по HTTP на любое имя
и на IP сервера. Для работы по домену нужна A-запись домена на IP сервера и две переменные
(в `group_vars/all/main.yml` или `host_vars/<имя>.yml`):

```yaml
gateway_server_name: "llm.example.com"
gateway_tls_enabled: true
```

ufw откроет порт 443, запросы по HTTP получат редирект на HTTPS. Сертификат подключается
одним из трех способов.

**Свой сертификат** (от CA или Let's Encrypt). Файлы лежат на машине, с которой запускается
Ansible, ключ зашифрован ansible-vault тем же паролем, что `vault.yml`. Плейбук копирует их
на сервер в `gateway_tls_cert` и `gateway_tls_key`, ключ расшифровывается при копировании.

```bash
mkdir -p files/tls                                # каталог в .gitignore
cp fullchain.pem files/tls/llm.example.com.crt    # сертификат с цепочкой
cp privkey.pem files/tls/llm.example.com.key
ansible-vault encrypt files/tls/llm.example.com.key
```

```yaml
gateway_tls_cert_src: "{{ playbook_dir }}/files/tls/llm.example.com.crt"
gateway_tls_key_src: "{{ playbook_dir }}/files/tls/llm.example.com.key"
```

**Самоподписанный** (по умолчанию, `gateway_tls_self_signed: true`). Выпускается на домен
(CN и subjectAltName) и подходит для проверки; клиенту нужен `curl -k` или `--cacert`.
Выпускается один раз: после смены домена удалить старый файл `gateway_tls_cert` на сервере.

**Уже на сервере** (например, certbot настроен отдельно): `gateway_tls_self_signed: false`
и пути к файлам в `gateway_tls_cert` и `gateway_tls_key`.

Запрос по домену:

```bash
curl -s https://llm.example.com/v1/models -H "Authorization: Bearer $LLM_API_KEY"
```

## Проверка работоспособности

Роль `verify` выполняется в конце каждого прогона и проверяет цепочку целиком:

1. `nvidia-smi` на хосте видит GPU и драйвер.
2. `/health` vLLM отвечает 200 (ожидание до 30 минут, пока скачивается и загружается модель).
3. Запрос без ключа через nginx получает 401.
4. `/v1/models` через nginx с ключом возвращает модель.
5. Тестовый `/v1/chat/completions` через nginx возвращает ответ; он выводится в лог прогона.

Ручная проверка на сервере:

```bash
nvidia-smi
docker compose -f /opt/vllm/docker-compose.yml ps     # STATUS: healthy
docker logs -f vllm
curl -s http://127.0.0.1:8000/health -o /dev/null -w '%{http_code}\n'
curl -s http://127.0.0.1:9400/metrics | grep -E '^DCGM_FI_(DEV_FB_USED|PROF_GR_ENGINE_ACTIVE)'
```

## Пример запроса к API

```bash
export LLM_HOST=203.0.113.10                     # IP сервера или домен
export LLM_API_KEY=<ключ из vault>                 # ansible-vault view group_vars/all/vault.yml

curl -s http://$LLM_HOST/v1/chat/completions \
  -H "Authorization: Bearer $LLM_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
        "model": "qwen3.8-27b",
        "messages": [
          {"role": "system", "content": "Ты полезный ассистент."},
          {"role": "user", "content": "Объясни в двух предложениях, что такое GPU."}
        ],
        "max_tokens": 200,
        "temperature": 0.7,
        "chat_template_kwargs": {"enable_thinking": false}
      }' | jq -r '.choices[0].message.content'
```

`chat_template_kwargs` выключает размышления, и модель сразу отвечает. Без него она сначала
размышляет: размышления приходят в поле `reasoning_content`, ответ в `content`, а `max_tokens`
нужен с запасом.

Стриминг (`"stream": true`) работает через nginx без буферизации:

```bash
curl -N http://$LLM_HOST/v1/chat/completions \
  -H "Authorization: Bearer $LLM_API_KEY" -H "Content-Type: application/json" \
  -d '{"model": "qwen3.8-27b", "stream": true,
       "chat_template_kwargs": {"enable_thinking": false},
       "messages": [{"role": "user", "content": "Привет!"}]}'
```

Тот же API подходит для любого OpenAI-клиента:

```python
from openai import OpenAI

client = OpenAI(base_url="http://203.0.113.10/v1", api_key="<ключ>")
resp = client.chat.completions.create(
    model="qwen3.8-27b",
    messages=[{"role": "user", "content": "Привет!"}],
    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
)
print(resp.choices[0].message.content)
```

## Проверка без GPU (CI)

Все, что не зависит от GPU, проверяет CI на GitHub Actions
([.github/workflows/ci.yml](.github/workflows/ci.yml)) на каждый push:

- **lint**: `yamllint`, `ansible-lint` (профиль production), `--syntax-check`, рендер
  шаблонов docker compose с переменными из `group_vars` и проверка `docker compose config`.
- **e2e**: плейбук с тегами `common`, `gateway`, `verify` применяется к одноразовой ВМ
  раннера (Ubuntu 22.04 и 24.04). Вместо vLLM работает заглушка
  [tests/mock_vllm.py](tests/mock_vllm.py) с тем же API и проверкой ключа. Три варианта:
  HTTP, HTTPS с самоподписанным сертификатом, HTTPS со своим сертификатом (ключ зашифрован
  ansible-vault), плюс прогон на ansible-core 2.16. Проверяются ufw (правило «провайдера»
  `3000:10000/tcp`, добавленное до прогона, снимается, остаются только правила плейбука), `nginx -t`, роль
  `verify` целиком (кроме `nvidia-smi`), 401 без ключа и с чужим ключом, 404 на `/docs` и
  `/metrics`, 429 при превышении rate limit, редирект на HTTPS, сертификат на домен
  ([tests/check_gateway.sh](tests/check_gateway.sh)) и то, что повторный прогон ничего
  не меняет (идемпотентность).

Драйвер, Docker с GPU, vLLM и экспортеры в CI не запускаются: для них нужна GPU.

Линтеры локально:

```bash
pip install -r requirements-dev.txt
ansible-galaxy collection install -r requirements.yml
yamllint --strict .
ansible-lint
ansible-playbook -i tests/inventory.yml site.yml --syntax-check
```

Прогон в check mode (`--check --diff`) ничего не меняет на сервере и показывает, что
изменилось бы. На уже развернутом сервере это проверка дрейфа конфигурации. На чистой ВМ
без GPU и без репозиториев Docker/NVIDIA осмысленно проверяется только базовая роль:

```bash
ansible-playbook site.yml --ask-vault-pass --check --diff --tags common
```

Задачи, которые меняют систему через `command` (docker compose, nvidia-ctk) или требуют
работающего сервиса (`nginx -t`, проверки HTTP), в check mode пропускаются явно.

## Безопасность

- Ключ API и токен HF только в ansible-vault; на сервере в `/opt/vllm/vllm.env` с правами `0600`;
  задачи с ключом помечены `no_log`.
- vLLM не доступен снаружи напрямую (порт только на 127.0.0.1), наружу отдаются только
  `/v1/*` и `/health`; `/metrics` и `/docs` vLLM закрыты.
- ufw: по умолчанию все входящие закрыты, открыты только SSH и порт nginx. Порты метрик
  открываются только для адресов из `metrics_allowed_cidrs`.
- Образы провайдеров бывают со своими правилами ufw, например с открытым диапазоном портов,
  куда попадают порты экспортеров. Поэтому первый прогон сбрасывает ufw
  (`ufw_reset_on_first_run`, старые правила ufw сохраняет в `/etc/ufw/*.rules.<дата>`) и
  ставит метку `/etc/ufw/.ansible-managed`; дальше ufw не сбрасывается.
- SSH по умолчанию `allow`, а не `limit`: ufw `limit` блокирует IP после 6 подключений
  за 30 секунд, и Ansible без ControlPersist (например, через `ssh.exe` Windows) упирается
  в этот лимит посреди прогона. С ControlPersist можно включить `ssh_ufw_rule: "limit"`.
- Rate limit и лимит тела запроса в nginx защищают GPU от перегрузки и простого DoS.
- Доступ к API можно ограничить сетями (`api_allowed_cidrs`) и перевести на HTTPS по домену
  (раздел «Домен и HTTPS»).

## Нюансы и диагностика

- **`unsatisfied condition: cuda>=13.0` в тесте GPU (роль `docker`)**: драйвер старше 580
  не подходит образу vLLM на CUDA 13.0. Обновить драйвер (`nvidia_driver_keep_existing: false`,
  нужна перезагрузка) или подобрать тег vLLM на CUDA 12.x (про `-cu129` см. «Ключевые решения»).
- **MIG включен без GPU-инстансов**: режим MIG хранится на самой карте, и у провайдеров он
  может остаться от прошлого арендатора. `nvidia-smi` видит GPU, а CUDA нет: vLLM падает с
  `No CUDA GPUs are available`, в `dmesg` строка `Channel allocation not allowed when MIG is
  enabled`, загрузка GPU `N/A`, метрики `DCGM_FI_DEV_GPU_UTIL` нет. Роль `nvidia_driver`
  выключает MIG и перезагружает сервер; чтобы оставить MIG, задать `nvidia_mig_disable: false`
  и MIG-инстанс в `vllm_gpus`.
- **vLLM не стартует с `Free memory on device ... is less than desired GPU memory utilization`**:
  часть памяти GPU занята другим процессом (рабочий стол в desktop-образе, другой контейнер).
  Уменьшить `vllm_gpu_memory_utilization`, например до `0.80`.
- **Secure Boot**: модуль, собранный через DKMS, не загрузится без подписи. Выключить Secure Boot
  или взять `nvidia_driver_version: auto` (ubuntu-drivers ставит подписанные Canonical модули,
  если они есть для текущего ядра).
- **Серверы HGX с NVSwitch (8 x A100/H100 SXM)**: дополнительно нужен
  `nvidia-fabricmanager` той же версии, что драйвер, иначе CUDA не инициализируется.
  Плейбук его не ставит; для одной SXM-карты в виртуальной машине он не нужен.
- **`Conflicting values set for option Signed-By` при `apt update`**: репозиторий Docker
  или NVIDIA уже подключен другим файлом с другим ключом (ставили вручную). Удалить старый
  файл из `/etc/apt/sources.list.d/` и запустить плейбук снова.
- **Смена `api_allowed_cidrs` с пустого списка на сети**: старое правило ufw «порт 80 для всех»
  остается, его нужно удалить вручную: `ufw delete allow 80/tcp`.
- **Rate limit считается по IP клиента**: за балансировщиком все запросы придут с одного IP,
  тогда нужен модуль `real_ip` в nginx.

## Что улучшить дальше

- **Let's Encrypt внутри плейбука**: выпуск certbot (webroot) по флагу и продление по таймеру;
  сейчас сертификат получают отдельно и подключают через `gateway_tls_cert_src`.
- **Ключи на клиента**: сейчас один общий ключ. Для нескольких команд поставить LiteLLM
  или API-gateway с отдельными ключами, квотами и учетом токенов.
- **Сбор метрик и алерты**: Prometheus + Grafana (дашборд DCGM, метрики vLLM `/metrics`:
  очередь запросов, время до первого токена, использование KV-кеша), алерты на перегрев GPU,
  ошибки XID, падение healthcheck.
- **Логи**: отправка логов nginx и vLLM в Loki/ELK.
- **Масштабирование**: несколько серверов за балансировщиком, несколько моделей на разных GPU
  (по контейнеру на модель с `vllm_gpus`), Kubernetes с GPU Operator при росте парка.
- **Быстрый холодный старт**: заранее скачанные модели (локальное зеркало HF или S3),
  чтобы новый сервер не качал веса из интернета.
- **Полный прогон в CI**: self-hosted раннер с GPU (или облачная GPU-ВМ на время прогона),
  чтобы проверять драйвер, Docker с GPU и vLLM, а не только шлюз.
- **Деплой из CI**: workflow с ручным запуском и подтверждением (GitHub environment),
  SSH-ключ и пароль vault в секретах.
- **HGX-серверы**: установка `nvidia-fabricmanager` по флагу, проверка NVLink (`nvidia-smi nvlink -s`).
- **Обновления без простоя**: rolling-обновление по серверам (`serial: 1`) с выводом узла
  из балансировщика.
