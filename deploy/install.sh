#!/usr/bin/env bash
# Phase 1 deployer — run with: sudo bash /home/glen/dualing-simulation/deploy/install.sh
set -euo pipefail

if [[ $EUID -ne 0 ]]; then
  echo "Please run with sudo: sudo bash $0" >&2
  exit 1
fi

REPO=/home/glen/dualing-simulation
DOMAIN=smartenergylab.software

echo "==> Apt: ensure python venv, certbot, apache modules present"
apt-get update -qq
apt-get install -y -qq python3-venv python3-pip certbot python3-certbot-apache git

echo "==> Create venv (as user glen) and install Python deps"
if [[ ! -x ${REPO}/.venv/bin/pip ]]; then
  rm -rf ${REPO}/.venv
  runuser -u glen -- python3 -m venv ${REPO}/.venv
fi
runuser -u glen -- ${REPO}/.venv/bin/pip install --quiet --upgrade pip
runuser -u glen -- ${REPO}/.venv/bin/pip install --quiet -r ${REPO}/requirements.txt
runuser -u glen -- ${REPO}/.venv/bin/python -c "import fastapi, pvlib; print('  python deps OK')"

echo "==> Enable Apache reverse-proxy modules"
a2enmod proxy proxy_http headers >/dev/null

echo "==> Install Apache vhost for ${DOMAIN}"
install -m 644 ${REPO}/deploy/smartenergylab.conf /etc/apache2/sites-available/smartenergylab.conf
a2ensite smartenergylab.conf >/dev/null
# Default site competes for ServerName-less requests on :80 — keep it disabled
a2dissite 000-default.conf >/dev/null 2>&1 || true
apache2ctl configtest
systemctl reload apache2

echo "==> Install & enable systemd unit for the FastAPI app"
install -m 644 ${REPO}/deploy/dualing-simulation.service /etc/systemd/system/dualing-simulation.service
systemctl daemon-reload
systemctl enable --now dualing-simulation.service
sleep 1
systemctl --no-pager --full status dualing-simulation.service | sed -n '1,8p'

echo
echo "==> Local check via Apache (HTTP):"
curl -fsS -H "Host: ${DOMAIN}" http://127.0.0.1/dualing-simulation/api/health && echo

echo
echo "==> Attempting TLS via certbot --apache (requires DNS to resolve to this server)"
if certbot --apache -n --agree-tos --redirect \
    --email glen.morris@solarquip.com.au \
    -d ${DOMAIN} -d www.${DOMAIN}; then
  echo "==> TLS issued. Reloading apache."
  systemctl reload apache2
else
  echo "!! certbot failed — likely DNS hasn't propagated yet."
  echo "   Re-run this when 'dig +short ${DOMAIN}' returns 178.104.27.195:"
  echo "     sudo certbot --apache -n --agree-tos --redirect \\"
  echo "       --email glen.morris@solarquip.com.au \\"
  echo "       -d ${DOMAIN} -d www.${DOMAIN}"
fi

echo
echo "Done. Try: https://${DOMAIN}/dualing-simulation/"
