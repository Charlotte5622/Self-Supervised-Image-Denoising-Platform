#!/usr/bin/env sh
set -eu

mkdir -p \
    /app/media/uploads \
    /app/media/results \
    /app/media/intermediate \
    /app/media/preset \
    /app/models/weights \
    /app/staticfiles

python manage.py collectstatic --noinput

exec "$@"
