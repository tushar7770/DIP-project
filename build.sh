#!/bin/bash

# Install node.js
curl -sL https://deb.nodesource.com/setup_16.x | bash -
apt-get install -y nodejs

# Install npm and tailwindcss
npm install
npm install -g tailwindcss

# Compile Tailwind CSS
npx tailwindcss -i ./static/src/input.css -o ./static/dist/css/output.css --watch

gunicorn app:app