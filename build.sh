#!/bin/bash

# brew install node

# Install tailwindcss globally
npm install -g tailwindcss

# Compile Tailwind CSS
npx tailwindcss -i ./static/src/input.css -o ./static/dist/css/output.css 

# gunicorn app:app