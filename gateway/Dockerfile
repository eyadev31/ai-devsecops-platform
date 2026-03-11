FROM nginx:alpine

# Update Alpine packages to fix vulnerabilities
RUN apk update && apk upgrade --no-cache

COPY nginx.conf /etc/nginx/nginx.conf
