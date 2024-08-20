.SILENT:

all:
	echo "Usage:"
	echo "	make image      # build Docker image"
	echo "	make up  		# start container"
	echo "	make down     	# stop container"
	echo "  make restart    # restart container"

image:
	docker build -t gddemotokens -f Dockerfile ..

up:
	USERID=$(shell id -u) GROUPID=$(shell id -g) USERNAME=$(shell id -un) GROUPNAME=$(shell id -gn) docker compose -f docker-compose.yml -p gddemotokens up -d 
down:
	docker-compose -p gddemotokens down

restart:
	docker-compose -p gddemotokens restart
