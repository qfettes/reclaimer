# Hotel Reservation

A hotel reservation microservice, built with Go and gRPC, based on https://github.com/harlow/go-micro-services.

<!-- ## Application Structure -->

<!-- ![Social Network Architecture](socialNet_arch.png) -->

Supported actions: 
* Get profile and rates of nearby hotels available during given time periods
* Recommend hotels based on user provided metrics
* Place reservations

## Pre-requirements
- Docker
- Docker-compose
- luarocks (apt-get install luarocks)
- luasocket (luarocks install luasocket)

## Running the social network application
### Before you start
- Install Docker and Docker Compose.

### Start docker containers
docker stack deploy --compose-file docker-compose-swarm.yml hotel_reserv

#### workload generation
```bash
$WRK_DIR/wrk -D exp -t <num-threads> -c <num-conns> -d <duration> -L -s ./wrk2_lua_scripts/mixed-workload_type_1.lua http://x.x.x.x:5000 -R <reqs-per-sec>
```

### Questions and contact

You are welcome to submit a pull request if you find a bug or have extended the application in an interesting way. For any questions please contact us at: <microservices-bench-L@list.cornell.edu>
