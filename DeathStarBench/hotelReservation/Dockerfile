FROM golang:1.9

COPY . /go/src/github.com/harlow/go-micro-services
COPY src /go/src
WORKDIR /go/src/github.com/harlow/go-micro-services
RUN go get gopkg.in/mgo.v2
RUN go install github.com/apache/thrift/lib/go/thrift
RUN go get github.com/bradfitz/gomemcache/memcache
#RUN go mod init
#RUN go mod vendor
RUN go install -ldflags="-s -w" ./cmd/...
