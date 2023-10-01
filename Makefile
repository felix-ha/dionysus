.PHONY: docker_build
docker_build:
	docker build . -t dl

.PHONY: test_release
test_release: docker_build
	docker run --name dl_release -dt dl
	docker exec dl_release /bin/bash ./bin/release_testpypi.sh
	docker stop dl_release
	docker rm dl_release

.PHONY: release
release: docker_build
	docker run --name dl_release -dt dl
	docker exec dl_release /bin/bash ./bin/release.sh
	docker stop dl_release
	docker rm dl_release
