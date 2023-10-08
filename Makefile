.PHONY: docker_build
docker_build:
	docker build . -t dionysus

.PHONY: test_release
test_release: docker_build
	docker run --name dionysus_release -dt dionysus
	docker exec dionysus_release /bin/bash ./bin/release_testpypi.sh
	docker stop dionysus_release
	docker rm dionysus_release

.PHONY: release
release: docker_build
	docker run --name dionysus_release -dt dionysus
	docker exec dionysus_release /bin/bash ./bin/release.sh
	docker stop dionysus_release
	docker rm dionysus_release

.PHONY: flake8
flake8:
	flake8 . --count --exit-zero --max-complexity=10 --max-line-length=127 --show-source --statistics
