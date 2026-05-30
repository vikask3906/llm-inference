.PHONY: help demo traffic smoke down logs test bench

GW ?= http://localhost:8000

help:  ## list available targets
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-9s\033[0m %s\n", $$1, $$2}'

demo:  ## up full stack (gateway+mocks+prometheus+grafana), wait healthy, drive traffic
	python scripts/demo.py

traffic:  ## drive 60s of prefix-heavy traffic at a running gateway
	python bench/loadtest.py --url $(GW) --duration 60

smoke:  ## one streaming request against a running gateway
	curl -N $(GW)/v1/chat/completions -H 'content-type: application/json' \
		-d '{"model":"mock-model","messages":[{"role":"user","content":"hello"}],"stream":true}'

cluster-demo:  ## 2 gateway replicas behind an LB sharing prefix state via Redis
	docker compose -f docker-compose.cluster.yml up --build

cluster-down:  ## tear the cluster stack down
	docker compose -f docker-compose.cluster.yml down -v

down:  ## tear the stack down (incl. volumes)
	docker compose down -v

logs:  ## tail gateway logs
	docker compose logs -f gateway

test:  ## run the test suite
	python -m pytest -q

bench:  ## run the routing benchmark matrix (charts + CSV + markdown)
	python bench/matrix.py

bench-all:  ## run every standalone-package benchmark (matrix + dag + rag + admission + disagg + multimodal)
	python bench/matrix.py
	python bench/dag_bench.py
	python bench/rag_bench.py
	python bench/admission_bench.py
	python bench/disagg_bench.py
	python bench/multimodal_bench.py

bench-dag:  ## DAG-scheduler benchmark: locality vs round-robin
	python bench/dag_bench.py

bench-rag:  ## RAG benchmark: chunk-affinity vs cache-blind
	python bench/rag_bench.py

bench-admission:  ## Admission-control benchmark: gold protected under pressure
	python bench/admission_bench.py

bench-disagg:  ## Disaggregation benchmark: adaptive vs static colocate/split
	python bench/disagg_bench.py

bench-multimodal:  ## Multimodal benchmark: media-affinity vs cache-blind
	python bench/multimodal_bench.py
