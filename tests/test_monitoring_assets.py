import json
from pathlib import Path


def test_prometheus_scrapes_api_metrics_endpoint() -> None:
    config = Path("monitoring/prometheus.yml").read_text(encoding="utf-8")

    assert "job_name: electrical-rag-api" in config
    assert "metrics_path: /metrics" in config
    assert "rag-api:8000" in config


def test_grafana_dashboard_has_expected_panels_and_datasource() -> None:
    dashboard = json.loads(
        Path("monitoring/grafana/dashboards/electrical-rag-overview.json").read_text(
            encoding="utf-8"
        )
    )
    titles = {panel["title"] for panel in dashboard["panels"]}

    assert dashboard["uid"] == "electrical_rag-rag-overview"
    assert "P95 RAG Stage Latency" in titles
    assert "Redis Cache Hit Ratio" in titles
    assert all(panel["datasource"]["uid"] == "prometheus" for panel in dashboard["panels"])


def test_grafana_provisioning_tree_contains_all_expected_sections() -> None:
    provisioning = Path("monitoring/grafana/provisioning")

    assert (provisioning / "datasources/prometheus.yml").is_file()
    assert (provisioning / "dashboards/dashboards.yml").is_file()
    assert (provisioning / "plugins/plugins.yml").is_file()
    assert (provisioning / "alerting/alerting.yml").is_file()
