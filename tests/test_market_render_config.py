from pathlib import Path


def test_market_cron_disables_bounded_provider_debugging_by_default():
    config = Path("render.yaml").read_text()
    market_cron = config.split("name: market-intelligence-refresh", 1)[1].split("- type: cron", 1)[0]
    assert "key: MARKET_DEBUG_PROVIDER_RESPONSES" in market_cron
    assert 'value: "0"' in market_cron
