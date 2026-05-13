from app.anomaly import detect_anomalies
from app.log_parsing import parse_zscaler_log


SAMPLE_WITH_HEADER = (
    "datetime,user,url,urlcategory,action,threatname,useragent,sourceip,"
    "requestmethod,responsesize,status\n"
    "2026-05-07T08:00:01Z,alice,https://docs.google.com,Office365,Allowed,-,"
    "Mozilla/5.0,10.0.0.12,GET,1234,200\n"
    "2026-05-07T08:01:00Z,bob,https://malware.example/x.exe,Malware,Blocked,"
    "Trojan.Test,Wget,10.0.0.99,GET,0,403\n"
)


def test_parse_zscaler_with_header():
    entries, skipped = parse_zscaler_log(SAMPLE_WITH_HEADER)
    assert skipped == 0
    assert len(entries) == 2

    a = entries[0]
    assert a["source_ip"] == "10.0.0.12"
    assert a["url_category"] == "Office365"
    assert a["status_code"] == 200
    assert a["action"] == "Allowed"
    assert a["timestamp"] is not None

    b = entries[1]
    assert b["action"] == "Blocked"
    assert b["threat_name"] == "Trojan.Test"


def test_detect_anomalies_flags_threats():
    entries, _ = parse_zscaler_log(SAMPLE_WITH_HEADER)
    annotated = detect_anomalies(entries)

    by_action = {e["action"]: e for e in annotated}
    assert by_action["Blocked"]["is_anomaly"] is True
    assert by_action["Blocked"]["anomaly_score"] >= 0.5
    assert by_action["Allowed"]["is_anomaly"] is False


def test_parse_handles_empty_input():
    assert parse_zscaler_log("") == ([], 0)
    assert parse_zscaler_log("   \n  \n") == ([], 0)


def test_parse_tolerates_dash_placeholder():
    content = (
        "datetime,sourceip,url,action,threatname,useragent\n"
        "2026-01-01T00:00:00Z,-,https://example.com,Allowed,-,Mozilla\n"
    )
    entries, _ = parse_zscaler_log(content)
    assert len(entries) == 1
    assert entries[0]["source_ip"] is None
    assert entries[0]["threat_name"] is None
