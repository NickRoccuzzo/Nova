# // UNUSUAL VOLUME REPORT -- uses the 'options_dictionary[]'
from pull_option_chains import options_dictionary

# Thresholds from 'interpret_unusualness' function defined earlier we want to capture
unusual_volume_report_thresholds = {"Unusual", "Very Unusual", "🔥 Highly Unusual"}

# Unusual Volume Report will track the 'options_dictionary' for the most unusual contracts for this ticker
unusual_volume_report = []
for entry in options_dictionary:
    # check call side
    if entry["call_unusualness"] in unusual_volume_report_thresholds:
        unusual_volume_report.append({
            "expiration_date": entry["expiration_date"],
            "side": "call",
            **dict(zip(
                ["strike", "volume", "openInterest"],
                entry["call_with_the_largest_volume"]
            )),
            "unusualness": entry["call_unusualness"]
        })
    # check put side
    if entry["put_unusualness"] in unusual_volume_report_thresholds:
        unusual_volume_report.append({
            "expiration_date": entry["expiration_date"],
            "side": "put",
            **dict(zip(
                ["strike", "volume", "openInterest"],
                entry["put_with_the_largest_volume"]
            )),
            "unusualness": entry["put_unusualness"]
        })