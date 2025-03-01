import json
import statistics

def build_play_dictionaries(json_file_path):
    """
    Reads summary_results2.json (or any similarly structured file) and returns
    six dictionaries based on the specified sorting criteria:

    1. bullish_plays_dict    -> Top 10 tickers by highest 'score'
    2. bearish_plays_dict    -> Top 10 tickers by lowest 'score'
    3. unusual_activity_dict -> Top 10 tickers by highest 'unusual_contracts_count'
    4. money_flow_dict       -> Top 10 tickers by highest 'total_unusual_spent'
    5. call_flow_dict        -> Top 10 tickers by highest (calls / puts) ratio
    6. put_flow_dict         -> Top 10 tickers by highest (puts / calls) ratio
    """

    def parse_dollar_str(dollar_str):
        # Converts something like "$3,900.0" -> 3900.0
        return float(dollar_str.replace('$', '').replace(',', ''))

    def parse_price_str(price_str):
        # Converts something like "$32.59" -> 32.59
        return float(price_str.replace('$', '').replace(',', ''))

    # Load the JSON data
    with open(json_file_path, 'r') as f:
        data = json.load(f)

    all_tickers = data.get("all_tickers", {})
    ticker_data_list = []

    # Parse each ticker and gather the needed numeric fields
    for ticker, info in all_tickers.items():
        score = float(info["score"])
        unusual_count = float(info["unusual_contracts_count"])

        total_unusual_spent = parse_dollar_str(info["total_unusual_spent"])
        calls_spent = parse_dollar_str(info["cumulative_total_spent_calls"])
        puts_spent = parse_dollar_str(info["cumulative_total_spent_puts"])

        current_price_str = info.get("current_price", "0")
        current_price = parse_price_str(current_price_str) if current_price_str else 0.0
        company_name = info.get("company_name", "Unknown Company")

        # Safely compute calls_to_puts_ratio
        # If puts = 0 or calls = 0, set ratio to 0 to avoid 'inf'
        if puts_spent > 0 and calls_spent > 0:
            calls_to_puts_ratio = calls_spent / puts_spent
            puts_to_calls_ratio = puts_spent / calls_spent
        else:
            calls_to_puts_ratio = 0
            puts_to_calls_ratio = 0

        ticker_data_list.append({
            "ticker": ticker,
            "score": score,
            "unusual_contracts_count": unusual_count,
            "total_unusual_spent": total_unusual_spent,
            "cumulative_total_spent_calls": calls_spent,
            "cumulative_total_spent_puts": puts_spent,
            "calls_to_puts_ratio": calls_to_puts_ratio,
            "puts_to_calls_ratio": puts_to_calls_ratio,
            "current_price": current_price,
            "company_name": company_name
        })

    # 1) Bullish plays by score (descending)
    sorted_by_score_desc = sorted(
        ticker_data_list, key=lambda x: x["score"], reverse=True
    )[:10]
    bullish_plays_dict = {
        item["ticker"]: {
            "score": item["score"],
            "unusual_contracts_count": item["unusual_contracts_count"],
            "total_unusual_spent": item["total_unusual_spent"],
            "cumulative_total_spent_calls": item["cumulative_total_spent_calls"],
            "cumulative_total_spent_puts": item["cumulative_total_spent_puts"],
            "current_price": item["current_price"],
            "company_name": item["company_name"]
        }
        for item in sorted_by_score_desc
    }

    # 2) Bearish plays by score (ascending)
    sorted_by_score_asc = sorted(
        ticker_data_list, key=lambda x: x["score"]
    )[:10]
    bearish_plays_dict = {
        item["ticker"]: {
            "score": item["score"],
            "unusual_contracts_count": item["unusual_contracts_count"],
            "total_unusual_spent": item["total_unusual_spent"],
            "cumulative_total_spent_calls": item["cumulative_total_spent_calls"],
            "cumulative_total_spent_puts": item["cumulative_total_spent_puts"],
            "current_price": item["current_price"],
            "company_name": item["company_name"]
        }
        for item in sorted_by_score_asc
    }

    # 3) Most unusual activity by count (descending)
    sorted_by_unusual_count_desc = sorted(
        ticker_data_list, key=lambda x: x["unusual_contracts_count"], reverse=True
    )[:10]
    unusual_activity_dict = {
        item["ticker"]: {
            "score": item["score"],
            "unusual_contracts_count": item["unusual_contracts_count"],
            "total_unusual_spent": item["total_unusual_spent"],
            "cumulative_total_spent_calls": item["cumulative_total_spent_calls"],
            "cumulative_total_spent_puts": item["cumulative_total_spent_puts"],
            "current_price": item["current_price"],
            "company_name": item["company_name"]
        }
        for item in sorted_by_unusual_count_desc
    }

    # 4) Money flow by total_unusual_spent (descending)
    sorted_by_money_flow_desc = sorted(
        ticker_data_list, key=lambda x: x["total_unusual_spent"], reverse=True
    )[:10]
    money_flow_dict = {
        item["ticker"]: {
            "score": item["score"],
            "unusual_contracts_count": item["unusual_contracts_count"],
            "total_unusual_spent": item["total_unusual_spent"],
            "cumulative_total_spent_calls": item["cumulative_total_spent_calls"],
            "cumulative_total_spent_puts": item["cumulative_total_spent_puts"],
            "current_price": item["current_price"],
            "company_name": item["company_name"]
        }
        for item in sorted_by_money_flow_desc
    }

    # 5) Call flow -> descending calls_to_puts_ratio
    sorted_by_call_flow_desc = sorted(
        ticker_data_list, key=lambda x: x["calls_to_puts_ratio"], reverse=True
    )[:10]
    call_flow_dict = {
        item["ticker"]: {
            "score": item["score"],
            "unusual_contracts_count": item["unusual_contracts_count"],
            "total_unusual_spent": item["total_unusual_spent"],
            "cumulative_total_spent_calls": item["cumulative_total_spent_calls"],
            "cumulative_total_spent_puts": item["cumulative_total_spent_puts"],
            "calls_to_puts_ratio": item["calls_to_puts_ratio"],
            "current_price": item["current_price"],
            "company_name": item["company_name"]
        }
        for item in sorted_by_call_flow_desc
    }

    # 6) Put flow -> descending puts_to_calls_ratio
    sorted_by_put_flow_desc = sorted(
        ticker_data_list, key=lambda x: x["puts_to_calls_ratio"], reverse=True
    )[:10]
    put_flow_dict = {
        item["ticker"]: {
            "score": item["score"],
            "unusual_contracts_count": item["unusual_contracts_count"],
            "total_unusual_spent": item["total_unusual_spent"],
            "cumulative_total_spent_calls": item["cumulative_total_spent_calls"],
            "cumulative_total_spent_puts": item["cumulative_total_spent_puts"],
            "puts_to_calls_ratio": item["puts_to_calls_ratio"],
            "current_price": item["current_price"],
            "company_name": item["company_name"]
        }
        for item in sorted_by_put_flow_desc
    }

    return (
        bullish_plays_dict,
        bearish_plays_dict,
        unusual_activity_dict,
        money_flow_dict,
        call_flow_dict,
        put_flow_dict
    )


def build_oi_volume_dictionaries(json_file_path, ratio_threshold=3.0, whale_threshold=4.0):
    """
    This function returns EIGHT dictionaries in total:

    1. most_volume_puts_dict   -> Top 10 tickers by SUM(puts_volume)
    2. most_volume_calls_dict  -> Top 10 tickers by SUM(calls_volume)
    3. highest_ratio_calls_oi_dict -> Tickers whose SUM(calls_oi)/SUM(puts_oi) >= ratio_threshold,
                                      then top 10 by ratio
    4. highest_ratio_puts_oi_dict  -> Tickers whose SUM(puts_oi)/SUM(calls_oi) >= ratio_threshold,
                                      then top 10 by ratio
    5. whale_call_dict -> Tickers that have daily outliers or a sum outlier in calls_volume
    6. whale_put_dict  -> Tickers that have daily outliers or a sum outlier in puts_volume

    7. whale_call_oi_dict -> Tickers that have daily outliers or a sum outlier in calls_oi
    8. whale_put_oi_dict  -> Tickers that have daily outliers or a sum outlier in puts_oi

    Outlier Logic (Volumes and OI):
      A) Daily outlier if ANY single date's value > mean + (whale_threshold * stdev).
      B) Sum outlier if the total across all dates is > universe_mean + (whale_threshold * universe_stdev).
    """

    # 1) Load data
    with open(json_file_path, 'r') as f:
        data = json.load(f)

    all_tickers = data.get("all_tickers", {})

    def parse_price_str(price_str):
        return float(price_str.replace('$', '').replace(',', '')) if price_str else 0.0

    # -------------------------------------------------------------------------
    # STEP A: GLOBAL SUM OUTLIER DETECTION THRESHOLDS (Volume + Open Interest)
    # -------------------------------------------------------------------------
    # Gather sums for calls_volume & puts_volume across all tickers
    calls_sums_for_all_tickers = []
    puts_sums_for_all_tickers  = []
    # Also gather sums for calls_oi & puts_oi across all tickers
    calls_oi_sums_for_all_tickers = []
    puts_oi_sums_for_all_tickers   = []

    for ticker, info in all_tickers.items():
        # Volume sums
        calls_vol = sum(info.get("calls_volume", {}).values())
        puts_vol  = sum(info.get("puts_volume", {}).values())
        calls_sums_for_all_tickers.append(calls_vol)
        puts_sums_for_all_tickers.append(puts_vol)

        # OI sums
        calls_oi = sum(info.get("calls_oi", {}).values())
        puts_oi  = sum(info.get("puts_oi", {}).values())
        calls_oi_sums_for_all_tickers.append(calls_oi)
        puts_oi_sums_for_all_tickers.append(puts_oi)

    # Universe mean & stdev for volume sums (calls + puts)
    mean_sum_calls_vol = statistics.mean(calls_sums_for_all_tickers) if calls_sums_for_all_tickers else 0
    stdev_sum_calls_vol = statistics.pstdev(calls_sums_for_all_tickers) if len(calls_sums_for_all_tickers) > 1 else 0
    mean_sum_puts_vol = statistics.mean(puts_sums_for_all_tickers) if puts_sums_for_all_tickers else 0
    stdev_sum_puts_vol = statistics.pstdev(puts_sums_for_all_tickers) if len(puts_sums_for_all_tickers) > 1 else 0

    sum_calls_vol_threshold = mean_sum_calls_vol + whale_threshold * stdev_sum_calls_vol
    sum_puts_vol_threshold  = mean_sum_puts_vol + whale_threshold * stdev_sum_puts_vol

    # Universe mean & stdev for OI sums (calls + puts)
    mean_sum_calls_oi = statistics.mean(calls_oi_sums_for_all_tickers) if calls_oi_sums_for_all_tickers else 0
    stdev_sum_calls_oi = statistics.pstdev(calls_oi_sums_for_all_tickers) if len(calls_oi_sums_for_all_tickers) > 1 else 0
    mean_sum_puts_oi = statistics.mean(puts_oi_sums_for_all_tickers) if puts_oi_sums_for_all_tickers else 0
    stdev_sum_puts_oi = statistics.pstdev(puts_oi_sums_for_all_tickers) if len(puts_oi_sums_for_all_tickers) > 1 else 0

    sum_calls_oi_threshold = mean_sum_calls_oi + whale_threshold * stdev_sum_calls_oi
    sum_puts_oi_threshold  = mean_sum_puts_oi + whale_threshold * stdev_sum_puts_oi

    # ---------------------------------------------------------------------
    # STEP B: BUILD TICKER DATA FOR 'SUM' / 'RATIO' DICTIONARIES (OI/Volume)
    # ---------------------------------------------------------------------
    ticker_data_oi_vol = []
    for ticker, info in all_tickers.items():
        calls_oi_dict   = info.get("calls_oi", {})
        puts_oi_dict    = info.get("puts_oi", {})
        calls_vol_dict  = info.get("calls_volume", {})
        puts_vol_dict   = info.get("puts_volume", {})

        current_price_str = info.get("current_price", "0")
        current_price = parse_price_str(current_price_str)
        company_name = info.get("company_name", "Unknown Company")

        # Sums
        sum_calls_oi  = sum(calls_oi_dict.values())
        sum_puts_oi   = sum(puts_oi_dict.values())
        sum_calls_vol = sum(calls_vol_dict.values())
        sum_puts_vol  = sum(puts_vol_dict.values())

        # Ratios (OI)
        calls_oi_ratio = (sum_calls_oi / sum_puts_oi) if sum_puts_oi > 0 else 0
        puts_oi_ratio  = (sum_puts_oi / sum_calls_oi) if sum_calls_oi > 0 else 0

        ticker_data_oi_vol.append({
            "ticker": ticker,
            "sum_calls_oi":  sum_calls_oi,
            "sum_puts_oi":   sum_puts_oi,
            "sum_calls_vol": sum_calls_vol,
            "sum_puts_vol":  sum_puts_vol,
            "calls_oi_ratio": calls_oi_ratio,
            "puts_oi_ratio":  puts_oi_ratio,
            "current_price":  current_price,
            "company_name":   company_name
        })

    # ---------------------------------------------------------------------
    # STEP C: BUILD THE FIRST 4 DICTIONARIES (AS BEFORE)
    # ---------------------------------------------------------------------
    # 1) most_volume_puts_dict -> Top 10 by sum_puts_vol
    sorted_puts_vol = sorted(
        ticker_data_oi_vol, key=lambda x: x["sum_puts_vol"], reverse=True
    )[:10]
    most_volume_puts_dict = {
        item["ticker"]: {
            "sum_puts_vol": item["sum_puts_vol"],
            "sum_calls_vol": item["sum_calls_vol"],
            "sum_calls_oi": item["sum_calls_oi"],
            "sum_puts_oi": item["sum_puts_oi"],
            "current_price": item["current_price"],
            "company_name": item["company_name"]
        }
        for item in sorted_puts_vol
    }

    # 2) most_volume_calls_dict -> Top 10 by sum_calls_vol
    sorted_calls_vol = sorted(
        ticker_data_oi_vol, key=lambda x: x["sum_calls_vol"], reverse=True
    )[:10]
    most_volume_calls_dict = {
        item["ticker"]: {
            "sum_calls_vol": item["sum_calls_vol"],
            "sum_puts_vol": item["sum_puts_vol"],
            "sum_calls_oi": item["sum_calls_oi"],
            "sum_puts_oi": item["sum_puts_oi"],
            "current_price": item["current_price"],
            "company_name": item["company_name"]
        }
        for item in sorted_calls_vol
    }

    # 3) highest_ratio_calls_oi_dict -> ratio = sum_calls_oi / sum_puts_oi >= ratio_threshold
    filtered_calls_oi_ratio = [t for t in ticker_data_oi_vol if t["calls_oi_ratio"] >= ratio_threshold]
    sorted_calls_oi_ratio = sorted(filtered_calls_oi_ratio, key=lambda x: x["calls_oi_ratio"], reverse=True)[:10]
    highest_ratio_calls_oi_dict = {
        item["ticker"]: {
            "calls_oi_ratio": item["calls_oi_ratio"],
            "sum_calls_oi":   item["sum_calls_oi"],
            "sum_puts_oi":    item["sum_puts_oi"],
            "current_price":  item["current_price"],
            "company_name":   item["company_name"]
        }
        for item in sorted_calls_oi_ratio
    }

    # 4) highest_ratio_puts_oi_dict -> ratio = sum_puts_oi / sum_calls_oi >= ratio_threshold
    filtered_puts_oi_ratio = [t for t in ticker_data_oi_vol if t["puts_oi_ratio"] >= ratio_threshold]
    sorted_puts_oi_ratio = sorted(filtered_puts_oi_ratio, key=lambda x: x["puts_oi_ratio"], reverse=True)[:10]
    highest_ratio_puts_oi_dict = {
        item["ticker"]: {
            "puts_oi_ratio": item["puts_oi_ratio"],
            "sum_puts_oi":   item["sum_puts_oi"],
            "sum_calls_oi":  item["sum_calls_oi"],
            "current_price": item["current_price"],
            "company_name":  item["company_name"]
        }
        for item in sorted_puts_oi_ratio
    }

    # ---------------------------------------------------------------------
    # STEP D: Whale detection for calls_volume & puts_volume (same as before)
    #         + Whale detection for calls_oi & puts_oi (new)
    # ---------------------------------------------------------------------
    whale_call_dict = {}
    whale_put_dict  = {}
    whale_call_oi_dict = {}
    whale_put_oi_dict  = {}

    for ticker, info in all_tickers.items():
        # ------- Common fields -------
        current_price_str = info.get("current_price", "0")
        current_price = parse_price_str(current_price_str)
        company_name = info.get("company_name", "Unknown Company")

        # ------- (1) Volume-based outliers -------
        calls_vol_dict = info.get("calls_volume", {})
        puts_vol_dict  = info.get("puts_volume", {})

        calls_values = list(calls_vol_dict.values())
        puts_values  = list(puts_vol_dict.values())

        # (1-A) calls_volume daily outliers
        if calls_values:
            mean_calls_daily = statistics.mean(calls_values)
            stdev_calls_daily = statistics.pstdev(calls_values) if len(calls_values) > 1 else 0
            daily_threshold_calls = mean_calls_daily + whale_threshold * stdev_calls_daily

            # Identify outlier dates
            call_outlier_points = [
                (d, v) for d, v in calls_vol_dict.items() if v > daily_threshold_calls
            ]
            call_outlier_points.sort(key=lambda x: x[1], reverse=True)

            # sum outlier check
            total_calls_volume = sum(calls_values)
            sum_outlier_calls = (total_calls_volume > sum_calls_vol_threshold)

            if call_outlier_points or sum_outlier_calls:
                whale_call_dict[ticker] = {
                    "calls_volume": calls_vol_dict,
                    "daily_outliers": [
                        {"date": d, "volume": v} for (d, v) in call_outlier_points
                    ],
                    "mean_calls_volume": mean_calls_daily,
                    "stdev_calls_volume": stdev_calls_daily,
                    "daily_threshold_calls": daily_threshold_calls,
                    "total_calls_volume": total_calls_volume,
                    "universe_calls_mean": mean_sum_calls_vol,
                    "universe_calls_stdev": stdev_sum_calls_vol,
                    "universe_calls_threshold": sum_calls_vol_threshold,
                    "sum_outlier_triggered": sum_outlier_calls,
                    "current_price": current_price,
                    "company_name": company_name
                }

        # (1-B) puts_volume daily outliers
        if puts_values:
            mean_puts_daily = statistics.mean(puts_values)
            stdev_puts_daily = statistics.pstdev(puts_values) if len(puts_values) > 1 else 0
            daily_threshold_puts = mean_puts_daily + whale_threshold * stdev_puts_daily

            put_outlier_points = [
                (d, v) for d, v in puts_vol_dict.items() if v > daily_threshold_puts
            ]
            put_outlier_points.sort(key=lambda x: x[1], reverse=True)

            total_puts_volume = sum(puts_values)
            sum_outlier_puts = (total_puts_volume > sum_puts_vol_threshold)

            if put_outlier_points or sum_outlier_puts:
                whale_put_dict[ticker] = {
                    "puts_volume": puts_vol_dict,
                    "daily_outliers": [
                        {"date": d, "volume": v} for (d, v) in put_outlier_points
                    ],
                    "mean_puts_volume": mean_puts_daily,
                    "stdev_puts_volume": stdev_puts_daily,
                    "daily_threshold_puts": daily_threshold_puts,
                    "total_puts_volume": total_puts_volume,
                    "universe_puts_mean": mean_sum_puts_vol,
                    "universe_puts_stdev": stdev_sum_puts_vol,
                    "universe_puts_threshold": sum_puts_vol_threshold,
                    "sum_outlier_triggered": sum_outlier_puts,
                    "current_price": current_price,
                    "company_name": company_name
                }

        # ------- (2) OI-based outliers -------
        calls_oi_dict = info.get("calls_oi", {})
        puts_oi_dict  = info.get("puts_oi", {})

        calls_oi_values = list(calls_oi_dict.values())
        puts_oi_values  = list(puts_oi_dict.values())

        # (2-A) calls_oi daily outliers
        if calls_oi_values:
            mean_calls_oi_daily = statistics.mean(calls_oi_values)
            stdev_calls_oi_daily = statistics.pstdev(calls_oi_values) if len(calls_oi_values) > 1 else 0
            daily_threshold_calls_oi = mean_calls_oi_daily + whale_threshold * stdev_calls_oi_daily

            call_oi_outlier_points = [
                (d, v) for d, v in calls_oi_dict.items() if v > daily_threshold_calls_oi
            ]
            call_oi_outlier_points.sort(key=lambda x: x[1], reverse=True)

            total_calls_oi = sum(calls_oi_values)
            sum_outlier_calls_oi = (total_calls_oi > sum_calls_oi_threshold)

            if call_oi_outlier_points or sum_outlier_calls_oi:
                whale_call_oi_dict[ticker] = {
                    "calls_oi": calls_oi_dict,
                    "daily_outliers": [
                        {"date": d, "oi": v} for (d, v) in call_oi_outlier_points
                    ],
                    "mean_calls_oi": mean_calls_oi_daily,
                    "stdev_calls_oi": stdev_calls_oi_daily,
                    "daily_threshold_calls_oi": daily_threshold_calls_oi,
                    "total_calls_oi": total_calls_oi,
                    "universe_calls_oi_mean": mean_sum_calls_oi,
                    "universe_calls_oi_stdev": stdev_sum_calls_oi,
                    "universe_calls_oi_threshold": sum_calls_oi_threshold,
                    "sum_outlier_triggered": sum_outlier_calls_oi,
                    "current_price": current_price,
                    "company_name": company_name
                }

        # (2-B) puts_oi daily outliers
        if puts_oi_values:
            mean_puts_oi_daily = statistics.mean(puts_oi_values)
            stdev_puts_oi_daily = statistics.pstdev(puts_oi_values) if len(puts_oi_values) > 1 else 0
            daily_threshold_puts_oi = mean_puts_oi_daily + whale_threshold * stdev_puts_oi_daily

            put_oi_outlier_points = [
                (d, v) for d, v in puts_oi_dict.items() if v > daily_threshold_puts_oi
            ]
            put_oi_outlier_points.sort(key=lambda x: x[1], reverse=True)

            total_puts_oi = sum(puts_oi_values)
            sum_outlier_puts_oi = (total_puts_oi > sum_puts_oi_threshold)

            if put_oi_outlier_points or sum_outlier_puts_oi:
                whale_put_oi_dict[ticker] = {
                    "puts_oi": puts_oi_dict,
                    "daily_outliers": [
                        {"date": d, "oi": v} for (d, v) in put_oi_outlier_points
                    ],
                    "mean_puts_oi": mean_puts_oi_daily,
                    "stdev_puts_oi": stdev_puts_oi_daily,
                    "daily_threshold_puts_oi": daily_threshold_puts_oi,
                    "total_puts_oi": total_puts_oi,
                    "universe_puts_oi_mean": mean_sum_puts_oi,
                    "universe_puts_oi_stdev": stdev_sum_puts_oi,
                    "universe_puts_oi_threshold": sum_puts_oi_threshold,
                    "sum_outlier_triggered": sum_outlier_puts_oi,
                    "current_price": current_price,
                    "company_name": company_name
                }

    # Return all eight dictionaries
    return (
        most_volume_puts_dict,
        most_volume_calls_dict,
        highest_ratio_calls_oi_dict,
        highest_ratio_puts_oi_dict,
        whale_call_dict,
        whale_put_dict,
        whale_call_oi_dict,
        whale_put_oi_dict
    )
