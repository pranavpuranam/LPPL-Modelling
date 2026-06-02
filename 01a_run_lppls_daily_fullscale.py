# 01a: run daily positive lppl fits

import os  # handle file paths
import pandas as pd  # handle dataframes
import numpy as np  # numerical operations
import matplotlib.pyplot as plt  # plot charts
from scipy.optimize import differential_evolution, minimize  # optimise lppl fits

PRICE_CSV = "eurostoxx600_prices.csv"  # input price data
OUTPUT_CSV = "eurostoxx600_daily_positive_lppls_results.csv"  # raw fit output
CONFIDENCE_CSV = "eurostoxx600_daily_positive_lppls_confidence.csv"  # confidence output
DATE_COL = "Date"  # date column
Y_COL = "log_price"  # fitted series
M_BOUNDS = (0.1, 0.9)  # m bounds
OMEGA_BOUNDS = (6.0, 13.0)  # omega bounds
DT_MIN = 125  # shortest window
DT_MAX = 750  # longest window
DT_STEP = 25  # window step
T2_STEP = 1  # endpoint step
TC_MIN_DAYS = 1  # minimum tc after t2
TC_MAX_DAYS = 500  # maximum tc after t2
SEED = 42  # random seed
MAXITER = 5000  # local optimiser iterations
REQUIRE_FULL_SCALE = True  # require full window range

def load_prices(csv_file=PRICE_CSV):
    df = pd.read_csv(csv_file)  # load price data
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])  # parse dates
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")  # force numeric prices
    df["log_price"] = pd.to_numeric(df["log_price"], errors="coerce")  # force numeric log prices
    return (
        df.dropna(subset=[DATE_COL, "Close", "log_price"])
        .sort_values(DATE_COL)
        .reset_index(drop=True)
    )  # clean and sort data

def generate_rolling_lppl_windows(
    df,
    date_col=DATE_COL,
    t2_step=T2_STEP,
    dt_min=DT_MIN,
    dt_max=DT_MAX,
    dt_step=DT_STEP,
    require_full_scale=REQUIRE_FULL_SCALE,
):
    df = df.copy()  # avoid editing original
    df[date_col] = pd.to_datetime(df[date_col])  # parse dates
    df = df.sort_values(date_col).reset_index(drop=True)  # sort chronologically
    rows = []  # store windows
    t2_start_idx = dt_max if require_full_scale else dt_min  # set first endpoint
    for t2_idx in range(t2_start_idx, len(df), t2_step):  # loop over endpoints
        t2_date = df.loc[t2_idx, date_col]  # get endpoint date
        for dt in range(dt_min, dt_max + 1, dt_step):  # loop over window lengths
            t1_idx = t2_idx - dt  # get start index
            if t1_idx < 0:  # skip invalid windows
                continue  # continue loop
            rows.append({
                "t1": df.loc[t1_idx, date_col].strftime("%Y-%m-%d"),
                "t2": t2_date.strftime("%Y-%m-%d"),
                "window_dt_requested": dt,
            })  # store window
    return pd.DataFrame(rows)  # return window table

def solve_linear_params(t, y, tc, m, omega):
    tau = tc - t  # time to critical time
    if np.any(tau <= 0):  # check valid domain
        return None, np.inf, None  # reject invalid fit
    f = tau ** m  # power-law term
    g = f * np.cos(omega * np.log(tau))  # cosine term
    h = f * np.sin(omega * np.log(tau))  # sine term
    X = np.column_stack([np.ones_like(t), f, g, h])  # build design matrix
    try:
        params, *_ = np.linalg.lstsq(X, y, rcond=None)  # estimate linear parameters
        y_hat = X @ params  # fitted values
        residuals = y - y_hat  # residuals
        sse = np.sum(residuals ** 2)  # sum squared error
        rmse = np.sqrt(np.mean(residuals ** 2))  # root mean squared error
        return params, sse, rmse  # return fit
    except np.linalg.LinAlgError:
        return None, np.inf, None  # reject failed solve

def fit_lppl_row(
    df,
    t1,
    t2,
    global_tc_upper_date,
    date_col=DATE_COL,
    y_col=Y_COL,
    tc_min_days=TC_MIN_DAYS,
    seed=SEED,
    maxiter=MAXITER,
):
    t1 = pd.to_datetime(t1)  # parse start date
    t2 = pd.to_datetime(t2)  # parse endpoint date
    global_tc_upper_date = pd.to_datetime(global_tc_upper_date)  # parse tc upper date
    window = df[
        (df[date_col] >= t1) &
        (df[date_col] <= t2)
    ].copy()  # slice fitting window
    if len(window) < 30:  # check minimum observations
        raise ValueError(f"Window has only {len(window)} observations.")  # stop if too short
    window = window.sort_values(date_col).reset_index(drop=True)  # sort window
    actual_t1 = window[date_col].iloc[0]  # actual start date
    actual_t2 = window[date_col].iloc[-1]  # actual endpoint date
    window["t"] = (window[date_col] - actual_t1).dt.days.astype(float)  # convert dates to time
    t = window["t"].to_numpy(dtype=float)  # get time array
    y = window[y_col].to_numpy(dtype=float)  # get log price array
    t_last = float(t.max())  # final time
    tc_lower = t_last + tc_min_days  # lower tc bound
    tc_upper = float((global_tc_upper_date - actual_t1).days)  # upper tc bound
    if tc_upper <= tc_lower:  # check bounds
        raise ValueError("Invalid tc bounds.")  # stop if invalid
    def objective(theta):
        tc, m, omega = theta  # unpack nonlinear parameters
        _, sse, _ = solve_linear_params(t, y, tc, m, omega)  # solve linear parameters
        return sse  # minimise sse
    bounds = [
        (tc_lower, tc_upper),
        M_BOUNDS,
        OMEGA_BOUNDS,
    ]  # nonlinear bounds
    result_de = differential_evolution(
        objective,
        bounds=bounds,
        seed=seed,
        polish=False,
        workers=1,
        updating="immediate",
    )  # run global optimisation
    result = minimize(
        objective,
        result_de.x,
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": maxiter},
    )  # refine with local optimisation
    tc, m, omega = result.x  # get nonlinear parameters
    linear_params, sse, rmse = solve_linear_params(t, y, tc, m, omega)  # solve final linear parameters
    if linear_params is None:  # check fit success
        raise RuntimeError("LPPL fit failed.")  # stop if failed
    A, B, C1, C2 = linear_params  # unpack linear parameters
    C = float(np.sqrt(C1 ** 2 + C2 ** 2))  # recover amplitude
    phi = float(np.arctan2(C2, C1))  # recover phase
    tc_date = actual_t1 + pd.Timedelta(days=float(tc))  # convert tc to date
    damping = (m * abs(B)) / (omega * abs(C)) if abs(C) > 0 else np.inf  # calculate damping
    positive_lppls_valid = (
        (B < 0) and
        (abs(C) < 1) and
        (M_BOUNDS[0] <= m <= M_BOUNDS[1]) and
        (OMEGA_BOUNDS[0] <= omega <= OMEGA_BOUNDS[1]) and
        (damping >= 1) and
        ((tc_date - actual_t2).days > 0)
    )  # apply validity filters
    return pd.DataFrame([{
        "t1": actual_t1.strftime("%Y-%m-%d"),
        "t2": actual_t2.strftime("%Y-%m-%d"),
        "observations": int(len(window)),
        "tc_predicted": tc_date.strftime("%Y-%m-%d"),
        "tc_days_after_t2": int((tc_date - actual_t2).days),
        "global_tc_upper_date": global_tc_upper_date.strftime("%Y-%m-%d"),
        "positive_lppls_valid": bool(positive_lppls_valid),
        "RMSE": float(rmse),
        "SSE": float(sse),
        "A": float(A),
        "B": float(B),
        "C1": float(C1),
        "C2": float(C2),
        "C": float(C),
        "m": float(m),
        "omega": float(omega),
        "phi": float(phi),
        "damping": float(damping),
        "optimizer_success": bool(result.success),
    }])  # return one fit row

def run_daily_positive_lppls(
    df,
    output_csv=OUTPUT_CSV,
    date_col=DATE_COL,
    y_col=Y_COL,
    t2_step=T2_STEP,
    dt_min=DT_MIN,
    dt_max=DT_MAX,
    dt_step=DT_STEP,
    tc_min_days=TC_MIN_DAYS,
    tc_max_days=TC_MAX_DAYS,
    seed=SEED,
    require_full_scale=REQUIRE_FULL_SCALE,
):
    windows = generate_rolling_lppl_windows(
        df=df,
        date_col=date_col,
        t2_step=t2_step,
        dt_min=dt_min,
        dt_max=dt_max,
        dt_step=dt_step,
        require_full_scale=require_full_scale,
    )  # create rolling windows
    if windows.empty:  # check windows exist
        raise ValueError("No valid rolling windows generated.")  # stop if none
    if os.path.exists(output_csv):  # check old output
        os.remove(output_csv)  # delete old output
    for i, row in windows.iterrows():  # loop over windows
        try:
            t2 = pd.to_datetime(row["t2"])  # parse endpoint
            global_tc_upper_date = t2 + pd.Timedelta(days=tc_max_days)  # set tc upper date
            fit_row = fit_lppl_row(
                df=df,
                t1=row["t1"],
                t2=row["t2"],
                global_tc_upper_date=global_tc_upper_date,
                date_col=date_col,
                y_col=y_col,
                tc_min_days=tc_min_days,
                seed=seed,
            )  # fit one window
            fit_row["window_dt_requested"] = row["window_dt_requested"]  # store requested window
            write_header = not os.path.exists(output_csv)  # write header if new file
            fit_row.to_csv(
                output_csv,
                mode="a",
                header=write_header,
                index=False,
            )  # append fit row
            print(
                f"{i + 1}/{len(windows)} done | "
                f"t2={row['t2']} | "
                f"dt={row['window_dt_requested']} | "
                f"valid={fit_row['positive_lppls_valid'].iloc[0]} | "
                f"tc={fit_row['tc_predicted'].iloc[0]}"
            )  # print progress
        except Exception as e:
            error_row = pd.DataFrame([{
                "t1": row["t1"],
                "t2": row["t2"],
                "window_dt_requested": row["window_dt_requested"],
                "error": str(e),
            }])  # create error row
            write_header = not os.path.exists(output_csv)  # write header if new file
            error_row.to_csv(
                output_csv,
                mode="a",
                header=write_header,
                index=False,
            )  # append error row
            print(
                f"{i + 1}/{len(windows)} failed | "
                f"t2={row['t2']} | "
                f"dt={row['window_dt_requested']} | "
                f"{e}"
            )  # print failure
    return pd.read_csv(output_csv)  # return raw results

def make_daily_positive_confidence(results):
    res = results.copy()  # copy results
    res["t2"] = pd.to_datetime(res["t2"], errors="coerce")  # parse endpoints
    res["tc_predicted"] = pd.to_datetime(res["tc_predicted"], errors="coerce")  # parse predictions
    res["positive_lppls_valid"] = (
        res["positive_lppls_valid"]
        .astype(str)
        .str.lower()
        .eq("true")
    )  # convert validity flag
    res = res.dropna(subset=["t2"])  # remove missing endpoints
    confidence = (
        res.groupby("t2")
        .agg(
            total_fits=("positive_lppls_valid", "size"),
            valid_fits=("positive_lppls_valid", "sum"),
            mean_tc=("tc_predicted", "mean"),
            median_tc=("tc_predicted", "median"),
            mean_rmse=("RMSE", "mean"),
        )
        .reset_index()
    )  # aggregate by endpoint
    confidence["positive_bubble_confidence"] = (
        confidence["valid_fits"] / confidence["total_fits"]
    )  # calculate confidence score
    return confidence  # return confidence table

def plot_price_and_daily_positive_confidence(
    prices,
    confidence,
    title="STOXX Europe 600: Daily Positive LPPL Bubble Confidence",
):
    plt.rcParams.update({
        "font.family": "Arial",
        "font.size": 14,
        "axes.labelsize": 14,
        "axes.titlesize": 14,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 14,
    })  # set plot style
    fig, ax1 = plt.subplots(figsize=(14, 6))  # create figure
    ax1.plot(
        prices[DATE_COL],
        prices["Close"],
        color="black",
        linewidth=1.4,
        label="STOXX Europe 600 Price",
    )  # plot price
    ax1.set_xlabel("Date")  # set x label
    ax1.set_ylabel("Price")  # set left y label
    ax1.grid(True, alpha=0.25)  # add grid
    ax2 = ax1.twinx()  # create second axis
    ax2.plot(
        confidence["t2"],
        confidence["positive_bubble_confidence"],
        color="red",
        linestyle="--",
        linewidth=2.0,
        label="Positive LPPL Confidence Score",
    )  # plot confidence score
    ax2.set_ylabel("Confidence Score")  # set right y label
    ax2.set_ylim(0, 1)  # set confidence limits
    ax1.set_title(title)  # set title
    lines_1, labels_1 = ax1.get_legend_handles_labels()  # get price legend
    lines_2, labels_2 = ax2.get_legend_handles_labels()  # get confidence legend
    ax1.legend(
        lines_1 + lines_2,
        labels_1 + labels_2,
        loc="upper left",
        frameon=False,
    )  # combine legends
    plt.tight_layout()  # tidy layout
    plt.savefig("eurostoxx600_daily_positive_lppls_confidence_plot.png", dpi=300)  # save plot
    plt.show()  # show plot

if __name__ == "__main__":
    prices = load_prices(PRICE_CSV)  # load price data
    expected_windows = generate_rolling_lppl_windows(prices)  # generate expected windows
    print("Rows in price file:", len(prices))  # print price rows
    print("Expected LPPL fits:", len(expected_windows))  # print fit count
    print("Output raw fits file:", OUTPUT_CSV)  # print raw output path
    print("Output confidence file:", CONFIDENCE_CSV)  # print confidence output path
    results = run_daily_positive_lppls(
        df=prices,
        output_csv=OUTPUT_CSV,
        t2_step=T2_STEP,
        dt_min=DT_MIN,
        dt_max=DT_MAX,
        dt_step=DT_STEP,
        tc_min_days=TC_MIN_DAYS,
        tc_max_days=TC_MAX_DAYS,
        seed=SEED,
        require_full_scale=REQUIRE_FULL_SCALE,
    )  # run all lppl fits
    confidence = make_daily_positive_confidence(results)  # calculate confidence scores
    confidence.to_csv(CONFIDENCE_CSV, index=False)  # save confidence scores
    plot_price_and_daily_positive_confidence(prices, confidence)  # plot confidence scores
    print("Saved raw daily positive LPPL fits to:", OUTPUT_CSV)  # print raw save path
    print("Saved daily positive confidence scores to:", CONFIDENCE_CSV)  # print confidence save path
    print(confidence.head())  # print first rows
    print(confidence.tail())  # print last rows