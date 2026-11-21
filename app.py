# =============================================================================
# RETAIL INTELLIGENCE SYSTEM
# Demand Forecasting | Inventory Optimization | Predictive Replenishment
# Version: 2.1
#
# Research prototype:
# - Transaction CSV ingestion and validation
# - Daily SKU demand aggregation
# - Forecasting and chronological holdout evaluation
# - Inventory simulation with an order pipeline
# - Supplier MOQ/order-multiple constraints
# - Replenishment recommendations and CSV exports
#
# Important:
# Inventory, supplier and purchase-order data are simulated unless supplied
# separately. This is not a live ERP or purchasing integration.
# =============================================================================

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from sklearn.ensemble import (
    RandomForestRegressor,
    HistGradientBoostingRegressor,
)
from sklearn.metrics import mean_absolute_error, mean_squared_error

warnings.filterwarnings("ignore")

# -----------------------------------------------------------------------------
# 1. Application configuration and theme
# -----------------------------------------------------------------------------

APP_TITLE = "Retail Intelligence System"
APP_VERSION = "2.1"
RANDOM_SEED = 42

FEATURE_COLUMNS = [
    "lag_1",
    "lag_7",
    "lag_14",
    "rolling_mean_7",
    "rolling_mean_28",
    "rolling_std_28",
    "day_of_week",
    "month",
    "is_weekend",
]

st.set_page_config(
    page_title=APP_TITLE,
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
/* App surfaces */
.stApp,
[data-testid="stAppViewContainer"],
[data-testid="stMain"] {
    background-color: #F5F7FA !important;
    color: #1F2937 !important;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #FFFFFF !important;
    border-right: 1px solid #E5E7EB;
}
[data-testid="stSidebar"] * {
    color: #1F2937;
}

/* Typography */
h1, h2, h3, h4 {
    color: #172554 !important;
}
p, label, .stMarkdown, [data-testid="stCaptionContainer"] {
    color: #374151;
}

/* KPI cards */
[data-testid="stMetric"] {
    background: #FFFFFF !important;
    border: 1px solid #E5E7EB;
    border-radius: 12px;
    padding: 16px 18px;
    box-shadow: 0 2px 8px rgba(15, 23, 42, 0.04);
}
[data-testid="stMetricLabel"],
[data-testid="stMetricValue"],
[data-testid="stMetricDelta"] {
    color: #1F2937 !important;
}

/* Tables and expanders */
[data-testid="stDataFrame"],
[data-testid="stExpander"] {
    background: #FFFFFF;
    border: 1px solid #E5E7EB;
    border-radius: 10px;
}

/* Tabs */
button[data-baseweb="tab"] {
    color: #374151 !important;
    font-weight: 600;
}
button[data-baseweb="tab"][aria-selected="true"] {
    color: #2563EB !important;
}

/* Buttons */
.stButton > button,
.stDownloadButton > button {
    border-radius: 8px;
    font-weight: 600;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# 2. Inventory policy configuration
# -----------------------------------------------------------------------------

@dataclass
class InventoryConfig:
    lead_time_days: int = 7
    minimum_order_quantity: int = 10
    order_multiple: int = 1
    service_level: float = 0.95
    daily_holding_cost: float = 0.02
    shortage_cost: float = 5.0
    initial_stock_days: float = 14.0


# -----------------------------------------------------------------------------
# 3. Synthetic transaction generator
# -----------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def generate_retail_transactions(
    n_skus: int = 25,
    history_days: int = 365,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """Create reproducible transaction-level demonstration data."""
    rng = np.random.default_rng(seed)
    end_date = pd.Timestamp.today().normalize()
    dates = pd.date_range(
        end=end_date,
        periods=history_days,
        freq="D",
    )

    categories = [
        "Home", "Electronics", "Clothing",
        "Grocery", "Accessories",
    ]
    weekly_pattern = np.array(
        [0.85, 0.95, 1.00, 1.10, 1.25, 1.45, 0.65]
    )

    rows = []
    invoice_number = 100000

    for sku_number in range(1, n_skus + 1):
        sku = f"SKU-{sku_number:04d}"
        category = categories[(sku_number - 1) % len(categories)]
        unit_price = float(rng.uniform(5, 150))
        base_demand = float(rng.uniform(3, 35))

        for current_date in dates:
            weekday_factor = weekly_pattern[current_date.weekday()]
            seasonal_factor = (
                1.0
                + 0.15 * np.sin(2 * np.pi * current_date.month / 12)
            )
            demand = (
                base_demand * weekday_factor * seasonal_factor
                + rng.normal(0, base_demand * 0.20)
            )
            demand = max(0, int(round(demand)))

            if rng.random() < 0.015:
                demand *= int(rng.integers(2, 5))

            remaining = demand
            while remaining > 0:
                quantity = min(
                    remaining,
                    int(rng.integers(1, 6)),
                )
                remaining -= quantity
                invoice_number += 1

                rows.append({
                    "Invoice": str(invoice_number),
                    "InvoiceDate": current_date,
                    "StockCode": sku,
                    "Description": f"{category} Product {sku}",
                    "Quantity": quantity,
                    "UnitPrice": round(unit_price, 2),
                    "CustomerID": int(rng.integers(10000, 99999)),
                    "Country": rng.choice([
                        "United Kingdom", "Germany", "France",
                        "India", "Netherlands",
                    ]),
                })

    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# 4. Input validation and normalization
# -----------------------------------------------------------------------------

def validate_transactions(
    source_df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict]:
    """
    Validate required columns and normalize optional columns.
    Required: InvoiceDate, StockCode, Quantity, UnitPrice.
    """
    required = [
        "InvoiceDate", "StockCode", "Quantity", "UnitPrice"
    ]
    missing = [column for column in required if column not in source_df.columns]
    if missing:
        raise ValueError(
            "Missing required CSV columns: " + ", ".join(missing)
        )

    df = source_df.copy()
    total_rows = len(df)

    # Optional fields are generated if the uploaded file does not contain them.
    if "Invoice" not in df.columns:
        df["Invoice"] = [f"ROW-{i}" for i in range(len(df))]
    if "Description" not in df.columns:
        df["Description"] = "Unclassified product"
    if "Country" not in df.columns:
        df["Country"] = "Unknown"

    df["InvoiceDate"] = pd.to_datetime(
        df["InvoiceDate"], errors="coerce"
    )
    df["Quantity"] = pd.to_numeric(
        df["Quantity"], errors="coerce"
    )
    df["UnitPrice"] = pd.to_numeric(
        df["UnitPrice"], errors="coerce"
    )

    report = {
        "total_rows": total_rows,
        "invalid_dates": int(df["InvoiceDate"].isna().sum()),
        "invalid_quantities": int(df["Quantity"].isna().sum()),
        "invalid_prices": int(df["UnitPrice"].isna().sum()),
        "missing_skus": int(df["StockCode"].isna().sum()),
    }

    df = df.dropna(
        subset=["InvoiceDate", "StockCode", "Quantity", "UnitPrice"]
    )
    df["StockCode"] = df["StockCode"].astype(str).str.strip()
    df = df[
        (df["StockCode"] != "")
        & (df["UnitPrice"] > 0)
        & (df["Quantity"] != 0)
    ].copy()

    df["Revenue"] = df["Quantity"] * df["UnitPrice"]
    report["valid_rows"] = len(df)
    report["return_rows"] = int((df["Quantity"] < 0).sum())

    return df, report


def prepare_transactions(
    raw_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    validated, quality = validate_transactions(raw_df)

    returns = validated[validated["Quantity"] < 0].copy()
    sales = validated[validated["Quantity"] > 0].copy()

    returns["ReturnQuantity"] = returns["Quantity"].abs()
    sales["Date"] = sales["InvoiceDate"].dt.floor("D")
    returns["Date"] = returns["InvoiceDate"].dt.floor("D")

    return sales, returns, quality


# -----------------------------------------------------------------------------
# 5. Product catalog and daily demand
# -----------------------------------------------------------------------------

def build_product_catalog(sales_df: pd.DataFrame) -> pd.DataFrame:
    catalog = (
        sales_df.groupby("StockCode")
        .agg(
            Description=("Description", "first"),
            UnitPrice=("UnitPrice", "median"),
            TotalUnits=("Quantity", "sum"),
            TransactionLines=("Invoice", "count"),
        )
        .reset_index()
    )

    catalog["Description"] = (
        catalog["Description"].fillna("Unclassified product").astype(str)
    )
    catalog["Category"] = (
        catalog["Description"].str.split().str[0].fillna("General")
    )
    return catalog


def aggregate_daily_demand(sales_df: pd.DataFrame) -> pd.DataFrame:
    if sales_df.empty:
        raise ValueError("No positive-quantity sales transactions were found.")

    daily = (
        sales_df.groupby(["StockCode", "Date"], as_index=False)
        .agg(
            demand=("Quantity", "sum"),
            revenue=("Revenue", "sum"),
            transaction_count=("Invoice", "nunique"),
        )
    )

    all_dates = pd.date_range(
        daily["Date"].min(),
        daily["Date"].max(),
        freq="D",
    )
    all_skus = daily["StockCode"].unique()
    full_index = pd.MultiIndex.from_product(
        [all_skus, all_dates],
        names=["StockCode", "Date"],
    )

    daily = (
        daily.set_index(["StockCode", "Date"])
        .reindex(full_index)
        .reset_index()
    )

    for column in ["demand", "revenue", "transaction_count"]:
        daily[column] = daily[column].fillna(0)

    daily["demand"] = daily["demand"].clip(lower=0)
    daily["revenue"] = daily["revenue"].clip(lower=0)
    daily["transaction_count"] = daily["transaction_count"].astype(int)

    return daily.sort_values(["StockCode", "Date"]).reset_index(drop=True)


# -----------------------------------------------------------------------------
# 6. Feature engineering
# -----------------------------------------------------------------------------

def create_features(daily_df: pd.DataFrame) -> pd.DataFrame:
    df = daily_df.sort_values(["StockCode", "Date"]).copy()
    grouped = df.groupby("StockCode")["demand"]

    df["lag_1"] = grouped.shift(1)
    df["lag_7"] = grouped.shift(7)
    df["lag_14"] = grouped.shift(14)

    df["rolling_mean_7"] = (
        df.groupby("StockCode")["demand"]
        .transform(lambda x: x.shift(1).rolling(7, min_periods=7).mean())
    )
    df["rolling_mean_28"] = (
        df.groupby("StockCode")["demand"]
        .transform(lambda x: x.shift(1).rolling(28, min_periods=14).mean())
    )
    df["rolling_std_28"] = (
        df.groupby("StockCode")["demand"]
        .transform(lambda x: x.shift(1).rolling(28, min_periods=14).std())
    )

    df["day_of_week"] = df["Date"].dt.dayofweek
    df["month"] = df["Date"].dt.month
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    # Do not silently train ML models on missing lag values.
    df[FEATURE_COLUMNS] = df[FEATURE_COLUMNS].replace(
        [np.inf, -np.inf], np.nan
    )

    return df


# -----------------------------------------------------------------------------
# 7. Forecasting models
# -----------------------------------------------------------------------------

def train_forecasting_model(
    train_df: pd.DataFrame,
    model_name: str,
):
    valid_train = train_df.dropna(subset=FEATURE_COLUMNS).copy()
    if valid_train.empty:
        raise ValueError(
            "Not enough usable history after feature engineering."
        )

    X_train = valid_train[FEATURE_COLUMNS]
    y_train = valid_train["demand"]

    if model_name == "Random Forest":
        model = RandomForestRegressor(
            n_estimators=250,
            max_depth=12,
            min_samples_leaf=3,
            max_features=0.9,
            random_state=RANDOM_SEED,
            n_jobs=-1,
        )
    elif model_name == "Histogram Gradient Boosting":
        model = HistGradientBoostingRegressor(
            max_iter=200,
            learning_rate=0.05,
            max_leaf_nodes=20,
            min_samples_leaf=15,
            l2_regularization=1.0,
            random_state=RANDOM_SEED,
        )
    else:
        raise ValueError(f"Unsupported machine-learning model: {model_name}")

    model.fit(X_train, y_train)
    return model


def forecast_holdout(
    feature_df: pd.DataFrame,
    model_name: str,
    holdout_days: int = 30,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    cutoff = feature_df["Date"].max() - pd.Timedelta(days=holdout_days)

    train_df = feature_df[
        (feature_df["Date"] < cutoff)
    ].dropna(subset=FEATURE_COLUMNS).copy()

    test_df = feature_df[
        feature_df["Date"] >= cutoff
    ].dropna(subset=FEATURE_COLUMNS).copy()

    if train_df.empty or test_df.empty:
        raise ValueError(
            "Insufficient history for the selected holdout period."
        )

    test_df = test_df.sort_values(["StockCode", "Date"]).copy()

    if model_name in ["Random Forest", "Histogram Gradient Boosting"]:
        model = train_forecasting_model(train_df, model_name)
        test_df["forecast"] = np.maximum(
            0, model.predict(test_df[FEATURE_COLUMNS])
        )

    elif model_name == "Rolling Mean":
        test_df["forecast"] = test_df["rolling_mean_7"].clip(lower=0)

    elif model_name == "Seasonal Naive":
        test_df["forecast"] = test_df["lag_7"].clip(lower=0)

    else:
        raise ValueError("Unknown forecasting model.")

    return train_df, test_df


def calculate_forecast_metrics(
    actual: np.ndarray,
    predicted: np.ndarray,
) -> dict:
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    denominator = actual.sum()
    wape = (
        100 * np.abs(actual - predicted).sum() / denominator
        if denominator > 0 else np.nan
    )
    bias = predicted.sum() - actual.sum()

    return {
        "MAE": float(mae),
        "RMSE": float(rmse),
        "WAPE": float(wape),
        "Bias": float(bias),
    }


# -----------------------------------------------------------------------------
# 8. Safety stock and inventory simulation
# -----------------------------------------------------------------------------

def service_level_z(service_level: float) -> float:
    # Common normal-approximation z values.
    if service_level <= 0.90:
        return 1.282
    if service_level <= 0.95:
        return 1.645
    if service_level <= 0.975:
        return 1.960
    if service_level <= 0.99:
        return 2.326
    return 3.090


def calculate_safety_stock(
    historical_errors: np.ndarray,
    lead_time_days: int,
    service_level: float,
) -> float:
    errors = np.asarray(historical_errors, dtype=float)
    errors = errors[np.isfinite(errors)]

    if len(errors) < 5:
        return 0.0

    error_std = float(np.std(errors, ddof=1))
    z = service_level_z(service_level)

    # Approximation assuming independent daily forecast errors.
    return max(0.0, z * error_std * np.sqrt(max(1, lead_time_days)))


def simulate_inventory(
    demand_df: pd.DataFrame,
    forecast_df: pd.DataFrame,
    config: InventoryConfig,
) -> pd.DataFrame:
    """
    Retrospective inventory simulation.

    Orders are placed after the day's demand is observed and arrive after
    the configured number of calendar days. Pipeline orders are tracked.
    """
    results = []

    forecast_lookup = forecast_df.set_index(
        ["StockCode", "Date"]
    )["forecast"].to_dict()

    error_lookup = {}
    for sku, group in forecast_df.groupby("StockCode"):
        # In a real prospective deployment, these errors must come from a
        # prior validation/calibration window, not future holdout outcomes.
        error_lookup[sku] = (
            group["demand"].to_numpy()
            - group["forecast"].to_numpy()
        )

    for sku, group in demand_df.groupby("StockCode"):
        group = group.sort_values("Date").copy()
        historical_mean = float(group["demand"].head(28).mean())
        on_hand = max(0.0, historical_mean * 14.0)
        backorders = 0.0
        pipeline = []

        safety_stock = calculate_safety_stock(
            error_lookup.get(sku, np.array([])),
            config.lead_time_days,
            config.service_level,
        )

        for row in group.itertuples(index=False):
            current_date = row.Date
            actual_demand = float(row.demand)

            # Receive orders due today.
            received = 0.0
            outstanding = []
            for order in pipeline:
                if order["arrival_date"] <= current_date:
                    received += order["quantity"]
                else:
                    outstanding.append(order)
            pipeline = outstanding
            on_hand += received

            # Fulfil today's demand.
            fulfilled = min(on_hand, actual_demand)
            shortage = actual_demand - fulfilled
            on_hand -= fulfilled
            backorders += shortage

            forecast = float(
                forecast_lookup.get((sku, current_date), 0.0)
            )

            # Simplified order-up-to target for constant daily forecast.
            target_stock = (
                forecast * config.lead_time_days + safety_stock
            )

            on_order = sum(
                order["quantity"] for order in pipeline
            )
            inventory_position = on_hand + on_order - backorders
            order_qty = max(0.0, target_stock - inventory_position)

            if order_qty > 0:
                order_qty = max(
                    order_qty,
                    float(config.minimum_order_quantity),
                )
                if config.order_multiple > 1:
                    order_qty = (
                        math.ceil(order_qty / config.order_multiple)
                        * config.order_multiple
                    )
                order_qty = int(math.ceil(order_qty))

                pipeline.append({
                    "quantity": order_qty,
                    "arrival_date": (
                        current_date
                        + pd.Timedelta(days=config.lead_time_days)
                    ),
                })
            else:
                order_qty = 0

            holding_cost_value = (
                on_hand * config.daily_holding_cost
            )
            shortage_cost_value = (
                shortage * config.shortage_cost
            )

            results.append({
                "Date": current_date,
                "StockCode": sku,
                "ActualDemand": actual_demand,
                "Forecast": forecast,
                "OnHand": on_hand,
                "OnOrder": sum(
                    order["quantity"] for order in pipeline
                ),
                "InventoryPosition": inventory_position,
                "ReceivedQuantity": received,
                "OrderQuantity": order_qty,
                "Shortfall": shortage,
                "Backorders": backorders,
                "SafetyStock": safety_stock,
                "TargetStock": target_stock,
                "HoldingCost": holding_cost_value,
                "ShortageCost": shortage_cost_value,
                "TotalCost": (
                    holding_cost_value + shortage_cost_value
                ),
            })

    return pd.DataFrame(results)


# -----------------------------------------------------------------------------
# 9. Replenishment recommendations and KPI summaries
# -----------------------------------------------------------------------------

def generate_replenishment_recommendations(
    inventory_df: pd.DataFrame,
    catalog_df: pd.DataFrame,
) -> pd.DataFrame:
    if inventory_df.empty:
        return pd.DataFrame()

    latest = (
        inventory_df.sort_values("Date")
        .groupby("StockCode")
        .tail(1)
        .copy()
    )

    latest = latest.merge(
        catalog_df,
        on="StockCode",
        how="left",
    )

    latest["Status"] = np.select(
        [
            latest["Shortfall"] > 0,
            latest["OnHand"] <= latest["SafetyStock"],
        ],
        [
            "STOCKOUT RISK",
            "LOW STOCK",
        ],
        default="HEALTHY",
    )

    latest["EstimatedOrderValue"] = (
        latest["OrderQuantity"]
        * latest["UnitPrice"].fillna(0)
    )

    return latest[[
        "StockCode",
        "Description",
        "Category",
        "ActualDemand",
        "Forecast",
        "OnHand",
        "OnOrder",
        "SafetyStock",
        "TargetStock",
        "OrderQuantity",
        "Status",
        "UnitPrice",
        "EstimatedOrderValue",
    ]].rename(
        columns={"OrderQuantity": "RecommendedOrder"}
    ).sort_values(
        "RecommendedOrder",
        ascending=False,
    )


def calculate_inventory_kpis(inventory_df: pd.DataFrame) -> dict:
    if inventory_df.empty:
        return {
            "Total Demand": 0,
            "Total Orders": 0,
            "Total Shortfall": 0,
            "Fill Rate": 0,
            "Holding Cost": 0,
            "Shortage Cost": 0,
            "Total Cost": 0,
        }

    demand = float(inventory_df["ActualDemand"].sum())
    shortage = float(inventory_df["Shortfall"].sum())
    fill_rate = 100 * (1 - shortage / max(demand, 1))

    return {
        "Total Demand": demand,
        "Total Orders": float(inventory_df["OrderQuantity"].sum()),
        "Total Shortfall": shortage,
        "Fill Rate": float(np.clip(fill_rate, 0, 100)),
        "Holding Cost": float(inventory_df["HoldingCost"].sum()),
        "Shortage Cost": float(inventory_df["ShortageCost"].sum()),
        "Total Cost": float(inventory_df["TotalCost"].sum()),
    }


def to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


# -----------------------------------------------------------------------------
# 10. Sidebar controls
# -----------------------------------------------------------------------------

st.sidebar.title("Retail Intelligence")
st.sidebar.caption(f"Version {APP_VERSION}")
st.sidebar.divider()

data_source = st.sidebar.radio(
    "Data Source",
    ["Synthetic Retail Data", "Upload Transaction CSV"],
)

uploaded_file = None
if data_source == "Upload Transaction CSV":
    uploaded_file = st.sidebar.file_uploader(
        "Upload transaction data",
        type=["csv"],
        help=(
            "Required columns: InvoiceDate, StockCode, Quantity, UnitPrice. "
            "Invoice and Description are optional."
        ),
    )

st.sidebar.divider()
st.sidebar.subheader("Forecasting")

model_name = st.sidebar.selectbox(
    "Forecasting Model",
    [
        "Seasonal Naive",
        "Rolling Mean",
        "Random Forest",
        "Histogram Gradient Boosting",
    ],
)

st.sidebar.divider()
st.sidebar.subheader("Inventory Policy")

lead_time = st.sidebar.slider(
    "Supplier Lead Time (days)", 1, 60, 7
)
minimum_order_quantity = st.sidebar.number_input(
    "Minimum Order Quantity", min_value=1, value=10
)
order_multiple = st.sidebar.number_input(
    "Order Multiple", min_value=1, value=1
)
service_level = st.sidebar.slider(
    "Target Service Level",
    min_value=0.50,
    max_value=0.99,
    value=0.95,
    step=0.01,
)
holding_cost = st.sidebar.number_input(
    "Daily Holding Cost per Unit",
    min_value=0.0,
    value=0.02,
    step=0.01,
    format="%.3f",
)
shortage_cost = st.sidebar.number_input(
    "Shortage Cost per Unit",
    min_value=0.0,
    value=5.0,
    step=0.5,
)

config = InventoryConfig(
    lead_time_days=int(lead_time),
    minimum_order_quantity=int(minimum_order_quantity),
    order_multiple=int(order_multiple),
    service_level=float(service_level),
    daily_holding_cost=float(holding_cost),
    shortage_cost=float(shortage_cost),
)


# -----------------------------------------------------------------------------
# 11. Load and prepare source data
# -----------------------------------------------------------------------------

st.title("📦 Retail Intelligence System")
st.markdown(
    "Demand forecasting, inventory planning and predictive replenishment "
    "using historical retail transactions."
)
st.caption(
    "Research prototype | Synthetic inventory and supplier assumptions"
)

if data_source == "Synthetic Retail Data":
    with st.spinner("Generating reproducible retail transaction data..."):
        raw_df = generate_retail_transactions(
            n_skus=25,
            history_days=365,
            seed=RANDOM_SEED,
        )
else:
    if uploaded_file is None:
        st.info("Upload a transaction CSV from the sidebar to begin.")
        st.stop()
    try:
        raw_df = pd.read_csv(uploaded_file)
    except Exception as exc:
        st.error(f"Unable to read the uploaded CSV: {exc}")
        st.stop()

try:
    sales_df, return_df, quality_report = prepare_transactions(raw_df)
    if sales_df.empty:
        st.error("No positive-quantity sales records were found.")
        st.stop()

    catalog_df = build_product_catalog(sales_df)
    daily_df = aggregate_daily_demand(sales_df)
    feature_df = create_features(daily_df)

except Exception as exc:
    st.error(f"Data preparation failed: {exc}")
    st.stop()


# -----------------------------------------------------------------------------
# 12. Data quality and product selection
# -----------------------------------------------------------------------------

with st.expander("Data Quality and Dataset Overview", expanded=False):
    q1, q2, q3, q4 = st.columns(4)
    q1.metric("Input Rows", f"{quality_report['total_rows']:,}")
    q2.metric("Valid Rows", f"{quality_report['valid_rows']:,}")
    q3.metric("Return Rows", f"{quality_report['return_rows']:,}")
    q4.metric("Products", f"{daily_df['StockCode'].nunique():,}")

    st.write("Validation details", quality_report)
    st.write(
        "Date range:",
        daily_df["Date"].min().date(),
        "to",
        daily_df["Date"].max().date(),
    )
    st.dataframe(catalog_df.head(20), use_container_width=True)

st.sidebar.divider()
st.sidebar.subheader("Product Filters")

sku_options = sorted(daily_df["StockCode"].unique().tolist())
default_skus = sku_options[:min(10, len(sku_options))]

selected_skus = st.sidebar.multiselect(
    "Select products",
    options=sku_options,
    default=default_skus,
)

if not selected_skus:
    st.warning("Select at least one product from the sidebar.")
    st.stop()

filtered_daily = daily_df[
    daily_df["StockCode"].isin(selected_skus)
].copy()

filtered_features = feature_df[
    feature_df["StockCode"].isin(selected_skus)
].copy()


# -----------------------------------------------------------------------------
# 13. Forecast and inventory calculations
# -----------------------------------------------------------------------------

st.header("Forecasting Performance")

try:
    train_df, test_df = forecast_holdout(
        filtered_features,
        model_name,
        holdout_days=30,
    )
except Exception as exc:
    st.error(f"Forecasting could not be completed: {exc}")
    st.stop()

metrics = calculate_forecast_metrics(
    test_df["demand"].to_numpy(),
    test_df["forecast"].to_numpy(),
)

simulation_demand = test_df[[
    "StockCode", "Date", "demand"
]].copy()

simulation_forecast = test_df[[
    "StockCode", "Date", "demand", "forecast"
]].copy()

inventory_df = simulate_inventory(
    simulation_demand,
    simulation_forecast,
    config,
)

inventory_kpis = calculate_inventory_kpis(inventory_df)

recommendations_df = generate_replenishment_recommendations(
    inventory_df,
    catalog_df,
)


# -----------------------------------------------------------------------------
# 14. KPI cards
# -----------------------------------------------------------------------------

k1, k2, k3, k4 = st.columns(4)

k1.metric("Forecast MAE", f"{metrics['MAE']:.2f}")
k2.metric(
    "Forecast WAPE",
    f"{metrics['WAPE']:.2f}%" if np.isfinite(metrics["WAPE"]) else "N/A",
)
k3.metric("Inventory Fill Rate", f"{inventory_kpis['Fill Rate']:.2f}%")
k4.metric("Simulated Inventory Cost", f"{inventory_kpis['Total Cost']:,.2f}")


# -----------------------------------------------------------------------------
# 15. Dashboard tabs
# -----------------------------------------------------------------------------

tab_overview, tab_forecast, tab_inventory, tab_replenishment, tab_data = st.tabs(
    [
        "Overview",
        "Demand Forecast",
        "Inventory Simulation",
        "Replenishment",
        "Data Explorer",
    ]
)


# Overview
with tab_overview:
    st.subheader("Business Performance Overview")
    left, right = st.columns(2)

    with left:
        top_products = (
            sales_df[sales_df["StockCode"].isin(selected_skus)]
            .groupby("StockCode", as_index=False)["Quantity"]
            .sum()
            .sort_values("Quantity", ascending=False)
            .head(15)
            .merge(
                catalog_df[["StockCode", "Category"]],
                on="StockCode",
                how="left",
            )
        )
        fig_products = px.bar(
            top_products,
            x="StockCode",
            y="Quantity",
            color="Category",
            title="Top Products by Units Sold",
            template="plotly_white",
        )
        st.plotly_chart(fig_products, use_container_width=True)

    with right:
        revenue_trend = (
            sales_df[sales_df["StockCode"].isin(selected_skus)]
            .groupby("Date", as_index=False)["Revenue"]
            .sum()
        )
        fig_revenue = px.line(
            revenue_trend,
            x="Date",
            y="Revenue",
            title="Daily Revenue Trend",
            template="plotly_white",
        )
        st.plotly_chart(fig_revenue, use_container_width=True)

    fig_distribution = px.histogram(
        filtered_daily,
        x="demand",
        color="StockCode",
        nbins=40,
        title="Daily Demand Distribution",
        template="plotly_white",
    )
    st.plotly_chart(fig_distribution, use_container_width=True)


# Forecast
with tab_forecast:
    st.subheader("Actual Demand vs Forecast")
    selected_forecast_sku = st.selectbox(
        "Forecast Product",
        options=selected_skus,
        key="forecast_sku",
    )

    forecast_plot = test_df[
        test_df["StockCode"] == selected_forecast_sku
    ].sort_values("Date").copy()

    fig_forecast = go.Figure()
    fig_forecast.add_trace(go.Scatter(
        x=forecast_plot["Date"],
        y=forecast_plot["demand"],
        name="Actual Demand",
        mode="lines",
    ))
    fig_forecast.add_trace(go.Scatter(
        x=forecast_plot["Date"],
        y=forecast_plot["forecast"],
        name="Forecast",
        mode="lines",
        line={"dash": "dash"},
    ))
    fig_forecast.update_layout(
        title=f"Demand Forecast — {selected_forecast_sku}",
        xaxis_title="Date",
        yaxis_title="Units",
        hovermode="x unified",
        template="plotly_white",
    )
    st.plotly_chart(fig_forecast, use_container_width=True)

    forecast_plot["ForecastError"] = (
        forecast_plot["demand"] - forecast_plot["forecast"]
    )
    fig_error = px.bar(
        forecast_plot,
        x="Date",
        y="ForecastError",
        title="Daily Forecast Error (Actual − Forecast)",
        template="plotly_white",
    )
    st.plotly_chart(fig_error, use_container_width=True)

    st.dataframe(
        forecast_plot[[
            "Date", "StockCode", "demand", "forecast", "ForecastError"
        ]],
        use_container_width=True,
    )


# Inventory simulation
with tab_inventory:
    st.subheader("Inventory Position and Stockout Risk")
    selected_inventory_sku = st.selectbox(
        "Inventory Product",
        options=selected_skus,
        key="inventory_sku",
    )

    inventory_plot = inventory_df[
        inventory_df["StockCode"] == selected_inventory_sku
    ].sort_values("Date").copy()

    fig_inventory = go.Figure()
    fig_inventory.add_trace(go.Scatter(
        x=inventory_plot["Date"],
        y=inventory_plot["OnHand"],
        name="On-Hand Inventory",
        mode="lines",
    ))
    fig_inventory.add_trace(go.Scatter(
        x=inventory_plot["Date"],
        y=inventory_plot["TargetStock"],
        name="Target Stock",
        mode="lines",
        line={"dash": "dash"},
    ))
    fig_inventory.add_trace(go.Bar(
        x=inventory_plot["Date"],
        y=inventory_plot["Shortfall"],
        name="Shortfall",
        opacity=0.45,
    ))
    fig_inventory.update_layout(
        title="Inventory Position",
        xaxis_title="Date",
        yaxis_title="Units",
        hovermode="x unified",
        template="plotly_white",
    )
    st.plotly_chart(fig_inventory, use_container_width=True)

    cost_breakdown = pd.DataFrame({
        "Cost Type": ["Holding Cost", "Shortage Cost"],
        "Cost": [
            inventory_plot["HoldingCost"].sum(),
            inventory_plot["ShortageCost"].sum(),
        ],
    })
    fig_cost = px.bar(
        cost_breakdown,
        x="Cost Type",
        y="Cost",
        title="Inventory Cost Breakdown",
        template="plotly_white",
    )
    st.plotly_chart(fig_cost, use_container_width=True)
    st.dataframe(inventory_plot, use_container_width=True)


# Replenishment
with tab_replenishment:
    st.subheader("Predictive Replenishment Recommendations")
    st.info(
        "These are simulated planning recommendations, not live purchase orders."
    )

    if recommendations_df.empty:
        st.warning("No replenishment recommendations are available.")
    else:
        status_summary = (
            recommendations_df["Status"]
            .value_counts()
            .rename_axis("Status")
            .reset_index(name="Products")
        )
        fig_status = px.bar(
            status_summary,
            x="Status",
            y="Products",
            title="Inventory Risk by Product",
            template="plotly_white",
        )
        st.plotly_chart(fig_status, use_container_width=True)

        st.dataframe(recommendations_df, use_container_width=True)

        r1, r2 = st.columns(2)
        r1.metric(
            "Recommended Units",
            f"{recommendations_df['RecommendedOrder'].sum():,.0f}",
        )
        r2.metric(
            "Estimated Purchase Value",
            f"{recommendations_df['EstimatedOrderValue'].sum():,.2f}",
        )


# Data explorer
with tab_data:
    st.subheader("Daily Demand Data")
    st.dataframe(filtered_daily, use_container_width=True)

    st.subheader("Transaction Records")
    st.dataframe(
        sales_df[sales_df["StockCode"].isin(selected_skus)].head(500),
        use_container_width=True,
    )

    st.subheader("Returns")
    st.dataframe(
        return_df[return_df["StockCode"].isin(selected_skus)].head(200),
        use_container_width=True,
    )


# -----------------------------------------------------------------------------
# 16. Model comparison
# -----------------------------------------------------------------------------

st.divider()
st.header("Model Comparison")

with st.expander("Run chronological model comparison", expanded=False):
    if st.button("Evaluate All Models", type="primary"):
        comparison_rows = []

        for candidate in [
            "Seasonal Naive",
            "Rolling Mean",
            "Random Forest",
            "Histogram Gradient Boosting",
        ]:
            try:
                _, candidate_test = forecast_holdout(
                    filtered_features,
                    candidate,
                    holdout_days=30,
                )
                candidate_metrics = calculate_forecast_metrics(
                    candidate_test["demand"].to_numpy(),
                    candidate_test["forecast"].to_numpy(),
                )
                comparison_rows.append({
                    "Model": candidate,
                    **candidate_metrics,
                })
            except Exception as exc:
                st.warning(f"{candidate} could not be evaluated: {exc}")

        if comparison_rows:
            comparison_df = pd.DataFrame(comparison_rows)
            st.dataframe(comparison_df, use_container_width=True)

            fig_comparison = px.bar(
                comparison_df,
                x="Model",
                y="WAPE",
                title="Forecast Model Comparison (Lower WAPE Is Better)",
                template="plotly_white",
            )
            st.plotly_chart(fig_comparison, use_container_width=True)


# -----------------------------------------------------------------------------
# 17. Business alerts
# -----------------------------------------------------------------------------

st.divider()
st.header("Business Recommendations")

if not recommendations_df.empty:
    stockout_count = int(
        (recommendations_df["Status"] == "STOCKOUT RISK").sum()
    )
    low_stock_count = int(
        (recommendations_df["Status"] == "LOW STOCK").sum()
    )

    alert1, alert2 = st.columns(2)
    alert1.metric("Stockout-Risk Products", stockout_count)
    alert2.metric("Low-Stock Products", low_stock_count)

    if stockout_count:
        st.warning(
            "Review stockout-risk products and validate supplier delivery dates."
        )
    elif low_stock_count:
        st.info(
            "Some products are approaching their simulated safety-stock threshold."
        )
    else:
        st.success(
            "No immediate stockout alerts were identified in this simulation."
        )


# -----------------------------------------------------------------------------
# 18. Exports
# -----------------------------------------------------------------------------

st.divider()
st.header("Export Reports")

export1, export2, export3 = st.columns(3)

with export1:
    st.download_button(
        "Download Daily Demand",
        data=to_csv_bytes(filtered_daily),
        file_name="daily_demand.csv",
        mime="text/csv",
    )

with export2:
    st.download_button(
        "Download Forecast Results",
        data=to_csv_bytes(test_df),
        file_name="forecast_results.csv",
        mime="text/csv",
    )

with export3:
    st.download_button(
        "Download Inventory Results",
        data=to_csv_bytes(inventory_df),
        file_name="inventory_simulation.csv",
        mime="text/csv",
    )

if not recommendations_df.empty:
    st.download_button(
        "Download Replenishment Plan",
        data=to_csv_bytes(recommendations_df),
        file_name="replenishment_plan.csv",
        mime="text/csv",
    )


# -----------------------------------------------------------------------------
# 19. Footer
# -----------------------------------------------------------------------------

st.divider()
st.caption(
    "Retail Intelligence System | Demand Forecasting and Inventory Optimization"
)
st.caption(
    "Research prototype. Forecasts, supplier constraints and inventory "
    "recommendations are estimates. Validate operational decisions against "
    "actual warehouse and supplier records."
)
