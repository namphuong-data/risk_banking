"""
fetch_census_api.py
────────────────────
Kéo dữ liệu địa lý và kinh tế từ Census ACS 5-Year 2018 API
và merge với file SimpleMaps để tạo bảng zip_geo_final.csv hoàn chỉnh.

Chuẩn bị:
  1. Đăng ký Census API key miễn phí tại:
     https://api.census.gov/data/key_signup.html
     (nhận key qua email trong vài phút)

  2. Đặt SimpleMaps file tại thư mục DATA_DIR với tên:
     uszips.csv  (download từ https://simplemaps.com/data/us-zips)

Cách chạy:
    python fetch_census_api.py --api_key YOUR_CENSUS_KEY_HERE

Output:
    zip_income_census.csv  — income theo ZCTA từ Census
    zip_geo_final.csv      — bảng hoàn chỉnh merge SimpleMaps + Census
"""

import argparse
import os
import sys
import time
import requests
import pandas as pd

# ── CONFIG ────────────────────────────────────────────────────────────────────

DATA_DIR   = "/home/namphuong/Desktop/DE_banking/data"
FILE_ZIPS  = os.path.join(DATA_DIR, "uszips.csv")          # SimpleMaps file

OUTPUT_INCOME  = os.path.join(DATA_DIR, "zip_income_census.csv")
OUTPUT_FINAL   = os.path.join(DATA_DIR, "zip_geo_final.csv")

CENSUS_YEAR    = "2018"     # Dùng 2018 để khớp giai đoạn Lending Club
ACS_DATASET    = "acs/acs5"

# Các biến Census ACS cần kéo
# Chi tiết tại: https://api.census.gov/data/2018/acs/acs5/variables.html
CENSUS_VARIABLES = {
    "B19013_001E": "median_household_income",   # Thu nhập hộ gia đình trung vị
    "B01003_001E": "total_population",           # Tổng dân số (cross-check)
    "B17001_002E": "population_below_poverty",   # Dân số dưới ngưỡng nghèo
    "B23025_005E": "unemployed_population",      # Dân số thất nghiệp
    "B25077_001E": "median_home_value",          # Giá trị nhà trung vị
    "B15003_022E": "bachelors_degree_pop",       # Dân số có bằng cử nhân
}

# ── Helpers ───────────────────────────────────────────────────────────────────

def check_files():
    if not os.path.exists(FILE_ZIPS):
        print(f"\n[LỖI] Không tìm thấy file SimpleMaps: {FILE_ZIPS}")
        print("      Download tại: https://simplemaps.com/data/us-zips")
        print("      Đặt file uszips.csv vào thư mục data/")
        sys.exit(1)
    print(f"[✓] SimpleMaps file: {FILE_ZIPS}")


def test_api_key(api_key: str) -> bool:
    """Kiểm tra API key hợp lệ bằng query thử."""
    url = (
        f"https://api.census.gov/data/{CENSUS_YEAR}/{ACS_DATASET}"
        f"?get=B19013_001E"
        f"&for=zip%20code%20tabulation%20area:*"  # dùng wildcard * thay vì zip cụ thể
        f"&in=state:36"  # giới hạn 1 bang (New York) để test nhanh
        f"&key={api_key}"
    )
    try:
        r = requests.get(url, timeout=30)
        if r.status_code == 200 and r.text.startswith("[["):
            print("[✓] Census API key hợp lệ")
            return True
        else:
            print(f"[LỖI] API key không hợp lệ: {r.text[:150]}")
            return False
    except Exception as e:
        print(f"[LỖI] Không kết nối được Census API: {e}")
        return False


def fetch_census_variable_batch(api_key: str, variables: dict) -> pd.DataFrame:
    var_list = list(variables.keys())
    var_string = ",".join(var_list)

    # Dùng wildcard * — bắt buộc với Census API mới
    url = (
        f"https://api.census.gov/data/{CENSUS_YEAR}/{ACS_DATASET}"
        f"?get=NAME,{var_string}"
        f"&for=zip%20code%20tabulation%20area:*"  # * = tất cả ZCTA
        f"&key={api_key}"
    )

    print(f"\n  Kéo {len(var_list)} biến cho tất cả ZCTA ...")

    retries = 3
    for attempt in range(retries):
        try:
            r = requests.get(url, timeout=120)
            if r.status_code == 200:
                break
            print(f"  [!] HTTP {r.status_code} — thử lại ({attempt+1}/{retries})")
            time.sleep(3)
        except requests.Timeout:
            print(f"  [!] Timeout — thử lại ({attempt+1}/{retries})")
            time.sleep(5)
    else:
        print("[LỖI] Census API không phản hồi sau 3 lần thử")
        sys.exit(1)

    data = r.json()
    df = pd.DataFrame(data[1:], columns=data[0])

    df = df.rename(
        columns={
            "zip code tabulation area": "zcta",
            **variables,
        }
    )

    for col in variables.values():
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df.loc[df[col] < 0, col] = None

    print(f"  [✓] Kéo xong: {len(df):,} ZCTA")
    return df


def process_census(df: pd.DataFrame) -> pd.DataFrame:
    """Tính các chỉ số dẫn xuất từ dữ liệu Census."""

    # Tỷ lệ nghèo (%)
    df["poverty_rate_pct"] = (
        df["population_below_poverty"] / df["total_population"] * 100
    ).round(2)

    # Tỷ lệ thất nghiệp địa phương (%)
    df["local_unemployment_pct"] = (
        df["unemployed_population"] / df["total_population"] * 100
    ).round(2)

    # Tỷ lệ có bằng đại học (%)
    df["college_rate_pct"] = (
        df["bachelors_degree_pop"] / df["total_population"] * 100
    ).round(2)

    # Tạo zip_prefix (3 ký tự) để join với Lending Club
    df["zip_prefix"] = df["zcta"].astype(str).str[:3]

    # Gộp theo zip_prefix — lấy median của các chỉ số
    zip_agg = (
        df.groupby("zip_prefix", as_index=False)
        .agg(
            median_household_income = ("median_household_income", "median"),
            median_home_value       = ("median_home_value",       "median"),
            avg_poverty_rate_pct    = ("poverty_rate_pct",        "mean"),
            avg_local_unemployment  = ("local_unemployment_pct",  "mean"),
            avg_college_rate_pct    = ("college_rate_pct",        "mean"),
            zcta_count              = ("zcta",                    "count"),
        )
        .round(2)
    )

    return zip_agg


def load_simplemaps() -> pd.DataFrame:
    """Load và chuẩn hóa file SimpleMaps."""
    print(f"\n[2/4] Load SimpleMaps ...")
    df = pd.read_csv(FILE_ZIPS, dtype={"zip": str, "zcta": str}, low_memory=False)

    # Chỉ giữ cột cần thiết
    keep = ["zip", "lat", "lng", "city", "state_id", "state_name",
            "county_name", "population", "density", "timezone"]
    keep = [c for c in keep if c in df.columns]
    df   = df[keep].copy()

    # Bỏ ZIP quân sự (military=True nếu có cột đó)
    if "military" in df.columns:
        df = df[df["military"] != True]

    # Tạo zip_prefix
    df["zip_prefix"] = df["zip"].str[:3]

    # Gộp theo zip_prefix
    zip_geo = (
        df.groupby("zip_prefix", as_index=False)
        .agg(
            state_id    = ("state_id",   "first"),
            state_name  = ("state_name", "first"),
            county_name = ("county_name","first"),
            population  = ("population", "sum"),
            avg_density = ("density",    "mean"),
            city_sample = ("city",       "first"),
        )
        .round(2)
    )

    print(f"  [✓] {len(zip_geo):,} zip_prefix từ SimpleMaps")
    return zip_geo


def add_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Thêm các cột phân loại dẫn xuất."""

    # Income bucket — phân tầng thu nhập
    df["income_bucket"] = pd.cut(
        df["median_household_income"],
        bins   = [0, 35000, 50000, 70000, 100000, 999999],
        labels = ["Very Low (<35k)", "Low (35-50k)", "Medium (50-70k)",
                  "High (70-100k)", "Very High (>100k)"],
    ).astype(str)
    df.loc[df["median_household_income"].isna(), "income_bucket"] = "Unknown"

    # Area type — phân loại đô thị hóa từ mật độ dân số
    df["area_type"] = pd.cut(
        df["avg_density"],
        bins   = [0, 50, 500, 2000, 10000, 999999],
        labels = ["Rural", "Suburban", "Urban", "Dense Urban", "Mega Urban"],
    ).astype(str)
    df.loc[df["avg_density"].isna(), "area_type"] = "Unknown"

    # Risk tier — tổng hợp rủi ro vùng cho concentration analysis
    # Vùng nghèo + thất nghiệp cao + thu nhập thấp = rủi ro cao hơn
    conditions = (
        (df["avg_poverty_rate_pct"].fillna(0)   > 20) |
        (df["avg_local_unemployment"].fillna(0) > 10) |
        (df["median_household_income"].fillna(999999) < 40000)
    )
    df["high_risk_area"] = conditions.astype(int)

    return df


def validate(df: pd.DataFrame):
    """Kiểm tra nhanh kết quả cuối."""
    print("\n[4/4] Validate kết quả ...")
    print(f"  Tổng zip_prefix: {len(df):,}")
    print(f"  Null median_income: {df['median_household_income'].isna().sum():,}")
    print(f"  Income range: ${df['median_household_income'].min():,.0f} "
          f"– ${df['median_household_income'].max():,.0f}")
    print(f"\n  Phân phối income_bucket:")
    print(df["income_bucket"].value_counts().to_string())
    print(f"\n  Phân phối area_type:")
    print(df["area_type"].value_counts().to_string())
    print(f"\n  High-risk areas: {df['high_risk_area'].sum():,} zip_prefix")
    print(f"\n  Preview 5 dòng:")
    cols = ["zip_prefix","state_id","median_household_income",
            "income_bucket","area_type","avg_poverty_rate_pct","high_risk_area"]
    print(df[cols].head().to_string(index=False))


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Kéo dữ liệu Census ACS API và merge với SimpleMaps"
    )
    parser.add_argument(
        "--api_key", required=True,
        help="Census API key — đăng ký miễn phí tại https://api.census.gov/data/key_signup.html"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  Census ACS Data Fetcher")
    print(f"  Dataset : ACS 5-Year {CENSUS_YEAR}")
    print(f"  Output  : {DATA_DIR}")
    print("=" * 60)

    # Bước 0 — Kiểm tra
    check_files()
    if not test_api_key(args.api_key):
        sys.exit(1)

    # Bước 1 — Kéo Census ACS
    print(f"\n[1/4] Kéo dữ liệu từ Census ACS {CENSUS_YEAR} API ...")
    df_raw    = fetch_census_variable_batch(args.api_key, CENSUS_VARIABLES)
    zip_census = process_census(df_raw)

    zip_census.to_csv(OUTPUT_INCOME, index=False, encoding="utf-8-sig")
    print(f"  [✓] Lưu: {OUTPUT_INCOME} — {len(zip_census):,} zip_prefix")

    # Bước 2 — Load SimpleMaps
    zip_geo = load_simplemaps()

    # Bước 3 — Merge
    print("\n[3/4] Merge SimpleMaps + Census ...")
    zip_final = zip_geo.merge(zip_census, on="zip_prefix", how="left")
    zip_final = add_derived_columns(zip_final)

    zip_final.to_csv(OUTPUT_FINAL, index=False, encoding="utf-8-sig")
    print(f"  [✓] Lưu: {OUTPUT_FINAL} — {len(zip_final):,} dòng × {len(zip_final.columns)} cột")

    # Bước 4 — Validate
    validate(zip_final)

    print("\n" + "=" * 60)
    print("  HOÀN THÀNH")
    print(f"  zip_income_census.csv : {len(zip_census):,} zip_prefix")
    print(f"  zip_geo_final.csv     : {len(zip_final):,} zip_prefix")
    print("=" * 60)
    print("""
  Hướng dẫn join với Lending Club (DuckDB):

  SELECT
      l.addr_state,
      z.income_bucket,
      z.area_type,
      COUNT(*)                              AS loan_count,
      AVG(z.avg_poverty_rate_pct)           AS avg_poverty,
      ROUND(100.0 * SUM(
          CASE WHEN l.debt_group >= 3 THEN 1 ELSE 0 END
      ) / COUNT(*), 2)                      AS npl_ratio_pct
  FROM accepted_loans l
  LEFT JOIN read_csv_auto('data/zip_geo_final.csv') z
      ON l.zip_code = z.zip_prefix
  GROUP BY l.addr_state, z.income_bucket, z.area_type
  ORDER BY npl_ratio_pct DESC
    """)


if __name__ == "__main__":
    main()
