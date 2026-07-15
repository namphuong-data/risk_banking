"""
fetch_fred_macro.py
───────────────────
Kéo dữ liệu macro kinh tế từ FRED API (2007–2018)
và xuất sang CSV tại thư mục chỉ định.

Cách chạy:
    pip install fredapi pandas pyarrow
    python fetch_fred_macro.py --api_key YOUR_KEY_HERE

Lấy API key miễn phí tại:
    https://fred.stlouisfed.org/docs/api/api_key.html
"""

import argparse
import os
import sys
import pandas as pd
from fredapi import Fred

# ── Cấu hình ──────────────────────────────────────────────────────────────────

OUTPUT_DIR = "/home/namphuong/Desktop/DE_banking/data"
OUTPUT_FILE = "fred_macro_2007_2018.csv"

START_DATE = "2006-01-01"
END_DATE = "2018-12-31"

# Các chuỗi FRED cần kéo: tên cột → series ID
SERIES = {
    "fed_funds_rate": "FEDFUNDS",  # Lãi suất Fed (tháng)
    "unemployment_rate": "UNRATE",  # Tỷ lệ thất nghiệp (tháng)
    "cpi": "CPIAUCSL",  # Lạm phát CPI (tháng)
    "gdp_growth": "A191RL1Q225SBEA",  # Tăng trưởng GDP thực (quý)
    "prime_rate": "DPRIME",  # Lãi suất prime (tháng) — proxy lãi suất cho vay
    "consumer_sentiment": "UMCSENT",  # Chỉ số niềm tin tiêu dùng (tháng)
}

# ── Hàm chính ─────────────────────────────────────────────────────────────────


def fetch_series(fred: Fred, series_id: str, col_name: str) -> pd.DataFrame:
    """Kéo một chuỗi FRED và trả về DataFrame với index là date."""
    print(f"  Đang kéo [{series_id}] → cột '{col_name}' ...", end=" ")
    try:
        s = fred.get_series(
            series_id,
            observation_start=START_DATE,
            observation_end=END_DATE,
        )
        print(f"OK ({len(s)} điểm dữ liệu)")
        return s.rename(col_name).to_frame()
    except Exception as e:
        print(f"LỖI: {e}")
        sys.exit(1)


def build_macro_table(api_key: str) -> pd.DataFrame:
    """Kéo toàn bộ chuỗi, chuẩn hóa về tháng và trả về bảng macro."""
    fred = Fred(api_key=api_key)

    print("\n[1/4] Kéo dữ liệu từ FRED API ...")
    frames = []
    for col_name, series_id in SERIES.items():
        df = fetch_series(fred, series_id, col_name)
        frames.append(df)

    # Merge tất cả theo date index
    macro_raw = frames[0].join(frames[1:], how="outer")
    macro_raw.index = pd.to_datetime(macro_raw.index)
    macro_raw = macro_raw.sort_index()

    print("\n[2/4] Chuẩn hóa về tần suất tháng ...")

    # GDP là chuỗi quý → forward-fill xuống tháng
    macro_raw["gdp_growth"] = macro_raw["gdp_growth"].ffill()

    # Tạo cột issue_month (format YYYY-MM) để join với Lending Club
    macro_raw["issue_month"] = macro_raw.index.to_period("M").astype(str)

    # Gộp theo tháng — lấy giá trị trung bình tháng
    macro = macro_raw.groupby("issue_month", as_index=False).agg(
        fed_funds_rate=("fed_funds_rate", "mean"),
        unemployment_rate=("unemployment_rate", "mean"),
        cpi=("cpi", "mean"),
        gdp_growth=("gdp_growth", "last"),
        prime_rate=("prime_rate", "mean"),
        consumer_sentiment=("consumer_sentiment", "mean"),
    )

    print(f"     → {len(macro)} tháng (kỳ vọng: 144 tháng cho 2007–2018)")

    # Tính thêm cột dẫn xuất hữu ích cho phân tích IFRS 9
    print("\n[3/4] Tính các chỉ số dẫn xuất ...")

    # Thay đổi tỷ lệ thất nghiệp so với tháng trước (leading indicator)
    macro["unemployment_mom_change"] = macro["unemployment_rate"].diff().round(3)

    # CPI YoY — lạm phát so với cùng kỳ năm trước
    macro["cpi_yoy_pct"] = macro["cpi"].pct_change(periods=12).mul(100).round(3)

    # Recession flag: GDP tăng trưởng âm 2 quý liên tiếp
    macro["recession_flag"] = (
        (macro["gdp_growth"] < 0) & (macro["gdp_growth"].shift(1) < 0)
    ).astype(int)

    print("     → Đã tính: unemployment_mom_change, cpi_yoy_pct, recession_flag")

    return macro


def validate(macro: pd.DataFrame) -> None:
    """Kiểm tra nhanh tính hợp lệ của dữ liệu."""
    print("\n[4/4] Kiểm tra dữ liệu ...")

    assert len(macro) == 156, f"Thiếu tháng: có {len(macro)}, cần 144"

    # Kiểm tra khủng hoảng 2008 — Fed rate phải giảm mạnh
    fed_2008 = macro.loc[macro["issue_month"] == "2008-12", "fed_funds_rate"].values[0]
    assert fed_2008 < 1.0, f"Fed rate 12/2008 phải < 1%, thực tế: {fed_2008}"

    # Đỉnh thất nghiệp 2009–2010 phải > 9%
    max_unemp = macro["unemployment_rate"].max()
    assert max_unemp > 9.0, f"Đỉnh thất nghiệp phải > 9%, thực tế: {max_unemp}"

    # Recession flag phải bật trong giai đoạn 2008–2009
    recession_months = macro.loc[macro["recession_flag"] == 1, "issue_month"].tolist()
    assert len(recession_months) > 0, "Không phát hiện được recession 2008–2009"

    null_counts = macro.isnull().sum()
    if null_counts.any():
        print(f"     ⚠ Cảnh báo: có giá trị null:\n{null_counts[null_counts > 0]}")
    else:
        print("     ✓ Không có giá trị null")

    print(f"     ✓ {len(macro)} tháng hợp lệ (2007-01 → 2018-12)")
    print(
        f"     ✓ Fed rate thấp nhất: {macro['fed_funds_rate'].min():.2f}% (hậu khủng hoảng)"
    )
    print(f"     ✓ Thất nghiệp đỉnh:   {max_unemp:.1f}% (khủng hoảng 2009)")
    print(f"     ✓ Recession months:   {recession_months[:3]} ...")


def save_csv(macro: pd.DataFrame) -> str:
    """Lưu DataFrame ra CSV tại OUTPUT_DIR."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, OUTPUT_FILE)
    macro.to_csv(output_path, index=False, encoding="utf-8-sig")
    size_kb = os.path.getsize(output_path) / 1024
    print(f"\n✅ Đã lưu: {output_path}")
    print(
        f"   Kích thước: {size_kb:.1f} KB  |  {len(macro)} dòng × {len(macro.columns)} cột"
    )
    return output_path


def preview(macro: pd.DataFrame) -> None:
    """In preview và mô tả thống kê."""
    print("\n── Preview 5 dòng đầu ──────────────────────────────────────────")
    print(macro.head().to_string(index=False))

    print("\n── Thống kê mô tả ──────────────────────────────────────────────")
    desc = macro.describe().loc[["mean", "min", "max"]].round(3)
    print(desc.to_string())

    print("\n── Giai đoạn khủng hoảng 2008–2009 ────────────────────────────")
    crisis = macro[macro["issue_month"].between("2008-09", "2009-06")]
    cols = [
        "issue_month",
        "fed_funds_rate",
        "unemployment_rate",
        "gdp_growth",
        "recession_flag",
    ]
    print(crisis[cols].to_string(index=False))


# ── Entry point ───────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Kéo dữ liệu macro FRED 2007–2018 và xuất CSV"
    )
    parser.add_argument(
        "--api_key",
        required=True,
        help="FRED API key — lấy miễn phí tại https://fred.stlouisfed.org/docs/api/api_key.html",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  FRED Macro Data Fetcher — Credit Risk Project")
    print(f"  Giai đoạn: {START_DATE} → {END_DATE}")
    print(f"  Output:    {os.path.join(OUTPUT_DIR, OUTPUT_FILE)}")
    print("=" * 60)

    macro = build_macro_table(args.api_key)
    validate(macro)
    save_csv(macro)
    preview(macro)

    print("\n── Hướng dẫn join với Lending Club (DuckDB) ───────────────────")
    print("""
import duckdb
conn = duckdb.connect()
result = conn.execute(\"\"\"
    SELECT
        l.issue_month,
        COUNT(*)                   AS loan_count,
        AVG(m.unemployment_rate)   AS avg_unemployment,
        AVG(m.fed_funds_rate)      AS avg_fed_rate,
        MAX(m.recession_flag)      AS in_recession
    FROM read_parquet('data/lending_club.parquet') l
    LEFT JOIN read_csv_auto('data/fred_macro_2007_2018.csv') m
        ON l.issue_month = m.issue_month
    GROUP BY l.issue_month
    ORDER BY l.issue_month
\"\"\").df()
""")


if __name__ == "__main__":
    main()
