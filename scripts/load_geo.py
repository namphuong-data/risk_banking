import pandas as pd
from sqlalchemy import create_engine, text

engine = create_engine("postgresql://namphuong:2104@localhost:5432/risk_banking")

print("Loading zip_geo_final → PostgreSQL ...")
df = pd.read_csv(
    "/home/namphuong/Desktop/DE_banking/data/zip_geo_final.csv",
    dtype={"zip_prefix": str, "state_id": str},
)

# Lọc lãnh thổ hải ngoại không có trong Lending Club
EXCLUDE = ["PR", "VI", "GU", "AS", "MP"]
df = df[~df["state_id"].isin(EXCLUDE)].reset_index(drop=True)
df = df[df["area_type"] != "nan"].reset_index(drop=True)

df.to_sql(
    "zip_geo",
    engine,
    if_exists="replace",
    index=False,
)

# Tạo index
with engine.begin() as conn:
    conn.execute(
        text("CREATE INDEX IF NOT EXISTS idx_zip_geo_prefix ON zip_geo (zip_prefix);")
    )
    conn.execute(
        text("CREATE INDEX IF NOT EXISTS idx_zip_geo_state ON zip_geo (state_id);")
    )

print(f"[✓] {len(df):,} zip_prefix loaded → bảng zip_geo")
print(df[["zip_prefix", "state_id", "income_bucket", "area_type"]].head())
