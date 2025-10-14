import pandas as pd
import os
from dotenv import load_dotenv

def load_env():
    """Load environment variables from .env file."""
    load_dotenv()
    return {
        "gas_file": os.getenv("GAS_RESOURCE_FILE"),
        "agg_file": os.getenv("AGGREGATED_MATRIX_FILE"),
        "output_file": os.getenv("OUTPUT_FILE")
    }

def load_gas_resources(excel_path):
    """Load gas resources and return a mapping from ERCOT_UnitCode to Name."""
    df = pd.read_excel(excel_path, engine="openpyxl")
    df["ERCOT_UnitCode"] = df["ERCOT_UnitCode"].astype(str).str.strip()
    df["Name"] = df["Name"].astype(str).str.strip()
    return dict(zip(df["ERCOT_UnitCode"], df["Name"]))

def update_resource_names(csv_path, mapping, output_path):
    """Replace Resource Name values in the CSV using the provided mapping."""
    df = pd.read_csv(csv_path)
    df["Resource Name"] = df["Resource Name"].astype(str).str.strip()
    df["Resource Name"] = df["Resource Name"].map(mapping).fillna(df["Resource Name"])
    df.to_csv(output_path, index=False)
    print(f"✅ Updated file saved to: {output_path}")

def main():
    config = load_env()
    print("🔄 Loading gas resource mapping...")
    mapping = load_gas_resources(config["gas_file"])
    
    print("🔄 Updating aggregated base point matrix...")
    update_resource_names(config["agg_file"], mapping, config["output_file"])

if __name__ == "__main__":
    main()