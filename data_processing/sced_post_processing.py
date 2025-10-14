import pandas as pd
import os
from dotenv import load_dotenv

def load_env():
    """Load environment variables from .env file."""
    load_dotenv()
    return {
        "gas_file": os.getenv("GAS_RESOURCE_FILE"),
        "agg_file": os.getenv("AGGREGATED_MATRIX_FILE"),
        "output_file": os.getenv("OUTPUT_FILE"),
        "tracker_file": os.getenv("TRACKER_FILE", "resource_name_tracker.csv")
    }

def load_gas_resources(excel_path):
    """Load gas resources and return a mapping from ERCOT_UnitCode to Name."""
    df = pd.read_excel(excel_path, engine="openpyxl")
    df["ERCOT_UnitCode"] = df["ERCOT_UnitCode"].astype(str).str.strip()
    df["Name"] = df["Name"].astype(str).str.strip()
    return dict(zip(df["ERCOT_UnitCode"], df["Name"]))

def update_resource_names(csv_path, mapping, output_path, tracker_path):
    """Replace Resource Name values and track matched/unmatched entries."""
    df = pd.read_csv(csv_path)
    df["Resource Name"] = df["Resource Name"].astype(str).str.strip()

    # Track matches
    df["Matched Name"] = df["Resource Name"].map(mapping)
    df["Was Replaced"] = df["Matched Name"].notna()

    # Save tracker
    tracker_df = df[["Resource Name", "Matched Name", "Was Replaced"]]
    tracker_df.to_csv(tracker_path, index=False)

    # Replace values
    df["Resource Name"] = df["Matched Name"].fillna(df["Resource Name"])
    df.drop(columns=["Matched Name", "Was Replaced"], inplace=True)

    df.to_csv(output_path, index=False)
    print(f"✅ Updated file saved to: {output_path}")
    print(f"📊 Tracker file saved to: {tracker_path}")

def main():
    config = load_env()
    print("🔄 Loading gas resource mapping...")
    mapping = load_gas_resources(config["gas_file"])
    
    print("🔄 Updating aggregated base point matrix and tracking replacements...")
    update_resource_names(config["agg_file"], mapping, config["output_file"], config["tracker_file"])

if __name__ == "__main__":
    main()