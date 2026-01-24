#!/usr/bin/env python3
import pandas as pd
import argparse
import os


def format_excel(input_file, header_row):
    """
    Convert DSF Excel file to formatted CSV.

    Parameters
    ----------
    input_file : str
        Path to Excel file
    header_row : int
        Row where real header starts (1-based)

    Returns
    -------
    output_path : str
    """

    skiprows = header_row - 1
    SHEET = "Melt Curve Raw Data"

    out_folder = "formatted_results"
    os.makedirs(out_folder, exist_ok=True)

    base_name = os.path.splitext(os.path.basename(input_file))[0]
    output_path = os.path.join(
        out_folder, f"{base_name}_formatted.csv"
    )

    df = pd.read_excel(
        input_file,
        sheet_name=SHEET,
        skiprows=skiprows
    )

    wide = df.pivot(
        index="Temperature",
        columns="Well Position",
        values="Fluorescence"
    ).sort_index()

    wide.to_csv(output_path)

    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Pivot fluorescence data from an XLS file."
    )
    parser.add_argument("input_file")
    parser.add_argument("header_row", type=int)

    args = parser.parse_args()

    out = format_excel(args.input_file, args.header_row)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
