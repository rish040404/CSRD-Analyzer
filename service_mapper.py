import os
import pandas as pd
from utils import (
    map_opportunity_to_service_dynamic,
    map_all_opportunities,
    fetch_ergosign_services_text
)
from tabulate import tabulate


async def map_services():
    input_file = "opportunities.xlsx"

    if os.path.exists(input_file):
        df = pd.read_excel(input_file)
        mapped_df = await map_all_opportunities(df)
        print("\n✅ Mapping completed and saved to 'mapped_services.xlsx'.")
    else:
        print("❌ 'opportunities.xlsx' not found.")
