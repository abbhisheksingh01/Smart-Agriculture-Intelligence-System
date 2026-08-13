from src.data_loader import (
    load_cost_yield,
    load_production_history,
    load_production_index,
    load_crop_variety,
    load_area_production_yield
)

def main():
    datasets = {
        "Cost & Yield": load_cost_yield(),
        "Production History": load_production_history(),
        "Production Index": load_production_index(),
        "Crop Variety": load_crop_variety(),
        "Area Production Yield": load_area_production_yield(),
    }

    for name, df in datasets.items():
        print("=" * 60)
        print(name)
        print("=" * 60)
        print(df.head())
        print("\nShape:", df.shape)
        print("\nColumns:")
        print(df.columns.tolist())
        print()

if __name__ == "__main__":
    main()