from rapidds import Dataset

files = ["clean.csv", "missing.csv", "messy.csv"]

for file in files:
    print(f"\n========== Testing {file} ==========\n")
    try:
        data = Dataset(file)
        data.analyze()
        data.suggest()
        data.explain()
    except Exception as e:
        print("Error occurred:", e)