from rapidds import Dataset

def main():
    # Replace with your own dataset file
    data = Dataset("students.csv")

    print("\n--- ANALYSIS ---\n")
    data.analyze()

    print("\n--- SUGGESTIONS ---\n")
    data.suggest()

    print("\n--- EXPLANATION ---\n")
    data.explain()


if __name__ == "__main__":
    main()
