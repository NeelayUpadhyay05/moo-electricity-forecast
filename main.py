from src.utils.seed import set_seed


def main():
    set_seed(42)
    print("Reproducibility initialized.")


if __name__ == "__main__":
    main()