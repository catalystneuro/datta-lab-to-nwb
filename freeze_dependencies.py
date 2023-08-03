import subprocess


def main():
    with open("frozen_dependencies.txt", "w") as f:
        subprocess.run(["pip", "freeze"], stdout=f)


if __name__ == "__main__":
    main()
