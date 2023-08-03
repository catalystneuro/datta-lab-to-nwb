import subprocess
import os


def main():
    # with open("frozen_dependencies.txt", "w") as f:
    #     subprocess.run(["pip", "freeze"], stdout=f)
    os.system("pip freeze > frozen_dependencies.txt")


if __name__ == "__main__":
    main()
