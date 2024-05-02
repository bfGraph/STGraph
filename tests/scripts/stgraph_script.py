import argparse
import os
import subprocess


def main(args):
    version_number = args.version
    testpack_names = args.testpack_names

    for testpack in testpack_names:
        script_path = "v" + version_number + "/" + testpack + "/" + testpack + ".py"
        output_folder_path = "v" + version_number + "/" + testpack + "/outputs"
        if os.path.exists(script_path):
            subprocess.run(["python3", script_path, "-o", output_folder_path])
        else:
            print(f"Script {script_path} doesn't exists")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="STGraph Test Scripts")

    parser.add_argument(
        "-v",
        "--version",
        type=str,
        default="1_1_0",
        choices=["1_1_0"],
        help="Version of STGraph",
    )

    parser.add_argument(
        "-t",
        "--testpack-names",
        nargs="*",
        default=["temporal_tgcn_dataloaders", "gcn_dataloaders"],
        choices=["temporal_tgcn_dataloaders", "gcn_dataloaders"],
        help="Names of the testpacks to be executed",
    )

    args = parser.parse_args()

    main(args=args)
