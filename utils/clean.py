import os
import shutil

main_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def clean_all():
    shutil.rmtree(main_dir + "/device_torch/", ignore_errors=True)
    shutil.rmtree(main_dir + "/origin_torch/", ignore_errors=True)
    shutil.rmtree(main_dir + "/tests/", ignore_errors=True)


if __name__ == "__main__":
    clean_all()
