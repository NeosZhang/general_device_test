import shutil

def clean_all():
    shutil.rmtree("./device_torch/", ignore_errors=True)
    shutil.rmtree("./origin_torch/", ignore_errors=True)
    shutil.rmtree("../modified_tests/", ignore_errors=True)
    shutil.rmtree("../to_process_tests/", ignore_errors=True)

if __name__ == "__main__":
    clean_all()