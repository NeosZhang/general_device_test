import os
import shutil

def sync_files(origin_torch_path, npu_torch_path):
    if os.path.exists(npu_torch_path + "/testfiles_synchronized.txt"):
        with open(npu_torch_path + "/testfiles_synchronized.txt", "r") as file_sync:
            files_to_sync = file_sync.readlines()
            for line in files_to_sync:
                if line != "":
                    shutil.copy(origin_torch_path+line, npu_torch_path+line)
    
    if os.path.exists(npu_torch_path + "/testfolder_synchronized.txt"):
        with open(npu_torch_path + "/testfolder_synchronized.txt") as folder_sync:
            folders_to_sync = folder_sync.readlines()
            for line in folders_to_sync:
                if line != "":
                    shutil.copytree(origin_torch_path+line, npu_torch_path+line)