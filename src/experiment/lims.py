import os
import re
src_folder = "."
result_pattern = "result(\w+)"
result_msn_pattern = "result_msn_(\d+)(\w+)"
result_gsn_pattern = "result_gsn_(\d+)(\w+)"
result_gsn_A_pattern = "result_gsn_A_(\d+)(\w+)"
result_gsn_B_pattern = "result_gsn_B_(\d+)(\w+)"

def isResultFile(filename):
    matched = re.search(result_pattern, filename)
    if matched:
        return True
    return False

def isResultMsnFile(filename):
    matched = re.search(result_msn_pattern, filename)
    if matched:
        return True
    return False

def isResultGsnFile(filename):
    matched = re.search(result_gsn_pattern, filename)
    if matched:
        return True
    return False

def isResultGsnAFile(filename):
    matched = re.search(result_gsn_A_pattern, filename)
    if matched:
        return True
    return False

def isResultGsnBFile(filename):
    matched = re.search(result_gsn_B_pattern, filename)
    if matched:
        return True
    return False

def handleGSNFile():
    all_files = [os.path.join(src_folder, name) for name in os.listdir(src_folder) if os.path.isfile(os.path.join(src_folder, name)) and isResultGsnFile(name)]
    with open("ALL_GSN", 'w') as fw:
        for file in all_files:
            with open(file, 'r') as f:
                content = f.read()
                fw.write(content)

def handleGSN_A_File():
    all_files = [os.path.join(src_folder, name) for name in os.listdir(src_folder) if os.path.isfile(os.path.join(src_folder, name)) and isResultGsnAFile(name)]
    with open("ALL_GSN_A", 'w') as fw:
        for file in all_files:
            with open(file, 'r') as f:
                content = f.read()
                fw.write(content)

def handleGSN_B_File():
    all_files = [os.path.join(src_folder, name) for name in os.listdir(src_folder) if os.path.isfile(os.path.join(src_folder, name)) and isResultGsnBFile(name)]
    with open("ALL_GSN_B", 'w') as fw:
        for file in all_files:
            with open(file, 'r') as f:
                content = f.read()
                fw.write(content)

def handleMSN_File():
    all_files = [os.path.join(src_folder, name) for name in os.listdir(src_folder) if os.path.isfile(os.path.join(src_folder, name)) and isResultMsnFile(name)]
    with open("ALL_MSN", 'w') as fw:
        for file in all_files:
            with open(file, 'r') as f:
                content = f.read()
                fw.write(content)

if __name__ == '__main__':
        handleGSN_B_File()
        handleGSN_A_File()
        handleMSN_File()
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

