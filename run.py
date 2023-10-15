import subprocess

paper_split_cmd = [
    "python",
    "./hw1_3/hw1_3_inf.py",
    "--arg1", './hw1_3_test_data',
    "--arg2", './pred_dir',
]

# subprocess.run(paper_split_cmd, shell=False)

paper_split_cmd = [
    "python",
    "mean_iou_evaluate.py",
    "-g", './hw1_3_test_data',
    "-p", './pred_dir',
]

subprocess.run(paper_split_cmd, shell=False)