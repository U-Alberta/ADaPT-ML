import subprocess
import sys


def check_log(log):
    try:
        subprocess.run(["grep", "-i", "-e", "\"error\"", "-e", "\"warning\"", log], check=True)
        print("ERROR: Problems with program execution found in the log file.")
        sys.exit(1)
    except subprocess.CalledProcessError:
        pass


def process_annotations(choice):
    try:
        subprocess.run(["python", "./ls/process_annotations.py", "example_annotations.json",
                        "example", choice], check=True)
    except subprocess.CalledProcessError:
        print("ERROR: Annotation processing failed.")
        sys.exit(1)


print("=== STARTING LABEL STUDIO TESTS WITH EXAMPLE DATA ===")

print("Try sampling some tasks...")
try:
    subprocess.run(["python", "./ls/sample_tasks.py",
                    "example_data", "txt", "30", "example", "--filename", "example_tasks"], check=True)
except subprocess.CalledProcessError:
    print("ERROR: Task sampling failed.")
    sys.exit(1)
check_log("/tasks/example_tasks_log.txt")

print("Try processing annotations with random gold choice...")
process_annotations("random")
check_log("/annotations/example/process_log.txt")

print("Try processing annotations with drop gold choice...")
process_annotations("drop")
check_log("/annotations/example/process_log.txt")

print("Try processing annotations with majority gold choice...")
process_annotations("majority")
check_log("/annotations/example/process_log.txt")

print("Try processing annotations with worker_1 gold choice...")
process_annotations("1")
check_log("/annotations/example/process_log.txt")

print("Check annotator agreement...")
try:
    subprocess.run(["python", "./ls/annotator_agreement.py", "example"], check=True)
except subprocess.CalledProcessError:
    print("ERROR: Annotator agreement check failed.")
    sys.exit(1)
check_log("/annotations/example/agreement_log.txt")
try:
    subprocess.run(["grep", "-q", "0.43847133757961776", "/annotations/example/agreement_log.txt"], check=True)
except subprocess.CalledProcessError:
    print("ERROR: Annotator agreement calculation is not 0.43847133757961776.")
    sys.exit(1)

print("=== LABEL STUDIO TESTS WITH EXAMPLE DATA COMPLETED SUCCESSFULLY ===")
