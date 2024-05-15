import subprocess

def run_script():
    python_exec_path = '/home/dpetresc/.conda/envs/instruct-qa-venv/bin/python3'
    output_file = "/home/dpetresc/instruct-qa/examples/data/nq/index/cagra/output.log"
    script_path = "get_started_cagra.py"  # Relative path to be used with cwd
    working_dir = "/home/dpetresc/instruct-qa/examples"  # Working directory

    command = [
        'sudo', '-u', 'dpetresc', python_exec_path, script_path
    ]

    with open(output_file, "w") as output:
        try:
            subprocess.run(command, stdout=output, stderr=output, check=True, cwd=working_dir)
            print("Script executed and output redirected to", output_file)
        except subprocess.CalledProcessError as e:
            print("Error executing the command:", str(e))
            print("Return code:", e.returncode)

if __name__ == "__main__":
    run_script()

