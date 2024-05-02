import subprocess

def run_script():
    python_exec_path = '/home/dpetresc/.conda/envs/instruct-qa-venv/bin/python3'
    output_file = "/home/dpetresc/instruct-qa/examples/data/nq/index/cagra/output.log"
    command = f"""
    su - dpetresc -c '
    cd instruct-qa/examples/ && 
    {python_exec_path} get_started_cagra.py > {output_file} 2>&1
    '
    """
    
    try:
        subprocess.run(command, shell=True, check=True)
        print("Script executed and output redirected to", output_file)
    except subprocess.CalledProcessError as e:
        print("Error executing the command:", str(e))
        print("Return code:", e.returncode)
        print("Command output:", e.output)

if __name__ == "__main__":
    run_script()
