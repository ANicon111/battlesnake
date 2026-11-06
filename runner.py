import subprocess, os, atexit

cmd = f"python3 {os.path.dirname(os.path.realpath(__file__))}/example_data/example.py"

p = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True)
out, err = p.communicate()
print(out)


def cleanup():
    p.terminate()


atexit.register(cleanup)

exit()
