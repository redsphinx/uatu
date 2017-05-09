from subprocess import check_output


def execute_process(command_shell):
    stdout = check_output(command_shell, shell=True).strip()
    if not isinstance(stdout, (str)):
        stdout = stdout.decode()
    return stdout


def number_of_gpus():
    return execute_process("nvidia-smi -q -d MEMORY | grep 'GPUs'")[-1]


def is_gpu_busy(number):
    total_memory = execute_process(
        "nvidia-smi -q -d MEMORY -i %d | grep 'Total'" % number).split(' MiB')[0][-4:].split(':')[-1].strip()
    used_memory = execute_process(
        "nvidia-smi -q -d MEMORY -i %d | grep 'Used'" % number).split(' MiB')[0][-4:].split(':')[-1].strip()
    use_percentage = float(used_memory) / float(total_memory)
    if use_percentage > 0.05:
        print('gpu is busy')
        return True
    else:
        print('gpu is not busy')
        return False
