import os
import re
import json
import glob
import pickle
import r2pipe
import bisect
import subprocess
import networkx as nx
from tqdm import tqdm

from capstone import *
md = Cs(CS_ARCH_X86, CS_MODE_64)

md.detail = True

syscall_path = '/usr/include/x86_64-linux-gnu/asm/unistd_64.h'
cfg_dir = '/root/access_control_program_level/coreutils-test/cfgs'
log_dir = '/root/access_control_program_level/coreutils-test'

slice_save_dir = '/root/access_control_program_level/coreutils-test/slice_syscall'
if not os.path.exists(slice_save_dir): os.makedirs(slice_save_dir)

lib_paths = [
    '/usr/lib/x86_64-linux-gnu/', 
    '/usr/local/lib/'
]

reg_map = {"rdi":"rdi", "rsi":"rsi", "rdx":"rdx", "rcx":"rcx",  "r8":"r8",  "r9":"r9",  "rax":"rax", "rsp":"rsp", "rbp":"rbp",
            "edi":"rdi", "esi":"rsi", "edx":"rdx", "ecx":"rcx", "r8d":"r8", "r9d":"r9", "eax":"rax", "esp":"rsp", "ebp":"rbp",
             "di":"rdi",  "si":"rsi",  "dx":"rdx",  "cx":"rcx", "r8w":"r8", "r9w":"r9",  "ax":"rax",  "sp":"rsp",  "bp":"rbp",
            "dil":"rdi", "sil":"rsi",  "dl":"rdx",  "cl":"rcx", "r8b":"r8", "r9b":"r9",  "al":"rax", "spl":"rsp", "bpl":"rbp" ,
            "xmm0":"zmm0","ymm0":"zmm0","zmm0":"zmm0",
            "xmm1":"zmm1","ymm1":"zmm1","zmm1":"zmm1",
            "xmm2":"zmm2","ymm2":"zmm2","zmm2":"zmm2",
            "xmm3":"zmm3","ymm3":"zmm3","zmm3":"zmm3",
            }


# 从 /usr/include/x86_64-linux-gnu/asm/unistd_64.h 中获取系统调用号和 具体系统调用的映射关系
def init_syscall_num(syscall_path):
    syscall_map = {}
    with open(syscall_path, 'r') as fd:
        lines = fd.readlines()
    for line in lines:
        if '__NR_' in line:
            n = int(line.split(' ')[2])
            syscall = line.split(' ')[1][5:]
            syscall_map[n] = syscall
    return syscall_map

def find_log_files(directory):
    log_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.log'):
                log_files.append(os.path.join(root, file))
    return log_files

def find_absolute_lib_path(lib):
    for path in lib_paths:
        ab_lib = os.path.join(path, lib)
        if os.path.exists(ab_lib):
            return ab_lib
    return None

def extract_cfg_pkl(target_dir):
    cfg_pkls = glob.glob(f'{target_dir}/*.pkl')
    cfg_list = []
    for pkl in cfg_pkls:
        with open(pkl, 'rb') as fd:
            g:nx.DiGraph = pickle.load(fd)
            cfg_list.append(g)

    return cfg_list

def get_capstone_disasm(instr):
    r, w = [], []

    code = bytes.fromhex(instr['bytes'])
    addr = instr['offset']
    i_cap = next(md.disasm(code, addr))

    (regs_read, regs_write) = i_cap.regs_access()
    if regs_read:
        for reg in regs_read:
            # print "\tRead REG: %s" %(i.reg_name(reg))
            r.append("%s"%i_cap.reg_name(reg))
    if regs_write:
        for reg in regs_write:
            # print "\tWrite REG: %s" %(i.reg_name(reg))
            w.append("%s"%i_cap.reg_name(reg))

    return r, w, i_cap

def check_param(instr, reg_count):
    r, w, i_cap = get_capstone_disasm(instr)

    for reg in w:
        if not reg in reg_map.keys(): continue
        reg = reg_map[reg]
        if reg_count[reg] >= 1: continue
        else: reg_count[reg] += 1
        
        return True
        
    return False

def check_rax(instr, reg_count):
    r, w, i_cap = get_capstone_disasm(instr)

    for reg in r:
        if not reg in reg_map.keys(): continue
        reg = reg_map[reg]

        if reg != 'rax': continue

        if reg_count[reg] >= 1: continue
        else: reg_count[reg] += 1

        return True

    return False

def slice_syscall(addr, r2):
    slice_dict, ninstrs = {}, 0

    func_info = r2.cmdj(f'pdfj @{addr}')
    instrs = func_info['ops']
    offsets = [instr['offset'] for instr in instrs]

    idx = bisect.bisect_right(offsets, addr) - 1

    while idx < len(instrs):
        instr = instrs[idx]
        if instr['opcode'] == 'syscall':
            syscall_addr, syscall_idx = instr['offset'], idx
            break
        idx += 1

    back_instr_list = []
    reg_count = {reg: 0 for reg in reg_map.keys()}
    idx = syscall_idx - 1
    while idx >= 0:
        instr = instrs[idx]
        if instr['type'] == 'call': break

        if check_param(instr, reg_count):  back_instr_list.append(instr['disasm'])
        idx -= 1

    back_instr_list.reverse()
    slice_dict = {i+1: instr for i, instr in enumerate(back_instr_list)}

    ninstrs = len(slice_dict.keys()) + 1
    slice_dict[ninstrs] = instrs[syscall_idx]['disasm']

    reg_count = {"rax": 0}
    idx = syscall_idx + 1
    while idx < len(instrs):
        instr = instrs[idx]
        if instr['type'] == 'call': break
        if check_rax(instr, reg_count):
            ninstrs += 1
            slice_dict[ninstrs] = instr['disasm']
        idx += 1

    return slice_dict

def slice_cfg(cfg, r2_dict, slice_save_dir):
    syscall_list = [nd for nd, data in cfg.nodes(data=True) if data['syscall'] == 1]
    # bin@func@offset
    for nd in tqdm(syscall_list):
        lib = nd.split('@')[0]
        offset = int(nd.split('@')[-1])

        save_path = os.path.join(slice_save_dir, f'{lib}_{offset}.json')
        if os.path.exists(save_path): continue

        if lib not in r2_dict.keys():
            ab_lib = find_absolute_lib_path(lib)
            if ab_lib is None: raise LookupError(f'can not find lib {lib}')
            lib_r2 = r2pipe.open(ab_lib)
            lib_r2.cmd('aaa')
            r2_dict[lib] = lib_r2
        
        r2 = r2_dict[lib]
        slice_dict = slice_syscall(offset, r2)

        with open(save_path, 'w') as fd:
            json.dump(slice_dict, fd, indent=4)
        # print(f'{nd} has been sliced to {save_path}')
    
def start_slice_all(custom_cfg_path=[]):

    cfg_list = extract_cfg_pkl(cfg_dir)

    r2_dict = {} # bin -> r2

    # expand other cfgs
    for cfg_path in custom_cfg_path:
        with open(cfg_path, 'rb') as fd:
            g:nx.DiGraph = pickle.load(fd)
            cfg_list.append(g)

    for i, cfg in enumerate(cfg_list):
        print(f'starting slice for cfg {i} / {len(cfg_list)}')
        slice_cfg(cfg, r2_dict, slice_save_dir)


if __name__ == '__main__':
    start_slice_all()

