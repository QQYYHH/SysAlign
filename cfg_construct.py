import r2pipe
import argparse
import os
import re
import sys
import networkx as nx
import pickle
import matplotlib.pyplot as plt
from networkx.drawing.nx_agraph import to_agraph

from queue import Queue
# sys.path = ['/root/access_control_program_level/test'] + sys.path
from find_address_taken import AddressTaken

lib_paths = [
    '/usr/lib/x86_64-linux-gnu/', 
    '/usr/local/lib/'
]

# 定义颜色
RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
BLUE = "\033[34m"
RESET = "\033[0m"


# proc@func or proc@func@addr
def make_proc_func_addr(proc, func, addr):
    if addr is None:
        return f'{proc}@{func}'
    
    return f'{proc}@{func}@{addr}'
    
# Control Flow Graph
class CFG:
    def __init__(self):
        self.nodes = set() # proc@func@addr
        self.edges = set() # (proc@func@addr1, proc@func@addr2)
        self.blocks = {} # proc@func@addr -> block json
        self.symbols = {} # (proc, func) -> addr
        self.address_taken = {} # proc -> [AT1, AT2, ..., ] address taken

        self.syscall_nodes = set() 
        self.main_node = None 

    def add(self, u, v):
        self.nodes.add(u)
        self.nodes.add(v)
        self.edges.add((u, v))    

    def get_symbol_addr(self, sym): # sym: proc@func@addr
        sym_list = sym.split('@')
        proc = sym_list[0]
        func = sym_list[1]
        addr = int(sym_list[2], 16)
        return proc, func, addr    

    def block_insert(self, proc, func, addr, bk_json):
        addr = make_proc_func_addr(proc, func, addr)
        if addr not in self.blocks:
            self.blocks[addr] = bk_json

    def construct_via_func_block_list(self, bb_list, proc, func, r2, r2_dict, func_resolved, func_queue, imp_addr_dict):

        low_addr = bb_list[0]['addr']
        high_addr = bb_list[-1]['addr'] + bb_list[-1]['size']

        for bb in bb_list:
            u = make_proc_func_addr(proc, func, bb.get('addr'))
            v1, v2 = bb.get('jump'), bb.get('fail')

            if v2 is not None:
                v2 = make_proc_func_addr(proc, func, v2)
                self.add(u, v2)
                
            # switch case
            switch_op = bb.get('switch_op')
            if switch_op is not None:
                for cs in switch_op['cases']:
                    v = make_proc_func_addr(proc, func, cs['addr'])
                    self.add(u, v)

            if v1 is None:
                # tail jump
                last_instr = bb['instrs'][-1]
                instr_info = r2.cmdj(f'pdj 1 @ {last_instr}')
                if len(instr_info) > 0:
                    # try again
                    v1 = instr_info[0].get('jump')
                
            if v1 is not None:
                if v1 < low_addr or v1 >= high_addr:
                    # tail jmp 
                    func_info = r2.cmdj(f'afij @ {v1}')
                    if len(func_info) == 0:

                        print(RED + f'Target address {v1} in {proc}@{func} can not be resolved as a function by radare2.' + RESET)
                        continue
                    func_info = func_info[0]
                    if v1 != func_info['offset']:

                        print(YELLOW + f'Warning: jmp to the middle of a function, we treat this case as a jump to the beginning of the function' + RESET)
                    

                    callee_name = func_info['name']
                    callee_addr = func_info['offset']
                    try:
                        callee_proc, callee_name, callee_addr = analyze_func(proc, r2, callee_name, callee_addr, r2_dict, imp_addr_dict)
                    except LookupError as e:
                        print(e)
                        continue


                    proc_func = make_proc_func_addr(callee_proc, callee_name, None)
                    if proc_func not in func_resolved:
                        func_queue.put(proc_func)
                        func_resolved.add(proc_func)

                    v1 = make_proc_func_addr(callee_proc, callee_name, callee_addr)
                    self.add(u, v1)
                    print(GREEN + f'Direct Tail Jmp: {u} -> {v1}' + RESET)
                    continue
                

                v1 = make_proc_func_addr(proc, func, v1)
                self.add(u, v1)

    def trans2nx(self):
        g = nx.DiGraph()
        g.add_nodes_from(self.nodes)
        g.add_edges_from(self.edges)

        nx.set_node_attributes(g, 0, 'syscall')
        nx.set_node_attributes(g, 0, 'main')

        for n in self.syscall_nodes:
            g.nodes[n]['syscall'] = 1

        g.nodes[self.main_node]['main'] = 1
        return g
            
def find_absolute_lib_path(lib):
    for path in lib_paths:
        ab_lib = os.path.join(path, lib)
        if os.path.exists(ab_lib):
            return ab_lib
    return None

# imp_addr_dict : proc -> imp_addr
def get_imp_func_addr(r2, proc, imp_addr_dict):
    imp_addr = imp_addr_dict.get(proc)
    if imp_addr is None:

        imp_list = r2.cmdj('iij')
        imp_addr = set([ imp['plt'] for imp in imp_list if imp['type'] == 'FUNC' and imp.get('plt') != None])
        imp_addr_dict[proc] = imp_addr

    return imp_addr


# ret: proc, func_name, func_addr
def analyze_func(cur_proc, cur_r2, func, callee_addr, r2_dict, imp_addr_dict):

    imp_addr = get_imp_func_addr(cur_r2, cur_proc, imp_addr_dict)

    split = func.split('.')
    if callee_addr in imp_addr:

        libs = cur_r2.cmdj('ilj')

        if len(split) > 2 and split[0] == 'sym' and split[1] == 'imp': # sym.imp.xxx

            extern_sym = 'sym.' + func.split('.', 2)[-1] # 

        elif split[0] == 'sym': # sym.xxx

            extern_sym = func.split('.', 1)[-1]
        
        elif split[0] == 'dbg': # dbg.xxx

            extern_sym = func.split('.', 1)[-1]

        else: 
            # method.std::basic_ostream_char__std::char_traits_char____std::operator____std.char_traits_char____std::basic_ostream_char__std::char_traits_char_____char_const_
            extern_sym = func


        for lib in libs:
            if lib == "ld-linux-x86-64.so.2" or lib == "linux-vdso.so.1": 
                continue
            lib_r2 = r2_dict.get(lib)  
            if lib_r2 is None:
                ab_lib = find_absolute_lib_path(lib)
                if ab_lib is None: raise LookupError(f'can not find lib {lib}')
                print(YELLOW + f'Loading Lib {ab_lib}' + RESET)
                lib_r2 = r2pipe.open(ab_lib)
                lib_r2.cmd('aaa')
                r2_dict[lib] = lib_r2 
            
            sym_info = lib_r2.cmdj(f'afij @ {extern_sym}')
            if len(sym_info) > 0:

                return lib, extern_sym, sym_info[0]['offset']
            
        raise LookupError(f'can not find extern sym {extern_sym}')
    
    return cur_proc, func, callee_addr


def parse_call_in_func(r2, func):
    # pdsf : disassemble summary: strings, calls, jumps, refs (b=N bytes f=function)
    s = r2.cmd(f'pdsf @ {func}')
    direct_call_set = set()


    pattern_fcn = r"^0x\w+.*call fcn\..*\n"
    func_matches = re.findall(pattern_fcn, s, re.MULTILINE)
    for row in func_matches:
        callsite_addr = int(row.split(' ')[0], base=16) 
        target = row.split(' ')[2]

        callee_type = 'direct_fcn.xxx'
        callee_name = target
        callee_addr = int(f"0x{callee_name.split('.')[1]}", base=16)
        if callee_name not in direct_call_set:
            direct_call_set.add(callee_name)
            yield callsite_addr, callee_type, callee_name, callee_addr

    pattern_rest = r"^0x\w+\s+(?!.*fcn\.).*\n"
    rows = re.findall(pattern_rest, s, re.MULTILINE) 

    for row in rows:
        addr = int(row.split(' ')[0], base=16) 
        callsite_addr = addr
        t = row.split(' ')[1].strip() 

        if t == 'call':
            # target = row.split(' ')[2] 
            instr_info = r2.cmdj(f'pdj 1 @{addr}')[0]


            if instr_info['disasm'] == 'syscall':

                print(BLUE + f'syscall in {func}' + RESET)
                yield callsite_addr, 'syscall', None, None
                continue

            ref = instr_info.get('jump')
            if ref != None:

                callee_info = r2.cmdj(f'afij @{ref}')
                if len(callee_info) == 0:

                    print(RED + f'{hex(instr_info["offset"])} : {instr_info["disasm"]}; Target address can not be resolved as a function by radare2.' + RESET)
                    continue

                callee_info = callee_info[0]
                callee_type = 'direct_call'
                callee_name = callee_info['name']
                callee_addr = ref

                if callee_name not in direct_call_set: 
                    direct_call_set.add(callee_name)
                    yield callsite_addr, callee_type, callee_name, callee_addr

            else:
                callee_type = 'indirect_call'
                yield callsite_addr, callee_type, None, None
                
def find_block_in_func_via_addr(addr, bb_list):
    # binary search
    l, r = 0, len(bb_list)-1
    while l < r:
        idx = (l + r) // 2 + 1
        bb_addr = bb_list[idx]['addr']
        if bb_addr <= addr: l = idx
        else: r = idx - 1

    return bb_list[l]['addr']


def analyze_blocks(r2, proc, func, func_resolved, cfg:CFG, r2_dict, func_queue, imp_addr_dict):
    bb_list = r2.cmdj(f"afbj @ {func}")

    for bb in bb_list:
        bb_addr = bb.get('addr')
        cfg.block_insert(proc, func, bb_addr, bb)

    cfg.construct_via_func_block_list(bb_list, proc, func, r2, r2_dict, func_resolved, func_queue, imp_addr_dict)

    for callsite_addr, callee_type, callee_name, callee_addr in parse_call_in_func(r2, func):
        if callee_type == 'syscall':

            bb_addr = find_block_in_func_via_addr(callsite_addr, bb_list)
            u = make_proc_func_addr(proc, func, bb_addr)

            cfg.syscall_nodes.add(u)

        elif callee_type != 'indirect_call':
            
            if callee_type == 'direct_fcn.xxx':

                callee_proc = proc
                # callee_addr = int(f"0x{callee_name.split('.')[1]}", base=16)

            elif callee_type == 'direct_call':

                try:
                    callee_proc, callee_name, callee_addr = analyze_func(proc, r2, callee_name, callee_addr, r2_dict, imp_addr_dict)
                except LookupError as e:
                    print(e)
                    continue

            proc_func = make_proc_func_addr(callee_proc, callee_name, None)

            if proc_func not in func_resolved:
                func_queue.put(proc_func)
                func_resolved.add(proc_func)


            bb_addr = find_block_in_func_via_addr(callsite_addr, bb_list) 

            u = make_proc_func_addr(proc, func, bb_addr)
            v = make_proc_func_addr(callee_proc, callee_name, callee_addr)

            cfg.add(u, v)
            print(GREEN + f'Direct Call: {u} -> {v}' + RESET)


        else:
            heuristic_indirect_call_fix(proc, func, callsite_addr, r2_dict, bb_list, cfg, func_resolved, func_queue)
            
    # PLT jmp: bnd jmp [GOT]
    pass

# Address Taken
def get_address_taken(proc, r2_dict, cfg:CFG):
    if proc in cfg.address_taken.keys():
        return cfg.address_taken[proc]
    
    addr_set, addr2name = None, None

    # if cached
    if os.path.exists(f'address_taken_cache/{os.path.basename(proc)}_address_taken.pkl'):
        with open(f'address_taken_cache/{os.path.basename(proc)}_address_taken.pkl', 'rb') as fd:
            (addr_set, addr2name, ref_as_data) = pickle.load(fd)

    else:
        print(YELLOW + f'Start calculating address taken for {proc}' + RESET)

        ab_path = find_absolute_lib_path(proc)
        if ab_path is None:
            ab_path = sys.argv[1]

        r2 = r2_dict[proc]
        at = AddressTaken(r2, ab_path)
        at.init() 
        
        addr_set = at.get_address_taken()
        addr2name = at.addr2name

    res = [make_proc_func_addr(proc, addr2name[addr], addr) for addr in addr_set]
    cfg.address_taken[proc] = res
    return res


def heuristic_indirect_call_fix(proc, func, callsite_addr, r2_dict, bb_list, cfg:CFG, func_resolved, func_queue):
    target = get_address_taken(proc, r2_dict, cfg)

    bb_addr = find_block_in_func_via_addr(callsite_addr, bb_list)
    u = make_proc_func_addr(proc, func, bb_addr)

    print(GREEN + f'Indirect Call at {u}' + RESET)

    for v in target: 
        cfg.add(u, v)

        proc_func = v[: v.rfind('@')]
        if proc_func not in func_resolved:
            func_queue.put(proc_func)
            func_resolved.add(proc_func)
    

# ret: r2, proc, func
def func_r2_dispatch(r2_dict, func):
    proc = func.split('@')[0]
    func = func.split('@')[1]
    return r2_dict[proc], proc, func

def main(binary_path):
    r2 = r2pipe.open(binary_path)
    r2.cmd("aaa")  # 全面分析

    binary_basename =os.path.basename(binary_path)


    r2_dict = {
        f'{binary_basename}': r2
    }


    imp_addr_dict = {}


    cfg = CFG()
    func_queue = Queue()
    func_resolved = set([f'{binary_basename}@main'])
    func_queue.put(f'{binary_basename}@main')


    main_info = r2.cmdj('afij @main')
    cfg.main_node = make_proc_func_addr(binary_basename, 'main', main_info[0]['offset'])

    while not func_queue.empty(): # BFS search
        proc_func = func_queue.get()
        r2, proc, func = func_r2_dispatch(r2_dict, proc_func)
        analyze_blocks(r2, proc, func, func_resolved, cfg, r2_dict, func_queue, imp_addr_dict)
        print(YELLOW + f'resolve quene size is {func_queue.qsize()}' + RESET)

    print('CFG Construct Done...')

    g = cfg.trans2nx()
    with open(f'{binary_path}.pkl', 'wb') as fd:
        pickle.dump(g, fd)

    print(f'The number of nodes is : {g.number_of_nodes()}')
    print(f'The number of edges is : {g.number_of_edges()}')
    r2.quit()

def arg_init():
    parser = argparse.ArgumentParser(description="CFG Construct")
    parser.add_argument("binary_path", help="binary path")
    args = parser.parse_args()
    return args

def relabel_networkx_nodes_id(node_name):
    if "@" in node_name:
        proc, func, addr = node_name.split("@")

        hex_addr = hex(int(addr))[2:]  
        return f"{proc}@{func}@0x{hex_addr}"
    return node_name

def simple_graph_visual(binary_path):
    with open(f'{binary_path}.pkl', 'rb') as fd:
        g:nx.DiGraph = pickle.load(fd)
    print(f'The number of nodes is : {g.number_of_nodes()}')
    print(f'The number of edges is : {g.number_of_edges()}')

    num_syscall = 0
    for nd, data in g.nodes(data=True):
        if data['main'] == 1: print(f'main node is : {nd}.')
        if data['syscall'] == 1: 
            num_syscall += 1
            print(f'{nd} contains syscall instruction.')
    print(f'totally {num_syscall} syscall instruction basic blocks find.')

    main_nodes = [node for node in g.nodes if node.startswith(f"{os.path.basename(binary_path)}@")]
    expand_nodes = set(main_nodes)

    for node in main_nodes:
        neighbors = set(g.successors(node)) | set(g.predecessors(node))
        expand_nodes.update(neighbors)
    
    subgraph = g.subgraph(expand_nodes).copy()

    subgraph = nx.relabel_nodes(subgraph, relabel_networkx_nodes_id)

    print(subgraph.number_of_nodes())
    print(subgraph.number_of_edges())

    dot_g = to_agraph(subgraph)
    dot_g.write(f'{binary_path}.dot')

if __name__ == '__main__':
    args = arg_init()

    binary_path = args.binary_path
    # binary_path = 'switch_case'

    main(binary_path)

    # simple_graph_visual(binary_path)


    
