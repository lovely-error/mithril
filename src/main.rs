#![feature(generic_arg_infer)]
use core::{mem::{size_of, size_of_val, transmute}, ptr::{addr_of, copy_nonoverlapping}};

const N: u16 = 10;

const _ : () = if N > (1 << 6) - 1 { panic!() };

const PRG_FIB: [u16;44] = [

    // setup
    Opcode::PUT as u16 | (0 << 5) | (1 << 10),
    Opcode::PUT as u16 | (1 << 5) | (1 << 10),
    Opcode::PUT as u16 | (2 << 5) | (0 << 10),
    Opcode::PUT as u16 | (3 << 5) | (8 << 10),

    Opcode::PUT as u16 | (6 << 5) | (N << 10),
    Opcode::PUT as u16 | (7 << 5) | (1 << 10),
    Opcode::PUT as u16 | (8 << 5) | (0 << 10),
    Opcode::PUT as u16 | (10 << 5) | (8 << 10),

    Opcode::PUT as u16 | (11 << 5) | (12 << 10),
    Opcode::PUT as u16 | (12 << 5) | (2 << 10),
    Opcode::DN as _,
    Opcode::DN as _,

    Opcode::BSL as u16 | (11 << 5) | (12 << 10),
    Opcode::BSL as u16 | (10 << 5) | (12 << 10),
    Opcode::DN as _,
    Opcode::DN as _,

    // br1
    Opcode::STV as u16 | (0 << 5) | (2 << 10),
    Opcode::DN as _,
    Opcode::PUT as u16 | (4 << 5) | (8 << 10),
    Opcode::BM as u16 | (6 << 5) | (7 << 10),

    Opcode::BS as u16 | (2 << 5) | (4 << 10),
    Opcode::BS as u16 | (1 << 5) | (0 << 10),
    Opcode::CP as u16 | (6 << 5) | (9 << 10),
    Opcode::DN as _,

    Opcode::CP as u16 | (0 << 5) | (5 << 10),
    Opcode::BET as u16 | (9 << 5) | (8 << 10),
    Opcode::DN as _,
    Opcode::DN as _,

    Opcode::CP as u16 | (1 << 5) | (0 << 10),
    Opcode::MUL as u16 | (9 << 5) | (11 << 10),
    Opcode::DN as _,
    Opcode::DN as _,

    Opcode::CP as u16 | (5 << 5) | (1 << 10),
    Opcode::BS as u16 | (9 << 5) | (10 << 10),
    Opcode::DN as _,
    Opcode::DN as _,

    Opcode::CP as u16 | (9 << 5) | (31 << 10),
    Opcode::DN as _,
    Opcode::DN as _,
    Opcode::DN as _,

    // br2
    Opcode::DBG_HALT as _,
    Opcode::DN as _,
    Opcode::DN as _,
    Opcode::DN as _,

];

fn gen_fibs(lim:usize) -> Vec<u64> {
    let mut vec = Vec::new();
    let (mut a, mut b) = (1,1);
    for _ in 0 .. lim {
        let tmp = b + a;
        b = a;
        a = tmp;
        vec.push(b);
    }
    return vec
}

fn main() { unsafe {

    let mut cpu = CpuState {
        rfile: RegFile { dw: [0;_] },
        lmem: LMem { b: [0;_] },
        gmem: GMem { B64: [[0;_];_] },
        icache: [0;_],
        ipage_bptr: u64::MAX,
    };

    let prog_bytes_ref = &PRG_FIB;
    let dst = cpu.gmem.BYTES.as_mut_ptr().cast::<u8>();
    copy_nonoverlapping(
        addr_of!(*prog_bytes_ref).cast::<u8>(),
        dst,
        size_of_val(prog_bytes_ref)
    );

    let mut stop = false;
    let mut mem_deps = Vec::<MMemOp>::new();
    let mut cycle_counter = 0;
    let mut execed_inst_count = 0;
    let mut nops = 0;
    let mut mem_stall = 0;
    loop {
        step(&mut cpu, &mut stop, &mut mem_deps, &mut execed_inst_count, &mut nops);
        if mem_deps.is_empty() {
            cycle_counter += 1;
        } else {
            for mem_dep in &mem_deps {
                let MMemOp { op_type, blk_sz, dst_addr, src_addr } = mem_dep;

                let blk_sz = match blk_sz {
                    MemBlkSize::B64 => 64,
                    MemBlkSize::B256 => 256,
                    MemBlkSize::B1024 => 1024,
                    MemBlkSize::B4096 => 4096,
                };
                let src_ptr ;
                let dst_ptr ;
                match op_type {
                    MMemOpType::Store => {
                        src_ptr = cpu.lmem.b.as_ptr().add(*dst_addr as usize);
                        dst_ptr = cpu.gmem.BYTES.as_mut_ptr().add(*src_addr as usize);
                    },
                    MMemOpType::Load => {
                        src_ptr = cpu.gmem.BYTES.as_ptr().add(*src_addr as usize);
                        dst_ptr = cpu.lmem.b.as_mut_ptr().add(*dst_addr as usize);
                    },
                }
                copy_nonoverlapping(src_ptr, dst_ptr, blk_sz); // this is kinda suppose to be slow XD
                let gmem_lat = 64; // is this how many cycles it takes to get smth from main mem?
                cycle_counter += gmem_lat;
                mem_stall += gmem_lat;
            }
            mem_deps.clear()
        }
        if stop { break; }
    }

    let fib_nums = core::slice::from_raw_parts(cpu.lmem.b.as_ptr().cast::<u64>(), N as _);
    assert!(fib_nums == gen_fibs(N as _));

    println!(
        "Executed {} instructions ({} nops) in {} cycles ({} on gmem access)",
        execed_inst_count, nops, cycle_counter, mem_stall)
} }

#[repr(C)]
union RegFile {
    dw: [u64;32]
}

#[allow(non_snake_case)]
#[repr(C)]
union GMem {
    B64: [[u8;64]; GLOBAL_MEM_SIZE / size_of::<[u8;64]>()],
    B256: [[u8;256]; GLOBAL_MEM_SIZE / size_of::<[u8;256]>()],
    // B1024: [[u8;1024]; GLOBAL_MEM_SIZE / size_of::<[u8;1024]>()],
    // B4096: [[u8;4096]; GLOBAL_MEM_SIZE / size_of::<[u8;4096]>()],
    IMEM: [[u64;64]; GLOBAL_MEM_SIZE / size_of::<[u64;64]>()],
    BYTES: [u8;GLOBAL_MEM_SIZE],
}

#[repr(C)]
union LMem {
    b: [u8;LOCAL_MEM_SIZE],
    dw: [u64; LOCAL_MEM_SIZE / size_of::<u64>()],
}

const LOCAL_MEM_SIZE: usize = 1024;
const GLOBAL_MEM_SIZE: usize = 1024*8;


struct CpuState {
    rfile: RegFile,
    lmem: LMem,
    gmem: GMem,
    icache: [u64;64],
    ipage_bptr: u64,
}


#[allow(dead_code)]
#[allow(non_camel_case_types)]
#[repr(u8)]
enum Opcode {
    DN = 0,
    LDV,
    STV,
    LDS_1,
    STS_1,
    CP,
    PUT,
    BET,
    BS,
    BM,
    BINV,
    BAND,
    BOR,
    BXOR,
    BSL,
    BSR,
    MUL,
    DIV,
    DBG_HALT,
    EXT = 31
}

#[derive(Clone, Copy, Debug)]
#[repr(u8)]
enum MMemOpType {
    Store, Load
}
#[derive(Debug)]
struct MMemOp {
    src_addr: u64,
    dst_addr: u64,
    blk_sz: MemBlkSize,
    op_type: MMemOpType
}
#[allow(dead_code)]
#[repr(u8)]
#[derive(Debug)]
enum MemBlkSize1 {
    B64 = 0, B1024 = 1
}
#[allow(dead_code)]
#[repr(u8)]
#[derive(Debug)]
enum MemBlkSize2 {
    B256 = 0, B4096 = 1
}

#[allow(dead_code)]
#[repr(u8)]
#[derive(Debug)]
enum MemBlkSize {
    B64, B1024, B256, B4096
}

fn proc(
    isnt: u16,
    stop: &mut bool,
    cpu: &mut CpuState,
    mem_deps: &mut Vec<MMemOp>,
    nop_count: &mut usize
) { unsafe {
    let opc = (isnt & ((1 << 5) - 1)) as u8;
    let opc = transmute::<_, Opcode>(opc);
    match opc {
        Opcode::DBG_HALT => {
            *stop = true;
        },
        Opcode::DN => {
            *nop_count += 1;
            return;
        },
        Opcode::LDV => {
            let ibits = isnt as u16 >> 5;
            let sr = ibits & ((1 << 5) - 1);
            let src_addr = cpu.rfile.dw[sr as usize];
            let val = if src_addr < 1 << 48 {
                *cpu.lmem.dw.as_ptr().cast::<u8>().add(src_addr as usize).cast::<u64>()
            } else {
                0
            };
            let dst = (ibits >> 5) & ((1 << 5) - 1);
            cpu.rfile.dw[dst as usize] = val;
        },
        Opcode::STV => {
            let ibits = isnt as u16 >> 5;
            let dar = (ibits >> 5) & ((1 << 5) - 1);
            let dst_addr = cpu.rfile.dw[dar as usize];
            if dst_addr < 1 << 48 {
                let vr = ibits & ((1 << 5) - 1);
                let val = cpu.rfile.dw[vr as usize];
                cpu.lmem.dw.as_mut_ptr().cast::<u8>().add(dst_addr as usize).cast::<u64>().write(val);
            }
        },
        Opcode::LDS_1 => {
            let ibits = isnt as u16 >> 5;
            let mmem_addr_r = ibits & ((1 << 5) - 1);
            let mmem_addr = cpu.rfile.dw[mmem_addr_r as usize];
            if mmem_addr >= 1 << 48 { return }
            let lmem_addr_r = (ibits >> 5) & ((1 << 2) - 1);
            let lmem_addr = cpu.rfile.dw[lmem_addr_r as usize];
            let bs = ibits >> 15;
            let bs = if bs == 1 { MemBlkSize::B4096 } else { MemBlkSize::B64 };
            let mem_op = MMemOp {
                src_addr: mmem_addr,
                dst_addr: lmem_addr,
                op_type: MMemOpType::Load,
                blk_sz: bs
            };
            mem_deps.push(mem_op);
        },
        Opcode::STS_1 => {
            let ibits = isnt as u16 >> 5;
            let lmem_addr_r = ibits & ((1 << 5) - 1);
            let lmem_addr = cpu.rfile.dw[lmem_addr_r as usize];
            if lmem_addr >= 1 << 48 { return }
            let mmem_addr_r = (ibits >> 5) & ((1 << 2) - 1);
            let mmem_addr = cpu.rfile.dw[mmem_addr_r as usize];
            let bs = ibits >> 15;
            let bs = if bs == 1 { MemBlkSize::B64 } else { MemBlkSize::B4096 };
            let mem_op = MMemOp {
                src_addr: lmem_addr,
                dst_addr: mmem_addr,
                op_type: MMemOpType::Store,
                blk_sz: bs
            };
            mem_deps.push(mem_op);
        },
        Opcode::CP => {
            let ibits = isnt as u16 >> 5;
            let a1 = ibits & ((1 << 5) - 1);
            let a2 = (ibits >> 5) & ((1 << 5) - 1);
            cpu.rfile.dw[a2 as usize] = cpu.rfile.dw[a1 as usize];
        },
        Opcode::PUT => {
            let ibits = isnt as u16 >> 5;
            let dst = ibits & ((1 << 5) - 1);
            let imm = (ibits >> 5) & ((1 << 6) - 1);
            cpu.rfile.dw[dst as usize] = imm as _;
        },
        Opcode::BET => {
            let ibits = isnt as u16 >> 5;
            let a1 = ibits & ((1 << 5) - 1);
            let a2 = (ibits >> 5) & ((1 << 5) - 1);
            cpu.rfile.dw[a1 as usize] =
                (cpu.rfile.dw[a1 as usize] == cpu.rfile.dw[a2 as usize]) as _;
        },
        Opcode::BS => {
            let ibits = isnt as u16 >> 5;
            let a1 = ibits & ((1 << 5) - 1);
            let a2 = (ibits >> 5) & ((1 << 5) - 1);
            cpu.rfile.dw[a1 as usize] =
                cpu.rfile.dw[a1 as usize] + cpu.rfile.dw[a2 as usize];
        },
        Opcode::BINV => {
            let ibits = isnt as u16 >> 5;
            let a1 = ibits & ((1 << 5) - 1);
            cpu.rfile.dw[a1 as usize] = !cpu.rfile.dw[a1 as usize] ;
        },
        Opcode::BAND => {
            let ibits = isnt as u16 >> 5;
            let a1 = ibits & ((1 << 5) - 1);
            let a2 = (ibits >> 5) & ((1 << 5) - 1);
            cpu.rfile.dw[a1 as usize] =
                cpu.rfile.dw[a1 as usize] & cpu.rfile.dw[a2 as usize];
        },
        Opcode::BOR => {
            let ibits = isnt as u16 >> 5;
            let a1 = ibits & ((1 << 5) - 1);
            let a2 = (ibits >> 5) & ((1 << 5) - 1);
            cpu.rfile.dw[a1 as usize] =
                cpu.rfile.dw[a1 as usize] | cpu.rfile.dw[a2 as usize];
        },
        Opcode::BXOR => {
            let ibits = isnt as u16 >> 5;
            let a1 = ibits & ((1 << 5) - 1);
            let a2 = (ibits >> 5) & ((1 << 5) - 1);
            cpu.rfile.dw[a1 as usize] =
                cpu.rfile.dw[a1 as usize] ^ cpu.rfile.dw[a2 as usize];
        },
        Opcode::BSL => {
            let ibits = isnt as u16 >> 5;
            let a1 = ibits & ((1 << 5) - 1);
            let a2 = (ibits >> 5) & ((1 << 5) - 1);
            cpu.rfile.dw[a1 as usize] =
                cpu.rfile.dw[a1 as usize] << cpu.rfile.dw[a2 as usize];
        },
        Opcode::BSR => {
            let ibits = isnt as u16 >> 5;
            let a1 = ibits & ((1 << 5) - 1);
            let a2 = (ibits >> 5) & ((1 << 5) - 1);
            cpu.rfile.dw[a1 as usize] =
                cpu.rfile.dw[a1 as usize] >> cpu.rfile.dw[a2 as usize];
        },
        Opcode::MUL => {
            let ibits = isnt as u16 >> 5;
            let a1 = ibits & ((1 << 5) - 1);
            let a2 = (ibits >> 5) & ((1 << 5) - 1);
            cpu.rfile.dw[a1 as usize] =
                cpu.rfile.dw[a1 as usize] * cpu.rfile.dw[a2 as usize];
        },
        Opcode::DIV => {
            let ibits = isnt as u16 >> 5;
            let a1 = ibits & ((1 << 5) - 1);
            let a2 = (ibits >> 5) & ((1 << 5) - 1);
            cpu.rfile.dw[a1 as usize] =
                cpu.rfile.dw[a1 as usize] / cpu.rfile.dw[a2 as usize];
        },
        Opcode::BM => {
            let ibits = isnt as u16 >> 5;
            let a1 = ibits & ((1 << 5) - 1);
            let a2 = (ibits >> 5) & ((1 << 5) - 1);
            cpu.rfile.dw[a1 as usize] =
                cpu.rfile.dw[a1 as usize] - cpu.rfile.dw[a2 as usize];
        },
        Opcode::EXT => todo!(),
    }
} }

fn step(
    cpu: &mut CpuState,
    stop: &mut bool,
    mem_deps: &mut Vec<MMemOp>,
    i_count: &mut usize,
    nop_count: &mut usize
) { unsafe {

    let ip = cpu.rfile.dw[31];
    let ipage_baddr = ip & !((1 << 9) - 1);

    let icache_miss = ipage_baddr != cpu.ipage_bptr ;
    if icache_miss {
        let page_ix = ip >> 9;
        cpu.icache = cpu.gmem.IMEM[page_ix as usize];
        cpu.ipage_bptr = ip & !((1 << 9) - 1);
    }
    let i_bofft = ip - ipage_baddr;
    let pack_ix = i_bofft >> 3;
    let inst_ix = i_bofft - (i_bofft & !((1 << 3) - 1));

    let pack = cpu.icache[pack_ix as usize];
    let pack = pack & (u64::MAX >> (8 * inst_ix));

    let rem_b = 8 - inst_ix;
    cpu.rfile.dw[31] += rem_b;

    let [i1, i2, i3, i4] = transmute::<_, [u16;4]>(pack);

    proc(i1, stop, cpu, mem_deps, nop_count);
    proc(i2, stop, cpu, mem_deps, nop_count);
    proc(i3, stop, cpu, mem_deps, nop_count);
    proc(i4, stop, cpu, mem_deps, nop_count);
    *i_count += 4;

    return ;
} }


