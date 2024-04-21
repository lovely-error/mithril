#![feature(generic_arg_infer)]

use core::{mem::{size_of, size_of_val, transmute}, ptr::{addr_of, copy_nonoverlapping}};

const N: u16 = 32;

const _ : () = if N > (1 << 6) - 1 { panic!() };

const PRG_FIB: [u16;44] = [

    // setup
    Opcode::PUC as u16 | (0 << 5) | (1 << 10),
    Opcode::PUC as u16 | (1 << 5) | (1 << 10),
    Opcode::PUC as u16 | (2 << 5) | (0 << 10),
    Opcode::PUC as u16 | (3 << 5) | (8 << 10),

    Opcode::PUC as u16 | (6 << 5) | (N << 10),
    Opcode::PUC as u16 | (7 << 5) | (1 << 10),
    Opcode::PUC as u16 | (8 << 5) | (0 << 10),
    Opcode::PUC as u16 | (10 << 5) | (8 << 10),

    Opcode::PUC as u16 | (11 << 5) | (12 << 10),
    Opcode::PUC as u16 | (12 << 5) | (2 << 10),
    Opcode::TMP_BSL as u16 | (10 << 5) | (12 << 10),
    Opcode::DNO as _,

    Opcode::TMP_BSL as u16 | (11 << 5) | (12 << 10),
    Opcode::DNO as _,
    Opcode::DNO as _,
    Opcode::DNO as _,

    // loop
    Opcode::STV as u16 | (0 << 5) | (2 << 10),
    Opcode::PUC as u16 | (4 << 5) | (8 << 10),
    Opcode::TMP_BM as u16 | (6 << 5) | (7 << 10),
    Opcode::DNO as _,

    Opcode::TMP_BS as u16 | (2 << 5) | (4 << 10),
    Opcode::TMP_BS as u16 | (1 << 5) | (0 << 10),
    Opcode::CPY as u16 | (6 << 5) | (9 << 10),
    Opcode::CPY as u16 | (0 << 5) | (5 << 10),

    Opcode::TMP_BET as u16 | (9 << 5) | (8 << 10),
    Opcode::CPY as u16 | (1 << 5) | (0 << 10),
    Opcode::DNO as _,
    Opcode::DNO as _,

    Opcode::TMP_MUL as u16 | (9 << 5) | (11 << 10),
    Opcode::CPY as u16 | (5 << 5) | (1 << 10),
    Opcode::DNO as _,
    Opcode::DNO as _,

    Opcode::TMP_BS as u16 | (9 << 5) | (10 << 10),
    Opcode::DNO as _,
    Opcode::DNO as _,
    Opcode::DNO as _,

    Opcode::CPY as u16 | (9 << 5) | (31 << 10),
    Opcode::DNO as _,
    Opcode::DNO as _,
    Opcode::DNO as _,

    // exit
    Opcode::DBG_HALT as _,
    Opcode::DNO as _,
    Opcode::DNO as _,
    Opcode::DNO as _,

];

fn gen_fibs() -> [u64;N as usize] {
    let mut vals = [0;N as _];
    let (mut a, mut b) = (1,1);
    for i in 0 .. N as _ {
        let tmp = b + a;
        b = a;
        a = tmp;
        vals[i] = b;
    }
    return vals;
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
    let mut cycle_count = 0;
    let mut total_inst_count = 0;
    let mut nops_count = 0;
    let mut mem_stall = 0;
    let mut i_cache_miss_count = 0;
    let mut i_cache_hit_count = 0;
    loop {
        step(&mut cpu, &mut stop, &mut mem_deps, &mut total_inst_count, &mut nops_count, &mut i_cache_miss_count, &mut i_cache_hit_count);
        if stop { break; }
        if mem_deps.is_empty() {
            cycle_count += 1;
        } else {
            for mem_dep in &mem_deps {
                let MMemOp { op_type, blk_sz, dst_addr, src_addr } = mem_dep;

                let blk_sz = match blk_sz {
                    MemBlkSize::B128 => 64,
                    MemBlkSize::B512 => 256,
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
                cycle_count += gmem_lat;
                mem_stall += gmem_lat;
            }
            mem_deps.clear()
        }
    }

    let fib_nums = core::slice::from_raw_parts(cpu.lmem.b.as_ptr().cast::<u64>(), N as _);
    let v = gen_fibs();
    assert!(fib_nums == v.as_slice());

    let mut report = String::new();
    use core::fmt::Write;
    let usefull = total_inst_count - nops_count;
    let ipc_rate = usefull as f32 / cycle_count as f32;
    let sparcity = (nops_count as f32 / total_inst_count as f32) * 100.0;

    writeln!(&mut report,
        "Executed total of {total_inst_count} instructions.\n{usefull} usefull and {nops_count} nops in {cycle_count} cycles ({ipc_rate} IPC rate) ({sparcity}% sparsity)"
    ).unwrap();
    writeln!(&mut report, "{mem_stall} cycles spent stalled on repsense from gmem").unwrap();
    writeln!(&mut report, "Experienced {i_cache_miss_count} i$ misses during execution ({i_cache_hit_count} hits)").unwrap();
    println!("{}", report);
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

    DNO = 0, // do nothing

    LDV, // load from scpad
    STV, // store to scpad
    LDS, // load segment from main to scpad
    STS, // store segment from scpad to main
    PUC, // put 5 bit constant into register
    LAV, // loads a value from gmem to a specified reg and atomically sets an ownership flag returning it in a specified register
    STAV, // tries to atomically store a value in specified register into memory to gmem, and returns a flag in specified register denoting if this operation succeded succeded or not
    CPY, // reg2reg copy

    TEZ, // test equiv to zero
    NAND, // bitwise not and
    BSL, // bitwise shift left 1,4, or 16 bits
    BSR, // bitwise shift right 1,4, or 16 bits

    CHKLDF, // checks if mem2scpad load issued on specific address range has finished
    CHKSTF, // checks if scpad2mem store issued on specific address range has finished
    WSTF, // wait for prior scpad2mem store to finish on specific address range

    DBG_HALT,


    // remove these
    TMP_BSL,
    TMP_BET,
    TMP_BS,
    TMP_BM,
    TMP_BOR,
    TMP_MUL,

    LIMIT = 31,
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
    B128 = 0, B1024 = 1
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
    B128, B1024, B512, B4096
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
        Opcode::DNO => {
            *nop_count += 1;
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
        Opcode::LDS => {
            let ibits = isnt as u16 >> 5;
            let mmem_addr_r = ibits & ((1 << 5) - 1);
            let mmem_addr = cpu.rfile.dw[mmem_addr_r as usize];
            if mmem_addr >= 1 << 48 { return }
            let lmem_addr_r = (ibits >> 5) & ((1 << 2) - 1);
            let lmem_addr = cpu.rfile.dw[lmem_addr_r as usize];
            let bs = ibits >> 15;
            let bs = if bs == 1 { MemBlkSize::B1024 } else { MemBlkSize::B128 };
            let mem_op = MMemOp {
                src_addr: mmem_addr,
                dst_addr: lmem_addr,
                op_type: MMemOpType::Load,
                blk_sz: bs
            };
            mem_deps.push(mem_op);
        },
        Opcode::STS => {
            let ibits = isnt as u16 >> 5;
            let lmem_addr_r = ibits & ((1 << 5) - 1);
            let lmem_addr = cpu.rfile.dw[lmem_addr_r as usize];
            if lmem_addr >= 1 << 48 { return }
            let mmem_addr_r = (ibits >> 5) & ((1 << 2) - 1);
            let mmem_addr = cpu.rfile.dw[mmem_addr_r as usize];
            let bs = ibits >> 15;
            let bs = if bs == 1 { MemBlkSize::B1024 } else { MemBlkSize::B128 };
            let mem_op = MMemOp {
                src_addr: lmem_addr,
                dst_addr: mmem_addr,
                op_type: MMemOpType::Store,
                blk_sz: bs
            };
            mem_deps.push(mem_op);
        },
        Opcode::CPY => {
            let ibits = isnt as u16 >> 5;
            let a1 = ibits & ((1 << 5) - 1);
            let a2 = (ibits >> 5) & ((1 << 5) - 1);
            cpu.rfile.dw[a2 as usize] = cpu.rfile.dw[a1 as usize];
        },
        Opcode::PUC => {
            let ibits = isnt as u16 >> 5;
            let dst = ibits & ((1 << 5) - 1);
            let imm = (ibits >> 5) & ((1 << 6) - 1);
            cpu.rfile.dw[dst as usize] = imm as _;
        },
        Opcode::TMP_BET => {
            let ibits = isnt as u16 >> 5;
            let a1 = ibits & ((1 << 5) - 1);
            let a2 = (ibits >> 5) & ((1 << 5) - 1);
            cpu.rfile.dw[a1 as usize] =
                (cpu.rfile.dw[a1 as usize] == cpu.rfile.dw[a2 as usize]) as _;
        },
        Opcode::TMP_BS => {
            let ibits = isnt as u16 >> 5;
            let a1 = ibits & ((1 << 5) - 1);
            let a2 = (ibits >> 5) & ((1 << 5) - 1);
            cpu.rfile.dw[a1 as usize] =
                cpu.rfile.dw[a1 as usize] + cpu.rfile.dw[a2 as usize];
        },
        Opcode::NAND => {
            let ibits = isnt as u16 >> 5;
            let a1 = ibits & ((1 << 5) - 1);
            let a2 = (ibits >> 5) & ((1 << 5) - 1);
            cpu.rfile.dw[a1 as usize] =
                !(cpu.rfile.dw[a1 as usize] & cpu.rfile.dw[a2 as usize]);
        },
        Opcode::TMP_BOR => {
            let ibits = isnt as u16 >> 5;
            let a1 = ibits & ((1 << 5) - 1);
            let a2 = (ibits >> 5) & ((1 << 5) - 1);
            cpu.rfile.dw[a1 as usize] =
                cpu.rfile.dw[a1 as usize] | cpu.rfile.dw[a2 as usize];
        },
        Opcode::BSL  => {
            let ibits = isnt as u16 >> 5;
            let a1 = ibits & ((1 << 5) - 1);
            let a2 = (ibits >> 5) & ((1 << 5) - 1);
            match a2 {
                0 => {
                    cpu.rfile.dw[a1 as usize] =
                        cpu.rfile.dw[a1 as usize] << 1 ;
                },
                1 => {
                    cpu.rfile.dw[a1 as usize] =
                        cpu.rfile.dw[a1 as usize] << 4 ;
                },
                2 => {
                    cpu.rfile.dw[a1 as usize] =
                        cpu.rfile.dw[a1 as usize] << 16 ;
                },
                _ => unreachable!()
            }
        },
        Opcode::TMP_BSL => {
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
        Opcode::TMP_MUL => {
            let ibits = isnt as u16 >> 5;
            let a1 = ibits & ((1 << 5) - 1);
            let a2 = (ibits >> 5) & ((1 << 5) - 1);
            cpu.rfile.dw[a1 as usize] =
                cpu.rfile.dw[a1 as usize] * cpu.rfile.dw[a2 as usize];
        },
        Opcode::TMP_BM => {
            let ibits = isnt as u16 >> 5;
            let a1 = ibits & ((1 << 5) - 1);
            let a2 = (ibits >> 5) & ((1 << 5) - 1);
            cpu.rfile.dw[a1 as usize] =
                cpu.rfile.dw[a1 as usize] - cpu.rfile.dw[a2 as usize];
        },
        Opcode::TEZ => {
            let ibits = isnt as u16 >> 5;
            let a1 = ibits & ((1 << 5) - 1);
            cpu.rfile.dw[a1 as usize] = (0 == cpu.rfile.dw[a1 as usize]) as _;
        },
        Opcode::CHKLDF => todo!(),
        Opcode::CHKSTF => todo!(),
        Opcode::WSTF => todo!(),
        Opcode::LAV => todo!(),
        Opcode::STAV => todo!(),
        Opcode::LIMIT => panic!("Invalid instruction"),
    }
} }

fn step(
    cpu: &mut CpuState,
    stop: &mut bool,
    mem_deps: &mut Vec<MMemOp>,
    i_count: &mut usize,
    nop_count: &mut usize,
    i_cache_miss_count: &mut usize,
    i_cache_hit_count: &mut usize
) { unsafe {

    let ip = cpu.rfile.dw[31];
    let ipage_baddr = ip & !((1 << 9) - 1);

    let icache_miss = ipage_baddr != cpu.ipage_bptr ;
    if icache_miss {
        *i_cache_miss_count += 1;
        let page_ix = ip >> 9;
        cpu.icache = cpu.gmem.IMEM[page_ix as usize];
        cpu.ipage_bptr = ip & !((1 << 9) - 1);
    } else {
        *i_cache_hit_count += 1;
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



#[test]
fn nand_puter() {
    // https://en.wikipedia.org/wiki/NAND_logic

    fn nand(a:u16, b:u16) -> u16 {
        !(a & b)
    }
    for k in 0 .. 256 {
        for r in 0 .. 256 {
            let c = nand(k, r);
            let g = !(k & r);
            assert!(g == c);
        }
    }
    fn bit_not(a:u16) -> u16 {
        nand(a, a)
    }
    for k in 0 .. 256 {
        let c = bit_not(k);
        let r = !k;
        assert!(c == r)
    }
    fn bit_and(a:u16,b:u16) -> u16 {
        let c = nand(a, b);
        nand(c, c)
    }
    for k in 0 .. 256 {
        for r in 0 .. 256 {
            let c = bit_and(k, r);
            let g = k & r;
            assert!(g == c);
        }
    }
    fn bit_or(a:u16, b:u16) -> u16 {
        let a = nand(a, a);
        let b = nand(b, b);
        let c = nand(a, b);
        return c;
    }
    for k in 0 .. 256 {
        for r in 0 .. 256 {
            let c = bit_or(k, r);
            let g = k | r;
            assert!(g == c);
        }
    }
    fn bit_xor(a:u16, b:u16) -> u16 {
        let i = nand(a, b);
        let k = nand(a, i);
        let r = nand(b, i);
        let c = nand(k, r);
        return c;
    }
    for k in 0 .. 256 {
        for r in 0 .. 256 {
            let c = bit_xor(k, r);
            let g = k ^ r;
            assert!(g == c);
        }
    }

    fn add(mut a: u16, mut b: u16) -> u16 {
        for _ in 0 .. 16 {
            let k = bit_xor(a, b);
            let om = bit_and(a , b) << 1;
            a = k;
            b = om;
        }
        return a;
    }
    for k in 0 .. u16::MAX {
        let l = u16::MAX - k;
        let c = add(k, l);
        let g = k + l;
        assert!(g == c);
    }
    fn bet(a:u16, b:u16) -> u16 {
        // a == b
        let k = bit_xor(a, b);
        let k = (k == 0) as u16;
        return k;
    }
    fn bool_not(a:u16) -> u16 {
        bet(0, a)
    }
    assert!(bool_not(1) == 0);
    assert!(bool_not(0) == 1);

    fn mul(a: u16, b: u16) -> u16 {
        let mut result = 0;
        for i in 0 .. 16 {
            let k = bit_and(b, 1 << i);
            let k = bet(k, 0);
            let k = bool_not(k);
            let r = (a * k as u16) << i;
            result = add(result, r);
        }
        return result;
    }
    for k in 0 .. 256 {
        for r in 0 .. 256 {
            let c = mul(k, r);
            let g = k * r;
            assert!(g == c);
        }
    }

}

