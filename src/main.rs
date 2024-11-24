#![feature(generic_arg_infer)]
#![feature(new_zeroed_alloc)]

use core::{mem::{size_of, transmute}, ptr::{addr_of_mut, copy_nonoverlapping}};

fn main() {
    fibs();
}

const N: u16 = 32;
const _ : () = if N > (1 << 6) - 1 { panic!() };

#[repr(C)]
union RegFile {
    dw: [u64;32]
}

#[allow(non_snake_case)]
#[repr(C)]
union ExternalMem {
    B64: [[u8;64]; EXTERNAL_MEM_SIZE / size_of::<[u8;64]>()],
    IMEM: [[u64;128]; EXTERNAL_MEM_SIZE / size_of::<[u64;128]>()],
    BYTES: [u8;EXTERNAL_MEM_SIZE],
}

#[repr(C)]
union Sracthpad {
    b: [u8;SCRATCHPAD_MEM_SIZE],
    dw: [u64; SCRATCHPAD_MEM_SIZE / size_of::<u64>()],
}

const SCRATCHPAD_MEM_SIZE: usize = 1024*64;
const EXTERNAL_MEM_SIZE: usize = 1024*1024*16;



type IStashCell = [u64;128];
#[derive(Debug)]
struct IStash {
    cells: [IStashCell;Self::BUCKET_COUNT],
    ident_addr_bits: [u64;Self::BUCKET_COUNT],
    counters: [u8;Self::BUCKET_COUNT],
    ip_addr_bits: u64,
}
impl IStash {
    const ISTASH_SIZE: usize = 1024*8;
    const BUCKET_COUNT: usize = Self::ISTASH_SIZE / size_of::<IStashCell>();
    const DECAY_COUNT: u8 = 64;
    fn new() -> Self {
        Self {
            cells: [[0;_];_],
            ident_addr_bits: [u64::MAX;_],
            counters: [0;_],
            ip_addr_bits: u64::MAX
        }
    }
    fn bucket_index_from_addr(addr: u64) -> u64 {
        (addr >> 10) & ((1 << 10) - 1)
    }
    fn addr_bits(addr: u64) -> u64 {
        addr >> 10
    }
    fn try_find(&mut self, addr: u64) -> Option<*const IStashCell> {
        let addr_bits = Self::addr_bits(addr);
        let guranteed_match = addr_bits == self.ip_addr_bits;
        let bucket_ix = if guranteed_match {
            let bucket_ix = Self::bucket_index_from_addr(addr);
            bucket_ix as usize
        } else {
            let mut ix = 0u64;
            for i in 0 .. Self::BUCKET_COUNT {
                ix |= ((self.ident_addr_bits[i] == addr_bits) as u64) << i;
            }
            if ix == 0 { return None }
            let ix = ix.trailing_zeros() as usize;
            self.ip_addr_bits = self.ident_addr_bits[ix];
            ix
        };
        for i in 0 .. Self::BUCKET_COUNT  {
            let cv = self.counters[i] ;
            self.counters[i] -= (i != bucket_ix && cv != 0) as u8;
        }
        let bucket = &self.cells[bucket_ix as usize];
        return Some(bucket);
    }
    fn load(&mut self, addr: u64, src: *const IStashCell) {
        let mut ix = 0u64;
        for i in 0 .. Self::BUCKET_COUNT {
           ix |= ((self.counters[i] == 0) as u64) << i;
        }
        let no_free = ix == 0;
        let index = if no_free { // just nuke some stuff
            Self::bucket_index_from_addr(addr)
        } else { // occupy free
            ix.trailing_zeros() as _
        } as usize;
        let ident_bits = Self::addr_bits(addr);
        self.ident_addr_bits[index as usize] = ident_bits;
        self.counters[index as usize] = Self::DECAY_COUNT;
        unsafe {
            copy_nonoverlapping(
                src.cast::<u8>(),
                addr_of_mut!(self.cells[index]).cast::<u8>(),
                size_of::<IStashCell>()
            )
        };
    }
}
#[test] #[ignore]
fn istash_test() {
    let mut stash = IStash::new();
    let v : IStashCell = [u64::MAX;_];
    for i in 0 .. IStash::BUCKET_COUNT {
        stash.load((1024*i) as _, &v);
    }
    assert!(stash.counters.iter().map(|e|*e==64).fold(true, |acc, e|acc && e));
    assert!(stash.ident_addr_bits.iter().zip(0u64..).fold(true, |acc, (l,r)| acc && (*l == r)));
    for _ in 0 .. 64 {
        let _ = stash.try_find(1024*4);
    }
    assert!(stash.counters.iter().zip(0..).fold(true, |acc, (e,i)| acc && if i == 4 { *e == 64 } else { *e == 0 }));
    stash.load(1024*3, &[0;_]);
    match stash.try_find(1024*3) {
        Some(i) => assert!((unsafe{&*i}).iter().fold(true, |acc, e|acc && (*e == 0))),
        None => panic!(),
    };
    println!("!")
}

#[repr(C)]
struct CpuState {
    instruction_stash: Box<IStash>,
    gp_reg_file: RegFile,
    scratchpad_mem: Box<Sracthpad>,
    external_mem: Box<ExternalMem>,
}
impl CpuState {
    fn new() -> Self {
        Self {
            gp_reg_file: RegFile { dw: [0;_] },
            scratchpad_mem: unsafe { Box::new_zeroed().assume_init() },
            external_mem: unsafe { Box::new_zeroed().assume_init() },
            instruction_stash: Box::new(IStash::new()),
        }
    }
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
    LAV, // loads a value from gmem to a specified reg and atomically sets an ownership flag returning it in a specified register.
    STAV, // tries to atomically store a value in specified register into memory to gmem, and returns a flag in specified register denoting if this operation succeded or not
    PUC, // put 6 bit constant into register
    CPY, // reg2reg copy

    TEZ, // test equiv to zero
    NAND, // bitwise not and
    BSL, // bitwise shift left 1,4, or 16 bits
    BROT, // bitwise rotate
    BS, // bitwise summ
    BRZ, // branch if zero

    CHKMOPS, // checks if issued load or store gmem-scpad operation on specific segment has finished

    NPUP, // indicates how to split next pack into constituents

    NSDINP, // next N packs do not have serialising dependencies

    DBG_HALT,


    I_EXT_32 = 30,
    I_EXT_64 = 31,

    LIMIT = 32,
}

#[derive(Debug, Clone, Copy)] #[repr(transparent)]
struct Inst16(u16);
#[allow(dead_code)]
impl Inst16 {
    fn dno() -> Self {
        Self(0)
    }
    fn ldv(src: u8, dst: u8) -> Self {
        assert!(src < 32 && dst < 32);
        Self(Opcode::LDV as u16 | ((src as u16) << 5) | ((dst as u16) << 10))
    }
    fn stv(src: u8, dst: u8) -> Self {
        assert!(src < 32 && dst < 32);
        Self(Opcode::STV as u16 | ((src as u16) << 5) | ((dst as u16) << 10))
    }
    fn lds(src: u8, dst: u8) -> Self {
        assert!(src < 32 && dst < 32);
        Self(Opcode::LDS as u16 | ((src as u16) << 5) | ((dst as u16) << 10))
    }
    fn sts(src: u8, dst: u8) -> Self {
        assert!(src < 32 && dst < 32);
        Self(Opcode::STS as u16 | ((src as u16) << 5) | ((dst as u16) << 10))
    }
    fn lav(src: u8, flag: u8) -> Self {
        assert!(src < 32 && flag < 32);
        Self(Opcode::LAV as u16 | ((src as u16) << 5) | ((flag as u16) << 10))
    }
    fn stav(src: u8, flag: u8) -> Self {
        assert!(src < 32 && flag < 32);
        Self(Opcode::STAV as u16 | ((src as u16) << 5) | ((flag as u16) << 10))
    }
    fn puc(src: u8, imm: u8) -> Self {
        assert!(src < 32 && imm < (1 << 6));
        Self(Opcode::PUC as u16 | ((src as u16) << 5) | ((imm as u16) << 10))
    }
    fn cpy(src: u8, dst: u8) -> Self {
        assert!(src < 32 && dst < 32);
        Self(Opcode::CPY as u16 | ((src as u16) << 5) | ((dst as u16) << 10))
    }
    fn tez(src: u8) -> Self {
        assert!(src < 32);
        Self(Opcode::TEZ as u16 | ((src as u16) << 5))
    }
    fn nand(src: u8, op: u8) -> Self {
        assert!(src < 32 && op < 32);
        Self(Opcode::NAND as u16 | ((src as u16) << 5) | ((op as u16) << 10))
    }
    fn bs(src: u8, op: u8) -> Self {
        assert!(src < 32 && op < 32);
        Self(Opcode::BS as u16 | ((src as u16) << 5) | ((op as u16) << 10))
    }
    fn bsl(src: u8, op: ShiftLength) -> Self {
        assert!(src < 32);
        Self(Opcode::BSL as u16 | ((src as u16) << 5) | ((op as u16) << 10))
    }
    fn brot(src: u8) -> Self {
        assert!(src < 32);
        Self(Opcode::BROT as u16 | ((src as u16) << 5))
    }
    fn chkmops(src: u8) -> Self {
        assert!(src < 32);
        Self(Opcode::CHKMOPS as u16 | ((src as u16) << 5))
    }
    fn npup(pattern: [u8;4]) -> Self {
        let mut val = 0u8;
        for i in 0 .. 4 {
            let seg = pattern[3 - i];
            assert!(seg <= 3);
            val |= seg;
            val <<= 2 * ((i != 3) as u8);
        }
        Self(Opcode::NPUP as u16 | ((val as u16) << 8))
    }
    fn dbg_halt() -> Self {
        Self(Opcode::DBG_HALT as u16)
    }
    fn get_opcode(&self) -> Opcode {
        unsafe { transmute(self.0 as u8 & ((1 << 5) - 1)) }
    }
    fn get_arg1(&self) -> u8 {
        ((self.0 >> 5) & ((1 << 5) - 1)) as u8
    }
    fn get_arg2(&self) -> u8 {
        ((self.0 >> 10) & ((1 << 5) - 1)) as u8
    }
    fn get_imm(&self) -> u8 {
        (self.0 >> 10) as u8
    }
    fn get_mem_size(&self) -> MemBlkSize1 {
        unsafe{transmute((self.0 >> 15) as u8)}
    }
    fn get_pattern(&self) -> u8 {
        (self.0 >> 8) as u8
    }
    fn as_bytes(&self) -> [u8;2] {
        unsafe { transmute(self.0) }
    }
    fn brz(dst: u8, flag: u8) -> Self {
        assert!(dst < 32 && flag < 32);
        Self(Opcode::BRZ as u16 | ((dst as u16) << 5) | ((flag as u16) << 10))
    }
}

#[allow(dead_code)]
#[repr(u8)] #[derive(Debug, Clone, Copy)]
enum ShiftLength {
    S1, S4, S16
}

#[derive(Debug, Clone, Copy)] #[allow(dead_code)]
struct Pack(u64);
impl Pack {
    fn i16x4(instrs: &[Inst16]) -> Self {
        assert!(instrs.len() <= 4);
        let mut ixs = [Inst16::dno();4];
        for i in 0 .. instrs.len() {
            ixs[i] = instrs[i];
        }
        Self(unsafe { transmute(ixs) })
    }
    #[allow(dead_code)]
    fn as_bytes(&self) -> [u8;8] {
        unsafe { transmute(self.0) }
    }
}

#[derive(Clone, Copy, Debug)]
#[repr(u8)]
enum MMemTransDir {
    SCPAD2MEM, MEM2SCPAD
}
#[derive(Debug)]
struct MemOp {
    src_addr: u64,
    dst_addr: u64,
    blk_sz: MemBlkSize,
    op_type: MMemTransDir
}
#[allow(dead_code)]
#[repr(u8)]
#[derive(Debug)]
enum MemBlkSize1 {
    B64 = 0, B1024 = 1
}
#[allow(dead_code)]
#[repr(u8)]
#[derive(Debug, Clone, Copy)]
enum MemBlkSize2 {
    B256 = 0, B4096 = 1
}

#[allow(dead_code)]
#[repr(u8)]
#[derive(Debug, Clone, Copy)]
enum MemBlkSize {
    B64,
    B256,
    B1024,
    B4096
}

struct MemCtrl {
    delay_count: [u8;16],
    blk_sz: [MemBlkSize;16],
    op_type: [MMemTransDir;16],
    saddr: [u64;16],
    daddr: [u64;16],
    occupation_map: u16,
}
impl MemCtrl {
    fn new() -> Self {
        MemCtrl {
            blk_sz: [MemBlkSize::B256;_],
            op_type: [MMemTransDir::SCPAD2MEM;_],
            saddr: [0;_],
            daddr: [0;_],
            occupation_map: 0,
            delay_count: [0;_],
        }
    }
    #[must_use]
    fn put(&mut self, mop: MemOp) -> bool {
        let MemOp { src_addr, dst_addr, blk_sz, op_type } = mop;
        let index = self.occupation_map.trailing_ones();
        if index == 16 { return false }
        let index = index as usize;
        self.blk_sz[index] = blk_sz;
        self.op_type[index] = op_type;
        self.saddr[index] = src_addr;
        self.daddr[index] = dst_addr;
        self.delay_count[index] = 100;
        return true;
    }
    fn check_mem_deps(&mut self, cpu: &mut CpuState) { unsafe {
        let mut ixs = self.occupation_map;
        loop {
            if ixs == 0 { break }
            let ix = ixs.trailing_zeros() as usize;
            let delc = self.delay_count[ix].saturating_sub(1);
            if delc == 0 {
                let src = self.saddr[ix];
                let dst = self.daddr[ix];
                let blk_sz = match self.blk_sz[ix] {
                    MemBlkSize::B64 => 64,
                    MemBlkSize::B256 => 256,
                    MemBlkSize::B1024 => 1024,
                    MemBlkSize::B4096 => 4096,
                };
                let src_ptr ;
                let dst_ptr ;
                match self.op_type[ix] {
                    MMemTransDir::MEM2SCPAD => {
                        src_ptr = cpu.external_mem.BYTES.as_ptr().add(src as usize);
                        dst_ptr = cpu.scratchpad_mem.b.as_mut_ptr().add(dst as usize);
                    },
                    MMemTransDir::SCPAD2MEM => {
                        src_ptr = cpu.scratchpad_mem.b.as_ptr().add(src as usize);
                        dst_ptr = cpu.external_mem.BYTES.as_mut_ptr().add(dst as usize);
                    }
                }
                copy_nonoverlapping(src_ptr, dst_ptr, blk_sz);
                ixs &= !(ix as u16);
            } else {
                self.delay_count[ix] = delc;
            }
        }
        self.occupation_map = ixs;
    } }
    fn _iter<'a>(&'a self) -> impl Iterator<Item = MemOp> + 'a {
        let mut ixs = self.occupation_map;
        core::iter::from_fn(move || {
            loop {
                let index = ixs.trailing_zeros() as usize;
                let memop = MemOp {
                    src_addr: self.saddr[index],
                    dst_addr: self.daddr[index],
                    blk_sz: self.blk_sz[index],
                    op_type: self.op_type[index],
                };
                ixs &= !(1 << index);
                return Some(memop);
            }
        })
    }
}

fn run_programm(state: &mut CpuState, prg_bytes: &[Pack]) -> String { unsafe {

    let byte_len = prg_bytes.len() * 8;
    let dst = state.external_mem.BYTES.as_mut_ptr().cast::<u8>();
    copy_nonoverlapping(
        prg_bytes.as_ptr().cast::<u8>(),
        dst,
        byte_len
    );

    let mut was_asked_to_stop = false;
    let mut mem_deps = MemCtrl::new();
    let mut instruction_cycle_count = 0;
    let mut noops_count = 0;
    let mut usefull_count = 0;
    let mut i_cache_miss_count = 0;
    let mut i_cache_hit_count = 0;
    let mut cache_fill_stall = 0;

    let mut decode_pattern = 0u8;
    loop {
        let ip = state.gp_reg_file.dw[31];
        let blob = match  state.instruction_stash.try_find(ip as _) {
            Some(blob) => { // hit
                i_cache_hit_count += 1;
                blob
            },
            None => { // miss
                i_cache_miss_count += 1;
                let block_addr = ip & !((1 << 10) - 1);
                let src = &state.external_mem.IMEM[block_addr as usize];
                state.instruction_stash.load(ip as _, src);
                cache_fill_stall += 100; // sorta stall on instruction mem fil
                continue;
            },
        };
        let pack_ix = ip >> 3;
        let pack = *blob.cast::<u64>().add(pack_ix as _);
        state.gp_reg_file.dw[31] += 8; // todo: change to allow jumps to arbitrary instructions

        let mut slam_pack = 0;
        if decode_pattern != 0 {
            let mut dec_pat = decode_pattern;
            let mut i = 0;
            loop {
                let mask =
                    ((dec_pat >> 0 & 0b11 == i) as u64 * ((u16::MAX as u64) << 0)) |
                    ((dec_pat >> 2 & 0b11 == i) as u64 * ((u16::MAX as u64) << 16)) |
                    ((dec_pat >> 4 & 0b11 == i) as u64 * ((u16::MAX as u64) << 32)) |
                    ((dec_pat >> 6 & 0b11 == i) as u64 * ((u16::MAX as u64) << 48)) ;
                if mask != 0 {
                    let pack = if slam_pack != 0 { slam_pack } else { pack };
                    let mut pack = pack & mask;
                    let [i1, i2, i3, i4] = transmute::<_, &mut [Inst16;4]>(&mut pack);
                    proc_i16(i1, &mut was_asked_to_stop, state, &mut mem_deps, &mut noops_count, &mut decode_pattern, &mut usefull_count);
                    proc_i16(i2, &mut was_asked_to_stop, state, &mut mem_deps, &mut noops_count, &mut decode_pattern, &mut usefull_count);
                    proc_i16(i3, &mut was_asked_to_stop, state, &mut mem_deps, &mut noops_count, &mut decode_pattern, &mut usefull_count);
                    proc_i16(i4, &mut was_asked_to_stop, state, &mut mem_deps, &mut noops_count, &mut decode_pattern, &mut usefull_count);
                    instruction_cycle_count += 1;
                    mem_deps.check_mem_deps(state);
                    if pack != 0 {
                        slam_pack = pack;
                        continue;
                    }
                }
                dec_pat &= !(
                    ((dec_pat >> 0 & 0b11 == i) as u8 * (0b11 << 0)) |
                    ((dec_pat >> 2 & 0b11 == i) as u8 * (0b11 << 2)) |
                    ((dec_pat >> 4 & 0b11 == i) as u8 * (0b11 << 4)) |
                    ((dec_pat >> 6 & 0b11 == i) as u8 * (0b11 << 6))
                );
                i += 1;
                if i == 4 { break }
            }
        } else {
            loop {
                let mut pack = if slam_pack != 0 { slam_pack } else { pack };
                let [i1, i2, i3, i4] = transmute::<_, &mut [Inst16;4]>(&mut pack);
                proc_i16(i1, &mut was_asked_to_stop, state, &mut mem_deps, &mut noops_count, &mut decode_pattern, &mut usefull_count);
                proc_i16(i2, &mut was_asked_to_stop, state, &mut mem_deps, &mut noops_count, &mut decode_pattern, &mut usefull_count);
                proc_i16(i3, &mut was_asked_to_stop, state, &mut mem_deps, &mut noops_count, &mut decode_pattern, &mut usefull_count);
                proc_i16(i4, &mut was_asked_to_stop, state, &mut mem_deps, &mut noops_count, &mut decode_pattern, &mut usefull_count);
                instruction_cycle_count += 1;
                mem_deps.check_mem_deps(state);
                if pack != 0 {
                    slam_pack = pack;
                    continue;
                }
                break;
            }
        }
        if was_asked_to_stop { break; }
    }

    let mut report = String::new();
    use core::fmt::Write;
    let usefull = usefull_count;
    let ipc_rate = usefull as f32 / instruction_cycle_count as f32;
    let total_instruction_count = usefull + noops_count;
    let sparcity = (noops_count as f32 / total_instruction_count as f32) * 100.0;
    let total_cycle_count = instruction_cycle_count + cache_fill_stall;

    writeln!(&mut report, "Executed total of {total_instruction_count} instructions in {total_cycle_count} cycles").unwrap();
    writeln!(&mut report, "{usefull} usefull and {noops_count} noops in {instruction_cycle_count} cycles (~{ipc_rate:.1} average IPC rate) (~{sparcity:.1}% average sparsity)").unwrap();
    writeln!(&mut report, "{cache_fill_stall} cycles spent stalled on instruction stash fill").unwrap();
    writeln!(&mut report, "Experienced {i_cache_miss_count} misses and {i_cache_hit_count} hits on i$ during execution").unwrap();
    writeln!(&mut report, "Programm size was {} bytes", byte_len).unwrap();
    return report;

} }

fn proc_i16(
    inst: &mut Inst16,
    stop: &mut bool,
    cpu: &mut CpuState,
    mem_ops: &mut MemCtrl,
    nop_count: &mut usize,
    dec_pat: &mut u8,
    use_count: &mut usize
) { unsafe {
    match inst.get_opcode() {
        Opcode::DBG_HALT => {
            *stop = true;
            *use_count += 1;
        },
        Opcode::DNO => {
            *nop_count += 1;
        },
        Opcode::LDV => {
            *use_count += 1;
            let sr = inst.get_arg1();
            let src_addr = cpu.gp_reg_file.dw[sr as usize];
            let val = if src_addr < 1 << 48 {
                *cpu.scratchpad_mem.dw.as_ptr().cast::<u8>().add(src_addr as usize).cast::<u64>()
            } else {
                0
            };
            let dst = inst.get_arg2();
            cpu.gp_reg_file.dw[dst as usize] = val;
        },
        Opcode::STV => {
            *use_count += 1;
            let dar = inst.get_arg1();
            let dst_addr = cpu.gp_reg_file.dw[dar as usize];
            if dst_addr < 1 << 48 {
                let vr = inst.get_arg2();
                let val = cpu.gp_reg_file.dw[vr as usize];
                cpu.scratchpad_mem.dw.as_mut_ptr().cast::<u8>().add(dst_addr as usize).cast::<u64>().write(val);
            }
        },
        Opcode::LDS => {
            *use_count += 1;
            let mmem_addr_r = inst.get_arg1();
            let mmem_addr = cpu.gp_reg_file.dw[mmem_addr_r as usize];
            if mmem_addr >= 1 << 48 {
                *inst = Inst16::dno();
                return
            }
            let lmem_addr_r = inst.get_arg2();
            let lmem_addr = cpu.gp_reg_file.dw[lmem_addr_r as usize];
            let mem_op_ty = inst.get_mem_size();
            let blk_sz_ty = match mem_op_ty {
                MemBlkSize1::B64 => MemBlkSize::B64,
                MemBlkSize1::B1024 => MemBlkSize::B1024,
            };
            let mem_op = MemOp {
                src_addr: mmem_addr,
                dst_addr: lmem_addr,
                op_type: MMemTransDir::MEM2SCPAD,
                blk_sz: blk_sz_ty
            };
            let ok = mem_ops.put(mem_op);
            if !ok { return }
            *inst = Inst16::dno();
        },
        Opcode::STS => {
            *use_count += 1;
            let lmem_addr_r = inst.get_arg1();
            let lmem_addr = cpu.gp_reg_file.dw[lmem_addr_r as usize];
            if lmem_addr >= 1 << 48 {
                *inst = Inst16::dno();
                return;
            }
            let mmem_addr_r = inst.get_arg2();
            let mmem_addr = cpu.gp_reg_file.dw[mmem_addr_r as usize];
            let blk_sz_ty = match inst.get_mem_size() {
                MemBlkSize1::B64 => MemBlkSize::B64,
                MemBlkSize1::B1024 => MemBlkSize::B1024,
            };
            let mem_op = MemOp {
                src_addr: lmem_addr,
                dst_addr: mmem_addr,
                op_type: MMemTransDir::SCPAD2MEM,
                blk_sz: blk_sz_ty
            };
            let ok = mem_ops.put(mem_op);
            if !ok { return }
            *inst = Inst16::dno();
        },
        Opcode::CPY => {
            *use_count += 1;
            let a1 = inst.get_arg1();
            let a2 = inst.get_arg2();
            cpu.gp_reg_file.dw[a2 as usize] = cpu.gp_reg_file.dw[a1 as usize];
        },
        Opcode::PUC => {
            *use_count += 1;
            let a1 = inst.get_arg1();
            let a2 = inst.get_imm();
            cpu.gp_reg_file.dw[a1 as usize] = a2 as _;
        },
        Opcode::NAND => {
            *use_count += 1;
            let a1 = inst.get_arg1();
            let a2 = inst.get_arg2();
            let a = cpu.gp_reg_file.dw[a1 as usize];
            let b = cpu.gp_reg_file.dw[a2 as usize];
            let c = a & b;
            let d = !c;
            cpu.gp_reg_file.dw[a1 as usize] = d;
        },
        Opcode::BSL  => {
            *use_count += 1;
            let a1 = inst.get_arg1();
            let a2 = inst.get_arg2();
            match a2 {
                0 => {
                    cpu.gp_reg_file.dw[a1 as usize] =
                        cpu.gp_reg_file.dw[a1 as usize] << 1 ;
                },
                1 => {
                    cpu.gp_reg_file.dw[a1 as usize] =
                        cpu.gp_reg_file.dw[a1 as usize] << 4 ;
                },
                2 => {
                    cpu.gp_reg_file.dw[a1 as usize] =
                        cpu.gp_reg_file.dw[a1 as usize] << 16 ;
                },
                _ => unreachable!()
            }
        },
        Opcode::BROT => {
            *use_count += 1;
            let a1 = inst.get_arg1();
            cpu.gp_reg_file.dw[a1 as usize] = cpu.gp_reg_file.dw[a1 as usize].reverse_bits();
        },
        Opcode::TEZ => {
            *use_count += 1;
            let a1 = inst.get_arg1();
            cpu.gp_reg_file.dw[a1 as usize] = (0 == cpu.gp_reg_file.dw[a1 as usize]) as _;
        },
        Opcode::BS => {
            *use_count += 1;
            let a1 = inst.get_arg1();
            let a2 = inst.get_arg2();
            cpu.gp_reg_file.dw[a1 as usize] += cpu.gp_reg_file.dw[a2 as usize];
        },
        Opcode::NPUP => {
            *use_count += 1;
            *dec_pat = (inst.0 >> 8) as u8;
        },
        Opcode::CHKMOPS => {
            *use_count += 1;
            let reg = inst.get_arg1() as usize;
            let addr = cpu.gp_reg_file.dw[reg];

            let mut source = &mem_ops.saddr;
            let mut mask = 0;
            for _ in 0 .. 2 {
                for i in 0 .. 16 {
                    mask |= (1u16 * (source[i] == addr) as u16) << i ;
                }
                source = &mem_ops.daddr;
            }
            let finished = mask == 0;
            cpu.gp_reg_file.dw[reg] = finished as _;
        },
        Opcode::BRZ => {
            *use_count += 1;
            let addr = inst.get_arg1();
            let reg = inst.get_arg2();
            let cond = cpu.gp_reg_file.dw[reg as usize] == 0;
            if cond {
                cpu.gp_reg_file.dw[31] = cpu.gp_reg_file.dw[addr as usize];
            }
        },
        Opcode::LAV => todo!(),
        Opcode::STAV => todo!(),
        Opcode::LIMIT => panic!("Invalid instruction"),
        Opcode::NSDINP => todo!(),
        Opcode::I_EXT_32 => todo!(),
        Opcode::I_EXT_64 => todo!(),
    }
    *inst = Inst16::dno();
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
            let m = a;
            let n = b;
            a = bit_xor(m, n);
            b = bit_and(m , n) << 1;
        }
        return a;
    }
    for k in 0 .. 1024 {
        for l in 0 .. 1024 {
            let c = add(k, l);
            let g = k + l;
            assert!(g == c);
        }
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
#[allow(dead_code)]
fn xor_u64(
    oper1: u8,
    oper2: u8,
    tmp: u8
) -> [Pack;2] {
    assert!((oper1 < 32) & (oper2 < 32) & (tmp < 32));
    [
        Pack::i16x4(&[
            Inst16::cpy(oper1, tmp),
            Inst16::npup([0,1,2,3])
        ]),
        Pack::i16x4(&[
            Inst16::nand(tmp, oper2), // i = a !& b
            Inst16::nand(oper1, tmp), // a = a !& i
            Inst16::nand(oper2, tmp), // b = b !& i
            Inst16::nand(oper1, oper2) // a = a !& b
        ])
    ]
}

#[test]
fn text_xor() {
    let mut cpu = CpuState::new();
    let a = 8;
    let b = 3;
    unsafe {
        cpu.gp_reg_file.dw[0] = a;
        cpu.gp_reg_file.dw[1] = b;
    }
    let mut prg = vec![];
    for i in xor_u64(0, 1, 2) {
        prg.push(i);
    }
    prg.push(Pack::i16x4(&[Inst16::dbg_halt()]));
    let rep = run_programm(&mut cpu, prg.as_slice());
    assert!(unsafe{ cpu.gp_reg_file.dw[0] } == a ^ b);
    println!("{}", rep);
}
#[allow(dead_code)]
fn and_u64(
    oper1: u8,
    oper2: u8,
) -> Pack {
    assert!((oper1 < 32) & (oper2 < 32));
    Pack::i16x4(&[
        Inst16::nand(oper1, oper2),
        Inst16::nand(oper1, oper1)
    ])
}
#[test]
fn test_and() {
    let mut cpu = CpuState::new();
    let a = 8;
    let b = 3;
    unsafe {
        cpu.gp_reg_file.dw[0] = a;
        cpu.gp_reg_file.dw[1] = b;
    }
    let mut prg = vec![];
    prg.push(and_u64(0, 1));
    prg.push(Pack::i16x4(&[Inst16::dbg_halt()]));
    let rep = run_programm(&mut cpu, prg.as_slice());
    println!("{}", rep);
    assert!(unsafe{ cpu.gp_reg_file.dw[0] } == a & b);
}

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
fn fib() -> [Pack;12] {
    [
        Pack::i16x4(&[
            Inst16::puc(0, 1), // a
            Inst16::puc(1, 1), // b
            Inst16::puc(2, 0), // ptr
            Inst16::puc(3, N as _) // end index
        ]),
        Pack::i16x4(&[
            Inst16::puc(8, 24), // loop start addr
            Inst16::puc(9, 0), // zero
            Inst16::puc(10, 8), // add bump len
            Inst16::npup([0,0,0,1]),
        ]),
        Pack::i16x4(&[
            Inst16::puc(5, 0), // loop counter
            Inst16::puc(6, 1), // bump counter
            Inst16::puc(7, 44),
            Inst16::bsl(7, ShiftLength::S1),
        ]),
        // loop
        Pack::i16x4(&[
            Inst16::npup([0,1,2,3])
        ]),
        Pack::i16x4(&[
            Inst16::puc(4, 0), // tmp = b + a
            Inst16::bs(4, 0),
            Inst16::bs(4, 1),
            Inst16::npup([0,1,2,3])
        ]),
        Pack::i16x4(&[
            Inst16::cpy(0, 1), // b = a
            Inst16::cpy(4, 0), // a = tmp
            Inst16::stv(2, 1), // vals[i] = b
            Inst16::npup([0,0,1,0])
        ]),
        Pack::i16x4(&[
            Inst16::bs(2, 10), // ptr += 8
            Inst16::bs(5, 6), // lc += 1
            Inst16::cpy(5, 11),
            Inst16::npup([0,0,1,2])
        ]),
        Pack::i16x4(&[
            Inst16::cpy(5, 12),
            Inst16::cpy(3, 13),
            Inst16::npup([0,1,2,3]),
        ]),
        Pack::i16x4(&[
            Inst16::nand(11, 3),
            Inst16::nand(13, 11),
            Inst16::nand(12, 11),
            Inst16::npup([0,1,0,0])
        ]),
        Pack::i16x4(&[
            Inst16::nand(13, 12),
            Inst16::brz(7, 13),
        ]),
        Pack::i16x4(&[
            Inst16::cpy(8, 31),
        ]),
        Pack::i16x4(&[
            Inst16::dbg_halt()
        ])
    ]
}

fn fibs() {
    let mut cpu = CpuState::new();
    let mut prg = vec![];
    for p in fib() {
        prg.push(p)
    }
    let rep = run_programm(&mut cpu, prg.as_slice());
    println!("{}", rep);
    let computed_fibs = unsafe {&cpu.scratchpad_mem.dw[..(N as usize)]};
    let ground_truth_fibs = gen_fibs();
    assert!(computed_fibs == &ground_truth_fibs[..])
}

#[test]
fn test_fibs() {
    fibs()
}

#[allow(dead_code)]
fn ctz(num: u32) -> usize {
    let mut res = 0;
    let n = (num & u32::MAX) == 0;
    let rem = if n {
        res += 32;
        return res;
    } else {
        let n = (num & ((1 << 16) - 1)) == 0;
        if n {
            res += 16;
            16
        } else {
            let n = (num & ((1 << 8) - 1)) == 0;
            if n {
                res += 8;
                8
            } else {
                0
            }
        }
    };
    for i in 0 .. (32 - rem) {
        let n = i + rem;
        let one = (num & (1 << n)) != 0;
        if one { break }
        res += 1;
    }
    return res;
}

#[test]
fn ctz_test() {
    for i in 0 .. 31 {
        let num = 1 << i;
        let res = ctz(num);
        assert!(res == i)
    }
}