> CPU *meant* Cringe Processing Unit !

Here I explore moving all complexity from hardware to software :3
1. `4ops-wide EIPA (Explicit instruction parallel architecture)` in RISC style.
2. `explicit stashing of data` . Directly addressable scratchpad memory local to each core (like `__local` in OpenCL when (N,0,0)) . Small fast memory that acts like an extension to the register file
   1. `LDS` and `STS` instructions are used to transfer memory between extmem and scpad
3. Varying-size instructions, but only with `B16, B32, B64` as alowed lengths
4. Only 4 operations for data processing! `TEZ, NAND, BSL, BSR`
   1. Minimises logic needed to be duplicated for each line of instruction packs
   2. 1-cycle long issue-retire latency on any of these ops!
   3. All DPOs behave like `rX = rX OP rY` (movement is done by separate instruction)
   4. No `add, sub, mul, div` operations implemented in hardware. all should be implemented in software to allow for promise of ~1 cycle execution of packs.
5. Control flow through address manipulation and stores/loads on ip register
   1. `31st register is rw` and represents address from which cpu reads intructions
   2. ISA does not have intructions for doing conditional jumps
   3. branching via "jump tables"
6. No implicit 'near' caching (scrathpad is in different address space. No L1)
7. No speculative execution (aka no attempts to runahead)
   1. speculative fetch on $i and L2 is not prohibited
8. Out of order beahaviour only for loads and stores to external memory ("main" memory) (completions of issued load/store ops is checkable)
   1. One instruction `CHMOPS` to check if issued load or store of segment in scratchpad has finished or not.
   2. Issues of loads/stores on extmem do not stall the execution.
      1. Only an attemp to load a value from scpad on a segment for which a load was issued and have not completed would stall.
      2. Segment stores need not stall at all!
9.  Simplified cache coherence (only E of MESI) on L2 (Only two operations that atomically operate on extmem) (scpad is coherent because only one core writes to it, and no speculation ever occurs) (memory subsystem is only responsible for tracking operations on extmem)
      1. One is `try load exclusive with flag` which returns a value in one register and flag in another indicating if a core has a cache line locked. If flag is false, the return value is not most recent . If true, it is most recent and core owns cache line.
      2. One is `try store exclusive with flag` which attempts to store a value to external mem and returns a flag if it succeded or not.

So far computing `32 iterations` of fib seqv in scratchpad mem shows this perf props
```
Executed total of 788 instructions.
429 usefull and 359 nops in 196 cycles (~2.2 average IPC rate) (~45.6% average sparsity)
0 cycles spent stalled on response from extmem
Experienced 1 misses and 196 hits on $i during execution
Programm size was 88 bytes
```

Learned so far...

1. Concerns about unavailable OOE-enabled features
   1. Wont execute ahead of jumps!
   2. Which also means that it wont execute ahead of loops!
      1. Teach compiler some tricks to merge loops (?) and skip parts based on condition mask? (conditon mask wont be a part of cpu)
         1. Wont help with "cross module" function calls
2. Code with no inherent ILP may be bloated with nops
   1. Fixable with explicit instruction in a pack that indicates how a consequent pack should be split into different packs `NPUP` (`best approach`) . This means that a few sparse packs can be compressed into one lush pack.
      1. Split `[I1:0, I2:1, I3:2, I4:3]` into 4 packs each containing 1 operation
      2. Split `[I1:0, I2:0, I3:1, I4:2]` into 1 pack with 2 operations and 2 packs with 1 operation
   2. Fixable with "parity check" on one bit in instructions. (suboptimal solution)
      1. Split `[I1:1, I2:0, I3:0, I4:0]`...
      2. ...into `[I1:1, _, _, _]` & `[_, I2:0, I3:0, I4:0]` where `_` is a noop
   3. Maybe, real code dont suffer from this much (no longer relevant)
3. Loops have to not be sparse (pack compression does not solve this).
   1. Not max occupancy of EUs in fib code (45% of nops in fib code)
      1. Fib code is not a good representative of real code (there is more to saturate EUs)
      2. tolerable cus there is no high price to pay for EU duplication (its a nand computer basically). Multiple mops within one bundle can be serialised.
4. Pack width tied to impl can cause binary-level portability issues
   1. Code compatibility should not be a cpu design concern (its an OS (or higher) level concern!)
5. Sracthpad mem size is limited to few dozen kilobytes cuz it should be accessible with at most 1 cycle latency


