> CPU is for Cringe Processing Unit !

Here I explore moving all complexity from hardware to software :3
1. `4ops-wide EIPA (Explicit instruction parallel architecture)`
2. `explicit stashing of data` . Directly addressable scratchpad memory local to each core (like `__local` in OpenCL when (N,0,0)) . Small fast memory that acts like an extension to the register file
   1. `LDS` and `STS` instructions are used to transfer memory between extmem and scpad
3. Only 4 operations for data processing! `TEZ, NAND, BSL, BSR`
   1. Minimises logic needed to be duplicated for each line of instruction packs
   2. 1-cycle long issue-retire latency on any of these ops!
4. Control flow through address manipulation and stores/loads from ip register
   1. `31st register is rw` and represents address from which cpu reads intructions
   2. ISA does not have intructions for doing conditional jumps
5. 32 special registers which can store only 1 bit of information. These are to be used for instructions which return binary flags.
6. No implicit near caching (scrathpad is in different address space. No L1)
7. No speculative execution (aka no attempts to runahead)
   1. speculative fetch on $i and L2 is not prohibited
8. Out of order beahaviour only for loads and stores to external memory ("main" memory) (completions of issue loads store ops is checkable)
   1. One instruction `CHMOPS` to check if issued load or store of segment in scratchpad has finished or not.
9.  Simplified cache coherence (only E of MESI) on L2 (Only two operations that atomically operate on extmem)
      1. One is `try load exclusive with flag` which returns a value in one register and flag in another indication if a core has a cache line locked. If flag is false, the return value is not most recent . If true, it is most recent.
      2. One is `try store exclusive with flag` which attempts to store a value to external mem and returns a flag if it succeded or not.

So far computing `32 iterations` of fib seqv in scratchpad mem shows this perf props
```
Executed total of 788 instructions.
429 usefull and 359 nops in 196 cycles (~2.2 average IPC rate) (~45.6% average sparsity)
0 cycles spent stalled on response from extmem
Experienced 1 misses and 196 hits on $i during execution
```

Learned so far...
1. Code with no inherent ILP may be bloated with nops
   1. This is the only concerning aspect, tbh
   2. Seems unfixable :(
      1. Maybe, real code dont suffer from this too much...
2. Loops have to not be sparse.
   1. Not max occupancy of EUs in fib code (45% of nops in fib code)
      1. Fib code is not a good representative of real code (there is more to saturate EUs)
      2. tolerable cus there is no high price to pay for EU duplication (its a nand computer basically)
3. OOE is limited
   1. Wont execute ahead of loops
      1. Teach compiler some tricks to merge loops (?)
   2. Wont execute ahead of jumps 4 sure!
4. Width tied to impl can cause binary-level portability issues
5. CPU needs large(dozens of MiBs) ~1-cycle-access scartchpad mem or busted


