> CPU is for Cringe Processing Unit !

Here I explore moving all complexity from hardware to software :3
1. `4ops-wide EIPA (Explicit instruction parallel architecture)`
2. `explicit stashing of data` . Directly addressable scratchpad (like `__local` in OpenCL) . Kind of extension to register file
3. Only 4 operations for data processing! `(TEZ, NAND, BSL, BSR)`
4. Control flow through address manipulation and stores/loads from ip register
   1. 31st register is rw and represents address from which cpu reads intructions
   2. ISA does not have intructions for doing conditional jumps

So far computing 10 iterations of fib seqv in scratchpad mem shows this perf props

`Executed 260 instructions (117 nops) in 65 cycles (0 on gmem access)`

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


