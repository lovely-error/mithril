> CPU is for Cringe Processing Unit

Here I explore
1. `4ops-wide VLIW`
2. `explicit caching`
3. Control flow through address manipulation and stores/loads from ip register
   1. 31st register is rw and represents address from which cpu reads intructions
   2. ISA does not have intructions for doing conditional jumps

So far computing 10 iterations of fib seqv in scratchpad mem shows this perf props

`Executed 260 instructions (117 nops) in 65 cycles (0 on gmem access)`

Not very promissing...
1. Loops have to not be sparse.
   1. 45% of wasted EUs in fib code
   2. Need 4x EUs (or more!)
      1. Semisolvable with unified ALUs?
2. OOE is limited
   1. Wont run many loops simultaneously
3. Width tied to impl can possibly cause binary-level portability issues
4. Need faf (large!) scartchpad mem or busted

---
Conclusion: join the dark side, embrace register renaming in chip 💀 and stuff (?)

Maybe lifting the restriction about in-block intruction dependencies would make it more viable? Sorta turn it into some blend of vliw and dataflow-ish design... (or do it as an implementation detail of a backend for some existing ISA...)

Do dataflow instead: each instruction pack may have inter-instruction deps and they physically configure pipes in a grid where each node is an EU.