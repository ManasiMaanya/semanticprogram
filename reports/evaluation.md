# Retrieval evaluation

Relevance is automatic and tag-based. A problem is relevant to a query when they share at least one specific topic tag. Tags used by more than 20% of tagged problems are ignored (`Algorithms`, `Mathematics`, `Data Structures`). Dynamic programming is kept. The query problem itself is excluded. Hybrid weight `alpha` is the semantic share, chosen by AP@10 on the tuning half and reported on the held-out half.

- Problems indexed: 1316
- Queries with at least 3 relevant problems: 1025 (tune 512, test 513)
- Median relevant problems per query: 152. Recall@10 stays small because most queries have many more than 10 tagged cousins.
- Selected hybrid alpha: 0.65
- Generic tags ignored: algorithms, basic math, basic programming concepts, data structures, math, mathematics, simple math

## Held-out comparison

| Method | P@5 | P@10 | R@10 | AP@10 | nDCG@10 | MRR |
| --- | --- | --- | --- | --- | --- | --- |
| keyword | 0.274 | 0.259 | 0.023 | 0.153 | 0.266 | 0.456 |
| semantic | 0.265 | 0.246 | 0.019 | 0.141 | 0.254 | 0.447 |
| hybrid | 0.289 | 0.269 | 0.023 | 0.162 | 0.277 | 0.464 |

## Same concept, different wording

Each pair shares a specific topic tag, has token Jaccard under 0.10, and semantic similarity of at least 0.55. The neighbor is inside the top 20 by embedding similarity, so the stories can differ while the technique matches.

- `DIVQUERY` Chef and The Divisibility Queries → `PARPERM` Partition It (shared: number theory; semantic 0.83; Jaccard 0.052)
- `AMXOR` XOR Sum → `XOR2` Consecutive Xor (shared: bit manipulation; semantic 0.87; Jaccard 0.093)
- `THREENUMBERS` Three Numbers → `PLMU` Plus Multiply (shared: observation; semantic 0.82; Jaccard 0.067)
- `CHANOQ` Chef and odd queries → `CHGORAM` Chef and Gordon Ramsay (shared: advanced data structures; semantic 0.81; Jaccard 0.071)
- `COOLCHEF` Chef and His Dish → `DIVQUERY` Chef and The Divisibility Queries (shared: queries; semantic 0.82; Jaccard 0.091)
- `INTXOR` Chef and Interactive XOR → `STRINGXOR` Consecutive XOR (shared: bit manipulation; semantic 0.83; Jaccard 0.099)
- `CHEF1010` Chef Loves 1010 → `S08E12` The One Where Joey Dates Rachel (shared: ad hoc, constructive, observation, string; semantic 0.82; Jaccard 0.096)
- `XORABC` Pairwise Xors → `INTXOR` Chef and Interactive XOR (shared: bit manipulation; semantic 0.80; Jaccard 0.085)
- `AMSGAME1` Subtraction Game 1 → `CHNUM` Chef and Number Game (shared: observation; semantic 0.81; Jaccard 0.097)
- `SEAND2` Sereja and Number Division 2 → `SEAVEC` Sereja and Vectors (shared: challenge problem; semantic 0.79; Jaccard 0.083)
- `SEREJA` Sereja and Data Structures  → `SEAPERMS` Sereja and Permutations (shared: advanced algorithms, modular arithmetic, modulo multiplicative inverse, pre processing; semantic 0.79; Jaccard 0.081)
- `GCDSUBARRAYS` Gcd of Subarrays → `D4` Primes in the GCD table (shared: number theory; semantic 0.80; Jaccard 0.092)

## Near misses

Nearest neighbor shares no specific tag. `kind` is a guess from lexical overlap.

- `GOOD2` Good Array 2 → `GOOD1` Good Array 1  (shared wording, different concept; semantic 0.96; keyword 0.73)
- `GOOD1` Good Array 1  → `GOOD2` Good Array 2 (shared wording, different concept; semantic 0.96; keyword 0.73)
- `GOODSUB6` Good Subarrays → `GOOD2` Good Array 2 (shared wording, different concept; semantic 0.94; keyword 0.29)
- `MAX1XOR` Max count of 1 → `MAXONES` Maximize Ones (shared wording, different concept; semantic 0.92; keyword 0.33)
- `MAXONES` Maximize Ones → `MAX1XOR` Max count of 1 (shared wording, different concept; semantic 0.92; keyword 0.33)
- `COMPCLUB` Company and Club Hierarchies → `SHDWCMP` Company Club Membership (shared wording, different concept; semantic 0.92; keyword 0.79)
- `SHDWCMP` Company Club Membership → `COMPCLUB` Company and Club Hierarchies (shared wording, different concept; semantic 0.92; keyword 0.79)
- `LUCKY4` Lucky Array → `LUCKY5` Lucky Long (shared wording, different concept; semantic 0.91; keyword 0.54)
- `LUCKY5` Lucky Long → `LUCKY4` Lucky Array (shared wording, different concept; semantic 0.91; keyword 0.54)
- `UNSORTSORT` Unsorted Sorting → `ARRSORT` Array sorting (shared wording, different concept; semantic 0.91; keyword 0.20)
- `ARRSORT` Array sorting → `UNSORTSORT` Unsorted Sorting (shared wording, different concept; semantic 0.91; keyword 0.20)
- `ENDSORTED` End Sorted → `ARRHALVES` Array Halves (shared wording, different concept; semantic 0.91; keyword 0.29)
