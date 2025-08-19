NDFILL-DSL(5) — Fill Insertion DSL Manual
NAME

ndfill-dsl — a small domain-specific language for inserting shifted number-row symbols relative to a preserved digit seed
SYNOPSIS

<seed> -> <action>

DESCRIPTION

ndfill-dsl describes how to generate filler strings from a seed of digits and the corresponding shifted symbols (US keyboard number row). The seed’s digits are preserved in order unless an action explicitly reverses them. Shifted symbols are derived per digit:

1→!   2→@   3→#   4→$   5→%
6→^   7→&   8→*   9→(   0→)

Generated fills can be used downstream (e.g., by nd_combiner) to join with word pairs.
SEED

Exactly one seed selector must be provided. It produces an ordered sequence of digits D = d1 d2 … dn and a matched symbol sequence S = s1 s2 … sn.
Literal digits

@<digits>

Examples:

@12          # D = 1 2
@1234        # D = 1 2 3 4

Mask (subset of hashcat maskprocessor)

#mask(<mask>)

Supported mask subset:

    ?d — any digit 0–9

    ?{…} — explicit set of digits per position (e.g., ?{12} means 1 or 2)

    Concatenation of tokens for fixed-length seeds

Examples:

#mask(?d?d)          # 00..99 (two digits, each 0–9)
#mask(?{12}?{12})    # 11,12,21,22

Numeric range (inclusive)

#range(<start>..<end>)

    Start/end must be digit strings of equal width; output is zero‑padded to that width.

Examples:

#range(10..99)       # 10,11,...,99
#range(0000..9999)   # 0000..9999

ACTION

An action defines placement of symbols relative to digits. Actions are composed of one or more layouts, separated by spaces. The final output is the concatenation of each layout’s expansion.
Layout tokens

D     emit digits in seed order (d1 d2 … dn)
D~    emit digits in reverse    (dn … d2 d1)
S     emit symbols in order     (s1 s2 … sn)
S~    emit symbols in reverse   (sn … s2 s1)
zip(...)  interleave digits/symbols by position (see below)

Notes

    The seed’s digits are not modified unless D~ is used; D preserves order.

    Repeating D or S re‑emits those streams; this is allowed and produces concatenation.

    If the seed length is n, both D and S contribute exactly n characters.

ZIP INTERLEAVING

zip interleaves digits and symbols position-wise. Both streams referenced must have equal length n.

Forms:

zip(ds)     => d1 s1 d2 s2 … dn sn
zip(sd)     => s1 d1 s2 d2 … sn dn
zip(d s~)   => d1 sn d2 s(n-1) … dn s1
zip(s~ d)   => sn d1 s(n-1) d2 … s1 dn

    d / s refer to the digit or symbol stream in order.

    d~ / s~ refer to the reversed stream prior to interleaving.

Errors:

    If streams have unequal length (should not occur for valid seeds), the program must error.

    Nested zip(...) is not supported.

GRAMMAR (ABNF-like)

program   = seed WSP "->" WSP action

seed      = "@" 1*DIGIT
          / "#mask(" mask ")"
          / "#range(" number ".." number ")"

mask      = 1*( mtoken )
mtoken    = "?d" / "?{" 1*DIGIT "}"

number    = 1*DIGIT  ; start and end must have equal width

action    = layout *( WSP layout )

layout    = "D" [ "~" ]
          / "S" [ "~" ]
          / "zip(" zspec ")"

zspec     = ( "d" [ "~" ] WSP "s" [ "~" ] )
          / ( "s" [ "~" ] WSP "d" [ "~" ] )

WSP       = 1*( SP / HTAB )

EXAMPLES
Prefix / Postfix

@12           -> D S      => 12!@
@12           -> S D      => !@12
#mask(?d?d)   -> D S~     => 00)@, 01)(, ..., 99@)

Interleave (zip)

@12           -> zip(ds)  => 1!2@
@12           -> zip(sd)  => !1@2
@12           -> zip(d s~)=> 1@2!
@12           -> zip(s~ d)=> @1!2

Reverse order

@12           -> D S~     => 12@!
@12           -> S~ D     => @!12
@12           -> D~ S     => 21!@

Mixed layouts (concatenation)

@12           -> S D S~   => !@12@!
@1234         -> D S      => 1234!@#$
@1234         -> D S~     => 1234$#@!
#range(0000..0002) -> D S => 00)) , 01)( , 02)(

Mask with restricted digits

#mask(?{12}?{12}) -> D S      => 11!!, 12!@, 21@!, 22@@

DIAGNOSTICS

    Invalid seed: non-digit characters in a literal seed; mismatched range width.

    Invalid mask: tokens outside the supported subset.

    Invalid action: unknown layout token; malformed zip(...).

    Length mismatch: zip(...) with unequal stream lengths (should not occur with valid seeds).

Exit codes (suggested):

    0 success

    2 parse error

    3 semantic error (e.g., length mismatch)

DESIGN NOTES

    The seed is authoritative; actions cannot insert, delete, or reorder digits except D~.

    zip is the only interleaving primitive; other compositions are concatenations of whole streams.

    Repetition of D/S is allowed to support patterns like “prefix+interleave,” but avoid unless needed.

SEE ALSO

maskprocessor(1), hashcat(1), nd_combiner(1)