''' Generate n sequential digits. e.g. 12, 23 .. 90'''

import re
from collections.abc import Sequence



def circular_sequences(charset: list[str], length: int) -> list[str]:
    '''
    Return a list of all forward string sequences from a charcter list

    Example:
    ['a', 'b', 'c'], length = 3 => ['abc', 'bca', cab']

    Note: string reversal is intentionally excluded to minimize the attack space.
    Reversals can be explicitly applied through a mask as a separate operation.
    '''
    
    n = len(charset)   
    results = []

    for i in range(n):
        seq = "".join(charset[(i + j) % n ] for j in range(length))
        results.append(seq)
    return results


def shifted_symbol(locale: str = "en-US", digit: int = 0):
    '''
    Return the keyboard shifted symbol for a provided digit (0-9)
    according to the specified keyboard layout.
    '''

    keyboard_layouts = {
        "en-US": [")", "!", "@", "#", "$", "%", "^", "&", "*", "("],
        "fr-FR": [")", "1", "2", "3", "4", "5", "6", "7", "8", "9"],
        "de-DE": ["!", '"', "§", "$", "%", "&", "/", "(", ")", "="],
        "en-GB": [")", "!", '"', "£", "$", "%", "^", "&", "*", "("],
        "es-ES": ["!", '"', "·", "$", "%", "&", "/", "(", ")", "="],
        "es-LA": ["!", '"', "#", "$", "%", "&", "/", "(", ")", "="],
        "it-IT": ["!", '"', "£", "$", "%", "&", "/", "(", ")", "="],
        "sv-SE": ["!", '"', "#", "¤", "%", "&", "/", "(", ")", "="],
        "ru-RU": ["!", '"', "№", ";", "%", ":", "?", "*", "(", ")"],
        "ja-JP": [")", "!", '"', "#", "$", "%", "&", "'", "(", ")"],
    }

    if locale not in keyboard_layouts:
        raise ValueError(f"Unsupported locale: {locale}")

    if not (0 <= digit <= 9):
        raise ValueError(f"Digit must in 0..9, got {digit}")
    return keyboard_layouts[locale][digit]


def parse_mask(mask: str) -> tuple[tuple[str, int], ...]:
    '''
    Map shifted symbols using a mask or map.
        dx: emits the digit in position x
        sx: emits the symbol mapped to the digit in position x
        Return a tuple of directive tuples in the form of <directive>, <position>. 
    
    Rules:
        1. Allowable chars: s, d, integer
        2. Each char must be followed by an int.
    '''

    # this is the mask regex
    _PAIR_RE = re.compile(r'([sd])(\d+)')


    # test for valid mask
    if not isinstance(mask, str):
        raise TypeError("mask must be a string")
    if not mask:
        raise ValueError("mask is empty")
    if mask[0] not in ("s", "d"):
        raise ValueError("mask must start with a directive")
    
    # define the output format
    out: list[tuple[str, int]] = []
    pos = 0 # stores current match position

    for m in _PAIR_RE.finditer(mask):
        if m.start() != pos:    # has to start where we think it does (zero for first match)
            bad_index = pos     #   if not, record where we are and note the error(s).
            ch = mask[bad_index]
            if ch in ("s", "d"):
                raise ValueError(f"expected integer after directive at index {bad_index}")
            raise ValueError(f"unexpected character {ch!r} at index {bad_index}")
        directive, digits = m.group(1), m.group(2)
        out.append((directive, int(digits)))
        pos = m.end()      # get ready for the next matched group.

    return tuple(out)


def apply_directives(template: Sequence[str], directives: tuple[tuple[str, int], ...]) -> str:
    '''
    Apply (directive, position) pairs to a template using 1-based positions.
    'd' => emit digit at position; 's' => emit shifted symbol of that digit.
    '''

    output: list[str] = []
    for directive, pos in directives:
        ch = template[pos - 1]
        if directive == 'd':
            output.append(ch)
        elif directive == 's':
            if not ch.isdigit():
                raise ValueError(f"template[{pos-1}]='{ch}' is not a digit for 's'")
            output.append(shifted_symbol("en-US", int(ch)))
        else:
            raise ValueError(f"Unknown directive: {directive}")
    return "".join(output)

                     

def main():

    keyboard_num_row: list[str] = ['1', '2' ,'3', '4', '5', '6', '7', '8', '9', '0']

    seqs = circular_sequences(keyboard_num_row, 4)
    print("sequences", seqs)

    mask = 'd000001d2d3s1s2s3'
    directives = parse_mask(mask)
    print("mask:", mask)
    print("directives: ", directives)

    result = [apply_directives(seq, directives) for seq in seqs]
    print("results", result)



if __name__ == '__main__':
    main()
