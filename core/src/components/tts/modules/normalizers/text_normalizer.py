# Copyright (C) 2024-2025 FableYard
# SPDX-License-Identifier: GPL-3.0-or-later

"""
Text Normalizer for TTS

Normalizes input text to a format suitable for TTS models:
- Expands abbreviations (Mr. -> Mister)
- Converts numbers to words (123 -> one hundred twenty three)
- Handles special characters and symbols
- Normalizes punctuation and whitespace

This is a self-contained implementation with no external dependencies
beyond the Python standard library.
"""

import re
from typing import List, Tuple


class TextNormalizer:
    """
    Text normalizer for TTS preprocessing.

    Converts raw text to a normalized form by:
    - Expanding abbreviations
    - Converting numbers to words
    - Handling special characters
    - Normalizing whitespace and punctuation
    """

    def __init__(self):
        # Build regex patterns
        self._build_patterns()

    def _build_patterns(self):
        """Build all regex patterns for normalization."""

        # Abbreviations with periods
        self._abbreviations = [(re.compile(r'\b%s\.' % x[0], re.IGNORECASE), x[1]) for x in [
            ('mrs', 'misess'),
            ('ms', 'miss'),
            ('mr', 'mister'),
            ('dr', 'doctor'),
            ('st', 'saint'),
            ('co', 'company'),
            ('jr', 'junior'),
            ('maj', 'major'),
            ('gen', 'general'),
            ('drs', 'doctors'),
            ('rev', 'reverend'),
            ('lt', 'lieutenant'),
            ('hon', 'honorable'),
            ('sgt', 'sergeant'),
            ('capt', 'captain'),
            ('esq', 'esquire'),
            ('ltd', 'limited'),
            ('col', 'colonel'),
            ('ft', 'fort'),
        ]]

        # Case-sensitive abbreviations
        self._cased_abbreviations = [(re.compile(r'\b%s\b' % x[0]), x[1]) for x in [
            ('Hz', 'hertz'),
            ('kHz', 'kilohertz'),
            ('KBs', 'kilobytes'),
            ('KB', 'kilobyte'),
            ('MBs', 'megabytes'),
            ('MB', 'megabyte'),
            ('GBs', 'gigabytes'),
            ('GB', 'gigabyte'),
            ('TBs', 'terabytes'),
            ('TB', 'terabyte'),
            ('APIs', "a p i's"),
            ('API', 'a p i'),
            ('CLIs', "c l i's"),
            ('CLI', 'c l i'),
            ('CPUs', "c p u's"),
            ('CPU', 'c p u'),
            ('GPUs', "g p u's"),
            ('GPU', 'g p u'),
            ('Ave', 'avenue'),
            ('etc', 'et cetera'),
            ('Mon', 'monday'),
            ('Tues', 'tuesday'),
            ('Wed', 'wednesday'),
            ('Thurs', 'thursday'),
            ('Fri', 'friday'),
            ('Sat', 'saturday'),
            ('Jan', 'january'),
            ('Feb', 'february'),
            ('Mar', 'march'),
            ('Apr', 'april'),
            ('Aug', 'august'),
            ('Sept', 'september'),
            ('Oct', 'october'),
            ('Nov', 'november'),
            ('Dec', 'december'),
            ('and/or', 'and or'),
        ]]

        # Number patterns
        self._num_prefix_re = re.compile(r'#\d')
        self._num_suffix_re = re.compile(r'\b\d+(K|M|B|T)\b', re.IGNORECASE)
        self._num_letter_split_re = re.compile(r'(\d[a-z]|[a-z]\d)', re.IGNORECASE)
        self._comma_number_re = re.compile(r'(\d[\d,]+\d)')
        self._date_re = re.compile(r'(^|[^/])(\d\d?[/-]\d\d?[/-]\d\d(?:\d\d)?)($|[^/])')
        self._phone_number_re = re.compile(r'(\(?\d{3}\)?[-.\s]\d{3}[-.\s]?\d{4})')
        self._time_re = re.compile(r'(\d\d?:\d\d(?::\d\d)?)')
        self._pounds_re = re.compile(r'£([\d,]*\d+)')
        self._dollars_re = re.compile(r'\$([\d.,]*\d+)')
        self._decimal_number_re = re.compile(r'(\d+(?:\.\d+)+)')
        self._multiply_re = re.compile(r'(\d\s?\*\s?\d)')
        self._divide_re = re.compile(r'(\d\s?/\s?\d)')
        self._add_re = re.compile(r'(\d\s?\+\s?\d)')
        self._subtract_re = re.compile(r'(\d?\s?-\s?\d)')
        self._fraction_re = re.compile(r'(\d+(?:/\d+)+)')
        self._ordinal_re = re.compile(r'\d+(st|nd|rd|th)')
        self._number_re = re.compile(r'\d+')

        # Special character patterns
        self._preunicode_special = [(re.compile(x[0]), x[1]) for x in [
            ('—', ' - '),
        ]]
        self._special_characters = [(re.compile(x[0]), x[1]) for x in [
            ('@', ' at '),
            ('&', ' and '),
            ('%', ' percent '),
            (':', '.'),
            (';', ','),
            (r'\+', ' plus '),
            (r'\\', ' backslash '),
            ('~', ' about '),
            ('(^| )<3', ' heart '),
            ('<=', ' less than or equal to '),
            ('>=', ' greater than or equal to '),
            ('<', ' less than '),
            ('>', ' greater than '),
            ('=', ' equals '),
            ('/', ' slash '),
            ('_', ' '),
            (r'\*', ' '),
        ]]

        # Misc patterns
        self._link_header_re = re.compile(r'(https?://)')
        self._dash_re = re.compile(r'(. - .)')
        self._dot_re = re.compile(r'([A-Z]\.[A-Z])', re.IGNORECASE)
        self._parentheses_re = re.compile(r'[\(\[\{].*[\)\]\}](.|$)')
        self._camelcase_re = re.compile(r'\b([A-Z][a-z]*)+\b')

    def normalize(self, text: str) -> str:
        """
        Normalize text for TTS.

        Args:
            text: Raw input text.

        Returns:
            Normalized text suitable for TTS.
        """
        # Pre-unicode special characters
        for regex, replacement in self._preunicode_special:
            text = re.sub(regex, replacement, text)

        # Convert to ASCII
        text = self._to_ascii(text)

        # Normalize newlines
        text = self._normalize_newlines(text)

        # Expand numbers
        text = self._normalize_numbers(text)

        # Expand special patterns
        text = self._normalize_special(text)

        # Expand abbreviations
        for regex, replacement in self._abbreviations + self._cased_abbreviations:
            text = re.sub(regex, replacement, text)

        # Handle mixed case (CamelCase)
        text = self._normalize_mixedcase(text)

        # Expand special characters
        for regex, replacement in self._special_characters:
            text = re.sub(regex, replacement, text)

        # Lowercase
        text = text.lower()

        # Remove unknown characters
        text = re.sub(r"[^A-Za-z !\$%&'\*\+,-./0123456789<>\?_]", "", text)
        text = re.sub(r"[<>/_+]", "", text)

        # Collapse whitespace
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r' [.\?!,]', lambda m: m.group(0)[1], text)
        text = text.strip()

        # Deduplicate punctuation
        text = self._dedup_punctuation(text)

        # Collapse triple letters
        text = re.sub(r'(\w)\1{2,}', lambda m: m.group(0)[:2], text)

        return text

    def _to_ascii(self, text: str) -> str:
        """Convert unicode to ASCII approximation."""
        import unicodedata
        # Normalize unicode and encode to ASCII, ignoring errors
        text = unicodedata.normalize('NFKD', text)
        text = text.encode('ascii', 'ignore').decode('ascii')
        return text

    def _normalize_newlines(self, text: str) -> str:
        """Normalize newlines to periods."""
        lines = text.split('\n')
        for i in range(len(lines)):
            lines[i] = lines[i].strip()
            if not lines[i]:
                continue
            if lines[i][-1] not in '.!?':
                lines[i] = f"{lines[i]}."
        return ' '.join(lines)

    def _normalize_numbers(self, text: str) -> str:
        """Convert numbers to words."""
        # Number prefix (#1 -> number 1)
        text = re.sub(self._num_prefix_re, lambda m: f"number {m.group(0)[1]}", text)

        # Number suffix (100K -> 100 thousand)
        def expand_suffix(m):
            match = m.group(0)
            suffixes = {'K': 'thousand', 'M': 'million', 'B': 'billion', 'T': 'trillion'}
            return f"{match[:-1]} {suffixes.get(match[-1].upper(), '')}"
        text = re.sub(self._num_suffix_re, expand_suffix, text)

        # Remove commas from numbers
        text = re.sub(self._comma_number_re, lambda m: m.group(1).replace(',', ''), text)

        # Dates
        text = re.sub(self._date_re, lambda m: m.group(1) + ' dash '.join(re.split('[./-]', m.group(2))) + m.group(3), text)

        # Phone numbers
        def expand_phone(m):
            digits = re.sub(r'\D', '', m.group(1))
            if len(digits) == 10:
                return f"{' '.join(list(digits[:3]))}, {' '.join(list(digits[3:6]))}, {' '.join(list(digits[6:]))}"
            return m.group(0)
        text = re.sub(self._phone_number_re, expand_phone, text)

        # Time
        def expand_time(m):
            parts = m.group(1).split(':')
            if len(parts) == 2:
                hours, minutes = parts
                if minutes == '00':
                    if int(hours) == 0:
                        return '0'
                    elif int(hours) > 12:
                        return f"{hours} minutes"
                    return f"{hours} o'clock"
                elif minutes.startswith('0'):
                    minutes = f'oh {minutes[1:]}'
                return f"{hours} {minutes}"
            return m.group(0)
        text = re.sub(self._time_re, expand_time, text)

        # Currency
        text = re.sub(self._pounds_re, r'\1 pounds', text)

        def expand_dollars(m):
            match = m.group(1).replace(',', '')
            parts = match.split('.')
            if len(parts) > 2:
                return match + ' dollars'
            dollars = int(parts[0]) if parts[0] else 0
            cents = int(parts[1]) if len(parts) > 1 and parts[1] else 0
            if dollars and cents:
                d_unit = 'dollar' if dollars == 1 else 'dollars'
                c_unit = 'cent' if cents == 1 else 'cents'
                return f'{dollars} {d_unit}, {cents} {c_unit}'
            elif dollars:
                d_unit = 'dollar' if dollars == 1 else 'dollars'
                return f'{dollars} {d_unit}'
            elif cents:
                c_unit = 'cent' if cents == 1 else 'cents'
                return f'{cents} {c_unit}'
            return 'zero dollars'
        text = re.sub(self._dollars_re, expand_dollars, text)

        # Decimal numbers
        def expand_decimal(m):
            parts = m.group(1).split('.')
            return parts[0] + ' point ' + ' point '.join(' '.join(list(p)) for p in parts[1:])
        text = re.sub(self._decimal_number_re, expand_decimal, text)

        # Math operations
        text = re.sub(self._multiply_re, lambda m: ' times '.join(m.group(1).split('*')), text)
        text = re.sub(self._divide_re, lambda m: ' over '.join(m.group(1).split('/')), text)
        text = re.sub(self._add_re, lambda m: ' plus '.join(m.group(1).split('+')), text)
        text = re.sub(self._subtract_re, lambda m: ' minus '.join(m.group(1).split('-')), text)

        # Fractions
        def expand_fraction(m):
            parts = m.group(1).split('/')
            return ' over '.join(parts) if len(parts) == 2 else ' slash '.join(parts)
        text = re.sub(self._fraction_re, expand_fraction, text)

        # Ordinals
        text = re.sub(self._ordinal_re, lambda m: self._number_to_words_ordinal(m.group(0)), text)

        # Split alphanumeric
        for _ in range(2):
            text = re.sub(self._num_letter_split_re, lambda m: f"{m.group(1)[0]} {m.group(1)[1]}", text)

        # Cardinal numbers
        text = re.sub(self._number_re, lambda m: self._number_to_words(int(m.group(0))), text)

        return text

    def _number_to_words(self, num: int) -> str:
        """Convert a number to words."""
        if num == 0:
            return 'zero'

        ones = ['', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine',
                'ten', 'eleven', 'twelve', 'thirteen', 'fourteen', 'fifteen', 'sixteen',
                'seventeen', 'eighteen', 'nineteen']
        tens = ['', '', 'twenty', 'thirty', 'forty', 'fifty', 'sixty', 'seventy', 'eighty', 'ninety']

        def below_thousand(n):
            if n < 20:
                return ones[n]
            elif n < 100:
                return tens[n // 10] + ('' if n % 10 == 0 else ' ' + ones[n % 10])
            else:
                return ones[n // 100] + ' hundred' + ('' if n % 100 == 0 else ' ' + below_thousand(n % 100))

        # Special handling for years (1900-2099)
        if 1900 <= num < 2100 and num != 2000:
            if num < 2000:
                return below_thousand(num // 100) + ' ' + (below_thousand(num % 100) if num % 100 >= 10 else 'oh ' + ones[num % 100] if num % 100 > 0 else 'hundred')
            elif num < 2010:
                return 'two thousand ' + ones[num % 100]
            else:
                return 'twenty ' + (below_thousand(num % 100) if num % 100 >= 10 else 'oh ' + ones[num % 100])

        # General number conversion
        if num < 1000:
            return below_thousand(num)

        result = []
        scales = [
            (1000000000000, 'trillion'),
            (1000000000, 'billion'),
            (1000000, 'million'),
            (1000, 'thousand'),
        ]

        for scale, name in scales:
            if num >= scale:
                result.append(below_thousand(num // scale) + ' ' + name)
                num %= scale

        if num > 0:
            result.append(below_thousand(num))

        return ' '.join(result)

    def _number_to_words_ordinal(self, s: str) -> str:
        """Convert ordinal number string (e.g., '1st') to words."""
        num = int(re.sub(r'[^\d]', '', s))
        cardinal = self._number_to_words(num)

        # Convert cardinal to ordinal
        if cardinal.endswith('one'):
            return cardinal[:-3] + 'first'
        elif cardinal.endswith('two'):
            return cardinal[:-3] + 'second'
        elif cardinal.endswith('three'):
            return cardinal[:-5] + 'third'
        elif cardinal.endswith('five'):
            return cardinal[:-4] + 'fifth'
        elif cardinal.endswith('eight'):
            return cardinal + 'h'
        elif cardinal.endswith('nine'):
            return cardinal[:-4] + 'ninth'
        elif cardinal.endswith('twelve'):
            return cardinal[:-6] + 'twelfth'
        elif cardinal.endswith('y'):
            return cardinal[:-1] + 'ieth'
        else:
            return cardinal + 'th'

    def _normalize_special(self, text: str) -> str:
        """Handle special patterns like URLs, dashes, dots."""
        # URL headers
        text = re.sub(self._link_header_re, 'h t t p s colon slash slash ', text)

        # Dashes between words
        text = re.sub(self._dash_re, lambda m: f"{m.group(0)[0]}, {m.group(0)[4]}", text)

        # Dots between letters (A.B. -> A dot B)
        text = re.sub(self._dot_re, lambda m: f"{m.group(0)[0]} dot {m.group(0)[2]}", text)

        # Parentheses
        def expand_parens(m):
            match = m.group(0)
            match = re.sub(r'[\(\[\{]', ', ', match)
            match = re.sub(r'[\)\]\}][^$.!?,]', ', ', match)
            match = re.sub(r'[\)\]\}]', '', match)
            return match
        text = re.sub(self._parentheses_re, expand_parens, text)

        return text

    def _normalize_mixedcase(self, text: str) -> str:
        """Split CamelCase words."""
        def split_camel(m):
            match = m.group(0)
            matches = re.findall('[A-Z][a-z]*', match)
            if len(matches) == 1:
                return match  # Single capital word
            if len(matches) == len(match):
                return match  # All uppercase
            if len(matches) == len(match) - 1 and match[-1] == 's':
                return f"{match[:-1]}'s"  # Plural uppercase
            return ' '.join(matches)
        return re.sub(self._camelcase_re, split_camel, text)

    def _dedup_punctuation(self, text: str) -> str:
        """Remove duplicate punctuation."""
        text = re.sub(r"\.\.\.+", "[ELLIPSIS]", text)
        text = re.sub(r",+", ",", text)
        text = re.sub(r"[\.,]*\.[\.,]*", ".", text)
        text = re.sub(r"[\.,!]*![\.,!]*", "!", text)
        text = re.sub(r"[\.,!\?]*\?[\.,!\?]*", "?", text)
        text = re.sub(r"\[ELLIPSIS\]", "...", text)
        return text


class TextSplitter:
    """
    Splits text into chunks suitable for TTS processing.

    Attempts to keep sentences intact while respecting length limits.
    """

    def __init__(self, desired_length: int = 1, max_length: int = 300):
        """
        Initialize text splitter.

        Args:
            desired_length: Minimum desired chunk length.
            max_length: Maximum allowed chunk length.
        """
        self.desired_length = desired_length
        self.max_length = max_length

    def split(self, text: str) -> List[str]:
        """
        Split text into chunks.

        Args:
            text: Input text to split.

        Returns:
            List of text chunks.
        """
        # Normalize whitespace and quotes
        text = re.sub(r'\n\n+', '\n', text)
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[""]', '"', text)

        rv = []
        in_quote = False
        current = ""
        split_pos = []
        pos = -1
        end_pos = len(text) - 1

        def seek(delta):
            nonlocal pos, in_quote, current
            is_neg = delta < 0
            for _ in range(abs(delta)):
                if is_neg:
                    pos -= 1
                    current = current[:-1]
                else:
                    pos += 1
                    current += text[pos]
                if text[pos] == '"':
                    in_quote = not in_quote
            return text[pos]

        def peek(delta):
            p = pos + delta
            return text[p] if 0 <= p < len(text) else ""

        def commit():
            nonlocal current, split_pos
            rv.append(current)
            current = ""
            split_pos = []

        while pos < end_pos:
            c = seek(1)

            # Force split if too long
            if len(current) >= self.max_length:
                if split_pos and len(current) > (self.desired_length / 2):
                    # Seek back to last sentence boundary
                    d = pos - split_pos[-1]
                    seek(-d)
                else:
                    # No sentence boundary, seek back to word boundary
                    while c not in '!?.\n ' and pos > 0 and len(current) > self.desired_length:
                        c = seek(-1)
                commit()

            # Check for sentence boundaries
            elif not in_quote and (c in '!?\n' or (c == '.' and peek(1) in '\n ')):
                # Consume consecutive boundary markers
                while pos < len(text) - 1 and len(current) < self.max_length and peek(1) in '!?.':
                    c = seek(1)
                split_pos.append(pos)
                if len(current) >= self.desired_length:
                    commit()

            # Quote boundaries
            elif in_quote and peek(1) == '"' and peek(2) in '\n ':
                seek(2)
                split_pos.append(pos)

        if current:
            rv.append(current)

        # Clean up
        rv = [s.strip() for s in rv]
        rv = [s for s in rv if s and not re.match(r'^[\s.,;:!?]*$', s)]

        return rv
