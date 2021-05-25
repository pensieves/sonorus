from fairseq.data import Dictionary


class FairseqTokenDictionary(Dictionary):
    """A mapping from symbols to consecutive integers. 
    A modified class which can take a python dictionary of 
    token vocab with already mapped integer values."""

    def __init__(
        self, bos="<s>", pad="<pad>", eos="</s>", unk="<unk>", symbols_int_map=None
    ):
        special_symbols = (bos, pad, eos, unk)
        self.bos_word, self.pad_word, self.eos_word, self.unk_word = special_symbols
        self.symbols = []
        self.count = []
        self.indices = {}

        if symbols_int_map is None:
            symbols = special_symbols
        else:
            assert_str = "special symbol {} should be in the provided symbols_int_map"
            for s in special_symbols:
                assert s in symbols_int_map, assert_str.format(s)

            symbols = sorted(symbols_int_map.keys(), key=lambda k: symbols_int_map[k])

        for s in symbols:
            self.add_symbol(s)

        self.bos_index = self.index(bos)
        self.pad_index = self.index(pad)
        self.eos_index = self.index(eos)
        self.unk_index = self.index(unk)
        self.nspecial = len(self.symbols)
