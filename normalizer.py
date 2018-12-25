import regex as re

ALEF = "\u0627"
ALEF_MADDA = "\u0622"
ALEF_HAMZA_ABOVE = "\u0623"
ALEF_HAMZA_BELOW = "\u0625"

WAW = "\u0648"
WAW_HAMZA = "\u0624"

YEH = "\u064A"
YEH_HAMZA = "\u0626"
DOTLESS_YEH = "\u0649"  # ALEF MAKSOURA

TEH_MARBOUTA = "\u0629"
HEH = "\u0647"

TATWEEL = "\u0640"

FATHATAN = "\u064B"
DAMMATAN = "\u064C"
KASRATAN = "\u064D"
FATHA = "\u064E"
DAMMA = "\u064F"
KASRA = "\u0650"
SHADDA = "\u0651"
SUKUN = "\u0652"
SIFR = "\u0660"
WAHID = "\u0661"
ETHNEN = "\u0662"
THALATHA = "\u0663"
ARBAA = "\u0664"
KHAMSA = "\u0665"
SITTA = "\u0666"
SABAA = "\u0667"
TAMANIA = "\u0668"
TISAA = "\u0669"

NORMALIZATION_RULES = {
    ALEF_MADDA: ALEF, ALEF_HAMZA_ABOVE: ALEF, ALEF_HAMZA_BELOW: ALEF,
    YEH_HAMZA: YEH, DOTLESS_YEH: YEH,
    TEH_MARBOUTA: HEH,
    WAW_HAMZA: WAW,
    TATWEEL: '', FATHATAN: '', DAMMATAN: '', KASRATAN: '', FATHA: '',
    DAMMA: '', KASRA: '', SHADDA: '', SUKUN: '',
    SIFR: '0', WAHID: '1', ETHNEN: '2', THALATHA: '3', ARBAA: '4',
    KHAMSA: '5', SITTA: '6', SABAA: '7', TAMANIA: '8', TISAA: '9'
}


def normalize(string):
    pattern = re.compile(r'\b(' + '|'.join(NORMALIZATION_RULES.keys()) + r')\b')
    liststr = list(string)
    for i in range(len(liststr)):
        liststr[i] = pattern.sub(lambda x: NORMALIZATION_RULES[x.group()], liststr[i])
    return ''.join(liststr)
