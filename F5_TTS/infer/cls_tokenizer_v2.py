import re
import string

try:
    from indic_unified_parser.uparser import wordparse
except Exception:  # noqa: BLE001
    wordparse = None

script_ranges = {
    # Indo-Aryan
    "devanagari": [("\u0900", "\u097F")],   # Hindi, Marathi
    "arabic": [("\u0600", "\u06FF")],       # Urdu
    "gurmukhi": [("\u0A00", "\u0A7F")],     # Punjabi
    "gujarati": [("\u0A80", "\u0AFF")],     # Gujarati
    "bengali": [("\u0980", "\u09FF")],      # Bengali
    "odia": [("\u0B00", "\u0B7F")],         # Odia

    # Dravidian
    "tamil": [("\u0B80", "\u0BFF")],        # Tamil
    "telugu": [("\u0C00", "\u0C7F")],       # Telugu
    "kannada": [("\u0C80", "\u0CFF")],      # Kannada
    "malayalam": [("\u0D00", "\u0D7F")],    # Malayalam

    # English + digits + common punctuation
    "latin_basic": [("\u0020", "\u007E")]   # English letters, digits, ASCII symbols
}

def has_non_indic_script(text):
    """
    Check if text contains any other characters from the specified Indic/Urdu scripts.
    
    Returns True if any character falls in the Unicode ranges outside of Hindi/Marathi (Devanagari),
    Gujarati, Punjabi (Gurmukhi), Urdu (Arabic), English (Latin) False otherwise.
    """
    for char in text:
        for lang, ranges in script_ranges.items():
            for start, end in ranges:
                if start <= char <= end:
                    return False
    return True

non_problematic_chars = set()

for lang, ranges in script_ranges.items():
    for start, end in ranges:
        for char in range(ord(start), ord(end)+1):
            parsed = None
            try:
                parsed = wordparse(chr(char), 0, 0, 1)
            except Exception as e:
                pass
            if parsed is not None and isinstance(parsed, str) and parsed.strip() != "":
                non_problematic_chars.add(chr(char))

def get_transliteration(text, language):
    if language.lower() == "urdu":
        # We will not add bias with transliteration as of now
        # text = ml_transliterate(text, from_script="ur-PK", to_script="hi-IN")
        pass
    elif language.lower() == "punjabi":
        # GURUMUKHI TO DEVNAGRI IS NOT THERE - SKIPPING FOR NOW, WILL REVIST IF CLS SHOWS EVIDENCE OF IMPROVEMENT
        # text = script_convert(text, from_script="pa-IN", to_script="hi-IN")
        pass
    return text

def normalize_indic_nasals(text):
    # Combined pattern and replacement using capturing groups for script-specific anusvara and consonant groups
    pattern = (
        r'(ं|ং|ં|ਂ|ಂ|ം|ଂ|ం|ஂ)'  # anusvara chars for Devanagari, Bengali, Gujarati, Punjabi, Kannada, Malayalam, Odia, Telugu, Tamil
        r'([कखगघङचछजझञटठडढणतथदधनपफबभम'
        r'কখগঘঙচছজঝঞটঠডঢণতথদধনপফবভম'
        r'કખગઘઙચછજઝઞટઠડઢણતથદધનપફબભમ'
        r'ਕਖਗਘਙਚਛਜਝਞਟਠਡਢਣਤਥਦਧਨਪਫਬਭਮ'
        r'ಕಖಗಘಙಚಛಜಝಞಟಠಡಢಣತಥದಧನಪಫಬಭಮ'
        r'കഖഗഘങചഛജഝഞടഠഡഢണതഥദധനപഫബഭമ'
        r'କଖଗଘଙଚଛଜଝଞଟଠଡଢଣତଥଦଧନପଫବଭମ'
        r'కఖగఘఙచఛజఝఞటఠడఢణతథదధనపఫబభమ'
        r'கஙசஜஞடணதநபம])'
    )
    
    replacement = lambda m: {
        # Mapping anusvara to conjunct nasal for each script block
        'ं': {'क': 'ङ्', 'ख': 'ङ्', 'ग': 'ङ्', 'घ': 'ङ्', 'ङ': 'ङ्',
              'च': 'ञ्', 'छ': 'ञ्', 'ज': 'ञ्', 'झ': 'ञ्', 'ञ': 'ञ्',
              'ट': 'ण्', 'ठ': 'ण्', 'ड': 'ण्', 'ढ': 'ण्', 'ण': 'ण्',
              'त': 'न्', 'थ': 'न्', 'द': 'न्', 'ध': 'न्', 'न': 'न्',
              'प': 'म्', 'फ': 'म्', 'ब': 'म्', 'भ': 'म्', 'म': 'म्'},
        'ং': {'ক': 'ঙ্', 'খ': 'ঙ্', 'গ': 'ঙ্', 'ঘ': 'ঙ্', 'ঙ': 'ঙ্',
              'চ': 'ঞ্', 'ছ': 'ঞ্', 'জ': 'ঞ্', 'ঝ': 'ঞ্', 'ঞ': 'ঞ্',
              'ট': 'ণ্', 'ঠ': 'ণ্', 'ড': 'ণ্', 'ঢ': 'ণ্', 'ণ': 'ণ্',
              'ত': 'ন্', 'থ': 'ন্', 'দ': 'ন্', 'ধ': 'ন্', 'ন': 'ন্',
              'প': 'ম্', 'ফ': 'ম্', 'ব': 'ম্', 'ভ': 'ম্', 'ম': 'ম্'},
        'ં': {'ક': 'ઙ્', 'ખ': 'ઙ્', 'ગ': 'ઙ્', 'ઘ': 'ઙ્', 'ઙ': 'ઙ્',
              'ચ': 'ઞ્', 'છ': 'ઞ્', 'જ': 'ઞ્', 'ઝ': 'ઞ્', 'ઞ': 'ઞ્',
              'ટ': 'ણ્', 'ઠ': 'ણ્', 'ડ': 'ણ્', 'ઢ': 'ણ્', 'ણ': 'ણ્',
              'ત': 'ન્', 'થ': 'ન્', 'દ': 'ન્', 'ધ': 'ન્', 'ન': 'ન્',
              'પ': 'મ્', 'ફ': 'મ્', 'બ': 'મ્', 'ભ': 'મ્', 'મ': 'મ્'},
        'ਂ': {'ਕ': 'ਙ੍', 'ਖ': 'ਙ੍', 'ਗ': 'ਙ੍', 'ਘ': 'ਙ੍', 'ਙ': 'ਙ੍',
              'ਚ': 'ਞ੍', 'ਛ': 'ਞ੍', 'ਜ': 'ਞ੍', 'ਝ': 'ਞ੍', 'ਞ': 'ਞ੍',
              'ਟ': 'ਣ੍', 'ਠ': 'ਣ੍', 'ਡ': 'ਣ੍', 'ਢ': 'ਣ੍', 'ਣ': 'ਣ੍',
              'ਤ': 'ਨ੍', 'ਥ': 'ਨ੍', 'ਦ': 'ਨ੍', 'ਧ': 'ਨ੍', 'ਨ': 'ਨ੍',
              'ਪ': 'ਮ੍', 'ਫ': 'ਮ੍', 'ਬ': 'ਮ੍', 'ਭ': 'ਮ੍', 'ਮ': 'ਮ੍'},
        'ಂ': {'ಕ': 'ಙ್', 'ಖ': 'ಙ್', 'ಗ': 'ಙ್', 'ಘ': 'ಙ್', 'ಙ': 'ಙ್',
              'ಚ': 'ಞ್', 'ಛ': 'ಞ್', 'ಜ': 'ಞ್', 'ಝ': 'ಞ್', 'ಞ': 'ಞ್',
              'ಟ': 'ಣ್', 'ಠ': 'ಣ್', 'ಡ': 'ಣ್', 'ಢ': 'ಣ್', 'ಣ': 'ಣ್',
              'ತ': 'ನ್', 'ಥ': 'ನ್', 'ದ': 'ನ್', 'ಧ': 'ನ್', 'ನ': 'ನ್',
              'ಪ': 'ಮ್', 'ಫ': 'ಮ್', 'ಬ': 'ಮ್', 'ಭ': 'ಮ್', 'ಮ': 'ಮ್'},
        'ം': {'ക': 'ങ്', 'ഖ': 'ങ്', 'ഗ': 'ങ്', 'ഘ': 'ങ്', 'ങ': 'ങ്',
              'ച': 'ഞ്', 'ഛ': 'ഞ്', 'ജ': 'ഞ്', 'ഝ': 'ഞ്', 'ഞ': 'ഞ്',
              'ട': 'ണ്', 'ഠ': 'ണ്', 'ഡ': 'ണ്', 'ഢ': 'ണ്', 'ണ': 'ണ',
              'ത': 'ന്', 'ഥ': 'ന്', 'ദ': 'ന്', 'ധ': 'ന്', 'ന': 'ന്',
              'പ': 'മ്', 'ഫ': 'മ്', 'ബ': 'മ്', 'ഭ': 'മ്', 'മ': 'മ്'},
        'ଂ': {'କ': 'ଙ୍', 'ଖ': 'ଙ୍', 'ଗ': 'ଙ୍', 'ଘ': 'ଙ୍', 'ଙ': 'ଙ୍',
              'ଚ': 'ଞ୍', 'ଛ': 'ଞ୍', 'ଜ': 'ଞ୍', 'ଝ': 'ଞ୍', 'ଞ': 'ଞ୍',
              'ଟ': 'ଣ୍', 'ଠ': 'ଣ୍', 'ଡ': 'ଣ୍', 'ଢ': 'ଣ୍', 'ଣ': 'ଣ୍',
              'ତ': 'ନ୍', 'ଥ': 'ନ୍', 'ଦ': 'ନ୍', 'ଧ': 'ନ୍', 'ନ': 'ନ୍',
              'ପ': 'ମ୍', 'ଫ': 'ମ୍', 'ବ': 'ମ୍', 'ଭ': 'ମ୍', 'ମ': 'ମ୍'},
        'ం': {'క': 'ఙ్', 'ఖ': 'ఙ్', 'గ': 'ఙ్', 'ఘ': 'ఙ్', 'ఙ': 'ఙ్',
              'చ': 'ఞ్', 'ఛ': 'ఞ్', 'జ': 'ఞ్', 'ఝ': 'ఞ్', 'ఞ': 'ఞ్',
              'ట': 'ణ్', 'ఠ': 'ణ్', 'డ': 'ణ్', 'ఢ': 'ణ్', 'ణ': 'ణ్',
              'త': 'న్', 'థ': 'న్', 'ద': 'న్', 'ధ': 'న్', 'న': 'న్',
              'ప': 'మ్', 'ఫ': 'మ్', 'బ': 'మ్', 'భ': 'మ్', 'మ': 'మ్'},
        'ஂ': {'க': 'ங்', 'ங': 'ங்',
              'ச': 'ஞ்', 'ஜ': 'ஞ்', 'ஞ': 'ஞ்',
              'ட': 'ண்', 'ண': 'ண்',
              'த': 'ந்', 'ந': 'ந்',
              'ப': 'ம்', 'ம': 'ம்'}
    }[m.group(1)][m.group(2)] + m.group(2)

    return re.sub(pattern, replacement, text)

def process_segment(segment, state, language):
    if state == "problematic":
        return list(segment)
    elif state == "english":
        return [f"en_{char}" for char in segment]
    else:
        try:
            return wordparse(segment, 0, 0, 1).split()
        except Exception as e:
            return list(segment)

def get_cls_token_list(text, language):
    cls_token_list = []
    state = "text"
    if has_non_indic_script(text):
        raise Exception("Non-indic script found in text.")
    for word in text.split():
        segment = ""
        for char in word:
            if char in string.ascii_letters:
                curr_state = "english"
            elif char in non_problematic_chars:
                curr_state = "text"
            else:
                curr_state = "problematic"
            if state != curr_state:
                cls_token_list.extend(process_segment(segment, state, language))
                segment = ""
            segment += char
            state = curr_state
        if segment:
            cls_token_list.extend(process_segment(segment, state, language))
        cls_token_list.append(" ")
    return cls_token_list[:-1]

def get_cls_for_out_of_mapping(text):
    cls_token_list = []
    for word in text.split():
        for char in word:
            processed_char = char
            if char in string.ascii_letters:
                processed_char = f"en_{char}"
            cls_token_list.append(processed_char)
        cls_token_list.append(" ")
    return cls_token_list[:-1]

def cls_tokenize_text(text: str, language: str):
    if wordparse is None:
        raise RuntimeError("indic_unified_parser is required for CLS tokenization but is not installed.")
    return get_cls_token_list(normalize_indic_nasals(get_transliteration(text.lower(), language)), language)
