import re
import string
import nltk

# Ensure NLTK datasets are downloaded quietly if missing
for resource in ['stopwords', 'punkt', 'punkt_tab']:
    try:
        nltk.download(resource, quiet=True)
    except Exception:
        pass

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

try:
    stop_words = set(stopwords.words('english'))
except Exception:
    stop_words = set()

# Broad regex pattern to match Unicode emojis as used in main.ipynb
EMOJI_PATTERN = re.compile(
    "["
    "\U0001F600-\U0001F64F"  # emoticons
    "\U0001F300-\U0001F5FF"  # symbols & pictographs
    "\U0001F680-\U0001F6FF"  # transport & map symbols
    "\U0001F1E0-\U0001F1FF"  # flags (iOS)
    "\U00002702-\U000027B0"
    "\U000024C2-\U0001F251"
    "]+",
    flags=re.UNICODE,
)

def preprocess_text(text: str) -> str:
    """
    Applies text preprocessing identical to main.ipynb:
    1. Lowercase
    2. Punctuation removal
    3. Number removal
    4. Emoji removal
    5. NLTK English stopword removal
    """
    if not text or not isinstance(text, str):
        return ""

    # 1. Lowercase
    text = text.lower()

    # 2. Punctuation removal
    text = text.translate(str.maketrans("", "", string.punctuation))

    # 3. Number removal
    text = "".join([char for char in text if not char.isdigit()])

    # 4. Emoji removal
    text = EMOJI_PATTERN.sub(r"", text)

    # 5. Tokenize and remove stopwords
    try:
        words = word_tokenize(text)
    except Exception:
        words = text.split()

    cleaned_words = [word for word in words if word.lower() not in stop_words and word.strip()]
    
    return " ".join(cleaned_words)
