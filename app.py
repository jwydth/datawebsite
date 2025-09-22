import os
import re
import math
from collections import defaultdict, Counter
from datetime import datetime
import random
from markupsafe import Markup
from flask import Flask, render_template, request, redirect, url_for, flash, Response
from flask_sqlalchemy import SQLAlchemy
import pandas as pd
import joblib
import numpy as np

from bs4 import BeautifulSoup
from difflib import get_close_matches
import html
from nltk.collocations import BigramCollocationFinder, BigramAssocMeasures, TrigramAssocMeasures, TrigramCollocationFinder, QuadgramCollocationFinder, QuadgramAssocMeasures
from nltk.probability import *
from nltk.util import ngrams
from sklearn.base import BaseEstimator, TransformerMixin
from collections import Counter as _Counter  # avoid shadowing
import gensim.downloader as api
from gensim.models import FastText
import joblib as _joblib
import lightgbm as lgb
from matplotlib import pyplot as plt
import nltk
from nltk.corpus import words, wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import sent_tokenize, RegexpTokenizer
import numpy as _np
import os as _os
import pandas as _pd
import re as _re
from scipy.sparse import hstack, csr_matrix
from scipy.stats import chi2_contingency
import seaborn as sns
from sklearn.ensemble import AdaBoostClassifier, VotingClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, make_scorer, classification_report
from sklearn.model_selection import cross_val_score, cross_validate, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, PowerTransformer, StandardScaler
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
import warnings

from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
warnings.filterwarnings("ignore")

# --- Ensure NLTK resources (use canonical names) ---
try:
    nltk.download("punkt", quiet=True)
    nltk.download("wordnet", quiet=True)
    nltk.download("omw-1.4", quiet=True)
    nltk.download("words", quiet=True)
    nltk.download("averaged_perceptron_tagger", quiet=True)
    nltk.download("universal_tagset", quiet=True)
    nltk.download('averaged_perceptron_tagger_eng')
except Exception:
    pass

# -------- Optional stemming for simple search
try:
    from nltk.stem import PorterStemmer
except Exception:
    PorterStemmer = None

# -------- Keep class name for joblib compatibility
try:
    from sklearn.base import BaseEstimator, TransformerMixin
except Exception:
    BaseEstimator = object
    class TransformerMixin: pass

# Function to read vocabulary file to a Python dict()
def read_vocab(filename):
  vocab = {}
  with open(filename, 'r') as f:
    vocab = {line.split(':')[0]: int(line.split(':')[1]) for line in f} # Convert index to integer
  return vocab

class TextToVectorTransformer(BaseEstimator, TransformerMixin):
    # (kept as-is from your original)
    def __init__(self):
        self.vocab = None
        self.fasttext_model = None
        self.tfidf_vectorizer = None
        self.data_is_preprocessed = False
    def fit(self, X, y=None):
        self.vocab = read_vocab('vocab_both.txt')
        self.fasttext_model = FastText.load('fasttext_model.model')
        self.tfidf_vectorizer = TfidfVectorizer(analyzer='word', vocabulary=self.vocab, lowercase=True)
        self.tfidf_vectorizer.fit(X)
        self.data_is_preprocessed = True
        return self
    def transform(self, X):
        if self.vocab is None or self.fasttext_model is None or self.tfidf_vectorizer is None:
            raise RuntimeError("This transformer has not been fitted yet. Call .fit() before .transform().")
        data = pd.DataFrame({'New Review': X if self.data_is_preprocessed else self.text_preprocessing(X)})
        weighted_vectors = self.calc_weighted_vectors(data, 'New Review', self.vocab, self.fasttext_model)
        self.data_is_preprocessed = False
        return weighted_vectors
    def tokenize(self, texts, get_vocab=False, print_process=False):
        regex_tokenizer = RegexpTokenizer(r"[a-zA-Z]+(?:[-'][a-zA-Z]+)?")
        ENTITY_RE = re.compile(r"&(?:[A-Za-z]+|#[0-9]+|#x[0-9A-Fa-f]+);")
        corpus = []
        for text in texts:
            tokens = []
            unescaped = html.unescape(text)
            soup = BeautifulSoup(unescaped, 'html.parser')
            text = soup.get_text()
            text = ENTITY_RE.sub(' ', text)
            for sent in sent_tokenize(text):
                words = regex_tokenizer.tokenize(sent)
                words = [w.lower() for w in words]
                tokens.extend(words)
            corpus.append(tokens)
        if get_vocab:
            unique_tokens = sorted({t for doc in corpus for t in doc})
            vocab = {token: idx for idx, token in enumerate(unique_tokens)}
            return corpus, vocab
        return corpus
    def lemmatize(self, corpus, print_process=False):
        result_corpus = []
        pos_map = {'ADJ':'a','ADP':'s','ADV':'r','NOUN':'n','VERB':'v'}
        lemmatizer = WordNetLemmatizer()
        for doc in corpus:
            doc_with_tag = nltk.pos_tag(doc, tagset='universal')
            lemmatized_doc = [lemmatizer.lemmatize(token, pos_map.get(tag, 'n')) for token, tag in doc_with_tag]
            result_corpus.append(lemmatized_doc)
        return result_corpus
    def remove_tokens(self, corpus, tokens_to_remove, remove_single_char=False, print_process=False):
        tokens_to_remove = set(tokens_to_remove)
        cleaned_corpus = []
        for doc in corpus:
            cleaned_doc = [w for w in doc if (w not in tokens_to_remove) and ((not remove_single_char) or len(w) >= 2)]
            cleaned_corpus.append(cleaned_doc)
        return cleaned_corpus
    def add_collocations(self, corpus, collocations, print_process=False):
        result_corpus = []
        for doc in corpus:
            doc = ' '.join(doc)
            for collocation in collocations:
                collocation_with_space = collocation.replace('-', ' ')
                doc = doc.replace(collocation_with_space, collocation)
            doc = doc.split(' ')
            result_corpus.append(doc)
        return result_corpus
    def text_preprocessing(self, texts):
        corpus = self.tokenize(texts)
        with open('collocations.txt', 'r') as f:
            collocations = set(w.strip().lower() for w in f if w.strip())
        corpus = self.add_collocations(corpus, collocations)
        with open('typos.txt', 'r') as f:
            typos_dict = {line.split(':')[0]: line.split(':')[1].strip() for line in f}
        corpus = [[typos_dict.get(token, token) for token in doc] for doc in corpus]
        with open('removed_tokens.txt', 'r') as f:
            removed_tokens = set(w.strip().lower() for w in f if w.strip())
        corpus = self.remove_tokens(corpus, removed_tokens)
        corpus = self.lemmatize(corpus)
        with open("stopwords_en.txt", "r", encoding="utf-8") as f:
            stop_words = set(w.strip().lower() for w in f if w.strip())
        corpus = self.remove_tokens(corpus, stop_words, remove_single_char=True)
        processed_texts = [' '.join(doc) for doc in corpus]
        return processed_texts
    def calc_weighted_vectors(self, df, attribute, vocab_dict, model):
        tfidf_matrix = self.tfidf_vectorizer.transform(df[attribute].fillna(''))
        embedding_matrix = np.zeros((len(vocab_dict), model.wv.vector_size))
        for token, idx in vocab_dict.items():
            if token in model.wv.key_to_index:
                embedding_matrix[idx] = model.wv[token]
        weighted_vectors = []
        for doc_idx in range(tfidf_matrix.shape[0]):
            row = tfidf_matrix.getrow(doc_idx)
            indices = row.indices
            weights = row.data
            if len(indices) == 0:
                weighted_vectors.append(np.zeros(model.wv.vector_size))
                continue
            word_vecs = embedding_matrix[indices]
            weighted_sum = np.dot(weights, word_vecs)
            weighted_avg = weighted_sum / weights.sum()
            weighted_vectors.append(weighted_avg)
        return np.vstack(weighted_vectors)

# -------------------------
# Helpers
# -------------------------
def get_category_image(class_name, item_id):
    """Get a random image for the given clothing class"""
    if not class_name:
        return None
    
    # Normalize class name to lowercase
    folder_name = class_name.lower().replace(' ', '_').replace('-', '_')
    
    # Map all dress-related classes to 'dresses' folder
    if 'dress' in folder_name:
        folder_name = 'dresses'
    
    folder_path = os.path.join('static', 'images', folder_name)
    
    # Check if folder exists
    if not os.path.exists(folder_path):
        return None
    
    # Get all jpg files in the folder
    try:
        image_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.jpg')]
        if not image_files:
            return None
        
        # Use item_id as seed for consistent image per item
        random.seed(item_id)
        selected_image = random.choice(image_files)
        return f"images/{folder_name}/{selected_image}"
    except:
        return None

# -------------------------
# Config
# -------------------------
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DB_PATH = os.path.join(BASE_DIR, "shop.db")
DATA_CSV = os.path.join(BASE_DIR, "data", "assignment3_II.csv")
MODEL_P = os.path.join(BASE_DIR, "models", "review_recommender.joblib")

app = Flask(__name__)
app.secret_key = "dev-secret"  # replace in production
app.config["SQLALCHEMY_DATABASE_URI"] = f"sqlite:///{DB_PATH}"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db = SQLAlchemy(app)

BRAND = "winter"
app.jinja_env.globals["BRAND"] = BRAND
app.jinja_env.globals['get_category_image'] = get_category_image


# -------------------------
# DB Models
# -------------------------
class Item(db.Model):
    __tablename__ = "items"
    id = db.Column(db.Integer, primary_key=True)
    source_clothing_id = db.Column(db.Integer, unique=True)
    title = db.Column(db.String, nullable=False)
    description = db.Column(db.Text)
    class_name = db.Column(db.String)
    department_name = db.Column(db.String)
    created_at = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    __table_args__ = (db.Index("idx_items_title", "title"),)
    reviews = db.relationship("Review", backref="item", lazy=True, cascade="all, delete-orphan")


class Review(db.Model):
    __tablename__ = "reviews"
    id = db.Column(db.Integer, primary_key=True)
    item_id = db.Column(db.Integer, db.ForeignKey("items.id", ondelete="CASCADE"), nullable=False)
    reviewer_age = db.Column(db.Integer)
    title = db.Column(db.String)
    body = db.Column(db.Text, nullable=False)
    rating = db.Column(db.Integer, nullable=False)
    recommend_label = db.Column(db.Integer, nullable=False)
    model_suggested = db.Column(db.Integer, nullable=False, default=0)
    positive_feedback_count = db.Column(db.Integer, nullable=False, default=0)
    created_at = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    __table_args__ = (
        db.CheckConstraint("rating BETWEEN 1 AND 5", name="chk_rating_range"),
        db.CheckConstraint("recommend_label IN (0,1)", name="chk_rec_label"),
        db.Index("idx_reviews_item_created", "item_id", "created_at"),
    )

# -------------------------
# Optional model loader (kept for /suggest and review submit)
# -------------------------
pipeline = None
if os.path.exists(MODEL_P):
    try:
        pipeline = joblib.load(MODEL_P)
        try:
            pipeline.predict(["warm up text"])
        except Exception:
            pass
        print("[Model] Loaded:", MODEL_P)
    except Exception as e:
        print("[Model] Failed to load:", e)
else:
    print("[Model] File not found:", MODEL_P)


class ModelUnavailable(Exception):
    pass


def predict_label_strict(text: str) -> int:
    if pipeline is None:
        raise ModelUnavailable("Model not loaded")
    text = (text or "").strip()
    if not text:
        raise ValueError("Empty input")
    y = int(pipeline.predict([text])[0])
    if y not in (0, 1):
        raise ValueError(f"Model returned non-binary label: {y}")
    return y

# -------------------------
# Search index (build at startup + after import)
# -------------------------
STEMMER = PorterStemmer() if PorterStemmer else None
INVERTED = defaultdict(set)
TOKENS_PER_ITEM = {}  # item_id -> Counter(tokens)
WEIGHTS = {"title": 3.0, "class": 2.0, "department": 2.0, "description": 1.0}
 
TFIDF_VECT = None
TFIDF_MAT  = None
TFIDF_IDS  = []

_norm_token_re = re.compile(r"[^a-z0-9]+")
_tokenize_re = re.compile(r"[A-Za-z0-9']+")


def normalize_token(w: str) -> str:
    w = w.lower()
    w = _norm_token_re.sub("", w)
    if not w:
        return w
    if len(w) <= 3:
        return w
    if STEMMER:
        try:
            return STEMMER.stem(w)
        except Exception:
            pass
    # crude plural -> singular
    # if w.endswith("ies") and len(w) > 4:
    #     return w[:-3] + "y"
    # if w.endswith("es") and len(w) > 3:
    #     return w[:-2]
    # if w.endswith("s") and len(w) > 3:
    #     return w[:-1]
    return w


def tokenize(text: str, normalize: bool = True):
    if not text:
        return []
    if normalize:
        return [normalize_token(t) for t in _tokenize_re.findall(text)]
    return _tokenize_re.findall(text)


def expand_query_tokens(q_tokens):
    """Fuzzy-map each query token to the nearest class_name token (same normalization)."""
    # build normalized vocabulary from class_name
    classes = set()
    for (val,) in db.session.query(Item.class_name).distinct().all():
        for t in tokenize(val or ""):
            if t:
                classes.add(t)
    expanded = []
    for qt in q_tokens:
        # try fuzzy match against class tokens
        m = get_close_matches(qt, list(classes), n=1, cutoff=0.75)
        expanded.append(m[0] if m else qt)
    return expanded


def index_item(item: Item):
    fields = {
        "title": item.title or "",
        # "description": item.description or "",
        "class": item.class_name or "",
        "department": item.department_name or "",
    }
    c = Counter()
    for fname, content in fields.items():
        toks = tokenize(content)
        w = WEIGHTS.get(fname, 1.0)
        for t in toks:
            if not t:
                continue
            INVERTED[t].add(item.id)
            c[t] += w
    TOKENS_PER_ITEM[item.id] = c


def build_tfidf_index():
    global TFIDF_VECT, TFIDF_MAT, TFIDF_IDS
    TFIDF_VECT = None
    TFIDF_MAT = None
    TFIDF_IDS = []

    try:
        items = Item.query.all()
    except Exception:
        items = []

    if not items:
        print("[Search][TFIDF] No items to index.")
        return

    docs: list[str] = []
    ids: list[int] = []
    for it in items:
        title = (it.title or "").strip()
        cls = (it.class_name or "").strip()
        dept = (it.department_name or "").strip()
        desc = (it.description or "").strip()
        doc = " ".join(
            [
                (title + " ") * 2,
                (cls + " ") * 3,
                (dept + " ") * 2,
                desc,
            ]
        )
        docs.append(doc)
        ids.append(it.id)

    try:
        vect = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=50000)
        mat = vect.fit_transform(docs)
        TFIDF_VECT, TFIDF_MAT, TFIDF_IDS = vect, mat, ids
        print(f"[Search][TFIDF] Indexed {len(ids)} items, vocab={len(vect.vocabulary_)}")
    except Exception as e:
        TFIDF_VECT = None
        TFIDF_MAT = None
        TFIDF_IDS = []
        print("[Search][TFIDF] Failed to build:", e)


def build_index():
    INVERTED.clear()
    TOKENS_PER_ITEM.clear()
    for item in Item.query.all():
        index_item(item)
    print(f"[Search] Indexed {len(TOKENS_PER_ITEM)} items, {len(INVERTED)} tokens")
    build_tfidf_index()


def fix_typos(q: str, marker: bool = False):
    q_tokens = tokenize(q)
    expanded: list[str] = []
    for qt in q_tokens:
        matches = get_close_matches(qt, INVERTED.keys(), n=1, cutoff=0.6)
        if not matches or matches[0] == qt:
            expanded.append(qt)
        else:
            expanded.append(matches[0])
    q_tokens_original = tokenize(q, normalize=False)
    for i, qt in enumerate(q_tokens_original):
        if i < len(expanded) and qt.lower() != expanded[i].lower() and marker:
            expanded[i] = f"<b>{expanded[i]}</b>"
    return Markup(" ".join(expanded))

def score_items_simple(q_tokens):
    if not q_tokens:
        return []
    # OR semantics with tf-weight scoring
    candidate_ids = set()
    for t in q_tokens:
        candidate_ids |= INVERTED.get(t, set())

    scored = []
    for item_id in candidate_ids:
        c = TOKENS_PER_ITEM.get(item_id, Counter())
        s = float(sum(c.get(t, 0.0) for t in q_tokens))
        if s > 0:
            scored.append((item_id, s))
    # stable sort by score desc then id asc
    scored.sort(key=lambda x: (-x[1], x[0]))
    return scored

def score_items_tfidf(query_text: str):
    """Return list[(item_id, sim)] like simple scorer."""
    if TFIDF_VECT is None or TFIDF_MAT is None:
        build_tfidf_index()
    if TFIDF_VECT is None or TFIDF_MAT is None:
        return []
    try:
        qvec = TFIDF_VECT.transform([query_text or ""])
        sims = (qvec @ TFIDF_MAT.T).toarray().ravel()
        # return tuples to unify shape
        tuples = [(TFIDF_IDS[i], float(sims[i])) for i in range(len(sims)) if sims[i] > 0]
        tuples.sort(key=lambda x: (-x[1], x[0]))
        return tuples
    except Exception as e:
        print("[Search][TFIDF] query failed:", e)
        return []

def score_items(q: str):
    q_tokens = [t for t in tokenize(q) if t]
    if not q_tokens:
        return []
    q_tokens = expand_query_tokens(q_tokens)

    candidate_ids: set[int] = set()
    for t in q_tokens:
        candidate_ids |= INVERTED.get(t, set())

    scored: list[tuple[int, float]] = []
    for item_id in candidate_ids:
        c = TOKENS_PER_ITEM.get(item_id, Counter())
        s = float(sum(c.get(t, 0) for t in q_tokens))
        if s > 0:
            scored.append((item_id, s))
    return scored

def class_name_exact_ids(q_tokens):
    """If the query exactly equals a normalized class_name label, return those ids (as high score)."""
    if not q_tokens:
        return []
    # only treat single-token exact class label as 'exact'
    if len(q_tokens) != 1:
        return []
    tok = q_tokens[0]
    ids = []
    for it in Item.query.with_entities(Item.id, Item.class_name).all():
        label = " ".join(tokenize(it.class_name or ""))
        if label == tok:
            ids.append(it.id)
    return [(i, 1e6) for i in ids]  # huge boost to show exact group

def highlight_corrections(original_query: str, expanded_tokens):
    """Return Markup for UI with <b>…</b> where we corrected a token."""
    orig_tokens = tokenize(original_query, normalize = False)
    out = []
    for o, e in zip(orig_tokens, expanded_tokens[:len(orig_tokens)]):
        out.append(f"<b>{e}</b>" if e != o else e)
    # append any remaining expanded tokens
    if len(expanded_tokens) > len(orig_tokens):
        out.extend(expanded_tokens[len(orig_tokens):])
    return Markup(" ".join(out) if out else original_query)


def get_multi_token_suggestions(query: str, max_suggestions: int = 3):
    """Generate suggestions for each token in multi-word queries."""
    if not query:
        return []
    
    query_tokens = tokenize(query.strip())
    if not query_tokens:
        return []
    
    suggestions = []
    
    # Get all class names
    class_stats = db.session.query(
        Item.class_name, 
        db.func.count(Item.id).label('count')
    ).filter(
        Item.class_name.isnot(None)
    ).group_by(Item.class_name).all()
    
    class_data = [(class_name.lower(), count) for class_name, count in class_stats if class_name]
    
    # For each query token, find the best class match
    for token in query_tokens:
        best_match = None
        best_score = 0
        
        for class_name, count in class_data:
            class_tokens = tokenize(class_name)
            
            # Check each class token against query token
            for class_token in class_tokens:
                # Calculate similarity
                matches = get_close_matches(token, [class_token], n=1, cutoff=0.6)
                if matches:
                    # Use a higher threshold for better matches
                    from difflib import SequenceMatcher
                    similarity = SequenceMatcher(None, token, class_token).ratio()

                    if similarity > best_score and similarity > 0.6:
                        best_score = similarity
                        best_match = {
                            'original_token': token,
                            'suggested_class': class_name,
                            'count': count,
                            'similarity': similarity
                        }
        
        if best_match:
            # Avoid duplicates
            existing_classes = [s['suggested_class'] for s in suggestions]
            if best_match['suggested_class'] not in existing_classes:
                suggestions.append(best_match)
    
    # Sort by similarity score (best matches first)
    suggestions.sort(key=lambda x: x['similarity'], reverse=True)
    return suggestions[:max_suggestions]

app.jinja_env.globals['get_multi_token_suggestions'] = get_multi_token_suggestions

def items_with_rec_order(
    scored_items: list[tuple[int, float]] | None = None, limit: int | None = None):
    """
    If scored_items is provided → accept [(id, score), ...] too.
    Sort by relevance (if present) → rec_sum → review_count → id.
    If not provided → homepage sort by rec_sum → review_count → id.
    """
    relevance_map: dict[int, float] = {}

    q = (
        db.session.query(
            Item,
            db.func.coalesce(db.func.sum(Review.recommend_label), 0).label("rec_sum"),
            db.func.count(Review.id).label("review_count"),
        )
        .outerjoin(Review, Review.item_id == Item.id)
    )

    if scored_items:
        # Normalize to list of (id, score)
        if scored_items and not isinstance(scored_items[0], (list, tuple)):
            scored_items = [(int(scored_items[0]), 0.0)]  # safeguard; not expected
        ids = [int(item_id) for item_id, _ in scored_items]
        relevance_map = {int(item_id): float(score) for item_id, score in scored_items}
        q = q.filter(Item.id.in_(ids))

    q = q.group_by(Item.id)
    rows = q.all()

    rec_map = {
        row[0].id: {
            "rec_sum": int(row[1] or 0),
            "review_count": int(row[2] or 0),
            "relevance": relevance_map.get(row[0].id, 0.0),
        }
        for row in rows
    }

    if relevance_map:
        items = sorted(
            [row[0] for row in rows],
            key=lambda it: (
                -rec_map[it.id]["relevance"],
                -rec_map[it.id]["rec_sum"],
                -rec_map[it.id]["review_count"],
                it.id,
            ),
        )
    else:
        items = sorted(
            [row[0] for row in rows],
            key=lambda it: (
                -rec_map[it.id]["rec_sum"],
                -rec_map[it.id]["review_count"],
                it.id,
            ),
        )

    if limit:
        items = items[:limit]

    return items, rec_map

def get_search_suggestions(query: str, max_suggestions: int = 3):
    """Generate detailed search suggestions with item counts."""
    if not query:
        return []
    
    suggestions = []
    query_lower = query.strip().lower()
    
    # Get class names with item counts
    class_stats = db.session.query(
        Item.class_name, 
        db.func.count(Item.id).label('count')
    ).filter(
        Item.class_name.isnot(None)
    ).group_by(Item.class_name).all()
    
    # Find fuzzy matches in class names
    for class_name, count in class_stats:
        if class_name and class_name.lower() != query_lower:
            # Check for partial matches or fuzzy similarity
            class_tokens = tokenize(class_name.lower())
            query_tokens = tokenize(query_lower)
            
            # Check if any query token is similar to class tokens
            has_match = False
            for q_token in query_tokens:
                for c_token in class_tokens:
                    if (q_token in c_token or c_token in q_token or 
                        len(get_close_matches(q_token, [c_token], n=1, cutoff=0.6)) > 0):
                        has_match = True
                        break
                if has_match:
                    break
            
            if has_match:
                suggestions.append({
                    'term': class_name.lower(),
                    'count': count,
                    'type': 'class'
                })
    
    # Sort by item count (most items first)
    suggestions.sort(key=lambda x: x['count'], reverse=True)
    
    return suggestions[:max_suggestions]

# Make function available in templates
app.jinja_env.globals['get_search_suggestions'] = get_search_suggestions


def rank_items(query: str, mode: str = "simple"):
    q_raw = (query or "").strip()
    q_tokens = [t for t in tokenize(q_raw) if t]
    if not q_tokens:
        return [], "simple", Markup(query or ""), False

    # Always try original query first
    scored_original = score_items_simple(q_tokens)
    
    # Always generate corrected version for comparison
    q_tokens_exp = expand_query_tokens(q_tokens)
    q_fixed_markup = highlight_corrections(q_raw, q_tokens_exp)
    was_corrected = (q_raw.lower() != " ".join(q_tokens_exp).lower())
    
    # If original query has results, return those
    if scored_original:
        return scored_original, "simple", q_fixed_markup, was_corrected
    
    # If no original results, try corrected version
    exact = class_name_exact_ids(q_tokens_exp)
    if exact:
        return exact, "class", q_fixed_markup, was_corrected

    if mode == "tfidf":
        scored = score_items_tfidf(q_raw)
        if scored:
            return scored, "tfidf", q_fixed_markup, was_corrected

    scored = score_items_simple(q_tokens_exp)
    return scored, "simple", q_fixed_markup, was_corrected

# -------------------------
# CSV import / bootstrap
# -------------------------
def load_items_from_csv(path: str = DATA_CSV) -> int:
    if not os.path.exists(path):
        print(f"[Import] CSV not found at {path}")
        return 0

    df = pd.read_csv(path)

    def norm(s):
        return str(s).strip().lower().replace("_", " ")

    cols = {norm(c): c for c in df.columns}

    def col(*names):
        for n in names:
            k = norm(n)
            if k in cols:
                return cols[k]
        return None

    c_id = col("clothing id", "clothing_id", "id")
    c_title = col("clothes title", "item title", "product title", "title")
    c_desc = col("clothes description", "description", "product description")
    c_class = col("class name", "class")
    c_dept = col("department name", "department")

    if not c_id:
        print("[Import] ERROR: 'Clothing ID' column not found — cannot create real items.")
        return 0

    df = df[df[c_id].notna()].copy()
    df[c_id] = df[c_id].astype(int)
    df = df.sort_values(by=[c_id]).drop_duplicates(subset=[c_id], keep="first")

    added = 0
    for _, r in df.iterrows():
        srcid = int(r[c_id])
        title = str(r.get(c_title, "") or "").strip()
        desc = str(r.get(c_desc, "") or "").strip()
        cls = str(r.get(c_class, "") or "").strip()
        dept = str(r.get(c_dept, "") or "").strip()

        item = Item.query.filter_by(source_clothing_id=srcid).first()
        if not item:
            item = Item(
                source_clothing_id=srcid,
                title=title or f"Item {srcid}",
                description=desc,
                class_name=cls,
                department_name=dept,
            )
            db.session.add(item)
            added += 1
        else:
            if not item.title and title:
                item.title = title
            if not item.description and desc:
                item.description = desc
            if not item.class_name and cls:
                item.class_name = cls
            if not item.department_name and dept:
                item.department_name = dept

    db.session.commit()
    print(f"[Import] Items in DB: {Item.query.count()} (+{added} added)")
    return added


def load_reviews_from_csv(path: str = DATA_CSV, limit: int | None = None) -> int:
    if not os.path.exists(path):
        print(f"[Import][Reviews] CSV not found at {path}")
        return 0

    df = pd.read_csv(path)
    id_map = {it.source_clothing_id: it.id for it in Item.query.all() if it.source_clothing_id is not None}
    title_map = {it.title.strip().lower(): it.id for it in Item.query.all() if it.title}

    existing_keys = set((rv.item_id, (rv.body or "")[:120], (rv.title or "")[:80]) for rv in Review.query.all())

    added = 0
    for _, row in df.iterrows():
        if limit and added >= limit:
            break

        item_id = None
        srcid = row.get("Clothing ID", None)
        if pd.notna(srcid):
            try:
                srcid = int(srcid)
                item_id = id_map.get(srcid)
            except Exception:
                item_id = None
        if not item_id:
            ctitle = str(row.get("Clothes Title", "") or "").strip().lower()
            if ctitle:
                item_id = title_map.get(ctitle)
        if not item_id:
            continue

        rev_title = str(row.get("Title", "") or "").strip()
        body = str(row.get("Review Text", "") or "").strip()
        if not body:
            continue

        try:
            rating = int(row.get("Rating", 5))
        except Exception:
            rating = 5
        rating = max(1, min(5, rating))

        try:
            rec = int(row.get("Recommended IND", 0))
            rec = 1 if rec == 1 else 0
        except Exception:
            rec = 0

        try:
            age = int(row.get("Age"))
        except Exception:
            age = None

        try:
            pfc = int(row.get("Positive Feedback Count", 0))
        except Exception:
            pfc = 0

        key = (item_id, body[:120], rev_title[:80])
        if key in existing_keys:
            continue

        rv = Review(
            item_id=item_id,
            title=rev_title if rev_title else None,
            body=body,
            rating=rating,
            recommend_label=rec,
            model_suggested=rec,
            positive_feedback_count=pfc,
            reviewer_age=age,
        )
        db.session.add(rv)
        existing_keys.add(key)
        added += 1

        if added % 1000 == 0:
            db.session.flush()

    db.session.commit()
    print(f"[Import][Reviews] Added {added} reviews.")
    return added


def bootstrap_if_needed():
    db.create_all()
    if Item.query.count() == 0:
        print("[Bootstrap] Items empty; importing CSV…")
        load_items_from_csv()
    if Review.query.count() == 0:
        print("[Bootstrap] Reviews empty; importing CSV reviews…")
        load_reviews_from_csv()


with app.app_context():
    bootstrap_if_needed()
    # Build search index once DB is ready (important)
    build_index()


def rank_items(query: str, mode: str = "simple"):
    q_raw = (query or "").strip()
    q_tokens = [t for t in tokenize(q_raw) if t]
    if not q_tokens:
        return [], "simple", Markup(query or ""), False

    # Always try original query first
    scored_original = score_items_simple(q_tokens)
    
    # Always generate corrected version for comparison
    q_tokens_exp = expand_query_tokens(q_tokens)
    q_fixed_markup = highlight_corrections(q_raw, q_tokens_exp)
    was_corrected = (q_raw.lower() != " ".join(q_tokens_exp).lower())
    
    # If original query has results, return those
    if scored_original:
        return scored_original, "simple", q_fixed_markup, was_corrected
    
    # If no original results, try corrected version
    exact = class_name_exact_ids(q_tokens_exp)
    if exact:
        return exact, "class", q_fixed_markup, was_corrected

    if mode == "tfidf":
        scored = score_items_tfidf(q_raw)
        if scored:
            return scored, "tfidf", q_fixed_markup, was_corrected

    scored = score_items_simple(q_tokens_exp)
    return scored, "simple", q_fixed_markup, was_corrected

# -------------------------
# Routes
# -------------------------
@app.route("/")
def index():
    q = request.args.get("q", "").strip()
    mode = request.args.get("mode", "simple")  # 'simple' | 'tfidf'
    count = None
    q_fixed = Markup(q)
    was_corrected = False

    if q:
        scored, used_mode, q_fixed, was_corrected = rank_items(q, mode)  # Now expects 4 values
        count = len(scored)
        items, rec_map = items_with_rec_order(scored)
    else:
        used_mode = "simple"
        items, rec_map = items_with_rec_order([], limit=24)

    return render_template(
        "index.html",
        items=items,
        q=q,
        count=count,
        mode=used_mode,
        rec_map=rec_map,
        q_fixed=q_fixed,
        was_corrected=was_corrected  # Add this new variable
    )
    
@app.route("/item/<int:item_id>")
def item_detail(item_id):
    item = Item.query.get_or_404(item_id)
    reviews = Review.query.filter_by(item_id=item.id).order_by(Review.created_at.desc()).all()
    return render_template("item.html", item=item, reviews=reviews)


@app.route("/item/<int:item_id>/review/new", methods=["GET", "POST"])
def new_review(item_id):
    item = Item.query.get_or_404(item_id)

    if request.method == "POST":
        title = request.form.get("title", "").strip()
        body = request.form.get("body", "").strip()
        rating = int(request.form.get("rating", "5"))

        if not body:
            flash("Review description is required.", "danger")
            return redirect(request.url)

        try:
            suggested = predict_label_strict((title + " " + body).strip())
            print(f"[Model] Final submit predicted {suggested} for: {title} {body}")
        except Exception:
            flash("Prediction unavailable right now. Please try again.", "danger")
            return redirect(request.url)

        final_lbl = request.form.get("recommend_label")
        final_lbl = int(final_lbl) if final_lbl in ("0", "1") else suggested

        rv = Review(
            item_id=item.id,
            title=title,
            body=body,
            rating=max(1, min(5, rating)),
            recommend_label=final_lbl,
            model_suggested=suggested,
        )
        db.session.add(rv)
        db.session.commit()
        flash("Review published.", "success")
        return redirect(url_for("review_detail", review_id=rv.id))

    return render_template("review_form.html", item=item, suggested=None)


@app.route("/suggest", methods=["POST"])
def suggest_label():
    title = request.form.get("title", "").strip()
    body = request.form.get("body", "").strip()
    text = (title + " " + body).strip()
    try:
        lbl = predict_label_strict(text)
        print(f"[Model] /suggest => {lbl} for: {text}")
        return Response(str(lbl), mimetype="text/plain")
    except Exception as e:
        print("[/suggest] error:", e)
        return Response("ERR", status=503, mimetype="text/plain")


@app.route("/reviews/<int:review_id>")
def review_detail(review_id):
    rv = Review.query.get_or_404(review_id)
    return render_template("review_detail.html", rv=rv)


@app.route("/admin/reindex")
def admin_reindex():
    build_index()  # actually rebuild
    flash("Search index rebuilt.", "success")
    return redirect(url_for("index"))


@app.route("/admin/import_csv")
def admin_import_csv():
    """Import/merge items and reviews from the CSV."""
    if request.args.get("wipe") == "1":
        Review.query.delete()
        Item.query.delete()
        db.session.commit()
        print("[Import] Wiped items + reviews")

    added_items = load_items_from_csv()
    limit = request.args.get("limit", type=int)
    added_reviews = load_reviews_from_csv(limit=limit)

    # Rebuild index after import
    build_index()

    flash(f"Imported CSV · items +{added_items}, reviews +{added_reviews}.", "success")
    return redirect(url_for("index"))


if __name__ == "__main__":
    app.run(debug=True)