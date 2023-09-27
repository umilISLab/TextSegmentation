import os.path
from .utils import *
from transformers import AutoTokenizer, AutoModel
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
import pickle
from tqdm.auto import tqdm

gpu = torch.cuda.is_available()
spacy.prefer_gpu()

class SyntacticEmbedder():

    def __init__(self, spacy_model: str):

        self.nlp = spacy.load(spacy_model)

    def word_feat_extraction(self, s: str):
        docs = self.nlp.pipe(s)
        word_feats = [[dict(**word.morph.to_dict(), Ent = word.ent_type_) for word in sent if word.morph] for sent in docs]
        return word_feats

    def __call__(self, corpus: Iterable[str], features: Iterable[str]):
        '''

        :param corpus:
        :param features: Features to be extracted (the order is relevant).
        :return:
        '''

        if not isinstance(corpus, List):
            corpus = corpus.tolist()

        word_feats = self.word_feat_extraction(corpus)

        all_modes = []
        feat_counts = {}
        print("-- SYNTACTIC FEATURE EXTRACTION --")
        for feat in tqdm(features):
            # Extract the feature from each document in the corpus
            feat_lists = list(map(lambda L: [d[feat] for d in L if feat in d], word_feats))

            # Retain only string modes of the features (the others are likely NaN) and that are not equal to modes of previous features
            feat_modes = [mode for mode in list(set(sum(feat_lists, []))) if isinstance(mode, str) and mode not in all_modes]
            all_modes.extend(feat_modes)

            # Filter feature modes
            feat_lists = map(lambda L: list(filter(lambda s: s in feat_modes, L)), feat_lists)

            # Count feature modes
            feat_counts[feat] = list(map(Counter, feat_lists))

        print(f"[Syntactic Embedder] Found a total of {len(all_modes)} modes for features {features}")

        feat_df = pd.DataFrame(feat_counts)

        synt_dicts = feat_df.apply(lambda row: {k:v for d in row for k,v in d.items()}, axis = 1)

        output = pd.DataFrame(synt_dicts.tolist()).fillna(0).to_dict("records")

        return output


class TransformerEmbedder():

    def __init__(self, model_name: str, pooling: bool = True):

        self.device = "cuda" if gpu else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.transformer = AutoModel.from_pretrained(model_name).to(self.device)
        self.pooling = pooling

    def __call__(self, corpus: Iterable[str], return_dicts: bool = False, batch_size: int = 256):

        if not isinstance(corpus, List):
            corpus = corpus.tolist()

        X_sbert = []
        batch_gen = batch_generator(corpus, batch_size)
        for doc_batch in tqdm(batch_gen):
            X_batch = sentence_embeddings(doc_batch, self.tokenizer, self.transformer, pooling=self.pooling)
            X_sbert.extend(X_batch)
            torch.cuda.empty_cache()

        X_sbert = list(map(np.array, X_sbert))

        if return_dicts:
            output = list(map(lambda a: {"B_" + str(i): x_i for i, x_i in enumerate(a)}, X_sbert))
        else:
            output = X_sbert

        return output


class TfidfEmbedder():

    def __init__(self, lang: str, **kwargs):

        self.lang = lang
        self.sw = stopwords.words(lang)
        self.tfidf = TfidfVectorizer(stop_words = self.sw, **kwargs)

    def __call__(self, corpus: Iterable[str], return_dicts: bool = False):

        if not isinstance(corpus, List):
            corpus = corpus.tolist()

        X_tfidf, tfidf_features = additional_features(corpus, self.tfidf)
        X_tfidf = [x for x in X_tfidf.toarray()]

        if return_dicts:
            output = list(map(lambda a: {s:x_i for s,x_i in zip(tfidf_features, a)}, X_tfidf))
        else:
            output = X_tfidf

        return output


def get_embeddings(folder, corpus, all_features, pooling = False, sbert = "", spacy_model = "", tfidf_dim = 0, N_prior_features = 0, try_loading: bool = True):

    tot_N_features = N_prior_features
    folder = os.path.abspath(folder)

    if "synt" in all_features:
        synt_path = os.path.join(folder, "synt_vectors.pkl")
        all_features.append("synt")
        synt_features = ['Mood','Tense','Ent']
        print(f"[Embedding] Computing syntactic vectors with features {synt_features}, saving in {synt_path}.")
        try:
            assert try_loading
            with open(synt_path, "rb") as f:
                synt_dicts = pickle.load(f)
            print("\tSyntactic embeddings loaded from memory.")
        except:
            print("\tUnable to load syntactic embeddings from memory, computing them...")
            syntactic_embedder = SyntacticEmbedder(spacy_model)
            synt_dicts = syntactic_embedder(corpus,
                                            features = synt_features)
            with open(synt_path, "wb") as f:
                pickle.dump(synt_dicts, f)
            print("\tDone.")

        tot_N_features += len(synt_dicts[0])
    else:
        synt_dicts = []

    # Compute SBERT embeddings
    if "sbert" in all_features:
        all_features.append("sbert")
        pooling_flag = "mean" if pooling else "cls"
        name = sbert.split("/")[-1].replace("-","")
        sbert_path = os.path.join(folder, f"{name}_{pooling_flag}_vectors.pkl")
        print(f"[Embedding] Computing SBERT vectors with model {name} and pooling {pooling_flag}, saving in {sbert_path}.")
        try:
            assert try_loading
            with open(sbert_path, "rb") as f:
                sbert_dicts = pickle.load(f)
            print("\tSBERT embeddings loaded from memory.")
        except:
            print("\tUnable to load SBERT embeddings from memory, computing them...")
            sbert_embedder = TransformerEmbedder(model_name = sbert,
                                                     pooling = pooling)
            sbert_dicts = sbert_embedder(corpus,
                                         return_dicts = True)
            with open(sbert_path, "wb") as f:
                pickle.dump(sbert_dicts, f)
            print("\tDone.")

        tot_N_features += len(sbert_dicts[0])
    else:
        name = ""
        sbert_dicts = []

    # Compute TFIDF embeddings
    if "tfidf" in all_features:
        all_features.append("tfidf")
        tfidf_path = os.path.join(folder, "tfidf_vectors.pkl")
        print(f"[Embedding] Computing TFIDF vectors with dimension {tfidf_dim}, saving in {tfidf_path}.")
        try:
            assert try_loading
            with open(tfidf_path, "rb") as f:
                tfidf_dicts = pickle.load(f)
            print("\tTFIDF embeddings loaded from memory.")
        except:
            print("\tUnable to load TFIDF embeddings from memory, computing them...")
            tfidf_embedder = TfidfEmbedder(lang="italian", max_features = tfidf_dim, ngram_range = (1,3))
            tfidf_dicts = tfidf_embedder(corpus,
                                         return_dicts = True)
            with open(tfidf_path, "wb") as f:
                pickle.dump(tfidf_dicts, f)
            print("\tDone.")

        tot_N_features += len(tfidf_dicts[0])
    else:
        tfidf_dicts = []

    return {"synt": synt_dicts, "sbert": sbert_dicts, "tfidf": tfidf_dicts}, tot_N_features